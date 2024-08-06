from __future__ import absolute_import, division, print_function

import math
import json
import logging
import os
import random
import datetime
import subprocess
import time

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from .pretrained.bert.bert import BertModel
from .pretrained.bert.tokenization import BertTokenizer
from .pretrained.bert.optimization import BertAdam, WarmupLinearSchedule
from .pretrained.zen2 import ZenModel as zen
from .modules import KVMN, CRF


from ..utils.io_utils import load_json, save_json, read_tsv, read_embedding
from ..utils.ngram_utils import get_ngram2id
from ..eval.Seg_eval import eval_sentence, cws_evaluate_word_PRF, cws_evaluate_OOV, pred_result, sentence_handle
from ..utils.Web_MAP import Seg_PRETRAINED_MODEL_ARCHIVE_MAP


DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'max_ngram_length': 5,
    'use_bilstm': False,
    'lstm_layer_number': 1,
    'lstm_hidden_size': 200,
    'embedding_dim': 100,
    'use_bert': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_memory': False,
    'decoder': 'crf'
}


class DSeg(nn.Module):
    def __init__(self, word2id, gram2id, labelmap, hpara, model_path, cache_dir='./',
                 emb_word2id=None):
        super().__init__()
        self.spec = locals()

        self.word2id = word2id
        self.gram2id = gram2id
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap)
        self.max_seq_length = self.hpara['max_seq_length']
        self.max_ngram_size = self.hpara['max_ngram_size']
        self.max_ngram_length = self.hpara['max_ngram_length']

        self.bilstm = None
        self.embedding = None
        self.emb_word2id = None
        if emb_word2id is not None:
            Warning('Pretrained word embedding file is given. Will use the pretrained embedding at %s' % model_path)

        self.bert_tokenizer = None
        self.bert = None
        self.zen_tokenizer = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bilstm']:
            if model_path is not None:
                self.emb_word2id, weight = read_embedding(model_path)
                self.hpara['embedding_dim'] = weight.shape[1]
                self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight), padding_idx=0, freeze=False)

            else:
                self.emb_word2id = emb_word2id
                self.embedding = nn.Embedding(len(self.emb_word2id), self.hpara['embedding_dim'], padding_idx=0)

            self.bilstm = nn.LSTM(self.hpara['embedding_dim'], self.hpara['lstm_hidden_size'],
                                  num_layers=self.hpara['lstm_layer_number'], batch_first=True,
                                  bidirectional=True, dropout=0.33)
            self.dropout = nn.Dropout(0.33)
            hidden_size = self.hpara['lstm_hidden_size'] * 2
        elif self.hpara['use_bert']:
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=cache_dir)
            self.bert = BertModel.from_pretrained(model_path, cache_dir=cache_dir)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_zen']:
            self.zen_tokenizer = zen.BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=cache_dir)
            self.zen_ngram_dict = zen.ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = zen.modeling.ZenModel.from_pretrained(model_path, cache_dir=cache_dir)
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.hpara['use_memory']:
            self.kv_memory = KVMN(hidden_size, len(gram2id), len(labelmap))
        else:
            self.kv_memory = None
        #线性分类器
        self.classifier = nn.Linear(hidden_size, self.num_labels, bias=False)
        #设置decoder
        if self.hpara['decoder'] == 'crf':
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
            self.loss_function = CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, word_seq=None, label_value_matrix=None, word_mask=None,
                input_ngram_ids=None, ngram_position_matrix=None):
        '''前四个是常规bert需要的数据，
            word_seq，label_value_matrix，word_mask 是 kv_memory,
            input_ngram_ids,ngram_position_matrix是用来做ngram'''

        if self.bilstm is not None:
            embedding = self.embedding(input_ids)
            sequence_output, _ = self.bilstm(embedding)
        elif self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        if self.kv_memory is not None:
            kv_memory = self.kv_memory(word_seq, sequence_output, label_value_matrix, word_mask)
            sequence_output = torch.add(kv_memory, sequence_output)

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        #crf 作为decoder

        if labels is not None:
            if self.crf is not None:
                total_loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask_label)
                pre_labels = self.crf.decode(logits, attention_mask_label)[0]
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
                total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                pre_labels = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            return total_loss, pre_labels
        else:
            if self.crf is not None:
                pre_labels = self.crf.decode(logits, attention_mask_label)[0]
            else:
                logits = logits[attention_mask_label]
                pre_labels = torch.argmax(logits, dim=2)
            return pre_labels


    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['max_ngram_length'] = args.max_ngram_length

        hyper_parameters['use_bilstm'] = args.use_bilstm
        hyper_parameters['lstm_layer_number'] = args.lstm_layer_number
        hyper_parameters['lstm_hidden_size'] = args.lstm_hidden_size
        hyper_parameters['embedding_dim'] = args.embedding_dim

        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_memory'] = args.use_memory
        hyper_parameters['decoder'] = args.decoder
        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def load_model(cls, model_path, use_memory=False, chemed=False, local_rank=-1, no_cuda=False):
        # assign model path
        model_path = (model_path, use_memory, chemed)
        # select the device
        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

        label_map = load_json(os.path.join(model_path, 'label_map.json'))
        hpara = load_json(os.path.join(model_path, 'hpara.json'))

        gram2id_path = os.path.join(model_path, 'gram2id.json')
        gram2id = load_json(gram2id_path) if os.path.exists(gram2id_path) else None
        if gram2id is not None:
            gram2id = {tuple(k.split('`')): v for k, v in gram2id.items()}

        word2id_path = os.path.join(model_path, 'word2id.json')
        word2id = load_json(word2id_path) if os.path.exists(word2id_path) else None

        emb_word2id_path = os.path.join(model_path, 'emb_word2id.json')
        emb_word2id = load_json(emb_word2id_path) if os.path.exists(emb_word2id_path) else None
        if emb_word2id:
            res = cls(model_path=None, labelmap=label_map, hpara=hpara,
                      gram2id=gram2id, word2id=word2id, emb_word2id=emb_word2id)
        else:
            res = cls(model_path=model_path, labelmap=label_map, hpara=hpara,
                  gram2id=gram2id, word2id=word2id, emb_word2id=emb_word2id)

        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        cls.device = device
        cls.n_gpu = n_gpu
        res.to(device)
        return res

    def save_model(self, output_dir, vocab_dir=None):
        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        label_map_file = os.path.join(output_dir, 'label_map.json')

        if not os.path.exists(label_map_file):
            save_json(label_map_file, self.labelmap)

            save_json(os.path.join(output_dir, 'hpara.json'), self.hpara)
            if self.gram2id is not None:
                gram2save = {'`'.join(list(k)): v for k, v in self.gram2id.items()}
                save_json(os.path.join(output_dir, 'gram2id.json'), gram2save)
            if self.word2id is not None:
                save_json(os.path.join(output_dir, 'word2id.json'), self.word2id)

            if self.bert or self.zen:
                output_config_file = os.path.join(output_dir, 'config.json')
                with open(output_config_file, "w", encoding='utf-8') as writer:
                    if self.bert:
                        writer.write(self.bert.config.to_json_string())
                    elif self.zen:
                        writer.write(self.zen.config.to_json_string())
                    else:
                        raise ValueError()
                output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
                command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
                subprocess.run(command, shell=True)


                vocab_name = 'vocab.txt'

                vocab_path = os.path.join(vocab_dir, vocab_name)
                command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
                subprocess.run(command, shell=True)

                if self.zen:
                    ngram_name = 'ngram.txt'
                    ngram_path = os.path.join(vocab_dir, ngram_name)
                    command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
                    subprocess.run(command, shell=True)
            elif self.bilstm:
                save_json(os.path.join(output_dir, 'emb_word2id.json'), self.emb_word2id)

    def load_data(self, data_path=None, sentence_list=None):
        if data_path is not None:
            '''找字符串中的数据类型'''
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
            sentence_list, label_list = read_tsv(data_path)
        elif sentence_list is not None:
            flag = 'predict'
            label_list = [['I'] * len(sentence) for sentence in sentence_list]
        else:
            raise ValueError()

        #加标注N-gram的分词结果
        data = []
        for sentence, label in zip(sentence_list, label_list):
            if self.kv_memory is not None:
                word_list = []
                matching_position = []
                # 从局子的第一个此开始循环
                for i in range(len(sentence)):
                    for j in range(self.max_ngram_length):  # 从单个字开始查找，最多到max_ngram_length个的ngram
                        if i + j > len(sentence):
                            break
                        word = ''.join(sentence[i: i + j + 1])
                        if word in self.gram2id:  # 如果ngram在已有的字典里
                            try:
                                index = word_list.index(word)  # 记录它在word_list中的初始位置
                            except ValueError:  # 否则把它添加到word_list中去
                                word_list.append(word)
                                index = len(word_list) - 1
                            word_len = len(word)
                            for k in range(j + 1):  # i+k表示某一个词，每个字在句子中的位置，index 记录在wordlist中的位置，l表示分词结果
                                if word_len == 1:
                                    l = 'S'
                                elif k == 0:
                                    l = 'B'
                                elif k == j:
                                    l = 'E'
                                else:
                                    l = 'I'
                                matching_position.append((i + k, index, l))
            else:
                word_list = None
                matching_position = None
            data.append((sentence, label, word_list, matching_position))
        # 返回一个InputExample对象list
        examples = []
        for i, (sentence, label, word_list, matching_position) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            if word_list is not None:
                word = ' '.join(word_list)
            else:
                word = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                                         label=label, word=word, matrix=matching_position))
        return examples

    def convert_examples_to_features(self, examples):
        # 指ngram
        max_seq_length = min(int(max([len(e.text_a.split(' ')) for e in examples]) * 1.1 + 2), self.max_seq_length)

        if self.kv_memory is not None:
            max_word_size = max(min(max([len(e.word.split(' ')) for e in examples]), self.max_ngram_size), 1)
        else:
            max_word_size = None

        features = []

        if self.bert or self.zen:
            tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer
        else:
            tokenizer = None

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []

            for i, word in enumerate(textlist):
                if tokenizer:
                    token = tokenizer.tokenize(word)
                else:
                    token = [word]
                tokens.extend(token)
                label_1 = labellist[i]

                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            # 留出CLS和SEP的位置
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []
            # 添加CLS
            ntokens.append("[CLS]")
            segment_ids.append(0)
            # CLS的标签
            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])
            # 把token好的结果放到加了CLS的list中
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.labelmap[labels[i]])  # 将label转成数字
            # 加SEP
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)  # label_mask1表示有效的,0表示无效
            label_ids.append(self.labelmap["[SEP]"])
            # tokens，中文即汉字，英文即wordpiece，转为对应的id
            if tokenizer:
                input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            else:
                input_ids = []
                for t in ntokens:
                    t_id = self.emb_word2id[t] if t in self.emb_word2id else self.emb_word2id['<UNK>']
                    input_ids.append(t_id)
            input_mask = [1] * len(input_ids)  # 1表示有效的值
            label_mask = [1] * len(label_ids)  # 1表示有效的值
            # padding
            while len(input_ids) < max_seq_length:
                input_ids.append(0)  # 0表示无效的值
                input_mask.append(0)  # 0表示无效的值
                segment_ids.append(0)  # 为什么padding时和文本都用0
                label_ids.append(0)
                valid.append(1)  # 为什么padding时和文本都用1
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length  # assert 的用法？
            # memory net需要的东西，一个矩阵，行是一句话，列放的ngram
            if self.kv_memory is not None:
                wordlist = example.word
                wordlist = wordlist.split(' ') if len(wordlist) > 0 else []  # 将文分按 ' 'split
                matching_position = example.matrix  # 即 (21, 2, 'S') 组成的list
                word_ids = []
                matching_matrix = np.zeros((max_seq_length, max_word_size),
                                           dtype=np.int)  # 生成以句子长度为行，ngram的总词数为列（max_word_size）
                '''某一个字在第几个词中充当的成分'''
                if len(wordlist) > max_word_size:
                    wordlist = wordlist[:max_word_size]
                for word in wordlist:
                    try:
                        word_ids.append(self.gram2id[word])  # 将词的id传入进去
                    except KeyError:
                        print(word)
                        print(wordlist)
                        print(textlist)
                        raise KeyError()
                while len(word_ids) < max_word_size:
                    word_ids.append(0)  # 相当于padding
                for position in matching_position:
                    char_p = position[0] + 1  # '''加1是给CLS留位置？'''
                    word_p = position[1]  # 在wordlist中的位置
                    if char_p > max_seq_length - 2 or word_p > max_word_size - 1:
                        # 句子特别长或者词在被wordlist[:max_word_size]删掉的那部分，就不管
                        continue
                    else:
                        matching_matrix[char_p][word_p] = self.labelmap[
                            position[2]]  # 一句话中的这个字，在几号词里充当的成分的 label对应的id
                        # 标上一句话中，所有的ngram（给定了最多ngram的个数）。
                assert len(word_ids) == max_word_size
            else:
                word_ids = None
                matching_matrix = None

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])  # p表示几个字的ngram，q表示起始位置

                # random.shuffle(ngram_matches)
                ngram_matches = sorted(ngram_matches, key=lambda s: s[0])  # 按ngram_index进行排序

                max_ngram_in_seq_proportion = math.ceil(
                    (len(tokens) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)  # 比较短的句子，对应的ngram比较少
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]  # 限制ngram的数量

                ngram_ids = [ngram[0] for ngram in ngram_matches]  # ngram在list中的编号
                ngram_positions = [ngram[1] for ngram in ngram_matches]  # ngram 在词中的起始位置
                ngram_lengths = [ngram[2] for ngram in ngram_matches]  # ngram 的长度
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1  # 将超过最大限制的词mask

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                  dtype=np.int32)
                for i in range(len(ngram_ids)):  # 列表示ngram的每一个词，行表示每一个词在句子中的位置
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              matching_matrix=matching_matrix,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array
                              ))
        return features

    def feature2input(self, device, feature):
        # bert需要的标准数据
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        # 到cuda
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        if self.hpara['use_memory']:
            all_word_ids = torch.tensor([f.word_ids for f in feature], dtype=torch.long)
            all_matching_matrix = torch.tensor([f.matching_matrix for f in feature], dtype=torch.long)
            all_word_mask = torch.tensor([f.matching_matrix for f in feature], dtype=torch.float)

            word_ids = all_word_ids.to(device)
            matching_matrix = all_matching_matrix.to(device)
            word_mask = all_word_mask.to(device)
        else:
            word_ids = None
            matching_matrix = None
            word_mask = None
        if self.hpara['use_zen']:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, segment_ids, valid_ids, word_ids, word_mask

    @classmethod
    def fit(cls, args):
        #creat logs files
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        # Event log
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_file_name = './logs/log-' + now_time
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            filename=log_file_name,
                            filemode='w',
                            level=logging.INFO)
        logger = logging.getLogger(__name__)
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
        logger = logging.getLogger(__name__)

        logger.info(vars(args))

        if args.server_ip and args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        #保存模型的目录
        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')
        #需要指定保存模型的名字
        if args.model_name is None:
            raise Warning('model name is not specified, the model will NOT be saved!')
        output_model_dir = os.path.join('./saved_models', args.model_name + '_' + now_time)

        word2id = get_word2id(args.train_data_path )
        logger.info('# of word in train: %d: ' % len(word2id))
        #使用memory网络，ngram 需要>1
        if args.use_memory:
            if args.ngram_num_threshold <= 1:
                raise Warning('The threshold of n-gram frequency is set to %d. '
                              'No n-grams will be filtered out by frequency. '
                              'We only filter out n-grams whose frequency is lower than that threshold!'
                              % args.ngram_num_threshold)
            train_sentences, _ = read_tsv(args.train_data_path )
            eval_sentences, _ = read_tsv(args.dev_data_path )
            all_sentence = train_sentences + eval_sentences
            gram2id, _ = get_ngram2id(all_sentence, args.max_ngram_length,
                                      args.ngram_num_threshold, args.ngram_type, args.av_threshold)
            logger.info('# of n-gram in memory: %d' % len(gram2id))
        else:
            gram2id = None

        if args.use_bilstm and args.model_path is None:
            emb_word2id = get_character2id(args.train_data_path)
        else:
            emb_word2id = None

        label_list = get_labels(args.train_data_path )
        label_map = {label: i for i, label in enumerate(label_list,0)}
        #载入超参数
        hpara = cls.init_hyper_parameters(args)
        # 初始化模型
        seg_model = cls(word2id, gram2id, label_map, hpara, args.model_path, cache_dir=args.cache_dir,
                        emb_word2id=emb_word2id)
        #初始化数据
        train_examples = seg_model.load_data(data_path=args.train_data_path)
        dev_examples = seg_model.load_data(data_path=args.dev_data_path)
        test_examples = seg_model.load_data(data_path=args.test_data_path)

        all_eval_examples = {
            'dev': dev_examples,
            'test': test_examples
        }
        num_labels = seg_model.num_labels
        convert_examples_to_features = seg_model.convert_examples_to_features
        feature2input = seg_model.feature2input

        label_map = {i: label for i, label in enumerate(label_list, 0)}

        #模型总参数量
        total_params = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
        logger.info('# of trainable parameters: %d' % total_params)
        #
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.fp16:
            seg_model.half()
        seg_model.to(device)

        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            seg_model = DDP(seg_model)
        elif n_gpu > 1:
            seg_model = torch.nn.DataParallel(seg_model)
        '''learning rate decay'''
        param_optimizer = list(seg_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.use_bilstm:
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
        elif args.use_bert or args.use_zen:
            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=args.learning_rate,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
                warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                     t_total=num_train_optimization_steps)

            else:
                # num_train_optimization_steps=-1
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
        else:
            ValueError()

        best_epoch = -1
        history = {'epoch': [],
                   'dev': {'cws_p': [], 'cws_r': [], 'cws_f': [], 'cws_oov': []},
                   'test': {'cws_p': [], 'cws_r': [], 'cws_f': [], 'cws_oov': []}
                   }
        best = {'dev': {'best_epoch': 0, 'best_p': -1, 'best_r': -1, 'best_f': -1, 'best_oov': -1},
                'test': {'best_epoch': 0, 'best_p': -1, 'best_r': -1, 'best_f': -1, 'best_oov': -1},
                }

        num_of_no_improvement = 0
        patient = args.patient

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            seg_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                seg_model.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples)
                input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                segment_ids, valid_ids, word_ids, word_mask = feature2input(device, train_features)

                loss, _ = seg_model.forward(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                    word_ids,matching_matrix, word_mask,
                                    ngram_ids, ngram_positions)
                if np.isnan(loss.to('cpu').detach().numpy()):
                    raise ValueError('loss is nan!')
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                         # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            seg_model.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                seg_model.eval()
                improved = False
                y_true = {'dev': [], 'test': []}
                y_pred = {'dev': [], 'test': []}
                for flag in ['dev', 'test']:
                    eval_examples = all_eval_examples[flag]
                    for start_index in range(0, len(eval_examples), args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                             len(eval_examples))]
                        eval_features = convert_examples_to_features(eval_batch_examples)

                        input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                        segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

                        with torch.no_grad():
                            _, tag_seq = seg_model.forward(input_ids, segment_ids, input_mask, labels=label_ids,
                                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                                       word_seq=word_ids, label_value_matrix=matching_matrix,
                                                       word_mask=word_mask,
                                                       input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                        logits = tag_seq.to('cpu').numpy()
                        label_ids = label_ids.to('cpu').numpy()

                        for i, label in enumerate(label_ids):
                            temp_1 = []
                            temp_2 = []
                            for j, m in enumerate(label):
                                if j == 0:
                                    continue
                                elif label_ids[i][j] == num_labels - 1:
                                    y_true[flag].append(temp_1)
                                    y_pred[flag].append(temp_2)
                                    break
                                else:
                                    temp_1.append(label_map[label_ids[i][j]])
                                    temp_2.append(label_map[logits[i][j]])

                    y_true_all = []
                    y_pred_all = []
                    sentence_all = []
                    for y_true_item in y_true[flag]:
                        y_true_all += y_true_item
                    for y_pred_item in y_pred[flag]:
                        y_pred_all += y_pred_item
                    for example, y_true_item in zip(all_eval_examples[flag], y_true[flag]):
                        sen = example.text_a
                        sen = sen.strip()
                        sen = sen.split(' ')
                        if len(y_true_item) != len(sen):
                            sen = sen[:len(y_true_item)]
                        sentence_all.append(sen)
                    cws_p, cws_r, cws_f = cws_evaluate_word_PRF(y_pred_all, y_true_all)
                    cws_oov = cws_evaluate_OOV(y_pred[flag], y_true[flag], sentence_all, word2id)

                    history['epoch'].append(epoch)
                    history[flag]['cws_p'].append(cws_p)
                    history[flag]['cws_r'].append(cws_r)
                    history[flag]['cws_f'].append(cws_f)
                    history[flag]['cws_oov'].append(cws_oov)
                    logger.info("=======entity level {}========".format(flag))
                    logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f", epoch + 1, cws_p, cws_r, cws_f, cws_oov)
                    logger.info("=======entity level {}========".format(flag))


                    if args.model_name is not None:
                        if not os.path.exists(output_model_dir):
                            os.mkdir(output_model_dir)


                    if flag == 'dev':
                        if history['dev']['cws_f'][epoch] > best['dev']['best_f']:
                            best_epoch = epoch + 1
                            num_of_no_improvement = 0
                            improved = True
                        else:
                            num_of_no_improvement += 1
                            improved = False

                if improved:
                    for flag in ['dev', 'test']:
                        best[flag]['best_p'] = history[flag]['cws_p'][epoch]
                        best[flag]['best_r'] = history[flag]['cws_r'][epoch]
                        best[flag]['best_f'] = history[flag]['cws_f'][epoch]
                        best[flag]['best_oov'] = history[flag]['cws_oov'][epoch]

                        if args.model_name:
                            with open(os.path.join(output_model_dir, 'CWS_result.%s.txt' % flag), "w") as writer:
                                writer.write("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f\n\n" %
                                             (epoch + 1, best[flag]['best_p'], best[flag]['best_r'],
                                              best[flag]['best_f'], best[flag]['best_oov']))
                                for i in range(len(y_pred[flag])):
                                    sentence = all_eval_examples[flag][i].text_a
                                    seg_true_str, seg_pred_str = eval_sentence(y_pred[flag][i], y_true[flag][i],
                                                                               sentence,
                                                                               word2id)
                                    writer.write('True: %s\n' % seg_true_str)
                                    writer.write('Pred: %s\n\n' % seg_pred_str)

                            save_model_dir = os.path.join(output_model_dir, 'model')
                            if not os.path.exists(save_model_dir):
                                os.makedirs(save_model_dir)

                            if args.model_path == None:
                                seg_model.save_model(save_model_dir)
                            elif '/' in args.model_path:
                                seg_model.save_model(save_model_dir, args.model_path)
                            elif '-' in args.model_path:
                                seg_model.save_model(save_model_dir, args.cache_dir)
                            else:
                                raise ValueError()


            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        for flag in ['dev', 'test']:
            logger.info("\n=======best %s f entity level========" % flag)
            logger.info("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f",
                        best_epoch, best[flag]['best_p'], best[flag]['best_r'],
                        best[flag]['best_f'], best[flag]['best_oov'])
            logger.info("\n=======best %s f entity level========" % flag)

        if os.path.exists(output_model_dir):
            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')

    def predict(self, sentence_list, eval_batch_size=16, seperated_type='list'):
        start_time = time.time()

        max_length = max(map(len, sentence_list))

        sentence_list, space_index_list = space_handle(sentence_list)

        if max_length > self.max_seq_length * 0.6:
            sentence_list = sentence_handle(sentence_list, self.max_seq_length)

        eval_examples = self.load_data(sentence_list=sentence_list, data_path=None)
        label_map = {v: k for k, v in self.labelmap.items()}

        self.eval()
        y_pred = []

        for start_index in trange(0, len(eval_examples), eval_batch_size):
            eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                 len(eval_examples))]
            eval_features = self.convert_examples_to_features(eval_batch_examples)

            input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
            segment_ids, valid_ids, word_ids, word_mask = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                _, tag_seq = self.forward(input_ids, segment_ids, input_mask, labels=label_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       word_seq=word_ids, label_value_matrix=matching_matrix,
                                       word_mask=word_mask,
                                       input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

            logits = tag_seq.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif len(eval_batch_examples[i].text_a.split()) < j:
                        y_pred.append(temp)
                        break
                    else:
                        temp.append(label_map[logits[i][j]])

        pred_list = []
        for pre_label, sentence, space_index in zip(y_pred, sentence_list, space_index_list):
            pred_list.append(pred_result(pre_label, sentence, space_index, seperated_type=seperated_type))
        print(time.time()-start_time, len(eval_examples))
        return pred_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None, matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.matrix = matrix


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readsentence(filename):
    data = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list = ['S' for _ in range(len(line))]
            data.append((line, label_list))
    return data


def get_labels(train_path):
    label_list = ['<PAD>', '<UNK>']

    with open(train_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            joint_label = splits[1]
            if joint_label not in label_list:
                label_list.append(joint_label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def get_word2id(train_data_path):
    word2id = {'<PAD>': 0}
    word = ''
    index = 1

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        splits = line.split()
        character = splits[0]
        label = splits[1]
        word += character
        if label in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    return word2id

def get_character2id(train_data_path):
    word2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}
    index = 4

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        splits = line.split()
        character = splits[0]
        if character not in word2id:
            word2id[character] = index
            index += 1
    return word2id


def cached_DNLP(model_path, use_memory, chemed):
    logger = logging.getLogger(__name__)
    task = 'Base'
    if chemed:
        task = 'chemed'
    if use_memory:
        task += '_KVMN'
    if os.path.exists(model_path):
        return model_path
    elif model_path in Seg_PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_web, model_name = Seg_PRETRAINED_MODEL_ARCHIVE_MAP[model_path][task]
        #
        model_path = os.path.join(os.getcwd(), model_name)
        #
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            try:
                os.system('wget -r -np -nd -R index.html -nH -P {} {}'.format(model_path, archive_web))
            except:
                os.system('rm -r {}'.format(model_path))
                logger.error(
                    "Automatic model download failed due to network instability. "
                    "Please try again or download the model manually.")
                return None
        return model_path
    else:
        logger.error(
            "Model name '{}' was not found in model name list ({}). "
            "We assumed '{}' was a path or url but couldn't find any file "
            "associated to this path or url.".format(
                model_path,
                ', '.join(Seg_PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                model_path))
        return None

def space_handle(sentence_list):
    space_index = []
    new_sentence_list = []
    for s in sentence_list:

        index_list = [i for i, c in enumerate(s) if c == ' ']
        while ' ' in s:
            s.remove(' ')
        new_sentence_list.append(s)
        space_index.append(index_list)
    return new_sentence_list, space_index
