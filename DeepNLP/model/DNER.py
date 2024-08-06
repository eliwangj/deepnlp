from __future__ import absolute_import, division, print_function

import math
import os
import re
import numpy as np
import torch
import subprocess
from torch import nn
import json
import logging
import random
import datetime
from tqdm import tqdm, trange

from .pretrained.bert import BertModel, BertTokenizer, BertAdam, WarmupLinearSchedule
from .pretrained.xlnet import XLNetModel, XLNetTokenizer
from .pretrained.zen2 import ZenModel, ZenNgramDict

from ..eval.NER_eval import eval_sentence, evaluate_word_PRF, pred_result
from ..utils.io_utils import load_json, save_json, read_tsv, read_embedding, get_language
from ..utils.ngram_utils import get_ngram2id


from .modules.crf import CRF
from seqeval.metrics import classification_report
from torch.nn import CrossEntropyLoss
from ..utils.Web_MAP import NER_PRETRAINED_MODEL_ARCHIVE_MAP
DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'use_bert': False,
    'use_zen': False,
    'use_xlnet': False,
    'do_lower_case': False,
    'use_memory': False,
    'cat_type': 'length',
    'cat_num': 10,
    'max_ngram_length': 10,
    'decoder': 'crf',
    'use_bilstm': False,
    'lstm_layer_number': 1,
    'lstm_hidden_size': 200,
    'embedding_dim': 100,
}


class DNER(nn.Module):

    def __init__(self, NE2id, gram2id, gram2count, labelmap, hpara, model_path, cache_dir='./', emb_NE2id=None):
        super().__init__()

        self.NE2id = NE2id
        self.gram2id = gram2id
        self.gram2count = gram2count
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap)
        self.max_seq_length = self.hpara['max_seq_length']
        self.use_memory = self.hpara['use_memory']
        self.cat_type = self.hpara['cat_type']
        self.cat_num = self.hpara['cat_num']
        self.max_ngram_length = self.hpara['max_ngram_length']
        self.cache_dir = cache_dir
        self.max_ngram_size = self.hpara['max_ngram_size']
        if self.cat_type == 'length':
            pass
            # assert self.cat_num == self.max_ngram_length


        self.bilstm = None
        self.embedding = None
        self.emb_NE2id = None
        if emb_NE2id is not None and model_path is not None:
            Warning('Pretrained word embedding file is given. Will use the pretrained embedding at %s' % model_path)

        self.tokenizer = None
        self.bert = None
        self.zen = None
        self.xlnet = None
        self.zen_ngram_dict = None

        if self.hpara['use_bilstm']:
            if model_path is not None:
                self.emb_NE2id, weight = read_embedding(model_path)
                self.hpara['embedding_dim'] = weight.shape[1]
                self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight), padding_idx=0, freeze=False)
            else:
                self.emb_NE2id = emb_NE2id
                self.embedding = nn.Embedding(len(self.emb_NE2id), self.hpara['embedding_dim'], padding_idx=0)

            self.bilstm = nn.LSTM(self.hpara['embedding_dim'], self.hpara['lstm_hidden_size'],
                                  num_layers=self.hpara['lstm_layer_number'], batch_first=True,
                                  bidirectional=True, dropout=0.33)
            self.dropout = nn.Dropout(0.33)
            hidden_size = self.hpara['lstm_hidden_size'] * 2
        elif self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=self.cache_dir)
            self.bert = BertModel.from_pretrained(model_path, cache_dir=self.cache_dir)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_zen']:
            self.tokenizer = ZenModel.BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=self.cache_dir)
            self.zen_ngram_dict = ZenModel.ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.modeling.ZenModel.from_pretrained(model_path, cache_dir=self.cache_dir)
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
            self.xlnet = XLNetModel.from_pretrained(model_path)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)
            self.xlnet.resize_token_embeddings(len(self.tokenizer))
        else:
            raise ValueError()
        # if self.use_memory:
        #     self.multi_attention = MultiChannelAttention(len(self.gram2id), hidden_size, self.cat_num)
        #     self.classifier = nn.Linear(hidden_size * (1 + self.cat_num), self.num_labels, bias=False)
        # else:

        self.multi_attention = None
        self.classifier = nn.Linear(hidden_size, self.num_labels, bias=False)

        if self.hpara['decoder'] == 'crf':
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
            self.loss_function = CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None,
                word_seq=None, label_value_matrix=None, word_mask=None, channel_ids=None,
                input_ngram_ids=None, ngram_position_matrix=None):

        if self.bilstm is not None:
            embedding = self.embedding(input_ids)
            sequence_output, _ = self.bilstm(embedding)
        elif self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.xlnet is not None:
            transformer_outputs = self.xlnet(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()
        #获得attention sequence_output -- (batch_size, character_seq_len, hidden_size)

        # if self.multi_attention is not None:
        #     attention_output = self.multi_attention(word_seq, sequence_output, word_mask, channel_ids)
        #     sequence_output = torch.cat([sequence_output, attention_output], dim=2)  #(batch_size, character_seq_len, (num_cat+1) * hidden_size)

        sequence_output = self.dropout(sequence_output)

        batch_size, _, feat_dim = sequence_output.shape
        max_len = attention_mask_label.shape[1]
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            sent_len = attention_mask_label[i].sum()
            valid_output[i][:sent_len] = temp[:sent_len]

        logits = self.classifier(valid_output)

        if labels is not None:
            if self.crf is not None:
                total_loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask_label)
                pre_labels = self.crf.decode(logits, attention_mask_label)[0]
            else:
                logits = logits[attention_mask_label]
                labels = labels[attention_mask_label]
                total_loss = self.loss_function(logits, labels)
                pre_labels = torch.argmax(logits, dim=2)
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
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_memory'] = args.use_memory
        hyper_parameters['cat_type'] = args.cat_type
        hyper_parameters['cat_num'] = args.cat_num
        hyper_parameters['max_ngram_length'] = args.max_ngram_length

        hyper_parameters['use_bilstm'] = args.use_bilstm
        hyper_parameters['lstm_layer_number'] = args.lstm_layer_number
        hyper_parameters['lstm_hidden_size'] = args.lstm_hidden_size
        hyper_parameters['embedding_dim'] = args.embedding_dim

        return hyper_parameters

    @classmethod
    def load_model(cls, model_path, chemed=False, language='en', local_rank=-1, no_cuda=False):
        if chemed:
            dataset = 'chemed'
        elif language == 'en':
            dataset = 'WN16'
        elif language == 'zh':
            dataset = 'RE'
        # assign model path
        model_path = cached_DNLP(model_path, dataset)
        # select the device
        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device( local_rank)
            device = torch.device("cuda",  local_rank)
            n_gpu = 1

        label_map = load_json(os.path.join(model_path, 'label_map.json'))
        hpara = load_json(os.path.join(model_path, 'hpara.json'))

        gram2id_path = os.path.join(model_path, 'gram2id.json')
        gram2id = load_json(gram2id_path) if os.path.exists(gram2id_path) else None
        if gram2id is not None:
            gram2id = {tuple(k.split('`')): v for k, v in gram2id.items()}

        NE2id_path = os.path.join(model_path, 'NE2id.json')
        NE2id = load_json(NE2id_path) if os.path.exists(NE2id_path) else None

        gram2count_path = os.path.join(model_path, 'gram2count.json')
        gram2count = load_json(gram2count_path) if os.path.exists(gram2count_path) else None

        emb_NE2id_path = os.path.join(model_path, 'emb_NE2id.json')
        emb_NE2id = load_json(emb_NE2id_path) if os.path.exists(emb_NE2id_path) else None
        if emb_NE2id:
            res = cls(model_path=None, labelmap=label_map, hpara=hpara,
                  gram2id=gram2id, NE2id=NE2id, gram2count=gram2count, emb_NE2id=emb_NE2id)
        else:
            res = cls(model_path=model_path, labelmap=label_map, hpara=hpara,
                      gram2id=gram2id, NE2id=NE2id, gram2count=gram2count, emb_NE2id=emb_NE2id)

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
        if self.NE2id is not None:
            save_json(os.path.join(output_dir, 'NE2id.json'), self.NE2id)
        if self.gram2count is not None:
            save_json(os.path.join(output_dir, 'gram2count.json'), self.gram2count)

        if self.bert or self.zen or self.xlnet:
            output_config_file = os.path.join(output_dir, 'config.json')
            with open(output_config_file, "w", encoding='utf-8') as writer:
                if self.bert:
                    writer.write(self.bert.config.to_json_string())
                elif self.zen:
                    writer.write(self.zen.config.to_json_string())
                elif self.xlnet:
                    writer.write(self.xlnet.config.to_json_string())
                else:
                    raise ValueError()
            output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
            command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
            subprocess.run(command, shell=True)

            if self.bert:
                vocab_name = 'vocab.txt'
            elif self.xlnet:
                vocab_name = 'spiece.model'
            elif self.zen:
                vocab_name = 'vocab.txt'
            else:
                raise ValueError()
            vocab_path = os.path.join(vocab_dir, vocab_name)
            command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
            subprocess.run(command, shell=True)

            if self.zen:
                ngram_name = 'ngram.txt'
                ngram_path = os.path.join(vocab_dir, ngram_name)
                command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
                subprocess.run(command, shell=True)
        elif self.bilstm:
            save_json(os.path.join(output_dir, 'emb_NE2id.json'), self.emb_NE2id)

    def load_data(self, data_path=None, sentence_list=None):
        if data_path is not None:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
            sentence_list, label_list = read_tsv(data_path)
        elif sentence_list is not None:
            label_list = [[0]*len(sentence) for sentence in sentence_list]
            flag = 'predict'
        else:
            raise ValueError()

        data = []
        for sentence, label in zip(sentence_list, label_list):
            if self.multi_attention is not None:
                ngram_list = []
                matching_position = []
                ngram_list_len = []
                for i in range(self.cat_num):
                    ngram_list.append([])
                    matching_position.append([])
                    ngram_list_len.append(0)
                for i in range(len(sentence)):
                    for j in range(0, self.max_ngram_length):
                        if i + j + 1 > len(sentence):
                            break
                        ngram = ''.join(sentence[i: i + j + 1])
                        if ngram in self.gram2id:
                            channel_index = self._ngram_category(ngram)  #确定在哪个 频率/长度 的第几类里
                            try:
                                index = ngram_list[channel_index].index(ngram)
                            except ValueError:
                                ngram_list[channel_index].append(ngram)
                                index = len(ngram_list[channel_index]) - 1
                                ngram_list_len[channel_index] += 1
                            for k in range(j + 1):
                                matching_position[channel_index].append((i + k, index)) #在某一个channel 中的index 和在句子中所处的位置
            else:
                ngram_list = None
                matching_position = None
                ngram_list_len = None

            max_ngram_len = max(ngram_list_len) if ngram_list_len is not None else None
            data.append((sentence, label, ngram_list, matching_position, max_ngram_len))

        examples = []
        for i, (sentence, label, word_list, matching_position, word_list_len) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = word_list
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word, matrix=matching_position,
                             sent_len=len(sentence), word_list_len=word_list_len))
        return examples

    def _ngram_category(self, ngram):
        if self.cat_type == 'length':
            index = int(min(self.cat_num, len(ngram))) - 1
            assert 0 <= index < self.cat_num
            return index
        elif self.cat_type == 'freq':
            index = int(min(self.cat_num, math.log2(self.gram2count[ngram]))) - 1
            assert 0 <= index < self.cat_num
            return index
        else:
            raise ValueError()

    def convert_examples_to_features(self, examples, language):
        if self.multi_attention is not None:
            max_word_size = max(max([e.word_list_len for e in examples]), 1)
        else:
            max_word_size = 1

        features = []

        tokenizer = self.tokenizer

        length_list = []
        tokens_list = []
        labels_list = []
        valid_list = []
        label_mask_list = []

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []

            if len(textlist) > self.max_seq_length - 2:
                textlist = textlist[:self.max_seq_length - 2]
                labellist = labellist[:self.max_seq_length - 2]

            for i, word in enumerate(textlist):
                if tokenizer:
                    token = tokenizer.tokenize(word)
                elif word in self.emb_NE2id:
                        token = [word]
                else:
                    if language == 'zh':
                        token = list(word)
                    elif language == 'en':
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
            length_list.append(len(tokens))
            tokens_list.append(tokens)
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        label_pad_length = max(label_len_list) + 2

        for indx, (example, tokens, labels, valid, label_mask) in \
                enumerate(zip(examples, tokens_list, labels_list, valid_list, label_mask_list)):

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])

            ntokens.append("[SEP]")
            segment_ids.append(0)

            label_ids.append(self.labelmap["[SEP]"])
            valid.append(1)
            label_mask.append(1)


            if tokenizer:
                input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            else:
                input_ids = []
                for t in ntokens:
                    t_id = self.emb_NE2id[t] if t in self.emb_NE2id else self.emb_NE2id['<UNK>']
                    input_ids.append(t_id)

            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                label_ids.append(0)
                label_mask.append(0)


            # ignore all punctuation if not specified


            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(label_mask) == label_pad_length


            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                max_gram_n = self.zen_ngram_dict.max_ngram_len

                for p in range(2, max_gram_n):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment,
                                                  self.zen_ngram_dict.ngram_to_freq_dict[character_segment]])

                ngram_matches = sorted(ngram_matches, key=lambda s: s[-1], reverse=True)

                max_ngram_in_seq_proportion = math.ceil((len(tokens) / self.max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(seq_pad_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
                for i in range(len(ngram_ids)):
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
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        if self.multi_attention is not None:
            all_word_ids = torch.tensor([f.word_ids for f in feature], dtype=torch.long)
            all_matching_matrix = torch.tensor([f.matching_matrix for f in feature], dtype=torch.long)
            all_word_mask = torch.tensor([f.matching_matrix for f in feature], dtype=torch.float)
            all_channel_ids = torch.tensor([f.channel_ids for f in feature], dtype=torch.long)

            word_ids = all_word_ids.to(device)
            matching_matrix = all_matching_matrix.to(device)
            word_mask = all_word_mask.to(device)
            channel_ids = all_channel_ids.to(device)
        else:
            word_ids = None
            matching_matrix = None
            word_mask = None
            channel_ids = None
        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, segment_ids, valid_ids, word_ids, word_mask

    @classmethod
    def fit(cls, args):

        if not os.path.exists('./logs'):
            os.makedirs('./logs')

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

        if args.use_memory and args.cat_type == 'length':
            if not args.cat_num == args.max_ngram_length:
                num = min(args.cat_num, args.max_ngram_length)
                logger.info('cat_num (%d) and max_ngram_length (%d) are not equal. Set them to %d' %
                            (args.cat_num, args.max_ngram_length, num))
                args.cat_num = num
                args.max_ngram_length = num

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
            n_gpu = torch.cuda.device_count()
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            # torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')

        if args.model_name is None:
            raise Warning('model name is not specified, the model will NOT be saved!')
        output_model_dir = os.path.join('./saved_models', args.model_name + '_' + now_time)

        NE2id, language = get_NE2id(args.train_data_path)
        logger.info('# of NE in train: %d: ' % len(NE2id))

        if args.use_memory:
            train_sentences, _ = read_tsv(args.train_data_path)
            eval_sentences, _ = read_tsv(args.dev_data_path)
            all_sentence = train_sentences + eval_sentences
            gram2id, gram2count = get_ngram2id(all_sentence, args.max_ngram_length,
                                               args.ngram_num_threshold, args.ngram_type, args.av_threshold)

            logger.info('# of n-gram in attention: %d' % len(gram2id))

            if not args.cat_type == 'freq':
                gram2count = None
        else:
            gram2id = None
            gram2count = None

        if args.use_bilstm and args.model_path is None:
            emb_NE2id = get_character2id(args.train_data_path)
        else:
            emb_NE2id = None

        label_list = get_labels(args.train_data_path)
        label_map = {label: i for i, label in enumerate(label_list, 0)}

        hpara = cls.init_hyper_parameters(args)
        tagger = cls(NE2id, gram2id, gram2count, label_map, hpara, model_path=args.model_path, cache_dir=args.cache_dir, emb_NE2id=emb_NE2id)

        train_examples = tagger.load_data(args.train_data_path)
        dev_examples = tagger.load_data(args.dev_data_path)
        test_examples = tagger.load_data(args.test_data_path)
        all_eval_examples = {
            'dev': dev_examples,
            'test': test_examples
        }
        num_labels = tagger.num_labels
        convert_examples_to_features = tagger.convert_examples_to_features
        feature2input = tagger.feature2input
        label_map = {i: label for i, label in enumerate(label_list, 0)}

        total_params = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
        logger.info('# of trainable parameters: %d' % total_params)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.fp16:
            tagger.half()
        tagger.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            tagger = DDP(tagger)
        elif n_gpu > 1:
            tagger = torch.nn.DataParallel(tagger)

        param_optimizer = list(tagger.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
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

        best_epoch = -1
        history = {'epoch': [],
                   'dev': {'ner_p': [], 'ner_r': [], 'ner_f': []},
                   'test': {'ner_p': [], 'ner_r': [], 'ner_f': []}
                   }
        best = {'dev': {'best_epoch': 0, 'best_pp': -1, 'best_pr': -1, 'best_pf': -1},
                'test': {'best_epoch': 0, 'best_pp': -1, 'best_pr': -1, 'best_pf': -1},
                }

        num_of_no_improvement = 0
        patient = args.patient

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            tagger.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                tagger.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue
                train_features = convert_examples_to_features(batch_examples, language)
                channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                segment_ids, valid_ids, word_ids, word_mask = feature2input(device, train_features)

                loss, _ = tagger(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                 word_ids, matching_matrix, word_mask, channel_ids,
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

            tagger.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                tagger.eval()
                improved = False
                y_true = {'dev': [], 'test': []}
                y_pred = {'dev': [], 'test': []}
                for flag in ['dev', 'test']:
                    eval_examples = all_eval_examples[flag]
                    for start_index in range(0, len(eval_examples), args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                             len(eval_examples))]
                        eval_features = convert_examples_to_features(eval_batch_examples, language)

                        channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                        segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

                        with torch.no_grad():
                            tag_seq = tagger(input_ids, segment_ids, input_mask, labels=None,
                                             valid_ids=valid_ids, attention_mask_label=l_mask,
                                             word_seq=word_ids, label_value_matrix=matching_matrix,
                                             word_mask=word_mask, channel_ids=channel_ids,
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


                    sentence_all = []
                    for example, y_true_item in zip(all_eval_examples[flag], y_true[flag]):
                        sen = example.text_a
                        sen = sen.strip()
                        sen = sen.split(' ')
                        if len(y_true_item) != len(sen):
                            sen = sen[:len(y_true_item)]
                        sentence_all.append(sen)
                    dev_pp, dev_pr, dev_pf = evaluate_word_PRF(y_pred[flag], y_true[flag])
                    # dev_poov = evaluate_OOV(y_pred[flag], y_true[flag], sentence_all, NE2id, language)

                    history['epoch'].append(epoch)
                    history[flag]['ner_p'].append(dev_pp)
                    history[flag]['ner_r'].append(dev_pr)
                    history[flag]['ner_f'].append(dev_pf)
                    # history[flag]['ner_oov'].append(dev_poov)
                    logger.info("======= %s entity level========" % flag)
                    logger.info("Epoch: %d, ner P: %f,  ner R: %f,  ner F: %f",
                                epoch + 1, dev_pp, dev_pr, dev_pf)
                    logger.info("======= %s entity level========" % flag)
                    # the evaluation method o
                    report = classification_report(y_true[flag], y_pred[flag], digits=4)
                    if not os.path.exists(output_model_dir):
                        os.makedirs(output_model_dir)

                    output_eval_file = os.path.join(output_model_dir, "results.%s.txt" % flag)
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** %s Eval results *****" % flag)
                        logger.info("=======token level========")
                        logger.info("\n%s", report)
                        logger.info("======= %s token level========" % flag)
                        writer.write(report)
                    if flag == 'dev':
                        if history['dev']['ner_f'][epoch] > best['dev']['best_pf']:
                            best_epoch = epoch + 1
                            num_of_no_improvement = 0
                            improved = True
                        else:
                            num_of_no_improvement += 1
                            improved = False

                if improved:
                    for flag in ['dev', 'test']:
                        best[flag]['best_pp'] = history[flag]['ner_p'][epoch]
                        best[flag]['best_pr'] = history[flag]['ner_r'][epoch]
                        best[flag]['best_pf'] = history[flag]['ner_f'][epoch]
                        # best[flag]['best_poov'] = history[flag]['ner_oov'][epoch]
                        with open(os.path.join(output_model_dir, 'POS_result.%s.txt' % flag), "w") as writer:
                            writer.write("Epoch: %d,  ner P: %f,  ner R: %f,  ner F: %f\n\n" %
                                         (epoch + 1, best[flag]['best_pp'], best[flag]['best_pr'],
                                          best[flag]['best_pf']))
                            for i in range(len(y_pred[flag])):
                                sentence = all_eval_examples[flag][i].text_a
                                seg_true_str, seg_pred_str = eval_sentence(y_pred[flag][i], y_true[flag][i], sentence,
                                                                           NE2id, language)
                                writer.write('True: %s\n' % seg_true_str)
                                writer.write('Pred: %s\n\n' % seg_pred_str)

                    model_to_save = tagger.module if hasattr(tagger, 'module') else tagger
                    save_model_dir = os.path.join(output_model_dir, 'model')
                    if not os.path.exists(save_model_dir):
                        os.makedirs(save_model_dir)

                    if args.model_path == None:
                        model_to_save.save_model(save_model_dir)
                    elif '/' in args.model_path:
                        model_to_save.save_model(save_model_dir, args.model_path)
                    elif '-' in args.model_path:
                        model_to_save.save_model(save_model_dir, args.cache_dir)
                    else:
                        raise ValueError()

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        for flag in ['dev', 'test']:
            logger.info("\n=======best %s f entity level========" % flag)
            logger.info("Epoch: %d,  ner P: %f,  ner R: %f,  ner F: %f",
                        best_epoch, best[flag]['best_pp'], best[flag]['best_pr'],
                        best[flag]['best_pf'])
            logger.info("\n=======best %s f entity level========" % flag)

        with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
            json.dump(history, f)
            f.write('\n')

    #
    def predict(self, sentence_list, seperated_type='list', eval_batch_size=16):
        # no_cuda = not next(self.parameters()).is_cuda
        eval_examples = self.load_data(sentence_list=sentence_list, data_path=None)
        label_map = {i: label for label, i in self.labelmap.items()}

        self.eval()
        y_pred = []
        language = get_language(''.join(eval_examples[0].text_a.strip().split(' ')))
        for start_index in range(0, len(eval_examples),  eval_batch_size):
            eval_batch_examples = eval_examples[start_index: min(start_index +  eval_batch_size,
                                                                 len(eval_examples))]
            eval_features = self.convert_examples_to_features(eval_batch_examples, language)

            channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
            segment_ids, valid_ids, word_ids, word_mask = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                tag_seq = self.forward(input_ids, segment_ids, input_mask, labels=None,
                                 valid_ids=valid_ids, attention_mask_label=l_mask,
                                 word_seq=word_ids, label_value_matrix=matching_matrix,
                                 word_mask=word_mask, channel_ids=channel_ids,
                                 input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

            logits = tag_seq.to('cpu').numpy()

            for i, label in enumerate(logits):
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif len(eval_batch_examples[i].text_a.split()) < j:
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_2.append(label_map[logits[i][j]])

        pred_list = []
        for pre_label, sentence in zip(y_pred, sentence_list):
            pred_list.append(pred_result(pre_label, sentence, language, seperated_type=seperated_type))
        return pred_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None, matrix=None, sent_len=None, word_list_len=None):
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
        self.sent_len = sent_len
        self.word_list_len = word_list_len


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None, channel_ids=None,
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
        self.channel_ids = channel_ids

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def get_NE2id(train_path):
    NE2id = {'<PAD>': 0, '<UNK>': 1}
    word = ''
    index = 2
    with open(train_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        language = get_language(lines[0].strip())
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = re.split('\\s+', line)
            character = splits[0]
            label = splits[-1][0]
            if label in ['B', 'O']:
                if word not in NE2id:
                    NE2id[word.strip()] = index
                    index += 1
                word = ''
            if label != 'O':
                if language == 'zh':
                    word += character
                elif language == 'en':
                    word += ' ' + character
    return NE2id, language


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


def cached_DNLP(model_path, dataset):
    logger = logging.getLogger(__name__)
    if os.path.exists(model_path):
        return model_path
    elif model_path in NER_PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_web, model_name = NER_PRETRAINED_MODEL_ARCHIVE_MAP[model_path][dataset]
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
                ', '.join(NER_PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                model_path))
        return None