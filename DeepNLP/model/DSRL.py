from __future__ import absolute_import, division, print_function

import math
from pickle import NONE   
import torch    
import json
import logging
import os
import warnings
import random
import datetime
import subprocess
import numpy as np
from torch import nn

from .pretrained.bert import BertModel, BertTokenizer, BertAdam, LinearWarmUpScheduler
from .pretrained.xlnet import XLNetModel, XLNetTokenizer
from .pretrained.zen2 import ZenModel, ZenNgramDict
from .modules import Biaffine, MLP, CRF
from ..utils.io_utils import save_json, load_json, read_embedding, get_language
from tqdm import tqdm, trange
from ..eval.SRL_eval import to_eval_file, get_prf, fix_verb
from ..utils.Web_MAP import SRL_PRETRAINED_MODEL_ARCHIVE_MAP

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'mlp_dropout': 0.33,
    'n_mlp': 400,
    'decoder': 'crf',
    'use_bilstm': False,
    'lstm_layer_number': 1,
    'lstm_hidden_size': 200,
    'embedding_dim': 100,
}


class DSRL(nn.Module):

    def __init__(self, labelmap, hpara, model_path, emb_word2id=None):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']


        if hpara['use_zen']:
            raise ValueError()

        self.bilstm = None
        self.embedding = None
        self.emb_word2id = None
        if emb_word2id is not None and model_path is not None:
            Warning('Pretrained word embedding file is given. Will use the pretrained embedding at %s' % model_path)

        self.tokenizer = None
        self.bert = None
        self.xlnet = None
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
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.xlnet = XLNetModel.from_pretrained(model_path)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()
        if self.tokenizer:
            self.tokenizer.add_never_split_tokens(["[V]", "[/V]"])
        else:
            pass
        self.mlp_pre_h = MLP(n_in=hidden_size,
                             n_hidden=self.hpara['n_mlp'],
                             dropout=self.hpara['mlp_dropout'])
        self.mlp_arg_h = MLP(n_in=hidden_size,
                             n_hidden=self.hpara['n_mlp'],
                             dropout=self.hpara['mlp_dropout'])

        self.srl_attn = Biaffine(n_in=self.hpara['n_mlp'],
                                 n_out=self.num_labels,
                                 bias_x=True,
                                 bias_y=True)
        if self.hpara['decoder'] == 'crf':
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                verb_index=None, labels=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                ):

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

        batch_size, _, feat_dim = sequence_output.shape
        max_len = attention_mask_label.shape[1]
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            sent_len = attention_mask_label[i].sum()
            valid_output[i][:sent_len] = temp[:sent_len]

        valid_output = self.dropout(valid_output)

        predicates = torch.zeros(batch_size, feat_dim, dtype=valid_output.dtype, device=valid_output.device)
        for i in range(batch_size):
            predicates[i] = valid_output[i][verb_index[i][0]]

        pre_h = self.mlp_pre_h(predicates)
        arg_h = self.mlp_arg_h(valid_output)

        # [batch_size, seq_len, n_labels]
        s_labels = self.srl_attn(arg_h, pre_h).permute(0, 2, 1)

        if labels is not None:
            if self.crf is not None:
                return -1 * self.crf(emissions=s_labels, tags=labels, mask=attention_mask_label)
            else:
                s_labels = s_labels[attention_mask_label]
                labels = labels[attention_mask_label]
                return self.loss_function(s_labels, labels)
        else:
            if self.crf is not None:
                return self.crf.decode(s_labels, attention_mask_label)[0]
            else:
                pre_labels = torch.argmax(s_labels, dim=2)
                return pre_labels


    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['mlp_dropout'] = args.mlp_dropout
        hyper_parameters['n_mlp'] = args.n_mlp
        hyper_parameters['decoder'] = args.decoder

        hyper_parameters['use_bilstm'] = args.use_bilstm
        hyper_parameters['lstm_layer_number'] = args.lstm_layer_number
        hyper_parameters['lstm_hidden_size'] = args.lstm_hidden_size
        hyper_parameters['embedding_dim'] = args.embedding_dim

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_model_dir, vocab_dir=None):
        best_eval_model_dir = os.path.join(output_model_dir, 'model')
        if not os.path.exists(best_eval_model_dir):
            os.makedirs(best_eval_model_dir)

        output_model_path = os.path.join(best_eval_model_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(best_eval_model_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(best_eval_model_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        output_config_file = os.path.join(best_eval_model_dir, 'config.json')
        if self.bert or self.zen or self.xlnet:
            with open(output_config_file, "w", encoding='utf-8') as writer:
                if self.bert:
                    writer.write(self.bert.config.to_json_string())
                elif self.xlnet:
                    writer.write(self.xlnet.config.to_json_string())
                elif self.zen:
                    writer.write(self.zen.config.to_json_string())
            output_bert_config_file = os.path.join(best_eval_model_dir, 'bert_config.json')
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
            command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(best_eval_model_dir, vocab_name))
            subprocess.run(command, shell=True)

        elif self.bilstm:
            save_json(os.path.join(best_eval_model_dir, 'emb_word2id.json'), self.emb_word2id)

    @classmethod
    def load_model(cls, model_path, language='en', dataset='CoNLL05', local_rank=-1, no_cuda=False):
        # assign model path
        model_path = cached_DNLP(model_path, language=language, dataset=dataset)
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
        # print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        #     device, n_gpu, bool(local_rank != -1), fp16))
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        emb_word2id_path = os.path.join(model_path, 'emb_word2id.json')
        emb_word2id = load_json(emb_word2id_path) if os.path.exists(emb_word2id_path) else None
        if emb_word2id:
            res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=None, emb_word2id=emb_word2id)
        else:
            res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path, emb_word2id=emb_word2id)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        cls.device = device
        cls.n_gpu = n_gpu
        res.to(device)
        return res

    def load_data(self, data_path=None, sentence_list=None, verb_index_list=None):
        if data_path is not None:
            flag = data_path[data_path.rfind('/') + 1: data_path.rfind('.')]
            lines = readfile(data_path)
        elif sentence_list is not None and verb_index_list is not None:
            flag = 'predict'
            lines = [(i, ['O'] * len(i), j) for i, j in zip(sentence_list, verb_index_list)]
        else:
            raise ValueError('You must input <data path> or <sentence_list and verb_index_list> together by list of list. ')
        examples = self.process_data(lines, flag)
        return examples

    @staticmethod
    def process_data(lines, flag):

        examples = []
        for i, (sentence, label, verb_index) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None,
                                         label=label, verb_index=verb_index))
        return examples

    def convert_examples_to_features(self, examples, language):
        tokenizer = self.tokenizer

        features = []

        length_list = []
        tokens_list = []
        labels_list = []
        valid_list = []
        label_mask_list = []
        eval_mask_list = []

        for (ex_index, example) in enumerate(examples):
            text_list = example.text_a
            label_list = example.label
            verb_index = example.verb_index
            tokens = []
            labels = []
            valid = []
            label_mask = []
            eval_mask = []

            if len(text_list) > self.max_seq_length - 2:
                continue

            assert verb_index[-1] - verb_index[0] == len(verb_index) - 1


            # add [V] and [\V] to the beginning and ending of the predicate
            new_textlist = [w for w in text_list[:verb_index[0]]]
            new_textlist.append('[V]')
            new_textlist.extend([w for w in text_list[verb_index[0]: verb_index[-1] + 1]])
            new_textlist.append('[/V]')
            new_textlist.extend([w for w in text_list[verb_index[-1] + 1:]])
            assert len(new_textlist) == len(label_list) + 2
            text_list = new_textlist

            tmp = 0
            for i, word in enumerate(text_list):

                if tokenizer:
                    token = tokenizer.tokenize(word)
                elif word in self.emb_word2id or word in ['[V]', '[/V]']:
                    token = [word]
                else:
                    if language == 'zh':
                        token = list(word)
                    elif language == 'en':
                        token = [word]

                tokens.extend(token)
                if word == '[V]' or word == '[/V]':
                    for _ in range(len(token)):
                        valid.append(0)
                    tmp += 1
                    continue
                label_1 = label_list[i - tmp]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        if label_1 == 'V':
                            eval_mask.append(0)
                        else:
                            eval_mask.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            assert tmp == 2
            assert len(tokens) == len(valid)
            assert len(eval_mask) == len(label_mask)

            length_list.append(len(tokens))
            tokens_list.append(tokens)
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            eval_mask_list.append(eval_mask)
            

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        #因为CLS和SEP用不着，所以直接musk掉了，就不需要加2
        label_pad_length = max(label_len_list)

        for indx, (example, tokens, labels, valid, label_mask, eval_mask) in \
                enumerate(zip(examples, tokens_list, labels_list, valid_list, label_mask_list, eval_mask_list)):

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, 0)

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
            valid.append(0)

            assert sum(valid) == len(label_ids)

            if tokenizer:
                input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            else:
                input_ids = []
                for t in ntokens:
                    t_id = self.emb_word2id[t] if t in self.emb_word2id else self.emb_word2id['<UNK>']
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
                eval_mask.append(0)

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            # below: n-gram
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
                            verb_index=example.verb_index,
                            label_id=label_ids,
                            valid_ids=valid,
                            label_mask=label_mask,
                            eval_mask=eval_mask,
                            ngram_ids=ngram_ids,
                            ngram_positions=ngram_positions_matrix,
                            ngram_lengths=ngram_lengths,
                            ngram_tuples=ngram_tuples,
                            ngram_seg_ids=ngram_seg_ids,
                            ngram_masks=ngram_mask_array,
                            ))
            ### Intentation ending
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_verb_idx = torch.tensor([[f.verb_index[0]] for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None

        return input_ids, input_mask, l_mask, eval_mask, all_verb_idx, label_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids

    @classmethod
    def fit(cls, args):

        if not os.path.exists('./logs'):
            os.mkdir('./logs')

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

        # 分配GPU
        if args.local_rank == -1 or args.no_cuda: 
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                                 world_size=args.world_size)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

        # 分配计算梯度
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
            raise ValueError('model name is not specified, the model will NOT be saved!')
        output_model_dir = os.path.join('./saved_models', args.model_name + '_' + now_time)

        if args.use_bilstm and args.model_path is None:
            emb_word2id = get_character2id(args.train_data_path)
        else:
            emb_word2id = None

        label_list = get_label(args.train_data_path)
        logger.info('# of tag types in train: %d: ' % (len(label_list) - 3))
        label_map = {label: i for i, label in enumerate(label_list, 1)}

        hpara = cls.init_hyper_parameters(args)
        sr_tagger = cls(label_map, hpara, args.model_path, emb_word2id=emb_word2id) #初始化BErt模型

        
        train_examples = sr_tagger.load_data(args.train_data_path)
        dev_examples = sr_tagger.load_data(args.dev_data_path)
        test_examples = sr_tagger.load_data(args.test_data_path)

        language = get_language(''.join(train_examples[0].text_a)) #确定语言

        eval_data = {
            'dev': dev_examples,
            'test': test_examples
        }

        # if args.brown_data_path is not None:
        #     brown_test_examples = sr_tagger.load_data(args.brown_data_path)
        #     eval_data['brown'] = brown_test_examples

        convert_examples_to_features = sr_tagger.convert_examples_to_features
        feature2input = sr_tagger.feature2input
        save_model = sr_tagger.save_model

        all_para = [p for p in sr_tagger.parameters()]
        all_named_para = [(p[0], p[1].shape, p[1].requires_grad) for p in sr_tagger.named_parameters()]

        total_params = sum(p.numel() for p in sr_tagger.parameters() if p.requires_grad)
        logger.info('# of trainable parameters: %d' % total_params)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.fp16:
            sr_tagger.half()
        sr_tagger.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            sr_tagger = DDP(sr_tagger)
        elif n_gpu > 1:
            sr_tagger = torch.nn.DataParallel(sr_tagger)

        param_optimizer = list(sr_tagger.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            print("using fp16")
            try:
                from apex.optimizers import FusedAdam
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)

            if args.loss_scale == 0:
                model, optimizer = amp.initialize(sr_tagger, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(sr_tagger, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale=args.loss_scale)
            scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                              total_steps=num_train_optimization_steps)

        else:
            #优化器
            # num_train_optimization_steps=-1
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
        best_epoch = -1
        best_dev_p = -1
        best_dev_r = -1
        best_dev_f = -1
        best_test_p = -1
        best_test_r = -1
        best_test_f = -1

        best_brown_p = -1
        best_brown_r = -1
        best_brown_f = -1

        history = {}

        for flag in eval_data.keys():
            history[flag] = {'epoch': [], 'p': [], 'r': [], 'f': []}

        num_of_no_improvement = 0
        patient = args.patient

        global_step = 0


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            sr_tagger.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size), colour='green', unit='sentence', unit_scale=args.train_batch_size)):
                sr_tagger.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples, language)

                input_ids, input_mask, l_mask, eval_mask, verb_index, labels, ngram_ids, ngram_positions, \
                segment_ids, valid_ids = feature2input(device, train_features)
                ### Skip the wrong data entries
                ### Intentation beginning
                loss = sr_tagger(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                verb_index=verb_index, labels=labels,
                                input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                if np.isnan(loss.to('cpu').detach().numpy()):
                    raise ValueError('loss is nan!')
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                ### Intentation ending

            sr_tagger.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                prediction = {flag: [] for flag in eval_data.keys()}
                logger.info('\n')
                for flag in eval_data.keys():
                    eval_examples = eval_data[flag]
                    sr_tagger.eval()
                    all_pred, all_gold = [], []
                    label_map = {i: label for i, label in enumerate(label_list, 1)}
                    for start_index in range(0, len(eval_examples), args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                             len(eval_examples))]

                        eval_features = convert_examples_to_features(eval_batch_examples, language)

                        input_ids, input_mask, l_mask, eval_mask, verb_index, labels, ngram_ids, ngram_positions, \
                        segment_ids, valid_ids = feature2input(device, eval_features)

                        with torch.no_grad():
                            pred = sr_tagger(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                             verb_index=verb_index, labels=None,
                                             input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                        lens = l_mask.sum(1).tolist()
                        all_pred.extend(pred[l_mask].split(lens))
                        all_gold.extend(labels[l_mask].split(lens))


                    label_map[0] = 'O'

                    all_pred = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_pred]
                    all_gold = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_gold]

                    prediction[flag] = all_pred
                    


                    if not os.path.exists(output_model_dir):
                        os.makedirs(output_model_dir)

                    # p, r, f = evaluate(all_gold, all_pred)
                    eval_dir = os.path.join(output_model_dir, 'eval')
                    if not os.path.exists(eval_dir):
                        os.makedirs(eval_dir)
                    all_sentence_list = [example.text_a for example in eval_examples]
                    gold_eval_file = os.path.join(eval_dir, flag + '.%d.gold.props' % (epoch+1))
                    to_eval_file(gold_eval_file, all_sentence_list, all_gold)
                    # import pdb; pdb.set_trace()

                    pred_eval_file = os.path.join(eval_dir, flag + '.%d.pred.props' % (epoch+1))
                    new_all_pred = fix_verb(all_gold, all_pred)
                    to_eval_file(pred_eval_file, all_sentence_list, new_all_pred)

                    output_report_file = os.path.join(eval_dir, flag + '.%d.eval.report' % (epoch+1))
                    eval_path = os.path.abspath(os.path.join(__file__, '../..')) # get back to the eval dir
                    command = eval_path+'/eval/eval.sh %s %s %s > %s' % (eval_path, gold_eval_file, pred_eval_file, output_report_file)
                    subprocess.run(command, shell=True)
                    p, r, f = get_prf(output_report_file)

                    report = '%s: Epoch: %d, precision:%.2f, recall:%.2f, f1:%.2f' \
                             % (flag, epoch+1, p, r, f)
                    logger.info(report)
                    history[flag]['epoch'].append(epoch)
                    history[flag]['p'].append(p)
                    history[flag]['r'].append(r)
                    history[flag]['f'].append(f)

                    output_eval_file = os.path.join(output_model_dir, flag + "_report.txt")
                    with open(output_eval_file, "a") as writer:
                        writer.write(report)
                        writer.write('\n')

                logger.info('\n')
                if history['dev']['f'][-1] > best_dev_f:
                    best_epoch = epoch + 1
                    best_dev_p = history['dev']['p'][-1]
                    best_dev_r = history['dev']['r'][-1]
                    best_dev_f = history['dev']['f'][-1]
                    best_test_p = history['test']['p'][-1]
                    best_test_r = history['test']['r'][-1]
                    best_test_f = history['test']['f'][-1]

                    if 'brown' in history:
                        best_brown_p = history['brown']['p'][-1]
                        best_brown_r = history['brown']['r'][-1]
                        best_brown_f = history['brown']['f'][-1]

                    num_of_no_improvement = 0

                    if args.model_name:
                        for flag in eval_data.keys():
                            with open(os.path.join(output_model_dir, flag + '_result.txt'), "w") as writer:
                                writer.write('word\tpred\tgold\n\n')
                                all_labels = prediction[flag]
                                examples = eval_data[flag]
                                for example, labels in zip(examples, all_labels):
                                    words = example.text_a
                                    gold_labels = example.label
                                    for word, label, gold_label in zip(words, labels, gold_labels):
                                        line = '%s\t%s\t%s\n' % (word, label, gold_label)
                                        writer.write(line)
                                    writer.write('\n')

                        if args.model_path == None:
                            save_model(output_model_dir)
                        elif '/' in args.model_path:
                            save_model(output_model_dir, args.model_path)
                        elif '-' in args.model_path:
                            save_model(output_model_dir, args.cache_dir)
                        else:
                            raise ValueError()
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        best_report = "Epoch: %d, dev_p: %f, dev_r: %f, dev_f: %f, " \
                      "test_p: %f, test_r: %f, test_f: %f" % (
            best_epoch, best_dev_p, best_dev_r, best_dev_f, best_test_p, best_test_r, best_test_f)

        if best_brown_f > 0:
            best_report += ', brown_p: %f, brown_r: %f, brown_f %f' % (best_brown_p, best_brown_r, best_brown_f)

        logger.info("\n=======best f at dev========")
        logger.info(best_report)
        logger.info("\n=======best f at dev========")

        if args.model_name is not None:
            output_eval_file = os.path.join(output_model_dir, "final_report.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(str(total_params))
                writer.write('\n')
                writer.write(best_report)

            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')

    def predict(self, sentence_list, verb_index_list, eval_batch_size=16):
        # no_cuda = not next(self.parameters()).is_cuda
        eval_examples = self.load_data(sentence_list=sentence_list, verb_index_list=verb_index_list)
        language = get_language(''.join(eval_examples[0].text_a))
        label_map = {v: k for k, v in self.labelmap.items()}
        self.eval()
        all_pred = []

        start_time = datetime.datetime.now()
        
        for start_index in tqdm(range(0, len(eval_examples), eval_batch_size), colour='green', unit='sentence', unit_scale=eval_batch_size): # unit_scale multiplies the original unit (iteration, or batch in this case) with its value (eval_batch_size in this case)
            eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                 len(eval_examples))]
            eval_features = self.convert_examples_to_features(eval_batch_examples, language)

            input_ids, input_mask, l_mask, eval_mask, verb_index, labels, ngram_ids, ngram_positions, \
            segment_ids, valid_ids = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                pred = self.forward(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                 verb_index=verb_index, labels=None,
                                 input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

            lens = l_mask.sum(1).tolist()
            all_pred.extend(pred[l_mask].split(lens))

        label_map[0] = 'O'
        all_pred = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_pred]

        result_list = []
        for pred, sentence in zip(all_pred, sentence_list):
            result_list.append([str(i)+'_'+str(j) for i, j in zip(sentence, pred)])
    
        end_time = datetime.datetime.now()
        time_spent = (end_time - start_time).total_seconds()

        return result_list, time_spent


#
# def verb_pre_process(sentence_list, verb_index_list):
#     new_sentence_list = []
#     verb_list = []
#     for sentence, verb in zip(sentence_list, verb_index_list):
#         if (type(verb) == tuple or type(verb) == list) and len(verb) == 2:
#             verb = sentence[verb[0]:verb[1]].strip()
#             verb_list.append(verb)
#             sentence = sentence.replace(verb, "<V> " + verb + " </V>")
#             new_sentence_list.append(sentence)
#         elif verb in sentence :
#             sentence = sentence.replace(verb, "<V> " + verb + " </V>")
#             new_sentence_list.append(sentence)
#             warnings.warn("Deprecation", RuntimeWarning)
#         else:
#             raise ValueError('verb_index_list should be list of string or list of list of index or entity')
#     return new_sentence_list, verb_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, verb_index=None):
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
        self.verb_index = verb_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, verb_index, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.verb_index = verb_index
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readfile(filename):
    data = []
    sentence = []
    label = []
    verb_index = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                # if len(sentence) > 0 and len(verb_index) > 0:
                data.append((sentence, label, verb_index))
                sentence = []
                label = []
                verb_index = []
                continue
            splits = line.split()
            sentence.append(splits[0])
            sr = splits[1]
            if sr == 'V':
                verb_index.append(len(label))
            label.append(sr)
        if len(sentence) > 0:
            data.append((sentence, label, verb_index))
    return data


def get_label(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            srl_label = splits[1]
            if srl_label not in label_list:
                label_list.append(srl_label)

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


def cached_DNLP(model_path, language, dataset):
    logger = logging.getLogger(__name__)
    if language == 'zh':
        dataset = 'CPB2.0'
    if os.path.exists(model_path):
        return model_path
    elif model_path in SRL_PRETRAINED_MODEL_ARCHIVE_MAP:
        if dataset in SRL_PRETRAINED_MODEL_ARCHIVE_MAP[model_path]:
            archive_web, model_name = SRL_PRETRAINED_MODEL_ARCHIVE_MAP[model_path][dataset]
        else:
            raise ValueError("Model fine-tuned on {} is not provided, "
                             "and you can choose from {} ".format(dataset), str(SRL_PRETRAINED_MODEL_ARCHIVE_MAP.keys()))
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
                ', '.join(SRL_PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                model_path))
        return None