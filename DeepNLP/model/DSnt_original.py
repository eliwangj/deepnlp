from __future__ import absolute_import, division, print_function

import math
import json
import logging
import os
import re
import random
import subprocess
import datetime

import numpy as np
import torch
from torch import nn
from .pretrained.bert import BertModel, BertTokenizer, BertAdam, LinearWarmUpScheduler
from .modules import TypeGraphConvolution
from .pretrained.zen2 import ZenModel, ZenNgramDict
from .pretrained.xlnet import XLNetModel, XLNetTokenizer
from sklearn import metrics

from ..utils.io_utils import save_json, load_json, read_embedding, get_language
from tqdm import tqdm, trange
import torch.nn.functional as F
from ..utils.Web_MAP import Snt_PRETRAINED_MODEL_ARCHIVE_MAP

DEFAULT_HPARA = {
    'max_seq_length': 300,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_tgcn': False,
    'layer_number': 3,
    'use_bilstm': False,
    'lstm_layer_number': 1,
    'lstm_hidden_size': 200,
    'embedding_dim': 100,
}


class DSnt(nn.Module):

    def __init__(self, labelmap, hpara, model_path, dep2id=None, cache_dir='./', emb_word2id=None):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.use_tgcn = self.hpara['use_tgcn']
        self.layer_number = self.hpara['layer_number']

        self.dep2id = dep2id

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
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=cache_dir)
            self.bert = BertModel.from_pretrained(model_path, cache_dir=cache_dir)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=cache_dir)
            self.xlnet = XLNetModel.from_pretrained(model_path, cache_dir=cache_dir)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)

        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'], cache_dir=cache_dir)
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir=cache_dir)
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)

        else:
            raise ValueError()

        if self.use_tgcn:
            self.TGCNLayers = nn.ModuleList(([TypeGraphConvolution(hidden_size, hidden_size)
                                              for _ in range(self.layer_number)]))
            self.ensemble_linear = nn.Linear(1, self.layer_number)
            self.ensemble = nn.Parameter(torch.FloatTensor(self.layer_number, 1))
            self.dep_embedding = nn.Embedding(len(self.dep2id), hidden_size, padding_idx=0)
        else:
            self.TGCNLayers = None
            self.ensemble_linear = None
            self.ensemble = None
            self.dep_embedding = None

        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                mem_valid_ids=None, dep_adj_matrix=None, dep_value_matrix=None
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

        if self.TGCNLayers is not None:
            batch_size, max_len, feat_dim = sequence_output.shape
            dep_embed = self.dep_embedding(dep_value_matrix)

            valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
            for i in range(batch_size):
                temp = sequence_output[i][valid_ids[i] == 1]
                valid_output[i][:temp.size(0)] = temp

            valid_output = self.dropout(valid_output)

            tgcn_layer_outputs = []
            seq_out = valid_output
            for tgcn in self.TGCNLayers:
                attention_score = TypeGraphConvolution.get_attention(seq_out, dep_embed, dep_adj_matrix)
                seq_out = F.relu(tgcn(seq_out, attention_score, dep_embed))
                tgcn_layer_outputs.append(seq_out)
            tgcn_layer_outputs_pool = [TypeGraphConvolution.get_avarage(mem_valid_ids, x_out) for x_out in tgcn_layer_outputs]

            x_pool = torch.stack(tgcn_layer_outputs_pool, -1)
            ensemble_out = torch.matmul(x_pool, F.softmax(self.ensemble_linear.weight, dim=0))
            ensemble_out = ensemble_out.squeeze(dim=-1)
            ensemble_out = self.dropout(ensemble_out)
            out = self.dropout(ensemble_out)
        else:
            out = sequence_output[:, 0]

        out = self.classifier(out)

        if labels is not None:
            return self.criterion(out, labels)
        else:
            return out

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_tgcn'] = args.use_tgcn
        hyper_parameters['layer_number'] = args.layer_number

        hyper_parameters['use_bilstm'] = args.use_bilstm
        hyper_parameters['lstm_layer_number'] = args.lstm_layer_number
        hyper_parameters['lstm_hidden_size'] = args.lstm_hidden_size
        hyper_parameters['embedding_dim'] = args.embedding_dim

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_dir, vocab_dir=None):

        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(output_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(output_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        if self.TGCNLayers is not None:
            output_dep2id_file = os.path.join(output_dir, 'dep2id.json')
            save_json(output_dep2id_file, self.dep2id)

        if self.bert or self.zen or self.xlnet:
            output_config_file = os.path.join(output_dir, 'config.json')
            with open(output_config_file, "w", encoding='utf-8') as writer:
                if self.bert:
                    writer.write(self.bert.config.to_json_string())
                elif self.xlnet:
                    writer.write(self.xlnet.config.to_json_string())
                elif self.zen:
                    writer.write(self.zen.config.to_json_string())
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
            save_json(os.path.join(output_dir, 'emb_word2id.json'), self.emb_word2id)

    @classmethod
    def load_model(cls, model_path, ABSA=False, language='en', dataset='MAMS', TGCN=False, local_rank=-1, no_cuda=False):
        '''
        --ABSA: There are two different sentiment analysis tasks.
            'ABSA==Flase' means general Sentiment Analysis and 'ABSA==True' means Aspect-based Sentiment Analysis.
        -- dataset: The models are fine-tuned on 6 different datasets that is Twitter, Laptop14, Rest14, Rest15, Rest16, MAMS,
            and you need to assign the specific model based on the dataset mentioned above to process your own data.
        -- TGCN: Utilize TGCN as decoder to extract information for ABSA.
        '''
        if ABSA:
            task = 'ABSA'
        else:
            task = 'SA'
            if language == 'zh':
                dataset = 'chnsenticorp'
            else:
                dataset = 'SST5'
        # assign model path
        model_path = cached_DNLP(model_path, task, dataset, TGCN=TGCN)
        # select the device
        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            # torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
            #                                      world_size=world_size)

        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        dep2id_file = os.path.join(model_path, 'dep2id.json')
        if os.path.exists(dep2id_file):
            dep2id = load_json(dep2id_file)
        else:
            dep2id = None

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        emb_word2id_path = os.path.join(model_path, 'emb_word2id.json')
        emb_word2id = load_json(emb_word2id_path) if os.path.exists(emb_word2id_path) else None
        if emb_word2id:
            res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=None, dep2id=dep2id,
                      emb_word2id=emb_word2id)
        else:
            res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path, dep2id=dep2id,  emb_word2id=emb_word2id)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))

        cls.device = device
        cls.n_gpu = n_gpu
        res.to(device)
        return res

    @staticmethod
    def set_not_grad(module):
        for para in module.parameters():
            para.requires_grad = False

    def load_data(self, data_path=None, sentence_list=None, aspect_list=None):
        if data_path is not None:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
            lines = readfile(data_path, flag)
        elif sentence_list is not None:
            flag = 'predict'
            if aspect_list is not None:
                aspect_index = [find_aspect_index(s, a) for s, a in zip(sentence_list, aspect_list)]
                lines = [(s, a, i, '0',  None, None) for s, a, i in zip(sentence_list, aspect_list, aspect_index)]
            else:
                aspect_index = None
                lines = [(s, ['<None>'], aspect_index, '0', None, None) for s in sentence_list]
        else:
            raise ValueError()
        examples = self.process_data(lines, flag)
        return examples

    @staticmethod
    def process_data(lines, flag):
        examples = []
        for i, (sentence, aspect, aspect_index, label, head, dep_type) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = ' '.join(aspect)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                         head=head, dep_type=dep_type,
                                         aspect_index=aspect_index))
        return examples

    def convert_examples_to_features(self, examples, language):
        tokenizer = self.tokenizer

        features = []

        length_list = []
        tokens_list = []
        aspect_list = []
        sent_valid_list = []
        aspect_valid_list = []

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            aspectlist = example.text_b.split(' ')

            tokens = []
            sent_valid = []
            aspect_valid = []

            if len(textlist) > self.max_seq_length:
                textlist = textlist[:self.max_seq_length]

            for i, word in enumerate(textlist):
                if tokenizer:
                    token = tokenizer.tokenize(word)
                elif word in self.emb_word2id:
                    token = [word]
                else:
                    if language == 'zh':
                        token = list(word)
                    elif language == 'en':
                        token = [word]

                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        sent_valid.append(1)
                    else:
                        sent_valid.append(0)

            aspects = []
            if aspectlist != ['<None>']:
                for i, word in enumerate(aspectlist):
                    if tokenizer:
                        token = tokenizer.tokenize(word)
                    elif word in self.emb_word2id:
                        token = [word]
                    else:
                        if language == 'zh':
                            token = list(word)
                        elif language == 'en':
                            token = [word]

                    aspects.extend(token)
                    for m in range(len(token)):
                        if m == 0:
                            aspect_valid.append(1)
                        else:
                            aspect_valid.append(0)
            else:
                aspects = []
                aspect_valid = []

            length_list.append(len(tokens) + len(aspects))

            tokens_list.append(tokens)
            aspect_list.append(aspects)
            sent_valid_list.append(sent_valid)
            aspect_valid_list.append(aspect_valid)

        seq_pad_length = max(length_list) + 3

        for indx, (example, sent_tokens, aspect_tokens, sent_valid, aspect_valid) in \
                enumerate(zip(examples, tokens_list, aspect_list,
                              sent_valid_list, aspect_valid_list)):

            ntokens = ['[CLS]'] + sent_tokens + ['[SEP]'] + aspect_tokens + ['[SEP]']
            valid = [1] + sent_valid + [1] + aspect_valid + [1]
            segment_ids = [0] * (len(sent_tokens) + len(aspect_tokens) + 3)
            label_id = self.labelmap[example.label]

            aspect_mask = [0] * seq_pad_length
            if example.aspect_index is not None:
                start_index = example.aspect_index[0] + 1
                end_index = example.aspect_index[1] + 1
                for i in range(start_index, end_index):
                    if i < len(aspect_mask):
                        aspect_mask[i] = 1

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

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            if self.TGCNLayers is not None:
                adj_matrix = np.zeros((seq_pad_length, seq_pad_length))
                type_matrix = np.zeros((seq_pad_length, seq_pad_length))
                head_list = example.head
                type_list = example.dep_type
                for i in range(len(head_list)):
                    head = head_list[i]
                    child = i + 1
                    dep_type = type_list[i]
                    adj_matrix[child][head] = 1
                    adj_matrix[child][child] = 1
                    adj_matrix[head][child] = 1

                    type_id = self.dep2id[dep_type] if dep_type in self.dep2id else self.dep2id['<UNK>']
                    type_matrix[child][head] = type_id
                    type_matrix[head][child] = type_id
                    type_matrix[child][child] = self.dep2id['self']
            else:
                adj_matrix = None
                type_matrix = None

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
                              label_id=label_id,
                              valid_ids=valid,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              aspect_mask=aspect_mask,
                              adj_matrix=adj_matrix,
                              type_matrix=type_matrix
                              ))
        return features

    def feature2input(self, device, feature):
        input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long).to(device)
        input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long).to(device)
        segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long).to(device)
        label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long).to(device)
        valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long).to(device)
        aspect_mask = torch.tensor([f.aspect_mask for f in feature], dtype=torch.long).to(device)

        if self.TGCNLayers is not None:
            adj_matrix = torch.tensor([f.adj_matrix for f in feature], dtype=torch.long).to(device)
            type_matrix = torch.tensor([f.type_matrix for f in feature], dtype=torch.long).to(device)
        else:
            adj_matrix = None
            type_matrix = None

        if self.zen is not None:
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

        return input_ids, input_mask, label_ids, segment_ids, valid_ids, aspect_mask, \
               ngram_ids, ngram_positions, \
               adj_matrix, type_matrix

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

        if args.server_ip and  args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(args.server_ip,  args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not  args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device( args.local_rank)
            device = torch.device("cuda",  args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank= args.rank,
                                                 world_size= args.world_size)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool( args.local_rank != -1),  args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed( args.seed)
        np.random.seed( args.seed)
        torch.manual_seed( args.seed)

        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')

        if args.model_name is None:
            raise Warning('model name is not specified, the model will NOT be saved!')
        output_model_dir = os.path.join('./saved_models',  args.model_name + '_' + now_time)

        word2id = get_word2id(args.train_data_path)
        logger.info('# of word in train: %d: ' % len(word2id))

        label_list = get_label_list(args.train_data_path)
        logger.info('# of sentiment types in train: %d: ' % (len(label_list) - 3))
        label_map = {label: i for i, label in enumerate(label_list, 1)}

        if args.use_tgcn:
            dep2id = get_dep2id(args.train_data_path)
        else:
            dep2id = None

        if args.use_bilstm and args.model_path is None:
            emb_word2id = get_character2id(args.train_data_path)
        else:
            emb_word2id = None

        hpara = cls.init_hyper_parameters(args)
        sentiment_analyzer = cls(label_map, hpara, model_path=args.model_path, cache_dir=args.cache_dir, dep2id=dep2id, emb_word2id=emb_word2id)

        train_examples = sentiment_analyzer.load_data(args.train_data_path)
        dev_examples = sentiment_analyzer.load_data(args.dev_data_path)
        test_examples = sentiment_analyzer.load_data(args.test_data_path)

        language = get_language(''.join(train_examples[0].text_a.strip().split(' ')))

        eval_data = {
            'dev': dev_examples,
            'test': test_examples
        }

        convert_examples_to_features = sentiment_analyzer.convert_examples_to_features
        feature2input = sentiment_analyzer.feature2input

        total_params = sum(p.numel() for p in sentiment_analyzer.parameters() if p.requires_grad)
        logger.info('# of trainable parameters: %d' % total_params)

        num_train_optimization_steps = int(
            len(train_examples) /  args.train_batch_size /  args.gradient_accumulation_steps) *  args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.fp16:
            sentiment_analyzer.half()
        sentiment_analyzer.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            sentiment_analyzer = DDP(sentiment_analyzer)
        elif n_gpu > 1:
            sentiment_analyzer = torch.nn.DataParallel(sentiment_analyzer)

        param_optimizer = list(sentiment_analyzer.named_parameters())
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
                                  lr= args.learning_rate,
                                  bias_correction=False)

            if args.loss_scale == 0:
                model, optimizer = amp.initialize(sentiment_analyzer, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(sentiment_analyzer, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale= args.loss_scale)
            scheduler = LinearWarmUpScheduler(optimizer, warmup= args.warmup_proportion,
                                              total_steps=num_train_optimization_steps)

        else:
            # num_train_optimization_steps=-1
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr= args.learning_rate,
                                 warmup= args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
        best_epoch = -1
        best_dev_acc = -1
        best_dev_f = -1
        best_test_acc = -1
        best_test_f = -1

        history = {
            'dev': {'epoch': [], 'acc': [], 'f': []},
            'test': {'epoch': [], 'acc': [], 'f': []},
        }
        num_of_no_improvement = 0
        patient = args.patient

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d",  args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int( args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            sentiment_analyzer.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples),  args.train_batch_size))):
                sentiment_analyzer.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                  args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples, language)

                input_ids, input_mask, label_ids, segment_ids, valid_ids, aspect_mask, \
                ngram_ids, ngram_positions, \
                adj_matrix, type_matrix = feature2input(device, train_features)

                loss = sentiment_analyzer(input_ids=input_ids, token_type_ids=segment_ids,
                                          attention_mask=input_mask, labels=label_ids,
                                          valid_ids=valid_ids,
                                          input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions,
                                          mem_valid_ids=aspect_mask,
                                          dep_adj_matrix=adj_matrix, dep_value_matrix=type_matrix
                                          )

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

            sentiment_analyzer.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                prediction = {
                    'dev': [],
                    'test': []
                }
                logger.info('\n')
                for flag in eval_data.keys():
                    eval_examples = eval_data[flag]
                    sentiment_analyzer.eval()

                    label_map = {i: label for i, label in enumerate(label_list, 1)}
                    label_map[0] = '<UNK>'
                    for start_index in range(0, len(eval_examples),  args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index +  args.eval_batch_size,
                                                                             len(eval_examples))]

                        eval_features = convert_examples_to_features(eval_batch_examples, language)

                        input_ids, input_mask, label_ids, segment_ids, valid_ids, aspect_mask, \
                        ngram_ids, ngram_positions, \
                        adj_matrix, type_matrix = feature2input(device, eval_features)

                        with torch.no_grad():
                            outputs = sentiment_analyzer(input_ids=input_ids, token_type_ids=segment_ids,
                                                        attention_mask=input_mask, labels=None,
                                                        valid_ids=valid_ids,
                                                        input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions,
                                                        mem_valid_ids=aspect_mask,
                                                        dep_adj_matrix=adj_matrix, dep_value_matrix=type_matrix
                                                        )

                        outputs = torch.argmax(outputs, -1).detach().to('cpu').tolist()
                        predicts = [label_map[o] for o in outputs]

                        prediction[flag].extend(predicts)

                    gold_labels = [example.label for example in eval_examples]
                    acc = metrics.accuracy_score(gold_labels, prediction[flag]) * 100
                    f = metrics.f1_score(gold_labels, prediction[flag], labels=['-1', '0', '1'], average='macro') * 100

                    report = '%s: Epoch: %d, acc:%.2f, f:%.2f' % (flag, epoch + 1, acc, f)
                    logger.info(report)
                    history[flag]['epoch'].append(epoch)
                    history[flag]['acc'].append(acc)
                    history[flag]['f'].append(f)

                    if args.model_name is not None:
                        if not os.path.exists(output_model_dir):
                            os.makedirs(output_model_dir)

                        output_eval_file = os.path.join(output_model_dir, flag + "_report.txt")
                        with open(output_eval_file, "a") as writer:
                            writer.write(report)
                            writer.write('\n')

                logger.info('\n')
                if history['dev']['f'][-1] > best_dev_f:
                    best_epoch = epoch + 1
                    best_dev_acc = history['dev']['acc'][-1]
                    best_dev_f = history['dev']['f'][-1]
                    best_test_acc = history['test']['acc'][-1]
                    best_test_f = history['test']['f'][-1]
                    num_of_no_improvement = 0

                    if args.model_name:
                        for flag in ['dev', 'test']:
                            with open(os.path.join(output_model_dir, flag + '_result.txt'), "w") as writer:
                                writer.write("Epoch: %d, dev_acc: %f, dev_f: %f, test_acc: %f, test_f: %f\n\n" % (
                                    best_epoch, best_dev_acc, best_dev_f, best_test_acc, best_test_f))

                                all_pred = prediction[flag]
                                examples = eval_data[flag]
                                for example, pred in zip(examples, all_pred):
                                    sentence = example.text_a
                                    aspect = example.text_b
                                    gold = example.label
                                    writer.write(sentence + '\n')
                                    writer.write(aspect + '\n')
                                    writer.write('gold: ' + gold + '\n')
                                    writer.write('pred: ' + pred + '\n')
                                    writer.write('\n')

                        # model_to_save = dep_parser.module if hasattr(dep_parser, 'module') else dep_parser
                        best_eval_model_dir = os.path.join(output_model_dir, 'model')
                        if not os.path.exists(best_eval_model_dir):
                            os.makedirs(best_eval_model_dir)

                        if args.model_path == None:
                            sentiment_analyzer.save_model(output_model_dir)
                        elif '/' in args.model_path:
                            sentiment_analyzer.save_model(best_eval_model_dir, args.model_path)
                        elif '-' in args.model_path:
                            sentiment_analyzer.save_model(best_eval_model_dir, args.cache_dir)
                        else:
                            raise ValueError()

                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        best_report = "Epoch: %d, dev_acc: %f, dev_f: %f, test_acc: %f, test_f: %f" % (
            best_epoch, best_dev_acc, best_dev_f, best_test_acc, best_test_f)
        logger.info("\n=======best las at dev========")
        logger.info(best_report)
        logger.info("\n=======best las at dev========")

        if args.model_name is not None:
            output_eval_file = os.path.join(output_model_dir, "final_report.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(best_report + '\n')

            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')

    def predict(self, sentence_list, aspect_list=None, eval_batch_size=16):
        eval_examples = self.load_data(sentence_list=sentence_list, aspect_list=aspect_list)
        language = get_language(''.join(eval_examples[0].text_a.strip().split(' ')))

        self.eval()
        prediction = []
        label_map = {v: k for k, v in self.labelmap.items()}
        label_map[0] = '<UNK>'
        for start_index in range(0, len(eval_examples), eval_batch_size):
            eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                 len(eval_examples))]

            eval_features = self.convert_examples_to_features(eval_batch_examples, language)

            input_ids, input_mask, label_ids, segment_ids, valid_ids, aspect_mask, \
            ngram_ids, ngram_positions, \
            adj_matrix, type_matrix = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                outputs = self.forward(input_ids=input_ids, token_type_ids=segment_ids,
                                             attention_mask=input_mask, labels=None,
                                             valid_ids=valid_ids,
                                             input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions,
                                             mem_valid_ids=aspect_mask,
                                             dep_adj_matrix=adj_matrix, dep_value_matrix=type_matrix
                                             )

            outputs = torch.argmax(outputs, -1).detach().to('cpu').tolist()
            predicts = [label_map[o] for o in outputs]

            prediction.extend(predicts)

        return prediction


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, aspect_index=None, head=None, label=None, dep_type=None):
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
        self.aspect_index = aspect_index
        self.head = head
        self.dep_type = dep_type
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None,
                 # label_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 aspect_mask=None,
                 adj_matrix=None, type_matrix=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        # self.label_mask = label_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks

        self.aspect_mask = aspect_mask
        self.adj_matrix = adj_matrix
        self.type_matrix = type_matrix


def find_aspect_index(sentence, aspect):

    if isinstance(sentence, str):
        sentence = sentence.split()
    if isinstance(aspect, str):
        aspect = aspect.split()

    for i in range(len(sentence) - len(aspect)):
        if sentence[i] == aspect[0]:
            span_text = ' '.join(sentence[i: i+len(aspect)])
            if span_text == ' '.join(aspect):
                return i, i + len(aspect)

    return None


def readfile(filename, flag):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        language = get_language(lines[0])
    if not flag == 'predict':
        for i in range(0, len(lines), 3):
            if lines[i] == '':
                break
            aspect = lines[i+1].strip().split()
            sentence = lines[i].strip().replace('$T$', ' $T$ ')
            if language == 'zh':
                sentence = re.sub('\\s+', '', sentence)
                sentence = list(sentence)
            elif language == 'en':
                sentence = sentence.split()

            new_sent = []
            aspect_index = None
            for word in sentence:
                if word == '$T$' and aspect != ['<None>']:
                    aspect_index = (len(new_sent), len(new_sent) + len(aspect))
                    new_sent.extend(aspect)
                else:
                    new_sent.append(word)
            label = lines[i+2].strip()
            if aspect != ['<None>']:
                assert aspect_index is not None

            data.append((new_sent, aspect, aspect_index, label, None, None))
    else:
        raise ValueError()

    dep_file = filename + '.dep'
    dep_data = []
    if os.path.exists(dep_file):
        with open(dep_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
        dep_list = []
        if not flag == 'predict':
            for line in lines:
                line = line.strip()
                if line == '' and len(dep_list) > 0:
                    dep_list.sort(key=lambda x: x[1])
                    dep_data.append(dep_list)
                    dep_list = []
                    continue

                split = line.split()
                head = int(split[0])
                dependent = int(split[1])
                dep_type = split[2]
                dep_list.append((head, dependent, dep_type))
            if len(dep_list) > 0:
                dep_list.sort(key=lambda x: x[1])
                dep_data.append(dep_list)
        else:
            raise ValueError()
        
        assert len(data) == len(dep_data)
        for i in range(len(data)):
            data[i] = (data[i][0], data[i][1], data[i][2], data[i][3],
                       [dep[0] for dep in dep_data[i]], [dep[2] for dep in dep_data[i]])
    return data


def get_word2id(train_data_path):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    index = 2
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            if lines[i] == '':
                break
            aspect = lines[i + 1].strip()
            sentence = lines[i].strip().replace('$T$', aspect)

            words = sentence.split()

            for word in words:
                if word not in word2id:
                    word2id[word] = index
                    index += 1
    return word2id


def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            if lines[i] == '':
                break
            label = lines[i + 2].strip()
            if label not in label_list:
                label_list.append(label)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def get_dep2id(train_data_path):
    train_data_path += '.dep'
    dep2id = {'<PAD>': 0, '<UNK>': 1, 'self': 2}
    index = 3

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label = line.strip().split()[-1]
            if label not in dep2id:
                dep2id[label] = index
                index += 1

    return dep2id


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


def cached_DNLP(model_path, task, dataset, TGCN):
    logger = logging.getLogger(__name__)
    if task == 'ABSA' and TGCN:
        task += '_tgcn'
    if os.path.exists(model_path):
        return model_path
    elif model_path in Snt_PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_web, model_name = Snt_PRETRAINED_MODEL_ARCHIVE_MAP[task][model_path][dataset]
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
                ', '.join(Snt_PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                model_path))
        return None