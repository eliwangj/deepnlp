from __future__ import absolute_import, division, print_function

import copy
import math
import json
import logging
import os
import random
import subprocess
import datetime

import numpy as np
import torch
from torch import nn
from .pretrained.bert import BertModel, BertTokenizer, BertAdam, LinearWarmUpScheduler
from .modules import Biaffine, MLP, eisner
from .pretrained.zen2 import ZenModel, ZenNgramDict
from .pretrained.xlnet import XLNetModel, XLNetTokenizer
from ..utils import ispunct

from ..utils.io_utils import save_json, load_json, read_embedding, get_language
from tqdm import tqdm, trange
from ..eval.Par_eval import Evaluator
from ..utils.Web_MAP import Par_PRETRAINED_MODEL_ARCHIVE_MAP

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_pos': False,
    'mlp_dropout': 0.33,
    'n_mlp_arc': 500,
    'n_mlp_rel': 100,
    'use_biaffine': True,
    'use_bilstm': False,
    'lstm_layer_number': 1,
    'lstm_hidden_size': 200,
    'embedding_dim': 100,
}


class DPar(nn.Module):

    def __init__(self, labelmap, hpara, model_path, cache_dir='./', emb_word2id=None):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.arc_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.rel_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.use_biaffine = self.hpara['use_biaffine']

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

        if self.use_biaffine:
            self.mlp_arc_h = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_arc'],
                                 dropout=self.hpara['mlp_dropout'])
            self.mlp_arc_d = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_arc'],
                                 dropout=self.hpara['mlp_dropout'])
            self.mlp_rel_h = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_rel'],
                                 dropout=self.hpara['mlp_dropout'])
            self.mlp_rel_d = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara['n_mlp_rel'],
                                 dropout=self.hpara['mlp_dropout'])

            self.arc_attn = Biaffine(n_in=self.hpara['n_mlp_arc'],
                                     bias_x=True,
                                     bias_y=False)
            self.rel_attn = Biaffine(n_in=self.hpara['n_mlp_rel'],
                                     n_out=self.num_labels,
                                     bias_x=True,
                                     bias_y=True)
        else:
            self.linear_arc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.rel_classifier_1 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.rel_classifier_2 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.bias = nn.Parameter(torch.tensor(self.num_labels, dtype=torch.float), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
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
        if self.use_biaffine:

            arc_h = self.mlp_arc_h(valid_output)
            arc_d = self.mlp_arc_d(valid_output)
            rel_h = self.mlp_rel_h(valid_output)
            rel_d = self.mlp_rel_d(valid_output)

            # get arc and rel scores from the bilinear attention
            # [batch_size, seq_len, seq_len]
            s_arc = self.arc_attn(arc_d, arc_h)
            # [batch_size, seq_len, seq_len, n_rels]
            s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))
        else:
            tmp_arc = self.linear_arc(valid_output).permute(0, 2, 1)
            s_arc = torch.bmm(valid_output, tmp_arc)

            # [batch_size, seq_len, seq_len, n_rels]
            rel_1 = self.rel_classifier_1(valid_output)
            rel_2 = self.rel_classifier_2(valid_output)
            rel_1 = torch.stack([rel_1] * max_len, dim=1)
            rel_2 = torch.stack([rel_2] * max_len, dim=2)
            s_rel = rel_1 + rel_2 + self.bias
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['mlp_dropout'] = args.mlp_dropout
        hyper_parameters['n_mlp_arc'] = args.n_mlp_arc
        hyper_parameters['n_mlp_rel'] = args.n_mlp_rel
        hyper_parameters['use_biaffine'] = args.use_biaffine

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
    def load_model(cls, model_path, language='en', use_Biaffine=False, local_rank=-1, no_cuda=False):
        # assign model path
        model_path = cached_DNLP(model_path, language, use_Biaffine)
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

    @staticmethod
    def set_not_grad(module):
        for para in module.parameters():
            para.requires_grad = False

    def load_data(self, data_path=None, sentence_list=None):
        if data_path is not None:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
            lines = readfile(data_path, flag)
        elif sentence_list is not None:
            flag = 'predict'
            lines = [(i,[0]*len(i),['advmod']*len(i)) for i in sentence_list]
        else:
            raise ValueError()
        examples = self.process_data(lines, flag)
        return examples

    @staticmethod
    def process_data(lines, flag):
        examples = []
        for i, (sentence, head, label) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, head=head,
                                         label=label))
        return examples

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        arc_loss = self.arc_criterion(arc_scores, arcs)
        rel_loss = self.rel_criterion(rel_scores, rels)
        loss = arc_loss + rel_loss

        return loss

    @staticmethod
    def decode(arc_scores, rel_scores, mask):
        arc_preds = eisner(arc_scores, mask)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def convert_examples_to_features(self, examples, language):

        tokenizer = self.tokenizer

        features = []

        length_list = []
        tokens_list = []
        head_idx_list = []
        labels_list = []
        valid_list = []
        label_mask_list = []
        punctuation_idx_list = []

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            head_list = example.head
            tokens = []
            head_idx = []
            labels = []
            valid = []
            label_mask = []

            punctuation_idx = []

            if len(textlist) > self.max_seq_length - 2:
                textlist = textlist[:self.max_seq_length - 2]
                labellist = labellist[:self.max_seq_length - 2]
                head_list = head_list[:self.max_seq_length - 2]

            for i, word in enumerate(textlist):
                if ispunct(word):
                    punctuation_idx.append(i+1)

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
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        head_idx.append(head_list[i])
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            length_list.append(len(tokens))
            tokens_list.append(tokens)
            head_idx_list.append(head_idx)
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            punctuation_idx_list.append(punctuation_idx)

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        label_pad_length = max(label_len_list) + 1

        for indx, (example, tokens, head_idxs, labels, valid, label_mask, punctuation_idx) in \
                enumerate(zip(examples, tokens_list, head_idx_list,
                              labels_list, valid_list, label_mask_list, punctuation_idx_list)):

            ntokens = []
            segment_ids = []
            label_ids = []
            head_idx = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            head_idx.append(-1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])
                head_idx.append(head_idxs[i])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)

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
                head_idx.append(-1)
                label_ids.append(0)
                label_mask.append(0)

            eval_mask = copy.deepcopy(label_mask)
            eval_mask[0] = 0
            # ignore all punctuation if not specified
            for idx in punctuation_idx:
                if idx < label_pad_length:
                    eval_mask[idx] = 0

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(head_idx) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

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
                              head_idx=head_idx,
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
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_head_idx = torch.tensor([f.head_idx for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        head_idx = all_head_idx.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)

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

        return input_ids, input_mask, l_mask, eval_mask, head_idx, label_ids, \
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

        logger.info(vars( args))

        if  args.server_ip and  args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=( args.server_ip,  args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not  args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device( args.local_rank)
            device = torch.device("cuda",  args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl', init_method= args.init_method, rank= args.rank,
                                                 world_size= args.world_size)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool( args.local_rank != -1),  args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

        args.train_batch_size =  args.train_batch_size //  args.gradient_accumulation_steps

        random.seed( args.seed)
        np.random.seed( args.seed)
        torch.manual_seed( args.seed)

        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')

        if  args.model_name is None:
            raise Warning('model name is not specified, the model will NOT be saved!')
        output_model_dir = os.path.join('./saved_models',  args.model_name + '_' + now_time)

        word2id = get_word2id( args.train_data_path)
        logger.info('# of word in train: %d: ' % len(word2id))

        if args.use_bilstm and args.model_path is None:
            emb_word2id = get_character2id(args.train_data_path)
        else:
            emb_word2id = None

        label_list = get_label_list( args.train_data_path)
        logger.info('# of tag types in train: %d: ' % (len(label_list) - 3))
        label_map = {label: i for i, label in enumerate(label_list, 1)}


        hpara = cls.init_hyper_parameters( args)
        dep_parser = cls(label_map, hpara, model_path=args.model_path, cache_dir= args.cache_dir, emb_word2id=emb_word2id)

        train_examples = dep_parser.load_data( args.train_data_path)
        dev_examples = dep_parser.load_data( args.dev_data_path)
        test_examples = dep_parser.load_data( args.test_data_path)

        language = get_language(''.join(train_examples[0].text_a.strip().split(' ')))

        eval_data = {
            'dev': dev_examples,
            'test': test_examples
        }

        convert_examples_to_features = dep_parser.convert_examples_to_features
        feature2input = dep_parser.feature2input
        get_loss = dep_parser.get_loss
        decode = dep_parser.decode

        total_params = sum(p.numel() for p in dep_parser.parameters() if p.requires_grad)
        logger.info('# of trainable parameters: %d' % total_params)

        num_train_optimization_steps = int(
            len(train_examples) /  args.train_batch_size /  args.gradient_accumulation_steps) *  args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.fp16:
            dep_parser.half()
        dep_parser.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            dep_parser = DDP(dep_parser)
        elif n_gpu > 1:
            dep_parser = torch.nn.DataParallel(dep_parser)

        param_optimizer = list(dep_parser.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if  args.fp16:
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

            if  args.loss_scale == 0:
                model, optimizer = amp.initialize(dep_parser, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(dep_parser, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
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
        best_dev_uas = -1
        best_dev_las = -1
        best_test_uas = -1
        best_test_las = -1

        history = {
            'dev': {'epoch': [], 'uas': [], 'las': []},
            'test': {'epoch': [], 'uas': [], 'las': []},
        }
        num_of_no_improvement = 0
        patient =  args.patient

        global_step = 0


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d",  args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int( args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            dep_parser.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples),  args.train_batch_size))):
                dep_parser.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                  args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples, language)

                input_ids, input_mask, l_mask, eval_mask, arcs, rels, ngram_ids, ngram_positions, \
                segment_ids, valid_ids = feature2input(device, train_features)

                arc_scores, rel_scores = dep_parser(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                                    ngram_ids, ngram_positions)
                l_mask[:, 0] = 0
                loss = get_loss(arc_scores, rel_scores, arcs, rels, l_mask)

                if np.isnan(loss.to('cpu').detach().numpy()):
                    raise ValueError('loss is nan!')
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if  args.gradient_accumulation_steps > 1:
                    loss = loss /  args.gradient_accumulation_steps

                if  args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) %  args.gradient_accumulation_steps == 0:
                    if  args.fp16:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            dep_parser.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                prediction = {
                    'dev': {},
                    'test': {}
                }
                logger.info('\n')
                for flag in eval_data.keys():
                    evaluator = Evaluator()
                    eval_examples = eval_data[flag]
                    dep_parser.eval()
                    all_arcs, all_rels = [], []
                    label_map = {i: label for i, label in enumerate(label_list, 1)}
                    label_map[0] = '<unk>'
                    for start_index in range(0, len(eval_examples),  args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index +  args.eval_batch_size,
                                                                             len(eval_examples))]

                        eval_features = convert_examples_to_features(eval_batch_examples, language)

                        input_ids, input_mask, l_mask, eval_mask, arcs, rels, ngram_ids, ngram_positions, \
                        segment_ids, valid_ids = feature2input(device, eval_features)

                        with torch.no_grad():
                            arc_scores, rel_scores = dep_parser(input_ids, segment_ids, input_mask, valid_ids,
                                                                l_mask,
                                                                ngram_ids, ngram_positions)
                        l_mask[:, 0] = 0
                        arc_preds, rel_preds = decode(arc_scores, rel_scores, l_mask)
                        evaluator(arc_preds, rel_preds, arcs, rels, eval_mask)

                        lens = l_mask.sum(1).tolist()
                        all_arcs.extend(arc_preds[l_mask].split(lens))
                        all_rels.extend(rel_preds[l_mask].split(lens))

                    all_arcs = [seq.tolist() for seq in all_arcs]
                    all_rels = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_rels]

                    prediction[flag]['all_arcs'] = all_arcs
                    prediction[flag]['all_rels'] = all_rels

                    uas, las = evaluator.uas * 100, evaluator.las * 100
                    report = '%s: Epoch: %d, UAS:%.2f, LAS:%.2f' % (flag, epoch + 1, uas, las)
                    logger.info(report)
                    history[flag]['epoch'].append(epoch)
                    history[flag]['uas'].append(uas)
                    history[flag]['las'].append(las)

                    if  args.model_name is not None:
                        if not os.path.exists(output_model_dir):
                            os.makedirs(output_model_dir)

                        output_eval_file = os.path.join(output_model_dir, flag + "_report.txt")
                        with open(output_eval_file, "a") as writer:
                            writer.write(report)
                            writer.write('\n')

                logger.info('\n')
                if history['dev']['las'][-1] > best_dev_las:
                    best_epoch = epoch + 1
                    best_dev_uas = history['dev']['uas'][-1]
                    best_dev_las = history['dev']['las'][-1]
                    best_test_uas = history['test']['uas'][-1]
                    best_test_las = history['test']['las'][-1]
                    num_of_no_improvement = 0

                    if args.model_name:
                        for flag in ['dev', 'test']:
                            with open(os.path.join(output_model_dir, flag + '_result.txt'), "w") as writer:
                                writer.write("Epoch: %d, dev_UAS: %f, dev_LAS: %f, test_UAS: %f, test_LAS: %f\n\n" % (
                                    best_epoch, best_dev_uas, best_dev_las, best_test_uas, best_test_las))
                                all_arcs = prediction[flag]['all_arcs']
                                all_rels = prediction[flag]['all_rels']
                                examples = eval_data[flag]
                                for example, arcs, rels in zip(examples, all_arcs, all_rels):
                                    words = example.text_a.split(' ')
                                    for word, arc, rel in zip(words, arcs, rels):
                                        line = '%s\t%s\t%s\n' % (word, arc, rel)
                                        writer.write(line)
                                    writer.write('\n')

                        # model_to_save = dep_parser.module if hasattr(dep_parser, 'module') else dep_parser
                        best_eval_model_dir = os.path.join(output_model_dir, 'model')
                        if not os.path.exists(best_eval_model_dir):
                            os.makedirs(best_eval_model_dir)

                        if args.model_path is None:
                            dep_parser.save_model(best_eval_model_dir)
                        elif '/' in args.model_path:
                            dep_parser.save_model(best_eval_model_dir, args.model_path)
                        elif '-' in args.model_path:
                            dep_parser.save_model(best_eval_model_dir, args.cache_dir)
                        else:
                            raise ValueError()

                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        best_report = "Epoch: %d, dev_UAS: %f, dev_LAS: %f, test_UAS: %f, test_LAS: %f" % (
            best_epoch, best_dev_uas, best_dev_las, best_test_uas, best_test_las)
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

    def predict(self, sentence_list, seperated_type='list', eval_batch_size=16):
        # no_cuda = not next(self.parameters()).is_cuda
        eval_examples = self.load_data(sentence_list=sentence_list)
        language = get_language(''.join(eval_examples[0].text_a.strip().split(' ')))

        self.eval()
        result_list = []
        label_map = {v: k for k, v in self.labelmap.items()}
        label_map[0] = '<unk>'
        for start_index in range(0, len(eval_examples), eval_batch_size):
            eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                 len(eval_examples))]

            eval_features = self.convert_examples_to_features(eval_batch_examples, language)

            input_ids, input_mask, l_mask, eval_mask, arcs, rels, ngram_ids, ngram_positions, \
            segment_ids, valid_ids = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                arc_scores, rel_scores = self.forward(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                                           ngram_ids, ngram_positions)
            l_mask[:, 0] = 0
            arc_preds, rel_preds = self.decode(arc_scores, rel_scores, l_mask)

            lens = l_mask.sum(1).tolist()

            sentence = [i.text_a.split(' ') for i in eval_batch_examples]
            arcs = arc_preds[l_mask].split(lens)
            rels = [[label_map[label_id] for label_id in seq.tolist()] for seq in rel_preds[l_mask].split(lens)]

            result_list.extend([(i, j.tolist(), k) for i, j, k in zip(sentence, arcs, rels)])
        return result_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, head=None, label=None):
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
        self.head = head
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_idx, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.head_idx = head_idx
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



def readfile(filename, flag):
    data = []
    sentence = []
    head = []
    label = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if not flag == 'predict':
            for line in lines:
                line = line.strip()
                if line == '' or line.startswith('#'):
                    if len(sentence) > 0:
                        data.append((sentence, head, label))
                        sentence = []
                        head = []
                        label = []
                    continue
                splits = line.split('\t')
                sentence.append(splits[1])
                head.append(int(splits[6]))
                label.append(splits[7])
            if len(sentence) > 0:
                data.append((sentence, head, label))
        else:
            raise ValueError()
            # for line in lines:
            #     line = line.strip()
            #     if line == '':
            #         continue
            #     label_list = ['NN' for _ in range(len(line))]
            #     data.append((line, label_list))
    return data


def get_word2id(train_data_path):
    word2id = {'<PAD>': 0}
    index = 1
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            splits = line.split('\t')
            word = splits[1]
            if word not in word2id:
                word2id[word] = index
                index += 1
    return word2id


def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            splits = line.split('\t')
            dep = splits[7]
            if dep not in label_list:
                label_list.append(dep)

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


def cached_DNLP(model_path, language, use_Biaffine):
    logger = logging.getLogger(__name__)
    if use_Biaffine:
        language += '_Biaffine'
    else:
        language += '_Base'
    if os.path.exists(model_path):
        return model_path
    elif model_path in Par_PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_web, model_name = Par_PRETRAINED_MODEL_ARCHIVE_MAP[model_path][language]
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
                ', '.join(Par_PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                model_path))
        return None