# >----------------- #
# -----------------< #
# import pdb; pdb.set_trace()
from __future__ import absolute_import, division, print_function

import math
from numpy.core.fromnumeric import _ravel_dispatcher   
import torch     # pytorch
import json
import logging
import os
import warnings
import random
import datetime
import subprocess
import numpy as np
import csv  # added
from torch import nn
from torch._C import NoneType

from .pretrained.bert import BertModel, BertTokenizer, BertAdam, LinearWarmUpScheduler
from .pretrained.xlnet import XLNetModel, XLNetTokenizer
from .pretrained.zen2 import ZenModel, ZenNgramDict
from .modules import Biaffine, MLP, CRF
from ..utils.io_utils import save_json, load_json, read_embedding, get_language
from tqdm import tqdm, trange
# from ..eval.SRL_eval import to_eval_file, get_prf, fix_verb
from ..eval.SRL_eval_dep import get_prf
from ..utils.Web_MAP import SRL_PRETRAINED_MODEL_ARCHIVE_MAP

DEFAULT_HPARA = {               # default hyper parameters
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
    'feat_dim_sense': 200
}


class DSRL(nn.Module):

    def __init__(self, labelmap, hpara, model_path, sense2id_dic, verb2sense_dic, emb_word2id=None):    # Usage: Initialize the class DSRL
        super().__init__()
        self.labelmap = labelmap              # labelmap: 
        self.hpara = hpara  # hyper parameters
        self.num_labels = len(self.labelmap) + 1            # why +1 ?
        self.max_seq_length = self.hpara['max_seq_length']  # 

        if hpara['use_zen']:        # we can't use ZEN?
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

        # >----------------- #
        self.sense2id_dic = sense2id_dic # initialization; a dictionary
        self.verb2sense_dic = verb2sense_dic
        self.sense_embedding = nn.Embedding(len(self.sense2id_dic), self.hpara['feat_dim_sense'])  # Initialize an Embedding module for senses, containing #senses tensors of size 200.
        # -----------------< #

        if self.hpara['use_bilstm']: # Which pretrained transformer-based models
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
            hidden_size = self.bert.config.hidden_size # is this 768 (base)?
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

        # ------------------ #
        # define the fully-connected layer (predicate embeddings -> sense embeddings)
        self.mlp_pred2sense_h = MLP(n_in=hidden_size,
                                    n_hidden=self.hpara['feat_dim_sense'])
        # ------------------ #

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
                attention_mask_label=None, batch_examples=None, verb_index=None, labels=None, sense_indices=None,
                input_ngram_ids=None, ngram_position_matrix=None):
                # attention_mask
                # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]: 1 indicates a value that should be attended to, 0 indicates a padded value.

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


        batch_size, _, feat_dim = sequence_output.shape # record shape 啥是'_'?
        max_len = attention_mask_label.shape[1]  # maximum length of a sentence
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device) # create a tensor filled with the scalar value 0
        # 将batch里每个句子的每个词的embedding提取到valid_output这个tensor里面
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            sent_len = attention_mask_label[i].sum()
            valid_output[i][:sent_len] = temp[:sent_len]

        valid_output = self.dropout(valid_output)

        # 将batch里每个predicate的embedding提取到predicates这个tensor里面
        predicates = torch.zeros(batch_size, feat_dim, dtype=valid_output.dtype, device=valid_output.device)  # torch.Size([batch_size, feat_dim]), e.g. torch.Size([16, 768])
        for i in range(batch_size):
            predicates[i] = valid_output[i][verb_index[i][0]]


        # >----------------- #
        predicate_converted = self.mlp_pred2sense_h(predicates)  # go through a fully-connected layer (predicate embeddings -> sense embeddings); torch.Size([batch_size, feat_dim_sense]), e.g. torch.Size([16, 200])
       
        sense_candidates = [] # indices of possible senses to be predicted from # (assume sense_pad_length = 4) [[97, 102, -1, -1], [10, 86, 37, 7], ...]
        sense_pad_length = 10
        for i in range(batch_size):
            target_verb = batch_examples[i].text_a[int(verb_index[i][0]) - 1]  # minus 1 b/c the index in a list starts from 0 where as verb_index starts from 1
            sense_ids = get_possible_sense_ids(target_verb, self.sense2id_dic, self.verb2sense_dic) # given the verb-to-be-disambiguated, we obtain the indices of possible senses it corresponds to
            # Padding
            while len(sense_ids) < sense_pad_length:
                sense_ids.append(0)   # b/c index 0 is already used for a sense
            sense_candidates.append(sense_ids)
            
        # print('\nvalid_output.device: \n', valid_output.device)
        senses_idx = torch.tensor(sense_candidates, device=valid_output.device)  # senses_idx.shape = torch.Size([16, 5]), if 5 if the maximum number of senses for a predicate; don't forget to put on the same device
        
        target_emb = self.sense_embedding(senses_idx) # retrieve embeddings of target senses using indices.

        predicate_converted = torch.unsqueeze(predicate_converted, 2)    # (batch_size, feat_dim_sense) -> (batch_size, feat_dim_sense, 1)

        ## multiply the two matrices to get the inner products
        product_scores = torch.bmm(target_emb, predicate_converted) # (batch_size, max(sense_number) , feat_dim_sense) * (batch_size, feat_dim_sense, 1) = (batch_size, max(sense_number), 1)
        product_scores = torch.squeeze(product_scores, 2)  # (batch_size, sense_pad_length, 1) -> (batch_size, sense_pad_length); values of inner products
        
        # -----------------< #


        pre_h = self.mlp_pre_h(predicates)
        arg_h = self.mlp_arg_h(valid_output)

        # [batch_size, seq_len, n_labels]
        s_labels = self.srl_attn(arg_h, pre_h).permute(0, 2, 1)


        # >----------------- #
        if labels is not None: # if training
            gold_senses = sense_indices

            if self.crf is not None:
                print('self.crf is not None')
                return -1 * self.crf(emissions=s_labels, tags=labels, mask=attention_mask_label)
            else:
                print('self.crf is None')
                s_labels = s_labels[attention_mask_label] # predicted labels; torch.Size([647, 14]), 14 = number of classes
                labels = labels[attention_mask_label]   # gold standard labels, 这个batch里所有的词（e.g. 647）
                return self.loss_function(s_labels, labels) + self.loss_function(product_scores, gold_senses)

        else: # if evaluation
            result_indices = []  # pure list
            for t in product_scores:
                index = torch.argmax(t).item()  # .item() converts tensor(x) to int x
                result_indices.append(index)
            # result_indices = torch.Tensor(result_indices)  # convert [10, 1, 8, 5, 19] to tensor([10., 1., 8., 5., 19.])

            if self.crf is not None:
                print('self.crf is not None')
                return self.crf.decode(s_labels, attention_mask_label)[0], result_indices
            else:
                print('self.crf is None')
                pre_labels = torch.argmax(s_labels, dim=2)  # s_labels.shape: torch.Size([16, 73, 14]). pick the highest value out of 14 to be the prediction
                return pre_labels, result_indices
        # -----------------< #

    @staticmethod
    def init_hyper_parameters(args): # Usage: Initialize the hyper parameters
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
        # hyper_parameters['feat_dim_sense'] is just the default value

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
        # print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(local_rank != -1), fp16))
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

    def load_data(self, data_path=None, sentence_list=None, verb_index_list=None):  #
        if data_path is not None:
            flag = data_path[data_path.rfind('/') + 1: data_path.rfind('.')]  # flag = 'train' or 'test' or 'dev'
            lines = readfile(data_path) # 有的时候不是readfile, 是read_tsv（只有句子和label, 没有verb_index）。只在这个DSRL.py里出现。DSeg等用的是read_csv.
        elif sentence_list is not None and verb_index_list is not None:
            flag = 'predict'
            lines = [(i, ['O'] * len(i), j) for i, j in zip(sentence_list, verb_index_list)] # lines will be processed into examples (in process_data)
            raise ValueError('You must input <data path> or <sentence_list and verb_index_list> together by list of list. ')
        examples = self.process_data(lines, flag)
        return examples

    @staticmethod
    def process_data(lines, flag):

        examples = [] # create an empty list to store InputExample objects 
        for i, (sentence, labels, verb_index, sense_label) in enumerate(lines): # lines are from raw data file; 
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, # convert raw data to structured objects (class InputExample)
                                         labels=labels, verb_index=verb_index, sense_label=sense_label))
        return examples  # return 'examples', which is a list of InputExample objects

    def convert_examples_to_features(self, examples, language): # this function converts untokenized data(InputExamples) to data(InputFeatures) that can be used to feed in BERT
        tokenizer = self.tokenizer  # rename the tokenizer

        features = []      # feature list

        length_list = []   # [20, 15, 30,...] stores the number of tokens: len(tokens)
        tokens_list = []   # stores tokens
        labels_list = []   # stores labels
        valid_list = []    # stores the index of valid token of each word
        label_mask_list = []  # list of the IDs of masked words
        eval_mask_list = [] # 0 if verbs; 1 for those to be predicted
        # >----------------- #
        # sense_indices = []
        # -----------------< #

        # senses_list = []
        # sense_mask_list = []

        for (ex_index, example) in enumerate(examples):
            text_list = example.text_a    # The untokenized text 这里是 a list of strings(words) 吗，一个example的text_list包含一句话的所有单词？
            label_list = example.labels  # The label, e.g, B-ARG1, I-ARG0
            verb_index = example.verb_index # The index of the predicate of the sentence example
            # >----------------- #
            # sense_label = example.sense_label
            # sense_idx = get_sense_id(sense_label)
            # sense_indices.append(sense_idx)
            # -----------------< #

            
            tokens = []  # tokens, e.g. ['Let', "'", 's', 'learn', 'deep', 'learning', '!']
            labels = []  # labels of tokens
            valid = []   # the index of valid token of each word
            label_mask = [] # 1 if labeled; 0 otherwise
            eval_mask = [] # 0 if verbs; 1 for those to be predicted
            

            if len(text_list) > self.max_seq_length - 2:  # minus 2 because [V] and [\V] are added in text_list
                continue # skip the loop if the length of the sentence is too long

            assert verb_index[-1] - verb_index[0] == len(verb_index) - 1 # check if the length of verb_index span matches


            # add [V] and [\V] to the beginning and ending of the predicate
            # self-attention BERT
            new_textlist = [w for w in text_list[:verb_index[0]]] # list of words until the first predicate
            new_textlist.append('[V]') # append '[V]' before the first predicate
            new_textlist.extend([w for w in text_list[verb_index[0]: verb_index[-1] + 1]])  # add the middle part into the list
            new_textlist.append('[/V]') # append '[/V]' after the last predicate
            new_textlist.extend([w for w in text_list[verb_index[-1] + 1:]])  # add the last part
            assert len(new_textlist) == len(label_list) + 2  # check if '[V]' and '[/V]' have been indeed added
            text_list = new_textlist # update text_list

            tmp = 0
            for i, word in enumerate(text_list): # i is the id; word is the actual word

                if tokenizer: # if tokenizer is valid
                    token = tokenizer.tokenize(word) # the token(s) of the word
                elif word in self.emb_word2id or word in ['[V]', '[/V]']:
                    token = [word]
                else: 
                    if language == 'zh': # chinese
                        token = list(word)
                    elif language == 'en': # english
                        token = [word]

                tokens.extend(token) # add token to the list of tokens
                if word == '[V]' or word == '[/V]': 
                    for _ in range(len(token)): # why do we use '_' ??
                        valid.append(0) # add all zeros to the valid list, each corresponds to a token
                    tmp += 1 # tmp indicates the number of '[V]' and '[/V]' been read
                    continue
                
                label_1 = label_list[i - tmp] # get rid of '[V]' and '[/V]' because label_list doesn't contain labels for them
                # sense_1 = sense_label[i - tmp] # the sense for this particular i(th) word
                
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1) 
                        labels.append(label_1) # add label of the current word
                        # senses.append(sense_1) # add sense of the current word. '_' for non-predicate words
                        if label_1 == 'V': 
                            eval_mask.append(0) # eval_mask里predicate都是0
                            # sense_mask.append(1) # 只有V的时候sense_mask才是1，表示这里是verb
                        else:
                            eval_mask.append(1) # 非predicate都是1；
                            # sense_mask.append(0)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            

            assert tmp == 2 # check if '[V]' and '[/V]' have been both added
            assert len(tokens) == len(valid) # check if the number of tokens matches
            assert len(eval_mask) == len(label_mask) # check if the length of eval_mask(which has the index of predicates) and label_mask matches


            length_list.append(len(tokens))  # add the number of tokens
            tokens_list.append(tokens)   # add tokens
            labels_list.append(labels)  # add labels
            valid_list.append(valid) # add valid
            label_mask_list.append(label_mask)
            eval_mask_list.append(eval_mask)
            # senses_list.append(senses)
            # sense_mask_list.append(sense_mask)
            

        label_len_list = [len(label) for label in labels_list] # a list of the length of labels
        seq_pad_length = max(length_list) + 2  # the maximum amount of tokens (including [CLS] and [SEP] after we already have [V] and [/V])
        label_pad_length = max(label_len_list) # #因为CLS和SEP用不着，所以直接musk掉了，就不需要加2 the maximum amount of the number of labels

        for indx, (example, tokens, labels, valid, label_mask, eval_mask) in \
                enumerate(zip(examples, tokens_list, labels_list, valid_list, label_mask_list, eval_mask_list)):

            ntokens = [] # tokens
            segment_ids = [] # not used in this task (some tasks have two sentences)
            label_ids = [] # label ids

            ntokens.append("[CLS]") # add [CLS] at the beginning 
            segment_ids.append(0)  # for [CLS]
            valid.insert(0, 0)  # for [CLS]

            for i, token in enumerate(tokens): # index, value of each token in tokens (a sentence)
                ntokens.append(token) # append tokens 
                segment_ids.append(0) 
            for i in range(len(labels)):
                if labels[i] in self.labelmap: # if label can be identified
                    label_ids.append(self.labelmap[labels[i]]) # add the ID of the corresponding label
                else:
                    label_ids.append(self.labelmap['<UNK>'])  # unknown label
            ntokens.append("[SEP]") # add [SEP] at the end of a sentence
            segment_ids.append(0)
            valid.append(0)

            assert sum(valid) == len(label_ids) # cheeck if the number of IDs matches

            if tokenizer: # True if tokenizer has been initialized; False otherwise
                input_ids = tokenizer.convert_tokens_to_ids(ntokens) #  each token was given a unique ID in pre-trained model
            else:
                input_ids = [] # IDs for each word
                for t in ntokens:
                    t_id = self.emb_word2id[t] if t in self.emb_word2id else self.emb_word2id['<UNK>'] # use read_embedding if no tokenizer
                    input_ids.append(t_id)

            input_mask = [1] * len(input_ids)
            ## [BERT-Padding Token] BERT receives a fixed length of sentence as input. For sentences that are shorter than the maximum length, we need to add paddings (empty tokens) to the sentences to make up the length.
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)  
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                label_ids.append(0)
                label_mask.append(0)
                eval_mask.append(0)

            ## Check if the padding has worked correctly
            assert len(input_ids) == seq_pad_length  
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            # below: deal with n-gram
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

            ## Append an object of class InputFeatures to the list 'features'
            features.append( 
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            verb_index=example.verb_index,
                            label_id=label_ids,
                            valid_ids=valid,
                            label_mask=label_mask,
                            eval_mask=eval_mask,
                            # >----------------- #
                            # sense_indices = sense_indices,
                            # -----------------< #  
                            ngram_ids=ngram_ids,
                            ngram_positions=ngram_positions_matrix,
                            ngram_lengths=ngram_lengths,
                            ngram_tuples=ngram_tuples,
                            ngram_seg_ids=ngram_seg_ids,
                            ngram_masks=ngram_mask_array,
                            ))
        return features  # return the list of 'InputFeatures' objects

    def feature2input(self, device, feature): # 把处理好的数据放到GPU上去；最后输进BERT的是这里出来的；如果有的在CUDA上有的不在，是这里出了问题
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long) # put input_ids into a tensor
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long) # put masked words into a tensor
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool) # put masked labels into a tensor
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool) # put eval indicators into a tensor
        all_verb_idx = torch.tensor([[f.verb_index[0]] for f in feature], dtype=torch.long) # put verb indexes into a tensor
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long) # put true label ids into a tensor
        # >----------------- #
        # all_sense_indices = torch.tensor([f.sense_indices for f in feature], dtype=torch.long) # put the indices of senses into a tensor
         # -----------------< #
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long) # put segment ids into a tensor
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long) # put ids of the first token of every sequence into a tensor


        ## put on GPU
        input_ids = all_input_ids.to(device) 
        input_mask = all_input_mask.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)
        label_ids = all_label_ids.to(device)
        # >----------------- #
        # sense_indices = all_sense_indices.to(device)
        # -----------------< #
        segment_ids = all_segment_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        # why didn't put all_verb_idx on GPU?


        if self.zen is not None: # if we use ZEN. n-gram
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else: # not using n-gram
            ngram_ids = None
            ngram_positions = None

        return input_ids, input_mask, l_mask, eval_mask, all_verb_idx, label_ids,\
               ngram_ids, ngram_positions, segment_ids, valid_ids

    @classmethod
    def fit(cls, args): # 从头到尾看
  
    ### ------------------------------------------------ ###
    ###                 Configuration
    ### ------------------------------------------------ ###
        
        if not os.path.exists('./logs'): # creates a logs file if there isn't one
            os.mkdir('./logs')

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # get the current time
        log_file_name = './logs/log-' + now_time # file name of the logger
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', # formatting the output
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
        if args.local_rank == -1 or args.no_cuda: # if there is no CUDA; CUDA is a parallel computing platform and programming model that makes using a GPU for general purpose computing simple and elegant.
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else: # if gpu and CUDA specified
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                                 world_size=args.world_size)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

        # 分配计算梯度的数据的多少
        if args.gradient_accumulation_steps < 1: #
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


        random.seed(args.seed)  # set a random seed
        np.random.seed(args.seed) # set a random seed for numpy
        torch.manual_seed(args.seed) # set a random seed for torch

        if not os.path.exists('./saved_models'): # folder to save models generated
            os.mkdir('./saved_models')

        if args.model_name is None:  # if no model_name is specified
            raise ValueError('model name is not specified, the model will NOT be saved!')

        output_model_name = args.model_name + '_' + now_time
        # output_model_dir = os.path.join('./saved_models', args.model_name + '_' + now_time) # path of saved models
        output_model_dir = os.path.join('./saved_models', output_model_name) # path of saved models

        if args.use_bilstm and args.model_path is None:             # if BiLSTM is used and no model_path is specified
            emb_word2id = get_character2id(args.train_data_path)
        else:
            emb_word2id = None # else there is no word2id embeddings

        # >----------------- #
        sense2id_dic = get_sense2id(args.train_data_path) 
        verb2sense_dic = get_verb2sense(args.train_data_path)
        # -----------------< #

        # Generate labelmap；每个标签对应一个数字；初始化的时候有初始化
        label_list = get_label(args.train_data_path)
        logger.info('# of tag types in train: %d: ' % (len(label_list) - 3))
        label_map = {label: i for i, label in enumerate(label_list, 1)}

        hpara = cls.init_hyper_parameters(args) # intialized hyper parameters

        # >----------------- #
        sr_tagger = cls(label_map, hpara, args.model_path, sense2id_dic, verb2sense_dic, emb_word2id=emb_word2id) # Initialize DSRL class
        # ------------------< #

        train_examples = sr_tagger.load_data(args.train_data_path) # read examples from training data
        dev_examples = sr_tagger.load_data(args.dev_data_path)    # read examples from development data
        test_examples = sr_tagger.load_data(args.test_data_path)  # read examples from test data

        language = get_language(''.join(train_examples[0].text_a)) # determine language

        eval_data = {  # these are the data to be evaluated
            'dev': dev_examples,
            'test': test_examples
        }

        # if args.brown_data_path is not None:  ## if there is brown data
        #     brown_test_examples = sr_tagger.load_data(args.brown_data_path)
        #     eval_data['brown'] = brown_test_examples

        convert_examples_to_features = sr_tagger.convert_examples_to_features # initialze this function
        feature2input = sr_tagger.feature2input  # initialze this function
        save_model = sr_tagger.save_model    # initialze this function

        all_para = [p for p in sr_tagger.parameters()]
        all_named_para = [(p[0], p[1].shape, p[1].requires_grad) for p in sr_tagger.named_parameters()]

        total_params = sum(p.numel() for p in sr_tagger.parameters() if p.requires_grad) # total number of trainable parameters
        logger.info('# of trainable parameters: %d' % total_params)  # write it into log file


    ### ------------------------------------------------ ###
    ###                 Optimization
    ### ------------------------------------------------ ###
        # Optimizer, optimization steps. 优化器 如果要换的话这里要改
        num_train_optimization_steps = int( # number of training optimization steps 这是啥？
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
            # num_train_optimization_steps=-1
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
        # evaluation matrics initialization
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


    ### ------------------------------------------------ ###
    ###                 Training Model
    ### ------------------------------------------------ ###

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"): # starting training from here, one epoch at a time
            np.random.shuffle(train_examples)
            sr_tagger.train()  # puts our model in training mode
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))): #遍历每个数据，one batch at a time
                sr_tagger.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                # >-------仍需修改---------- #  get gold sense_indices
                sense_indices = []
                for example in batch_examples:
                    sense_indices.append(1)
                # -----------------< #

                train_features = convert_examples_to_features(batch_examples, language) #examples转换成features的结果, objects of InputFeatures

                input_ids, input_mask, l_mask, eval_mask, verb_index, labels, ngram_ids, ngram_positions, \
                segment_ids, valid_ids = feature2input(device, train_features) #features to input 变成输入bert的数据

                loss = sr_tagger(input_ids, segment_ids, input_mask, valid_ids, l_mask, # training mode: get a loss from BERT
                                batch_examples=batch_examples, verb_index=verb_index, labels=labels, sense_indices=sense_indices,
                                input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                if np.isnan(loss.to('cpu').detach().numpy()):
                    raise ValueError('loss is nan!')
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                ##------ back probagation ------##
                if args.fp16: 
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0: # update parameters of net
                    if args.fp16:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        scheduler.step()
                    optimizer.step()  # optimize the net
                    optimizer.zero_grad()
                    global_step += 1
            

            sr_tagger.to(device) # write to gpu


    ### ------------------------------------------------ ###
    ###                   Evaluation
    ### ------------------------------------------------ ###
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:  # 把前面的重复了一遍 但是不会做gradient descent
                arg_prediction = {flag: [] for flag in eval_data.keys()}  # eval_data.keys() are dev and test
                sense_prediction = {flag: [] for flag in eval_data.keys()}  # eval_data.keys() are dev and test

                logger.info('\n')
                for flag in eval_data.keys():
                    eval_examples = eval_data[flag]
                    sr_tagger.eval()  # puts our model in evaluation mode
                    all_arg_pred, all_sense_pred = [], []
                    label_map = {i: label for i, label in enumerate(label_list, 1)}
                    for start_index in range(0, len(eval_examples), args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                             len(eval_examples))]

                        eval_features = convert_examples_to_features(eval_batch_examples, language)

                        input_ids, input_mask, l_mask, eval_mask, verb_index, labels, ngram_ids, ngram_positions, \
                        segment_ids, valid_ids = feature2input(device, eval_features)

                        with torch.no_grad():
                            arg_pred, sense_pred = sr_tagger(input_ids, segment_ids, input_mask, valid_ids, l_mask,  # evaluation mode: get predictions
                                             batch_examples=eval_batch_examples, verb_index=verb_index, labels=None, sense_indices=sense_indices,
                                             input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions) 

                        # import pdb; pdb.set_trace()
                        lens = l_mask.sum(1).tolist() # length of a sentence (#rows)
                        all_arg_pred.extend(arg_pred[l_mask].split(lens)) # .split(lens) to seperate different sentences (of a batch). what has been added to all_pred here are a few (batch size) tensors whose values are the ids of arguments
                        all_sense_pred.extend(sense_pred) # sense_pred is a list, whose length equals batch_size

                    label_map[0] = 'O'  # add 'O' to label_map


                    all_arg_pred = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_arg_pred]
                    all_sense_pred = [sense2id_dic[sense_id] for sense_id in all_sense_pred] # only indices for predicted sense, no '_'

                    arg_prediction[flag] = all_arg_pred
                    sense_prediction[flag] = all_sense_pred
                    
                    if not os.path.exists(output_model_dir): # create the directory for this trained model
                        os.makedirs(output_model_dir)


                    # >----------------- #
                    predict_filepath = os.path.join(output_model_dir, flag + '_output.txt')
                    eval_filepath = os.path.join(output_model_dir, flag + '_eval.txt')
                    # eval_filepath = os.path.join('/data1/junqiang/dnlptk-main-sep2/examples/DSRL/saved_models', output_model_name, flag + '_eval.txt')
                    if flag == 'dev':
                        gold_filepath = args.dev_data_path
                    elif flag == 'test':
                        gold_filepath = args.test_data_path
                    convert_back(arg_prediction[flag], sense_prediction[flag], gold_filepath, predict_filepath) # a function to convert the predictions back into the official tabular form
                    # -----------------< #

                    # eval_dir = os.path.join(output_model_dir, 'eval')
                    # if not os.path.exists(eval_dir):
                    #     os.makedirs(eval_dir)
                    # all_sentence_list = [example.text_a for example in eval_examples]
                    # gold_eval_file = os.path.join(eval_dir, flag + '.%d.gold.props' % (epoch+1))
                    # to_eval_file(gold_eval_file, all_sentence_list, all_gold)

                    # pred_eval_file = os.path.join(eval_dir, flag + '.%d.pred.props' % (epoch+1))
                    # new_all_pred = fix_verb(all_gold, all_pred)
                    # to_eval_file(pred_eval_file, all_sentence_list, new_all_pred)

                    # output_report_file = os.path.join(eval_dir, flag + '.%d.eval.report' % (epoch+1))
                    # eval_path = os.path.abspath(os.path.join(__file__, '../..'))

                    evalscript_path = '/data1/junqiang/dnlptk-main-sep2/DeepNLP/eval/srlconll-1.1/bin/eval_09.pl'
                    # command = 'perl ' + evalscript_path + ' -o %s -g %s -s %s' % (eval_filepath, gold_filepath, predict_filepath) # define the command that executes the eval script
                    command = 'perl ' + evalscript_path + ' -g %s -s %s > %s' % (gold_filepath, predict_filepath, eval_filepath) # define the command that executes the eval script
                    subprocess.run(command, shell=True) # excute the command
                    
                    p, r, f = get_prf(eval_filepath) # get the p, r, f from evaluation output file

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

                    # num_of_no_improvement = 0

                    # if args.model_name:
                    #     for flag in eval_data.keys():
                    #         with open(os.path.join(output_model_dir, flag + '_result.txt'), "w") as writer:
                    #             writer.write('word\tpred\tgold\n\n')
                    #             all_labels = prediction[flag]   # predicted labels
                    #             examples = eval_data[flag]      # gold standard labels
                    #             for example, labels in zip(examples, all_labels):
                    #                 words = example.text_a
                    #                 gold_labels = example.labels
                    #                 for word, label, gold_label in zip(words, labels, gold_labels):
                    #                     line = '%s\t%s\t%s\n' % (word, label, gold_label)
                    #                     writer.write(line)
                    #                 writer.write('\n')

        #                 if args.model_path == None:
        #                     save_model(output_model_dir)
        #                 elif '/' in args.model_path:
        #                     save_model(output_model_dir, args.model_path)
        #                 elif '-' in args.model_path:
        #                     save_model(output_model_dir, args.cache_dir)
        #                 else:
        #                     raise ValueError()
        #         else:
        #             num_of_no_improvement += 1

        #     if num_of_no_improvement >= patient:
        #         logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
        #         break

        best_report = "Epoch: %d, dev_p: %.2f, dev_r: %.2f, dev_f: %.2f, " \
                      "test_p: %.2f, test_r: %.2f, test_f: %.2f" % (
            best_epoch, best_dev_p, best_dev_r, best_dev_f, best_test_p, best_test_r, best_test_f)

        if best_brown_f > 0:
            best_report += ', brown_p: %f, brown_r: %f, brown_f %f' % (best_brown_p, best_brown_r, best_brown_f)

        logger.info("\n=======best f at dev========")
        logger.info(best_report)
        logger.info("\n=======best f at dev========")

        if args.model_name is not None:
            final_report_path = os.path.join(output_model_dir, "final_report.txt")
            with open(final_report_path, "w") as writer:
                writer.write("total #parameters: ")
                writer.write(str(total_params))
                writer.write('\n Best epoch: \n')
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

        for start_index in tqdm(range(0, len(eval_examples), eval_batch_size)):
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
        # print('write results to %s' % str(args.output_file))
        # with open(args.output_file, 'w', encoding='utf8') as writer:
        #     for i in range(len(y_pred)):
        #         sentence = eval_examples[i].text_a
        #         _, seg_pred_str = eval_sentence(y_pred[i], None, sentence, word2id)
        #         writer.write('%s\n' % seg_pred_str)

        return result_list



class InputExample(object): # the objects of InputExample are used in load_data function
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None, verb_index=None, sense_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            verb_index: The index of the predicate in this sentence.
            sense_label: (Optional) The label of sense of this single current predicate
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.verb_index = verb_index
        self.sense_label = sense_label


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


def readfile(filepath): # anything related to file/data reading is here, no need to modify other places
    data = []  

    I_FORM = 1
    I_FILLPRED = 12
    I_PoneA = 14 # Pone = the first predicate; A = argument

    data_dic = {} # {'predicate1': (sentence, label, verb_index), 'predicate2': (sentence, label, verb_index), ...}
    pred_count = -1

    with open(filepath, 'r', encoding='utf8') as f:
        lines = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in lines:
            ## if this is an empty line, which indicates the end of a sentence, reset the dictionary and all sentence info
            if line == []:  
                for v in range(NUM_PRED): # one predicate at a time
                    data.append(data_dic[v]) # add the data entry, v is the index and also the key of a corresponding predicate
                
                data_dic = {}   # reset the dictionary
                pred_count = -1 # reset the predicate count
                continue
            
            ## info of a data item
            NUM_PRED = len(line) - I_PoneA
            COL_SENSE = 13  # the index of the column of senses
            fillv = line[I_FILLPRED]


            ## Initialize the data_dic dictionary when reading the first line
            if line[0] == '1':
                for v in range(NUM_PRED):
                    data_dic[v] = ([], [], [], [])  # (sentence, labels, verb_index, sense_label)



            ## record the predicates (in index: 0 if there's one, 1 if there's two, etc.)
            if fillv == 'Y':
                pred_count += 1  

            ## Construct the data_dic dictionary
            for v in range(NUM_PRED): # if NUM_PRED = 2: v = 0, 1
                current_sentence = data_dic[v][0]
                current_labels = data_dic[v][1]
                current_verb_index = data_dic[v][2]
                current_sense = data_dic[v][3]

                sr = line[I_PoneA + v] # the column of arguments regarding the current predicate
                word = line[I_FORM]
                sense_label = line[COL_SENSE]

                current_sentence.append(word) # append the word into the sentence list

                if fillv == 'Y':
                    if pred_count == v: # if this is the current verb
                        current_labels.append('V')  # record label V
                        current_verb_index.append(len(current_labels))  # record the index of this predicate
                        current_sense.append(sense_label)  # record the sense of this predicate
                    else:   # if it's a verb but not the current verb
                        current_labels.append('O')
                        current_sense.append('_')  # the sense label is '_' if this is not the current verb

                # if fillv != 'Y':
                #     current_sense.append('_') # label '_' as sense otherwise

                if sr == '_' and fillv != 'Y':
                    current_labels.append('O')  # record label O
                elif sr != '_' and fillv != 'Y':
                    current_labels.append(sr)  # record argument label
            
                assert len(current_sentence) == len(current_labels)  # check if the number of words and number of labels match

    return data # a list of sentences of the structure (sentence, labels, verb_index, sense_label)


def get_label(train_data_path): # get a list of all types of labels 
    label_list = ['<UNK>']  # initialize the list with an unknown token

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            NUM_PRED = len(splits) - 14
            for v in range(NUM_PRED):
                srl_label = splits[14+v] # go over the argument positions horizontally
                if srl_label not in label_list and srl_label != '_':  # new label (exclude placeholder)
                    label_list.append(srl_label)    # add this label if it hasn't been recorded

    label_list.extend(['[CLS]', '[SEP]']) # beginning and seperating tokens
    return label_list


def convert_back(all_pred, sense_predictions, gold_filepath, output_filepath): # convert the predictions back into the official tabular form
    with open(gold_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        lines = infile.readlines()
        verb_count = 0
        sense_count = 0
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                verb_count += NUM_PRED
                outfile.write('\n')
                continue
            splits = line.split()
            word_index = int(splits[0]) - 1      # index of the verb in a sentence, starts from 0
            NUM_PRED = len(splits) - 14     
            
            
            ## predicate disambiguation
            if splits[12] == 'Y':
                splits[13] = sense_predictions[sense_count]
                sense_count += 1

            ## argument identification
            for v in range(NUM_PRED):  # remember that v starts from 0
                prediction = all_pred[verb_count + v][word_index]
                if prediction == '<UNK>':   # replace the gold arguments with predicted ones horizontally
                    splits[14 + v] = '_'
                else:
                    splits[14 + v] = prediction
            adding = '\t'.join(splits) + '\n'
            outfile.write(adding)
    outfile.close()


def get_character2id(train_data_path): # get a dictionary where keys are words and values are corresponding unique ids
    word2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}
    index = 4

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        splits = line.split()
        character = splits[1]  # the second column is word
        if character not in word2id:
            word2id[character] = index
            index += 1
    return word2id     

# >----------------- #
def get_sense2id(train_data_path): # get a dictionary where keys are unique ids and values are corresponding predicate senses
    sense2id = {}
    index = 0

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    for line in lines:  # Indenting back ensures that I don't hold the file open longer than necessary.
        line = line.strip()
        if line == '':
            continue
        splits = line.split()
        sense = splits[13]  # the 14th column. this column has the sense of predicate
        # if sense != '_' and splits[12] == 'Y':  # the second condition is for double check
        if splits[12] == 'Y':  # '_' should also have an index
            if sense not in sense2id:
                sense2id[index] = sense
                index += 1
    return sense2id     # {0: 'predict.01', 1: 'predict.02', ...}
# -----------------< #

# >----------------- #
def get_verb2sense(train_data_path): #每个predicates对应的可能的senses
    verb2sense = {}

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        splits = line.split()

        if splits[12] == 'Y':  # when reading the line of predicates
            verb = splits[1]    # the 2nd column, this column has the word itself
            sense = splits[13]  # the 14th column. this column has the sense of predicate
            if verb not in verb2sense:     # if this verb has been recorded
                verb2sense[verb] = [sense]
            elif sense not in verb2sense[verb]: # if this verb has been recorded but the sense has been linked with this verb
                verb2sense[verb].append(sense)
    return verb2sense
# -----------------< #

# >----------------- #
def get_possible_sense_ids(target_verb, sense2id_dic, verb2sense_dic): # given the target verb, we want to find the possible sense ids it corresponds to
    sense_ids = []
    for sense in verb2sense_dic[target_verb]:
        sense_id = get_sense_id(sense, sense2id_dic)
        sense_ids.append(sense_id)
    return sense_ids
# -----------------< #

# >----------------- #
def get_sense_id(sense, sense2id_dic): # given a sense, return its index
    position = list(sense2id_dic.values()).index(sense) # get the position of the given sense in the dictionary
    sense_id = list(sense2id_dic.keys())[position]      # get the id of the given sense
    return sense_id
# -----------------< #


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