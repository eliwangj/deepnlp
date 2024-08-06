from __future__ import absolute_import, division, print_function
import subprocess
import json
import logging
import os
import random
import math
import torch
import numpy as np
from torch import nn
from .pretrained.bert import BertModel, BertTokenizer, BertAdam, LinearWarmUpScheduler
from .pretrained.xlnet import XLNetModel, XLNetTokenizer
from .pretrained.zen2 import ZenModel, ZenNgramDict
from .modules import Biaffine, MLP
from ..utils.io_utils import save_json, load_json, read_embedding, get_language
from tqdm import tqdm, trange
from ..eval.Rel_eval import compute_metrics, compute_micro_f1, semeval_official_eval, tacred_official_eval
import datetime
from ..utils.Web_MAP import Rel_PRETRAINED_MODEL_ARCHIVE_MAP

DEFAULT_HPARA = {
    'max_seq_length': 508,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'mlp_dropout': 0.33,
    'n_mlp': 200,
    'use_biaffine': True,
    'use_bilstm': False,
    'lstm_layer_number': 1,
    'lstm_hidden_size': 200,
    'embedding_dim': 100,
}


class DRel(nn.Module):

    def __init__(self, labelmap, hpara, model_path, emb_word2id=None):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.use_biaffine = hpara['use_biaffine']

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
            self.tokenizer.add_never_split_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
            self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
            self.xlnet = XLNetModel.from_pretrained(model_path)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)

            self.xlnet.resize_token_embeddings(len(self.tokenizer))
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        self.mlp_e1 = MLP(n_in=hidden_size, n_hidden=self.hpara['n_mlp'], dropout=self.hpara['mlp_dropout'])
        self.mlp_e2 = MLP(n_in=hidden_size, n_hidden=self.hpara['n_mlp'], dropout=self.hpara['mlp_dropout'])

        if self.use_biaffine:
            self.biaffine = Biaffine(n_in=self.hpara['n_mlp'], n_out=self.num_labels, bias_x=True, bias_y=True)
        else:
            self.linear = nn.Linear(self.hpara['n_mlp'] * 2, self.num_labels, bias=True)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                entity_mark=None, labels=None,
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

        e1_h = sequence_output[range(sequence_output.size(0)), entity_mark[:, 0]]
        e2_h = sequence_output[range(sequence_output.size(0)), entity_mark[:, 1]]

        tmp_e1_h = self.mlp_e1(e1_h)
        tmp_e2_h = self.mlp_e2(e2_h)

        if self.use_biaffine:
            logits = self.biaffine(tmp_e1_h, tmp_e2_h)
        else:
            tmp = torch.cat([tmp_e1_h, tmp_e2_h], dim=1)
            logits = self.linear(tmp)

        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

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
        hyper_parameters['use_biaffine'] = args.use_biaffine

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
        if self.bert or self.zen or self.xlnet:
            output_config_file = os.path.join(best_eval_model_dir, 'config.json')
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
    def load_model(cls, model_path, dataset='ace2005en', local_rank=-1, no_cuda=False):
        # assign model path
        model_path = cached_DNLP(model_path, dataset)
        # select the device
        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1

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
        res.to(device) # supposed to be added
        return res

    def load_data(self, data_path=None, sentence_list=False, e1_list=None , e2_list=None):
        if data_path is not None:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
            lines = readfile(data_path)
        elif sentence_list is not None:
            flag = 'predict'
            lines = [(j, k, '<UNK>', i) for i, j, k in zip(sentence_list, e1_list, e2_list)]
        else:
            raise ValueError(
                'You must input <data path> or <sentence_list, e1_list and e2_list> together. ')

        examples = self.process_data(lines, flag)

        return examples

    @staticmethod
    def process_data(lines, flag):
        examples = []
        for i, (e1, e2, label, sentence) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, label=label, e1=e1, e2=e2))
        return examples

    def convert_examples_to_features(self, examples, language):

        tokenizer = self.tokenizer

        features = []

        length_list = []
        input_tokens_list = []
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        entity_mark_list = []

        for (ex_index, example) in enumerate(examples):

            tokens = ["[CLS]"]
            entity_mark = [0, 0]
            # previous_id = 1
            for word in example.text_a.split():
                if tokenizer:
                    token = tokenizer.tokenize(word)
                elif word in self.emb_word2id:
                    token = [word]
                elif word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                    token = [word]
                else:
                    if language == 'zh':
                        token = list(word)
                    elif language == 'en':
                        token = [word]



                if word == '<e1>':
                    entity_mark[0] = len(tokens)
                if word == '<e2>':
                    entity_mark[1] = len(tokens)
                # if word in ["</e1>"]:
                #     entity_mark[0] = previous_id
                # if word in ["</e2>"]:
                #     entity_mark[1] = previous_id
                tokens.extend(token)
                # previous_id = len(tokens) - len(token)

            if len(tokens) > 510:
                continue
            tokens.append("[SEP]")
            segment_ids = [0] * len(tokens)
            if tokenizer:
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            else:
                input_ids = []
                for t in tokens:
                    t_id = self.emb_word2id[t] if t in self.emb_word2id else self.emb_word2id['<UNK>']
                    input_ids.append(t_id)
            input_mask = [1] * len(input_ids)

            length_list.append(len(input_ids))
            input_tokens_list.append(tokens)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            entity_mark_list.append(entity_mark)

        max_seq_length = max(length_list) + 2

        for indx, (example, tokens, input_ids, input_mask, segment_ids, entity_mark) in \
                enumerate(zip(examples, input_tokens_list, input_ids_list, input_mask_list,
                              segment_ids_list, entity_mark_list)):

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = self.labelmap[example.label] if example.label in self.labelmap else self.labelmap['<UNK>']

            assert not label_id == 0

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
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
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
                              entity_mark=entity_mark,
                              label_id=label_id,
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
        all_entity_mark = torch.tensor([f.entity_mark for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)

        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)

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

        return input_ids, input_mask, all_entity_mark, label_ids, \
               ngram_ids, ngram_positions, segment_ids

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

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            # torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
            #                                      world_size=args.world_size)
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
            raise ValueError('model name is not specified, the model will NOT be saved!')
        output_model_dir = os.path.join('./saved_models', args.model_name + '_' + now_time)

        label_list = get_label_list(args.train_data_path)
        logger.info('# of relation types in train: %d: ' % (len(label_list) - 1))
        label_map = {label: i for i, label in enumerate(label_list, 1)}

        if args.use_bilstm and args.model_path is None:
            emb_word2id = get_character2id(args.train_data_path)
        else:
            emb_word2id = None

        hpara = cls.init_hyper_parameters(args)
        relation_extractor = cls(label_map, hpara, args.model_path, emb_word2id=emb_word2id)

        train_examples = relation_extractor.load_data(args.train_data_path)
        dev_examples = relation_extractor.load_data(args.dev_data_path)
        test_examples = relation_extractor.load_data(args.test_data_path)

        language = get_language(''.join(train_examples[0].text_a.strip().split(' ')))

        eval_data = {
            'dev': dev_examples,
            'test': test_examples
        }

        convert_examples_to_features = relation_extractor.convert_examples_to_features
        feature2input = relation_extractor.feature2input
        save_model = relation_extractor.save_model

        total_params = sum(p.numel() for p in relation_extractor.parameters() if p.requires_grad)
        logger.info('# of trainable parameters: %d' % total_params)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // 1
            # num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.fp16:
            relation_extractor.half()
        relation_extractor.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            relation_extractor = DDP(relation_extractor)
        elif n_gpu > 1:
            relation_extractor = torch.nn.DataParallel(relation_extractor)

        param_optimizer = list(relation_extractor.named_parameters())
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
                model, optimizer = amp.initialize(relation_extractor, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(relation_extractor, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale=args.loss_scale)
            scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                              total_steps=num_train_optimization_steps)

        else:
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
            relation_extractor.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size), colour='green', unit='sentence', unit_scale=args.train_batch_size)):
                relation_extractor.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples, language)

                input_ids, input_mask, entity_mark, labels, ngram_ids, ngram_positions, \
                segment_ids = feature2input(device, train_features)

                loss = relation_extractor(input_ids, segment_ids, input_mask,
                                          entity_mark=entity_mark, labels=labels,
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

            relation_extractor.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                prediction = {flag: [] for flag in eval_data.keys()}
                logger.info('\n')

                for flag in eval_data.keys():
                    eval_examples = eval_data[flag]
                    relation_extractor.eval()

                    pred_scores = None
                    out_label_ids = None

                    id2label = {i: label for i, label in enumerate(label_list, 1)}
                    for start_index in range(0, len(eval_examples), args.eval_batch_size):
                        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                             len(eval_examples))]

                        eval_features = convert_examples_to_features(eval_batch_examples, language)

                        input_ids, input_mask, entity_mark, labels, ngram_ids, ngram_positions, \
                        segment_ids = feature2input(device, eval_features)

                        with torch.no_grad():
                            logits = relation_extractor(input_ids, segment_ids, input_mask,
                                                        entity_mark=entity_mark, labels=None,
                                                        input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                        if pred_scores is None:
                            pred_scores = logits.detach().cpu().numpy()
                            out_label_ids = labels.detach().cpu().numpy()
                        else:
                            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
                            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

                    id2label[0] = '<UNK>'

                    all_pred_ids = np.argmax(pred_scores, axis=1)
                    all_gold_ids = out_label_ids

                    all_pred = [id2label[label_id] for label_id in all_pred_ids]
                    all_gold = [id2label[label_id] for label_id in all_gold_ids]

                    prediction[flag] = all_pred

                    if not os.path.exists(output_model_dir):
                        os.makedirs(output_model_dir)

                    if args.dataset_name == 'semeval':
                        result = semeval_official_eval(all_pred, all_gold, output_model_dir)
                    
                    elif "tacred" in args.dataset_name:
                        result = tacred_official_eval(all_pred, all_gold, output_model_dir)
                    elif "kbp37" in args.dataset_name:
                        result = compute_metrics(all_pred_ids, all_gold_ids, len(label_map),
                                                 ignore_label=label_map['no_relation'])
                        result["micro-f1"] = compute_micro_f1(all_pred_ids, all_gold_ids, id2label,
                                                              ignore_label='no_relation',
                                                              output_dir=output_model_dir)
                        result["f1"] = result["micro-f1"]
                    else:
                        result = compute_metrics(all_pred_ids, all_gold_ids, len(label_map), label_map['Other'])
                        result["micro-f1"] = compute_micro_f1(all_pred_ids, all_gold_ids, label_map, ignore_label='Other',
                                                              output_dir=output_model_dir)
                        result["f1"] = result["micro-f1"]

                    logging.info(result)

                    p, r, f = result["precision"], result["recall"], result['f1']

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

                    num_of_no_improvement = 0

                    if args.model_name:
                        for flag in eval_data.keys():
                            with open(os.path.join(output_model_dir, flag + '_result.txt'), "w") as writer:
                                writer.write("Epoch: %d, dev_p: %f, dev_r: %f, dev_f: %f, "
                                             "test_p: %f, test_r: %f, test_f: %f\n\n" % (
                                    best_epoch, best_dev_p, best_dev_r, best_dev_f,
                                    best_test_p, best_test_r, best_test_f))
                                writer.write('pred\tgold\te1\te2\t\\text\n\n')
                                all_labels = prediction[flag]
                                examples = eval_data[flag]
                                for example, labels in zip(examples, all_labels):
                                    words = example.text_a
                                    gold_labels = example.label
                                    line = '\t'.join([labels, gold_labels, example.e1, example.e2, words])
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

                        arg_file = os.path.join(output_model_dir, 'args.txt')
                        with open(arg_file, 'w', encoding='utf8') as f:
                            f.write(str(vars(args)))
                            f.write('\n')
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        best_report = "Epoch: %d, dev_p: %f, dev_r: %f, dev_f: %f, " \
                      "test_p: %f, test_r: %f, test_f: %f" % (
            best_epoch, best_dev_p, best_dev_r, best_dev_f, best_test_p, best_test_r, best_test_f)

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

    def predict(self, sentence_list, e1_list, e2_list, eval_batch_size=16):

        new_sentence_list, new_e1_list, new_e2_list = entity_pre_process(sentence_list, e1_list, e2_list)
        eval_examples = self.load_data(sentence_list=new_sentence_list, e1_list=new_e1_list, e2_list=new_e2_list)
        language = get_language(''.join(eval_examples[0].text_a.strip().split(' ')))
        label_map = {v: k for k, v in self.labelmap.items()}

        self.eval()
        pred_scores = None
        out_label_ids = None

        for start_index in tqdm(range(0, len(eval_examples), eval_batch_size), colour='green', unit='sentence', unit_scale=eval_batch_size):
            eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                 len(eval_examples))]

            eval_features = self.convert_examples_to_features(eval_batch_examples, language)

            input_ids, input_mask, entity_mark, labels, ngram_ids, ngram_positions, \
            segment_ids = self.feature2input(self.device, eval_features)

            with torch.no_grad():
                logits = self.forward(input_ids, segment_ids, input_mask,
                                      entity_mark=entity_mark, labels=None,
                                      input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

            if pred_scores is None:
                pred_scores = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        label_map[0] = '<UNK>'
        all_pred_ids = np.argmax(pred_scores, axis=1)
        all_pred = [label_map[label_id] for label_id in all_pred_ids]

        return all_pred


def entity_pre_process(sentence_list, e1_list, e2_list):
    new_sentence_list = []
    new_e1_list = []
    new_e2_list = []
    for sentence, e1, e2 in zip(sentence_list, e1_list, e2_list):
        if (type(e1 + e2) == tuple or type(e1 + e2) == list) and len(e1) == len(e2) == 2:
            e1 = sentence[e1[0]:e1[1]].strip()
            e2 = sentence[e2[0]:e2[1]].strip()
            new_e1_list.append(e1)
            new_e2_list.append(e2)
            sentence = sentence.replace(e1, "<e1> " + e1 + " </e1>").replace(e2, "<e2> " + e2 + " </e2>")
            new_sentence_list.append(sentence)
        elif e1 in sentence and e2 in sentence:
            sentence = sentence.replace(e1, "<e1> " + e1 + " </e1>").replace(e2, "<e2> " + e2 + " </e2>")
            new_sentence_list.append(sentence)
        else:
            raise ValueError('e1_list should be list of string or list of list index or entity')

    if new_e1_list and new_e2_list:
        return new_sentence_list, new_e1_list, new_e2_list
    else:
        return new_sentence_list, e1_list, e2_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, e1=None, e2=None):
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
        self.e1 = e1
        self.e2 = e2


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, entity_mark, label_id,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.entity_mark = entity_mark
        self.label_id = label_id

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readfile(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace("<e1> </e1>", "<e1> [E] </e1>").replace("<e2> </e2>", "<e2> [E] </e2>")

            splits = line.split('\t')
            if len(splits) < 1:
                continue
            e1, e2, label, sentence = splits
            sentence = sentence.strip()

            for token in ['<e1>', '</e1>', '<e2>', '</e2>']:
                sentence = sentence.replace(token, ' ' + token + ' ').replace('  ', ' ').strip()

            if len(e1) == 0:
                e1 = "[E]"
            if len(e2) == 0:
                e2 = "[E]"

            e11_p = sentence.index("<e1>")  # the start position of entity1
            e12_p = sentence.index("</e1>")  # the end position of entity1
            e21_p = sentence.index("<e2>")  # the start position of entity2
            e22_p = sentence.index("</e2>")  # the end position of entity2

            # try:
            #     e11_p = sentence.index("<e1>")  # the start position of entity1
            #     e12_p = sentence.index("</e1>")  # the end position of entity1
            #     e21_p = sentence.index("<e2>")  # the start position of entity2
            #     e22_p = sentence.index("</e2>")  # the end position of entity2
            # except:
            #     from pdb import set_trace
            #     set_trace()


            if e1 in sentence[e11_p:e12_p] and e2 in sentence[e21_p:e22_p]:
                data.append(splits)
            elif e2 in sentence[e11_p:e12_p] and e1 in sentence[e21_p:e22_p]:
                splits[0] = e2
                splits[1] = e1
                data.append(splits)
            else:
                print("data format error: {}".format(line))

    return data # each item is a list: [e1, e2, label, sentence]


def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0 or line == '\n':
                continue
            splits = line.split('\t')
            relation = splits[2]
            if relation not in label_list:
                label_list.append(relation)

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
    elif model_path in Rel_PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_web, model_name = Rel_PRETRAINED_MODEL_ARCHIVE_MAP[model_path][dataset]
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
                ', '.join(Rel_PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                model_path))
        return None