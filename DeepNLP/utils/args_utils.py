# -*- coding:utf-8 -*-
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--task",
    #                    default=None,
    #                    type=str,
    #                    required=True,
    #                    help="Which task to run training.")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        help="The path of local pre-trained model or pre-trained model selected in the list: model-base-uncased, "
                             "model-large-uncased, model-base-cased, model-large-cased, model-base-multilingual-uncased, "
                             "model-base-multilingual-cased, model-base-chinese.")
    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        required=True,
                        help="The name of saved model")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The training data path. Should contain the .tsv or .conllu files for the task.")
    parser.add_argument("--dev_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The dev data path. Should contain the .tsv or .conllu files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The test data path. Should contain the .tsv or .conllu files for the task.")
    parser.add_argument("--decoder",
                        default='crf',
                        type=str,
                        help="Which task to run training.")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--use_xlnet",
                        action='store_true',
                        help="Whether to use XLNET.")
    parser.add_argument("--use_bilstm",
                        action='store_true',
                        help="Whether to use biLSTM.")
    parser.add_argument("--lstm_layer_number",
                        default=2,
                        type=int,
                        help="The number of biLSTM Layer")
    parser.add_argument("--lstm_hidden_size",
                        default=200,
                        type=int,
                        help="The size of LSTM hidden layer")
    parser.add_argument("--embedding_dim",
                        default=200,
                        type=int,
                        help="The dimension of word embedding")
    # parser.add_argument("--pretrained_embedding_file",
    #                     default=None,
    #                     type=str,
    #                     help="The path to pretrained word embedding")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=30, help="Patient for the early stop.")

    parser.add_argument('--ngram_num_threshold', type=int, default=5, help="The threshold of n-gram frequency")
    parser.add_argument('--av_threshold', type=int, default=5, help="av threshold")
    parser.add_argument("--use_memory", default=False, action='store_true')
    parser.add_argument('--ngram_type', type=str, default='av', help="")
    parser.add_argument('--cat_type', type=str, default='length', help="")
    parser.add_argument('--cat_num', type=int, default=10, help="")
    parser.add_argument('--max_ngram_length', type=int, default=10, help="The maximum number of characters")


    parser.add_argument('--mlp_dropout', type=float, default=0.33, help='')
    #dependency
    parser.add_argument('--n_mlp_arc', type=int, default=500, help='')
    parser.add_argument('--n_mlp_rel', type=int, default=100, help='')
    parser.add_argument("--use_biaffine", default=False, action='store_true')
    #srl
    parser.add_argument('--n_mlp', type=int, default=400, help='')
    #relation
    parser.add_argument('--dataset_name', type=str, help='Choose the evaluation method according to dataset')
    #sentiment
    parser.add_argument('--use_tgcn', action='store_true', help='Whether use the tgcn')
    parser.add_argument('--layer_number', default=2, type=int, help='Whether use the tgcn')
    parser.add_argument('--joint_pos', action='store_true', help='')  #联合pos
    
    return parser
