#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models


python DSnt_train.py  --use_bilstm --train_data_path=../../data/ASA/laptop/train.txt --dev_data_path=../../data/ASA/laptop/test.txt --test_data_path=../../data/ASA/laptop/test.txt --model_path=//data/shishaoli/embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=ASA_bilstm_en
#python DSnt_train.py --train_data_path=../../data/SA/cn/train.tsv --dev_data_path=../../data/SA/cn/test.tsv --test_data_path=../../data/SA/cn/test.tsv --model_path=../../data/models/bert-base-chinese --max_seq_length=300  --model_name=SA_cn
# ASA T-GCN
#python DSnt_train.py  --train_data_path=./data/ASA/sample_data/train.txt --dev_data_path=./data/ASA/sample_data/test.txt --test_data_path=./data/ASA/sample_data/test.txt --model_path=/data/tianyuanhe/bert_model/bert_base_cased --max_seq_length=300  --model_name=asa --train_batch_size=2 --eval_batch_size=2
