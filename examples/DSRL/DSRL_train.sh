#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models


# ---------------- SRL baseline ----------------
#python DSRL_train.py --use_bert --train_data_path=../../data/SRL/en/train.tsv --dev_data_path=../../data/SRL/en/test.tsv --test_data_path=../../data/SRL/en/test.tsv --model_path=../../data/models/bert-large-uncased --max_seq_length=300  --model_name=SRL_baseline_en --train_batch_size=32 --eval_batch_size=32
python DSRL_train.py --use_bilstm --train_data_path=../../data/SRL/en/train.tsv --dev_data_path=../../data/SRL/en/test.tsv --test_data_path=../../data/SRL/en/test.tsv --model_path=//data/shishaoli/embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=SRL_bilstm_en --train_batch_size=32 --eval_batch_size=32
