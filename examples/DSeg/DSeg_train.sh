#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models


# random embedding
#python DSeg_train.py --use_bilstm --train_data_path=/data/tianyuanhe/CWS/WMSeg/data/CTB6/dev.tsv --dev_data_path=/data/tianyuanhe/CWS/WMSeg/data/CTB6/test.tsv --test_data_path=/data/tianyuanhe/CWS/WMSeg/data/CTB6/test.tsv --max_seq_length=300  --model_name=cws_r --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
python DSeg_train.py --use_bilstm --train_data_path=../../data/Seg/cn/train.tsv --dev_data_path=../../data/Seg/cn/dev.tsv --test_data_path=../../data/Seg/cn/test.tsv --model_path=//data/shishaoli/embedding/character_embedding.txt --max_seq_length=300  --model_name=Seg_bilstm_cn --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8

# pretrained embedding
#python DSeg_train.py --pretrained_embedding_file=../../embedding/vec100.txt --use_bilstm --train_data_path=../../data/Seg/cn/train.tsv --dev_data_path=../../data/Seg/cn/dev.tsv --test_data_path=../../data/Seg/cn/test.tsv --max_seq_length=300  --model_name=cws_p --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
