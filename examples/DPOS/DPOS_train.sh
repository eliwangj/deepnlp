#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models

# ---------------- POS ----------------
#python DPOS_train.py --use_bert --train_data_path=../../data/POS/cn/train.tsv --dev_data_path=../../data/POS/cn/dev.tsv --test_data_path=../../data/POS/cn/test.tsv --model_path=../../data/models/bert-base-chinese --max_seq_length=300  --model_name=POS_baseline_cn
#python DPOS_train.py --use_bert --train_data_path=../../data/POS/en/train.tsv --dev_data_path=../../data/POS/en/dev.tsv --test_data_path=../../data/POS/en/test.tsv --model_path=../../data/models/bert-large-uncased --max_seq_length=300  --model_name=POS_baseline_cn
python DPOS_train.py --joint_pos --use_bert --train_data_path=../../data/SP/cn/train.tsv --dev_data_path=../../data/SP/cn/dev.tsv --test_data_path=../../data/SP/cn/test.tsv --model_path=../../data/models/bert-base-chinese --max_seq_length=300  --model_name=JointCWS_baseline_cn

