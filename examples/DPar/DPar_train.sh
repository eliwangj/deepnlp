#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models


#python DPar_train.py --use_biaffine  --use_bert --train_data_path=../../data/Par/cn/zh_gsdsimp-ud-train.conllu --dev_data_path=../../data/Par/cn/zh_gsdsimp-ud-dev.conllu --test_data_path=../../data/Par/cn/zh_gsdsimp-ud-test.conllu --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300   --model_name=par_use_biaffine_cn
python DPar_train.py --use_biaffine  --use_bert --train_data_path=../../data/Par/en/train.conllu --dev_data_path=../../data/Par/en/dev.conllu --test_data_path=../../data/Par/en/test.conllu --model_path=../../data/models/bert-large-uncased --max_seq_length=300   --model_name=par_use_biaffine_en --do_lower_case
