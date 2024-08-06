#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models


# ---------------- Rel baseline ----------------
python DRel_train.py  --task=rel --use_bert --dataset_name=semeval --train_data_path=../../data/Rel/semeval/train.tsv --dev_data_path=../../data/Rel/semeval/test.tsv --test_data_path=../../data/Rel/semeval/test.tsv --model_path=../../data/models/bert-large-uncased --max_seq_length=300  --model_name=Rel_baseline_en
