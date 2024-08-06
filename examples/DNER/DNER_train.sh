#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models

#python DNER_train.py --use_bert --train_data_path=../../data/NER/cn/train.tsv --dev_data_path=../../data/NER/cn/dev.tsv --test_data_path=../../data/NER/cn/test.tsv --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300  --model_name=NEWTEST_NER_baseline_cn
#python DNER_train.py --use_bert --train_data_path=../../data/NER/en/train.tsv --dev_data_path=../../data/NER/en/dev.tsv --test_data_path=../../data/NER/en/test.tsv --model_path=../../data/models/bert-large-uncased --max_seq_length=300  --model_name=NEWTEST_NER_baseline_en --do_lower_case
#python DNER_train.py --use_xlnet --train_data_path=../../data/NER/en/train.tsv --dev_data_path=../../data/NER/en/dev.tsv --test_data_path=../../data/NER/en/test.tsv --model_path=../../data/models/XLNet_large_cased --max_seq_length=300  --model_name=NER_baseline_en

#CUDA_VISIBLE_DEVICES=2 python DSRL_train.py --use_bert --train_data_path=../../../resources/DSRL/data/CoNLL05/train.tsv --dev_data_path=../../../resources/DSRL/data/CoNLL05/dev.tsv --test_data_path=../../../resources/DSRL/data/CoNLL05/test.tsv --model_path=../../../dnlptk-main-sep2/examples/DSRL/saved_models/en_SRL_BERT_CoNLL05_bs_0.1.0/model --max_seq_length=300  --model_name=en_SRL_BERT_CoNLL05_test   --do_lower_case --num_train_epochs=1
