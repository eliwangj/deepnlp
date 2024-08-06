#!/bin/bash
#SBATCH -J YY_1
#SBATCH -p p-RTX2080
#SBATCH -N 1
#SBATCH --gres=gpu:1


#MODEL_HOME=./pretrained_models

#python main.py
#python train_main.py  --task=seg --use_memory --use_bert --train_data_path=./data/Seg/cn/train.tsv --dev_data_path=./data/Seg/cn/dev.tsv --test_data_path=./data/Seg/cn/test.tsv --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300  --model_name=Seg_use_memory
#python train_main.py  --task=sp --use_memory --use_bert --train_data_path=./data/SP/cn/train.tsv --dev_data_path=./data/SP/cn/dev.tsv --test_data_path=./data/SP/cn/test.tsv --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300  --model_name=SP_use_memory

# ---------------- POS ----------------
#python train_main.py  --task=pos --use_bert --train_data_path=./data/POS/cn/train.tsv --dev_data_path=./data/POS/cn/dev.tsv --test_data_path=./data/POS/cn/test.tsv --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300  --model_name=POS_baseline_cn
#python train_main.py  --task=pos --use_bert --train_data_path=./data/POS/en/train.tsv --dev_data_path=./data/POS/en/dev.tsv --test_data_path=./data/POS/en/test.tsv --model_path=//data/Yangyang/models/bert-large-uncased --max_seq_length=300  --model_name=POS_baseline_en --do_lower_case

# ---------------- SP ----------------
#python train_main.py  --task=sp --use_bert --train_data_path=./data/POS/cn/train.tsv --dev_data_path=./data/POS/cn/dev.tsv --test_data_path=./data/POS/cn/test.tsv --model_path=./data/models/bert-base-chinese --max_seq_length=300  --model_name=POS_baseline_cn
#python train_main.py  --task=sp --use_bert --train_data_path=./data/POS/en/train.tsv --dev_data_path=./data/POS/en/dev.tsv --test_data_path=./data/POS/en/test.tsv --model_path=./data/models/bert-large-uncased --max_seq_length=300  --model_name=POS_baseline_cn
#python train_main.py  --task=sp --joint_pos --use_bert --train_data_path=./data/SP/cn/train.tsv --dev_data_path=./data/SP/cn/dev.tsv --test_data_path=./data/SP/cn/test.tsv --model_path=./data/models/bert-base-chinese --max_seq_length=300  --model_name=POS_baseline_cn

# ---------------- Par ----------------
#python train_main.py  --task=par --use_biaffine  --use_bert --train_data_path=./data/Par/cn/zh_gsdsimp-ud-train.conllu --dev_data_path=./data/Par/cn/zh_gsdsimp-ud-dev.conllu --test_data_path=./data/Par/cn/zh_gsdsimp-ud-test.conllu --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300   --model_name=par_use_biaffine_cn
#python train_main.py  --task=par --use_biaffine  --use_bert --train_data_path=./data/Par/en/train.conllu --dev_data_path=./data/Par/en/dev.conllu --test_data_path=./data/Par/en/test.conllu --model_path=//data/Yangyang/models/bert-large-uncased --max_seq_length=300   --model_name=par_use_biaffine_en --do_lower_case

# ---------------- NER ----------------
#python train_main.py  --task=ner --use_bert --train_data_path=./data/NER/cn/train.tsv --dev_data_path=./data/NER/cn/dev.tsv --test_data_path=./data/NER/cn/test.tsv --model_path=//data/Yangyang/Core/bert-base-chinese --max_seq_length=300  --model_name=NEWTEST_NER_baseline_cn
python train_main.py  --task=ner --use_bert --train_data_path=./data/NER/en/train.tsv --dev_data_path=./data/NER/en/dev.tsv --test_data_path=./data/NER/en/test.tsv --model_path=//data/Yangyang/models/bert-large-uncased --max_seq_length=300  --model_name=NEWTEST_NER_baseline_en --do_lower_case
#python train_main.py  --task=ner --use_xlnet --train_data_path=./data/NER/en/train.tsv --dev_data_path=./data/NER/en/dev.tsv --test_data_path=./data/NER/en/test.tsv --model_path=/data/Yangyang/models/XLNet_large_cased --max_seq_length=300  --model_name=NER_baseline_en
# ---------------- SRL baseline ----------------
#python train_main.py  --task=srl --use_bert --train_data_path=./data/SRL/en/train.tsv --dev_data_path=./data/SRL/en/test.tsv --test_data_path=./data/SRL/en/test.tsv --model_path=//data/Yangyang/models/bert-large-uncased --max_seq_length=300  --model_name=SRL_baseline_en --train_batch_size=32 --eval_batch_size=32

# ---------------- Rel baseline ----------------
#python train_main.py  --task=rel --use_bert --dataset_name=semeval --train_data_path=./data/Rel/semeval/train.tsv --dev_data_path=./data/Rel/semeval/test.tsv --test_data_path=./data/Rel/semeval/test.tsv --model_path=//data/Yangyang/models/bert-large-uncased --max_seq_length=300  --model_name=Rel_baseline_en

# ---------------- ASA baseline ----------------
#python train_main.py  --task=snt --use_bert --train_data_path=./data/ASA/laptop/train.txt --dev_data_path=./data/ASA/laptop/test.txt --test_data_path=./data/ASA/laptop/test.txt --model_path=//data/Yangyang/models/bert-large-uncased --max_seq_length=300  --model_name=ASA_baseline_en






#----not finish----

# ASA T-GCN
#python train_main.py  --task=snt --train_data_path=./data/ASA/sample_data/train.txt --dev_data_path=./data/ASA/sample_data/test.txt --test_data_path=./data/ASA/sample_data/test.txt --model_path=/data/tianyuanhe/bert_model/bert_base_cased --max_seq_length=300  --model_name=asa --train_batch_size=2 --eval_batch_size=2

# random embedding
#python train_main.py --use_bilstm --task=seg --train_data_path=/data/tianyuanhe/CWS/WMSeg/data/CTB6/dev.tsv --dev_data_path=/data/tianyuanhe/CWS/WMSeg/data/CTB6/test.tsv --test_data_path=/data/tianyuanhe/CWS/WMSeg/data/CTB6/test.tsv --max_seq_length=300  --model_name=cws_r --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8

# pretrained embedding
#python train_main.py --pretrained_embedding_file=./embedding/vec100.txt --use_bilstm --task=seg --train_data_path=./data/Seg/cn/train.tsv --dev_data_path=./data/Seg/cn/dev.tsv --test_data_path=./data/Seg/cn/test.tsv --max_seq_length=300  --model_name=cws_p --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8


#python train_main.py  --task=pos --use_bilstm --train_data_path=./data/POS/cn/train.tsv --dev_data_path=./data/POS/cn/dev.tsv --test_data_path=./data/POS/cn/test.tsv --model_path=./embedding/Tencent_AILab_ChineseEmbedding.txt --max_seq_length=300  --model_name=POS_bilstm_cn --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=sp --use_bilstm --train_data_path=./data/SP/cn/train.tsv --dev_data_path=./data/SP/cn/dev.tsv --test_data_path=./data/SP/cn/test.tsv --model_path=./embedding/character_embedding.txt --max_seq_length=300  --model_name=SP_bilstm_cn --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=seg --use_bilstm --train_data_path=./data/Seg/cn/train.tsv --dev_data_path=./data/Seg/cn/dev.tsv --test_data_path=./data/Seg/cn/test.tsv --model_path=./embedding/character_embedding.txt --max_seq_length=300  --model_name=Seg_bilstm_cn --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=ner --use_bilstm --train_data_path=./data/NER/cn/train.tsv --dev_data_path=./data/NER/cn/dev.tsv --test_data_path=./data/NER/cn/test.tsv --model_path=./embedding/word_embedding.txt --max_seq_length=300  --model_name=NER_bilstm_cn --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=par --use_bilstm --use_biaffine --train_data_path=./data/Par/cn/zh_gsdsimp-ud-train.conllu --dev_data_path=./data/Par/cn/zh_gsdsimp-ud-dev.conllu --test_data_path=./data/Par/cn/zh_gsdsimp-ud-test.conllu --model_path=./embedding/word_embedding.txt --max_seq_length=300  --model_name=Par_bilstm_cn_use_biaffine --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8



#python train_main.py  --task=pos --use_bilstm --train_data_path=./data/POS/en/train.tsv --dev_data_path=./data/POS/en/dev.tsv --test_data_path=./data/POS/en/test.tsv --model_path=./embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=POS_bilstm_en --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=ner --use_bilstm --train_data_path=./data/NER/en/train.tsv --dev_data_path=./data/NER/en/dev.tsv --test_data_path=./data/NER/en/test.tsv --model_path=./embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=NER_bilstm_en --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=par --use_bilstm --use_biaffine --train_data_path=./data/Par/en/train.conllu --dev_data_path=./data/Par/en/dev.conllu --test_data_path=./data/Par/en/test.conllu --model_path=./embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=Par_bilstm_cn_use_biaffine --learning_rate=0.01 --train_batch_size=8 --eval_batch_size=8
#python train_main.py  --task=srl --use_bilstm --train_data_path=./data/SRL/en/train.tsv --dev_data_path=./data/SRL/en/test.tsv --test_data_path=./data/SRL/en/test.tsv --model_path=./embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=SRL_bilstm_en --train_batch_size=32 --eval_batch_size=32
#python train_main.py  --task=rel --use_bilstm --dataset_name=semeval --train_data_path=./data/Rel/semeval/train.tsv --dev_data_path=./data/Rel/semeval/test.tsv --test_data_path=./data/Rel/semeval/test.tsv --model_path=./embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=Rel_bilstm_en
#python train_main.py  --task=snt --use_bilstm --train_data_path=./data/ASA/laptop/train.txt --dev_data_path=./data/ASA/laptop/test.txt --test_data_path=./data/ASA/laptop/test.txt --model_path=./embedding/glove/glove.twitter.27B.200d.txt --max_seq_length=300  --model_name=ASA_bilstm_en

