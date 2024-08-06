# DSRL

This is the model designed for Semantic Role Labeling task in Chinese and English. 

With a few lines of code, you can call a method based on deep learning to get the semantic roles of every word in English texts or every chinese character in Chinese texts.


## Requirements

Our code works with the following environment.

* `python=3.7`
* `pytorch=1.3`

## Downloading model for inference

For this task, we provided pipeline to download model automatically with assigned name of pre-trained model. For example, you can load `bert-base-uncased` to inference the sentiment analysis results as below.

```python
#DSRL(English)
from DeepNLP.model.DSRL import DSRL
deepnlp = DSRL.load_model(model_path='bert-base-uncased', language='en', dataset='CoNLL05', local_rank=-1, no_cuda=False)
verb_list = [[6], [2]]
sentence = [
    ['The', 'economy', "'s", 'temperature', 'will', 'be', 'taken', 'from', 'several', 'vantage', 'points', 'this', 'week', ',', 'with', 'readings', 'on', 'trade', ',', 'output', ',', 'housing', 'and', 'inflation', '.'],
    ['Exports', 'are', 'thought', 'to', 'have', 'risen', 'strongly', 'in', 'August', ',', 'but', 'probably', 'not', 'enough', 'to', 'offset', 'the', 'jump', 'in', 'imports', ',', 'economists', 'said', '.']
]
result_list = deepnlp.predict(sentence_list=sentence, verb_index_list=verb_list)
print(result_list)
# [['The_B-A1', 'economy_I-A1', "'s_I-A1", 'temperature_I-A1', 'will_B-AM-MOD', 'be_O', 'taken_V', 'from_B-A2', 'several_I-A2',
#   'vantage_I-A2', 'points_I-A2', 'this_B-AM-TMP', 'week_I-AM-TMP', ',_O', 'with_B-AM-ADV', 'readings_I-AM-ADV', 'on_I-AM-ADV',
#   'trade_I-AM-ADV', ',_I-AM-ADV', 'output_I-AM-ADV', ',_I-AM-ADV', 'housing_I-AM-ADV', 'and_I-AM-ADV', 'inflation_I-AM-ADV', '._O'],
#  ['Exports_B-A1', 'are_O', 'thought_V', 'to_B-C-A1', 'have_I-C-A1', 'risen_I-C-A1', 'strongly_I-C-A1', 'in_I-C-A1', 'August_I-C-A1',
#   ',_I-C-A1', 'but_I-C-A1', 'probably_I-C-A1', 'not_I-C-A1', 'enough_I-C-A1', 'to_I-C-A1', 'offset_I-C-A1', 'the_I-C-A1', 'jump_I-C-A1',
#   'in_I-C-A1', 'imports_I-C-A1', ',_O', 'economists_O', 'said_O', '._O']]
```

Here we provide another example under a Chinese context:
```python
#DSRL(Chinese)
from DeepNLP.model.DSRL import DSRL
deepnlp = DSRL.load_model(model_path='bert-base-chinese', language='zh', dataset='CoNLL05', local_rank=-1, no_cuda=False)
verb_list = [[6], [2]]
sentence = [
    ['The', 'economy', "'s", 'temperature', 'will', 'be', 'taken', 'from', 'several', 'vantage', 'points', 'this', 'week', ',', 'with', 'readings', 'on', 'trade', ',', 'output', ',', 'housing', 'and', 'inflation', '.'],
    ['Exports', 'are', 'thought', 'to', 'have', 'risen', 'strongly', 'in', 'August', ',', 'but', 'probably', 'not', 'enough', 'to', 'offset', 'the', 'jump', 'in', 'imports', ',', 'economists', 'said', '.']
]
result_list = deepnlp.predict(sentence_list=sentence, verb_index_list=verb_list)
print(result_list)

```


### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this: 
```python
#SRL
from DeepNLP.model.DSRL import DSRL
deepnlp = DSRL.load_model(model_path='./zh_SRL_BERT_CPB2.0_md_0.1.0/model', local_rank=-1, no_cuda=False)
```
Note: /model has to be added after the model directory.

Please refer to the table below for the specific performance and download links of each different model.

| Model name                     | Language | Size  | CPU/GPU Predict*   | CPU/GPU Train   | Memory Size | Performance**   |
|--------------------------------|----------|-------|--------------------|-----------------|-------------|-----------------|
| zh_SRL_BiLSTM_CPB2.0_sm_0.1.0  | Chinese  | 90MB  | sentence/s         | sentence/s      | GB          | 9.05% (9.05%)   |
| zh_SRL_BERT_CPB2.0_bs_0.1.0    | Chinese  | 427MB | sentence/s         | ~65 sentence/s  | 7863 MiB    | 72.53% (70.76%) |
| zh_SRL_BERT_CPB2.0_md_0.1.0    | Chinese  | 264MB | sentence/s         | sentence/s      | GB          | &cross;         |
| en_SRL_BiLSTM_CoNLL05_sm_0.1.0 | English  | 1.1GB | 961.48 sentence/s  | sentence/s      | GB          | 47.15% (48.80)  |
| en_SRL_BiLSTM_CoNLL12_sm_0.1.0 | English  | 749MB | 1004.48 sentence/s | sentence/s      | GB          | &cross;         |
| en_SRL_BERT_CoNLL05_bs_0.1.0   | English  | 539MB | 156.57 sentence/s  | ~70 sentence/s  | ~9000 MiB   | 87.28% (87.96%) |
| en_SRL_BERT_CoNLL05_md_0.1.0   | English  | 354MB | 169.53 sentence/s  | ~105 sentence/s | 7817 MiB    | &cross;         |
| en_SRL_BERT_CoNLL05_ls_0.1.0   | English  | 1.4GB | 117.01 sentence/s  | ~35 sentence/s  | ~17200 MiB  | 87.91% (88.69%) |
| en_SRL_BERT_CoNLL12_bs_0.1.0   | English  | 651MB | 157.64  sentence/s | ~68 sentence/s  | 12515 MiB   | &cross;         |
| en_SRL_BERT_CoNLL12_md_0.1.0   | English  | 597MB | 170.07 sentence/s  | ~100 sentence/s | 7805 MiB    | 83.86% (84.13%) |
| en_SRL_BERT_CoNLL12_ls_0.1.0   | English  | 1.5GB | 117.12 sentence/s  | ~30 sentence/s  | 28453 MiB   | &cross;         |

__*__  The average speed of predicting the first 1000 sentences of test dataset.

__**__ Best dev F1 (Best test F1).

## Fine-tune your own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own Semantic Role Labeling model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
In details, you have to convert a sentence with SRL tags into a column of words and a column of SRL tags. Besides, line break is an instruction for dividing two sentences.
You can see the [demo data](../../examples/DSRL/data_demo) here.

### Examples for Training and Testing

You can find [example](../../examples/DSRL/DSRL_train.py) here. 
We recommend using the command line in [`DSRL_train.sh`](./examples/DSRL/DSRL_train.sh) to fine-tune the model.

Here are some important parameters :

* `--train_data_path`: The training data path. Should contain the .tsv for the task.
* `--dev_data_path`: The validation data path. Should contain the .tsv for the task.
* `--test_data_path`: The test data path. Should contain the .tsv for the task.
* `--use_bert`: Use BERT as encoder.
* `--use_zen`: Use ZEN as encoder.
* `--model_path`: The directory of pre-trained BERT/ZEN model.
* `--model_name`: The name of model to save.
* `--do_lower_case`: Convert uppercase in text to lowercase.
* `--max_seq_length`: Maximum length of input text.