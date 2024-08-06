# DPar
This is the model designed for Dependency Parsing task in Chinese and English. 
With a few lines of code, you can call a method based on deep learning to get the dependency trees of the text.
Moreover, you can use the parameter `use_Biaffine` to call the enhanced model for Dependency Parsing.

[comment]: <> (following the implementation of []&#40;https://www.aclweb.org/anthology/2020.coling-main.187/&#41; at COLING 2020.)

## Requirements
Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Downloading model for inference
For Dependency Parsing, we provided pipeline to download model automatically with assigned name of pre-train model. For example, you can load `bert-base-uncased` to inference the Par tagging result as below.
```python
#Par
from DeepNLP.model.DPar import DPar
deepnlp = DPar.load_model(model_path='bert-base-uncased', language='en', use_Biaffine=False, local_rank=-1, no_cuda=False)
sentence = [['From', 'the', 'AP', 'comes', 'this', 'story', ':'], 
            ['The', 'sheikh', 'in', 'wheel', '-', 'chair', 'has', 'been', 'attacked', 'with', 'a', 'F', '-', '16', '-', 'launched', 'bomb', '.']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#[(['From', 'the', 'AP', 'comes', 'this', 'story', ':'], 
#  [3, 3, 4, 0, 6, 4, 4], 
#  ['case', 'det', 'obl', 'root', 'det', 'nsubj', 'punct']),
# (['The', 'sheikh', 'in', 'wheel', '-', 'chair', 'has', 'been', 'attacked', 'with', 'a', 'F', '-', '16', '-', 'launched', 'bomb', '.'], 
#  [2, 9, 6, 6, 6, 2, 9, 9, 0, 17, 17, 16, 12, 12, 16, 17, 9, 9],
#  ['det', 'nsubj:pass', 'case', 'compound', 'punct', 'nmod', 'aux', 'aux:pass', 'root', 'case', 'det', 'obl:npmod', 'punct', 'nummod', 'punct', 'amod', 'obl', 'punct'])
# ]
```

If you want obatin the dependency trees from a chinese text, you need to send 
```python
#Par
from DeepNLP.model.DPar import DPar
deepnlp = DPar.load_model(model_path='bert-base-chinese', language='zh', use_Biaffine=False, local_rank=-1, no_cuda=False)
sentence = [['大', '多数', '的', '加长', '型', '礼车', '则是', '租车', '公司', '的', '财产', '。'],
            ['1355', '年', '，', '勃兰登堡', '被', '神圣', '罗马', '帝国', '皇帝', '查理', '四世', '升', '为', '选侯', '国', '。']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
# [(['大', '多数', '的', '加长', '型', '礼车', '则是', '租车', '公司', '的', '财产', '。'],
#   [2, 6, 2, 5, 6, 11, 11, 9, 11, 9, 0, 11],
#   ['advmod', 'amod', 'case', 'compound', 'p', 'nmod', 'nmod', 'case', 'root', 'punct']),
#  (['1355', '年', '，', '勃兰登堡', '被', '神圣', '罗马', '帝国', '皇帝', '查理', '四世', '升', '为', '选侯', '国', '。'],
#   [2, 12, 9, 10, 0, 12, 15, 12, 12],
#   ['nummod', 'nmod:tmod', 'punct', 'nsubj:pass', 'aux:pass', 'amod', 'nmod', 'nmod', 'nsubj', 'appos', 'flat:name', 'root', 'mark', 'compound', 'obj', 'punct'])
#  ]
```
Call the enhanced model for Dependency Parsing by assigning parameter `use_Biaffine=True`.

### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this: 
```python
#Par
from DeepNLP.model.DPar import DPar
deepnlp = DPar.load_model(model_path='./zh_Par_BERT_CTB5_sm_0.1.0',no_cuda=False)
```
Please refer to the table below for the specific performance and download links of each different model.

| Model name | Language | Size | CPU/GPU Predict | CPU/GPU Train | Memory Size | Performance |
| --- | --- | --- | --- |  --- |  --- |  --- |
| zh_DepPar_BiLSTM_UD(GSDSimp)_sm_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_DepPar_BERT_UD(GSDSimp)_md_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_DepPar_BERT_UD(GSDSimp)_bs_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| en_DepPar_BiLSTM_UD(EWT)_sm_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_DepPar_BERT_UD(EWT)_md_0.1.0  | English | MB | it/s | it/s | GB | % |
| en_DepPar_BERT_UD(EWT)_bs_0.1.0  | English | MB | it/s | it/s | GB | % |
| en_DepPar_BERT_UD(EWT)_md_0.1.0 | English | MB | it/s | it/s | GB | % |
| zh_DepPar_BiLSTM_Biaffine_UD(GSDSimp)_sm_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_DepPar_BERT_Biaffine_UD(GSDSimp)_md_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_DepPar_BERT_Biaffine_UD(GSDSimp)_bs_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| en_DepPar_BiLSTM_Biaffine_UD(EWT)_sm_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_DepPar_BERT_Biaffine_UD(EWT)_md_0.1.0  | English | MB | it/s | it/s | GB | % |
| en_DepPar_BERT_Biaffine_UD(EWT)_bs_0.1.0  | English | MB | it/s | it/s | GB | % |
| en_DepPar_BERT_Biaffine_UD(EWT)_md_0.1.0 | English | MB | it/s | it/s | GB | % |


## Fine-tune own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own Dependency Parsing model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
In details, you have to convert a sentence with head tags and relations tags into a column of words and two column of head and relations tags.
Besides, line break is an instruction for dividing two sentences.
You can see the [demo data](../../examples/DPar/data_demo) here.

### Examples for Training and Testing

You can find [example](../../examples/DPar/DPar_train.py) here. 
We recommend using the command line in [`DPar_train.sh`](./examples/DPar/DPar_train.sh) to fine-tune the model.

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


[comment]: <> (* `--ngram_length`: the max length of n-grams to be considered.)

[comment]: <> (* `--ngram_type`: use `av`, `dlg`, or `pmi` to construct the lexicon N.)

[comment]: <> (* `--av_threshold`: when using `av` to construct the lexicon N, n-grams whose AV score is lower than the threshold will be excluded from the lexicon N.)

[comment]: <> (* `--ngram_threshold`: n-grams whose frequency is lower than the threshold will be excluded from the lexicon N. Note that, when the threshold is set to 1, no n-gram is filtered out by its frequency. We therefore **DO NOT** recommend you to use 1 as the n-gram frequency threshold.)

