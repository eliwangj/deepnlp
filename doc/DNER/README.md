# DNER
This is the model designed for Named Entity Recognition task in Chinese and English. 
With a few lines of code, you can call a method based on deep learning to obtain the named entities of Chinese or English words in the text.

## Requirements
Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Downloading model for inference
For NER, we provided pipeline to download model automatically with assigned name of pre-train model. 
For example, you can load `bert-base-uncased` to inference the NER result as below.
The model will be saved in the current working directory.
```python
#NER
from DeepNLP.model.DNER import DNER
deepnlp = DNER.load_model(model_path='bert-base-uncased', chemed=False, language='en', local_rank=-1, no_cuda=False)
sentence = [
    ['The', 'Arizona', 'Corporations', 'Commission', 'authorized', 'an', '11.5', '%', 'rate', 'increase', 'at', 'Tucson', 'Electric', 'Power', 'Co.', ',', 'substantially', 'lower', 'than', 'recommended', 'last', 'month', 'by', 'a', 'commission', 'hearing', 'officer', 'and', 'barely', 'half', 'the', 'rise', 'sought', 'by', 'the', 'utility', '.'], 
    ['RT', '@Phil_Heim', ':', 'Safe', 'to', 'say', 'Super', 'Bowl', 'Sunday', 'is', 'my', 'favourite', 'holiday', 'of', 'the', 'year']
    ]

predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#[['The_', 'Arizona Corporations_ORG', 'Commission_', 'authorized_', 'an_', '11.5_', '%_', 'rate_', 'increase_', 'at_', 'Tucson Electric PowerCo._ORG', ',_', 'substantially_', 'lower_', 'than_', 'recommended_', 'last_', 'month_', 'by_', 'a_', 'commission_', 'hearing_', 'officer_', 'and_', 'barely_', 'half_', 'the_', 'rise_', 'sought_', 'by_', 'the_', 'utility_', '._'], 
# ['RT_', '@Phil_Heim_', ':_', 'Safe_', 'to_', 'say_', 'Supe Bowl Sunday_other', 'is_', 'my_', 'favourite_', 'holiday_', 'of_', 'the_', 'year_']]
```
Here we provide another example under a Chinese context:
```python
# NER-Chinese
from DeepNLP.model.DNER import DNER
deepnlp = DNER.load_model(model_path='bert-base-chinese', chemed=False, language='zh', local_rank=-1, no_cuda=False)
sentence = [
        ['常', '建', '良', '，', '男', '，', '1', '9', '6', '3', '年', '出', '生', '，', '工', '科', '学', '士', '，', '高', '级', '工', '程', '师', '，', '北', '京', '物', '资', '学', '院', '客', '座', '副', '教', '授', '。'],
        ['陈', '宝', '杰', '，', '男', '，', '汉', '族', '。', '毕', '业', '于', '解', '放', '军', '南', '京', '政', '治', '学', '院', '，', '大', '学', '学', '历', '。']
    ]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
# [['常建良_NAME', '，_', '男_', '，_', '1_', '9_', '6_', '3_', '年_', '出_', '生_', '，_', '工科_PRO', '学士_EDU', '，_', '高级工程师_TITLE', '，_', '北京物资学院_ORG', '客座副教授_TITLE', '。_'],
#  ['陈宝杰_NAME', '，_', '男_', '，_', '汉族_RACE', '。_', '毕_', '业_', '于_', '解放军南京政治学院_ORG', '，_', '大学学历_EDU', '。_']]
```
### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this: 
```python
#NER
from DeepNLP.model.DNER import DNER
deepnlp = DNER.load_model(model_path='./zh_NER_BERT_CTB5_sm_0.1.0', local_rank=-1, no_cuda=False)
```
Please refer to the table below for the specific performance and download links of each different model.

| Model name                    | Language| Size | CPU/GPU Predict | CPU/GPU Train | Memory Size | Performance |
| ------------------------------| --------| ---- | --------------- |  ------------ |  ---------- | ----------- |
| zh_NER_BiLSTM_RE_sm_0.1.0     | Chinese | MB   | it/s            | it/s          | GB | % |
| zh_NER_BERT_RE_md_0.1.0       | Chinese | MB   | it/s            | it/s          | GB | % |
| zh_NER_BERT_RE_bs_0.1.0       | Chinese | MB   | it/s            | it/s          | GB | % |
| chemed_NER_BiLSTM_RE_sm_0.1.0 | Chinese | MB   | it/s            | it/s          | GB | % |
| chemed_NER_BERT_RE_md_0.1.0   | Chinese | MB   | it/s            | it/s          | GB | % |
| chemed_NER_BERT_RE_bs_0.1.0   | Chinese | MB   | it/s            | it/s          | GB | % |
| en_NER_BiLSTM_WN16_sm_0.1.0   | English | MB   | it/s            | it/s          | GB | % |
| en_NER_BERT_WN16_md_0.1.0     | English | MB   | it/s            | it/s          | GB | % |
| en_NER_BERT_WN16_bs_0.1.0     | English | MB   | it/s            | it/s          | GB | % |
| en_NER_BERT_WN16_ls_0.1.0     | English | MB   | it/s            | it/s          | GB | % |

## Fine-tune own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own Named Entity Recognition model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
In details, you have to convert a sentence with NER tags into a column of words and a column of NER tags. Besides, line break is an instruction for dividing two sentences.
You can see the [demo data](../../examples/DNER/data_demo) here.

### Examples for Training and Testing

You can find [example](../../examples/DNER/DNER_train.py) here. 
We recommend using the command line in [`DNER_train.sh`](./examples/DNER/DNER_train.sh) to fine-tune the model.

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

