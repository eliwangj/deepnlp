# DSnt

This is the model designed for Sentiment Analysis task in Chinese and English, including Sentiment Analysis task and Aspect-Based Sentiment Analysis task. 

With a few lines of code, you can call a method based on deep model to get the attitude of the given text, or the fine-grained emotional attitude towards a specific aspect term in the text.

For the Aspect-Based Sentiment Analysis task, you can use Type-aware Graph Convolutional Networks (TGCNs)  to improve the performance by using the parameter `use_TGCN` , following the implementation of [Aspect-based Sentiment Analysis with Type-aware Graph Convolutional Networks and Layer Ensemble](https://aclanthology.org/2021.naacl-main.231.pdf) at NAACL 2021.

## Requirements

Our code works with the following environment.

* `python=3.7`
* `pytorch=1.3`


## Downloading model for inference

### Aspect-based Sentiment Analysis
For this task, we provided pipeline to download model automatically with assigned name of pre-trained model. 
For example, you can load `bert-base-uncased` to inference the sentiment analysis results as below.
Please note that aspect terms in every input sequence are required. You could give their strings or the index pairs of the list composed of words.
```python
#Snt
from DeepNLP.model.DSnt import DSnt
deepnlp = DSnt.load_model(model_path='bert-base-uncased', ABSA=True, language='en',dataset='MAMS', TGCN=False, local_rank=-1, no_cuda=False)
sentence = [['boot', 'time', 'is', 'super', 'fast', ',', 'around', 'anywhere', 'from', '35', 'seconds', 'to', '1', 'minute', '.'], 
            ['but', 'the', 'mountain', 'lion', 'is', 'just', 'too', 'slow', '.'], 
            ['osx', 'mountain', 'lion', 'soon', 'to', 'upgrade', 'to', 'mavericks', '.']]
# index pairs of aspect-terms
aspect_list = [(0, 2), (2, 4),(0, 3)]
# strings of aspect-terms
# aspect_list = ["boot time",
#                "mountain lion",
#                "osx mountain lion"]

predict_result = deepnlp.predict(sentence_list=sentence, aspect_list=aspect_list)
print(predict_result)
#['1','-1','0']
```
There are three sentiment labels: `1` represents the positive sentiment, `0` represents the neutral sentiment, and `-1` represents the negative sentiment.


### Sentiment Analysis
For the general Sentiment Analysis task, an example based on Chinese texts is listed below:
```python
#Snt
from DeepNLP.model.DSnt import DSnt
deepnlp = DSnt.load_model(model_path='bert-base-chinese', ABSA=False, language='zh', local_rank=-1, no_cuda=False)
sentence = [['这', '个', '宾', '馆', '比', '较', '陈', '旧', '了', '，', '特', '价', '的', '房', '间', '也', '很', '一', '般', '。', '总', '体', '来', '说', '一', '般'],
            ['交', '通', '方', '便', '，', '环', '境', '很', '好', '。', ' ', '风', '景', '美', '丽']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#['0','1']
```

There are two sentiment labels: `0` represents the negative sentiment, and `1` represents the positive sentiment.

### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this: 
```python
#Snt
from DeepNLP.model.DSnt import DSnt
deepnlp = DSnt.load_model(model_path='./zh_SA_BERT_chnsenticorp_md_0.1.0', local_rank=-1, no_cuda=False)
```
Please refer to the table below for the specific performance and download links of each different model.

| Model name                           | Language | Size   | CPU/GPU Predict   | CPU/GPU Train   | Memory Size | Performance*    |
|--------------------------------------|----------|--------|-------------------|-----------------|-------------|-----------------|
| zh_SA_BiLSTM_chnsenticorp_sm_0.1.0   | Chinese  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| zh_SA_BERT_chnsenticorp_md_0.1.0     | Chinese  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| zh_SA_BERT_chnsenticorp_bs_0.1.0     | Chinese  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_SA_BiLSTM_SST5_sm_0.1.0           | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_SA_BERT_SST5_md_0.1.0             | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_SA_BERT_SST5_bs_0.1.0             | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_SA_BERT_SST5_ls_0.1.0             | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_laptop_sm_0.1.0       | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_laptop_md_0.1.0         | English  | MB     | 200.89 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_laptop_bs_0.1.0         | English  | MB     | 188.17 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_laptop_ls_0.1.0         | English  | MB     | 145.60 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_MAMS_sm_0.1.0         | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_MAMS_md_0.1.0           | English  | 256 MB | 203.54 sentence/s | ~155 sentence/s | 3803 MiB    | 80.33% (79.36%) |
| en_ABSA_BERT_MAMS_bs_0.1.0           | English  | 418 MB | 171.88 sentence/s | ~100 sentence/s | 5475 MiB    | 81.94% (83.43%) |
| en_ABSA_BERT_MAMS_ls_0.1.0           | English  | 1.2 GB | 152.21 sentence/s | ~50 sentence/s  | 15791 MiB   | 81.96% (83.46%) |
| en_ABSA_BiLSTM_rest14_sm_0.1.0       | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest14_md_0.1.0         | English  | MB     | 203.84 sentence/s | ~170 sentence/s | 3663 MiB    | 67.88% (67.88%)                |
| en_ABSA_BERT_rest14_bs_0.1.0         | English  | MB     | 189.10 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest14_ls_0.1.0         | English  | MB     | 151.92 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_rest15_sm_0.1.0       | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest15_md_0.1.0         | English  | MB     | 203.45 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest15_bs_0.1.0         | English  | MB     | 189.45 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest15_ls_0.1.0         | English  | MB     | 150.43 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_rest16_sm_0.1.0       | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest16_md_0.1.0         | English  | MB     | 205.23 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest16_bs_0.1.0         | English  | MB     | 188.58 sentence/s | sentence/s      | MiB         | %               |
| en_ABSA_BERT_rest16_ls_0.1.0         | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_twitter_sm_0.1.0      | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_twitter_md_0.1.0        | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_twitter_bs_0.1.0        | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_twitter_ls_0.1.0        | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_tgcn_laptop_sm_0.1.0  | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_laptop_md_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_laptop_bs_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_laptop_ls_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_tgcn_MAMS_sm_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_MAMS_md_0.1.0      | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_MAMS_bs_0.1.0      | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_MAMS_ls_0.1.0      | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_tgcn_rest14_sm_0.1.0  | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest14_md_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest14_bs_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest14_ls_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_tgcn_rest15_sm_0.1.0  | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest15_md_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest15_bs_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest15_ls_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_tgcn_rest16_sm_0.1.0  | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest16_md_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest16_bs_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_rest16_ls_0.1.0    | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BiLSTM_tgcn_twitter_sm_0.1.0 | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_twitter_md_0.1.0   | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_twitter_bs_0.1.0   | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |
| en_ABSA_BERT_tgcn_twitter_ls_0.1.0   | English  | MB     | sentence/s        | sentence/s      | MiB         | %               |

__*__ Best dev F1 (Best test F1)

## Fine-tune own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own Sentiment Analysis model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
For fine tuning, the difference between ABSA and SA lies in their training data.

You can see two versions of training [demo datas](../../examples/DSnt/data_demo), e.i., `SA` and `ABSA`, here.

### Examples for Training and Testing

You can find [example](../../examples/DSnt/DSnt_train.py) here. 
We recommend using the command line in [`DSnt_train.sh`](./examples/DSnt/DSnt_train.sh) to fine-tune the model.

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
* `--use_tgcn`: For ABSA, you can use TGCN to enhance the model.


