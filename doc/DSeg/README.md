# DSeg
This is the model designed for Word segmentation for simplified standard Chinese or ancient Chinese. 
With a few lines of code, you can call a method based on deep learning to get the Chinese Word Segmentation (CWS) in the text.

Moreover, you can use the parameter `use_memory` to call the enhanced model for this task, following the implementation of [Improving Chinese Word Segmentation with Wordhood Memory Networks](https://www.aclweb.org/anthology/2020.acl-main.734/) at ACL 2020.


## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Downloading model for inference
For this Chinese Word Segmentation task, we provided pipeline to downloading model automatically with assigned name of pre-trained model. For example, you can load `bert-base-chinese` to do the inference as below. 

The argument `use_memory` determines whether the [Wordhood Memory Networks](https://www.aclweb.org/anthology/2020.acl-main.734/) enhanced methods will be used.

The prediction includes 3 parts. Take the example as blow, the number `0.038615322674560547` shows the the total calculation time of the model on the whole input sequences; the number `2` shows the number of sequence inputs; And at the end are the CWS results on the input sequences.

```python
#Seg
from DeepNLP.model.DSeg import DSeg
deepnlp = DSeg.load_model(model_path='bert-base-chinese', use_memory=False, chemed=False, local_rank=-1, no_cuda=False)

sentence = [['在', '维', '护', '奥', '林', '匹', '克', '运', '动', '非', '商', '业', '化', '的', '宗', '旨', '的', '同', '时', '，', '体', '育', '和', '经', '济', '可', '以', '起', '到', '相', '互', '推', '动', '的', '作', '用', '。'],
            ['新', '华', '社', '北', '京', '九', '月', '一', '日', '电', '（', '记', '者', '杨', '国', '军', '）', '。']]

predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#0.038615322674560547 2
#[['在', '维护', '奥林匹克', '运动', '非商', '业化', '的', '宗', '旨', '的', '同', '时', '，', '体', '育', '和', '经', '济', '可', '以', '起', '到', '相', '互', '推', '动', '的', '作', '用', '。'],['新华社', '北京', '九月', '一日', '电', '（', '记者', '杨国军', '）', '。']]
```

### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this:
```python
#DSeg
from DeepNLP.model.DSeg import DSeg
deepnlp = DSeg.load_model(model_path='./Language_Task_PretrainedModel_Dataset_',no_cuda=False)
```
Please refer to the table below for the specific performance and download links of each different model.<u>**表格改**</u>

| Model name                                                  | Language | Size | CPU/GPU Predict | CPU/GPU Train | Memory Size | Performance |
| ----------------------------------------------------------- | -------- | ---- | --------------- | ------------- | ----------- | ----------- |
| zh_Seg_BERT_CTB5_sm_0.1.0                                   | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_CTB5_md_0.1.0                                   | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_CTB5_bs_0.1.0                                   | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_KVMN_CTB5_sm_0.1.0                              | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_KVMN_CTB5_md_0.1.0                              | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_KVMN_CTB5_bs_0.1.0                              | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_chemed_sm_0.1.0                                   | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_chemed_md_0.1.0                                   | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_chemed_bs_0.1.0                                   | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_KVMN_chemed_sm_0.1.0                              | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_KVMN_chemed_md_0.1.0                              | Chinese  | MB   | it/s            | it/s          | GB          | %           |
| zh_Seg_BERT_KVMN_chemed_bs_0.1.0                              | Chinese  | MB   | it/s            | it/s          | GB          | %           |

## Fine-tune own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own Word segmentation model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
In details, you have to convert a sentence with Seg tags into a column of words and a column of Seg tags. Besides, line break is an instruction for dividing two sentences.
You can see the [demo data](../../examples/DSeg/data_demo) here.

### Examples for Training and Testing
You can find [example](../../examples/DSeg/DSeg_train.py) here. 
We recommend using the command line in [`DSeg_train.sh`](./examples/DSeg/DSeg_train.sh) to fine-tune the model.

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
* `--use_memory`: use KVMN .
* `--ngram_length`: the max length of n-grams to be considered.
* `--ngram_type`: use `av`, `dlg`, or `pmi` to construct the lexicon N.
* `--av_threshold`: when using `av` to construct the lexicon N, n-grams whose AV score is lower than the threshold will be excluded from the lexicon N.
* `--ngram_threshold`: n-grams whose frequency is lower than the threshold will be excluded from the lexicon N. Note that, when the threshold is set to 1, no n-gram is filtered out by its frequency. We therefore **DO NOT** recommend you to use 1 as the n-gram frequency threshold.

