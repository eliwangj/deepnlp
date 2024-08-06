# DPOS
This is the model designed for Part-of-speech Tagging task in Chinese and English. 
With a few lines of code, you can call a method based on deep learning to get the POS tags of Chinese or English words in the text.

For Chinese texts, a joint-task mode of word segmentation and POS tagging, `joint_cws_pos`,  is provided which can directly process the unsegmented Chinese corpus and return the POS tags.
Moreover, you can use the parameter `use_memory` to call the enhanced model for joint-task, following the implementation of [Joint Chinese Word Segmentation and Part-of-speech Tagging via Multi-channel Attention of Character N-grams](https://www.aclweb.org/anthology/2020.coling-main.187/) at COLING 2020.


## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

Use `pip install -r requirements.txt` to install the required packages.


## Downloading model for inference
### General POS tagging
For POS tagging, we provided pipeline to download model automatically with assigned name of pre-train model. For example, you can load `bert-base-uncased` to inference the POS tagging result as below.
```python
#POS
from DeepNLP.model.DPOS import DPOS
deepnlp = DPOS.load_model(model_path='bert-base-cased', language='en', joint_cws_pos=False, use_memory=False, local_rank=-1, no_cuda=False)
sentence = [['The', 'Arizona', 'Corporations', 'Commission', 'authorized', 'an', '11.5', '%', 'rate', 'increase', 'at', 'Tucson', 'Electric', 'Power', 'Co.', ',', 'substantially', 'lower', 'than', 'recommended', 'last', 'month', 'by', 'a', 'commission', 'hearing', 'officer', 'and', 'barely', 'half', 'the', 'rise', 'sought', 'by', 'the', 'utility', '.'], 
            ['The', 'ruling', 'follows', 'a', 'host', 'of', 'problems', 'at', 'Tucson', 'Electric', ',', 'including', 'major', 'write-downs', ',', 'a', '60', '%', 'slash', 'in', 'the', 'common', 'stock', 'dividend', 'and', 'the', 'departure', 'of', 'former', 'Chairman', 'Einar', 'Greve', 'during', 'a', 'company', 'investigation', 'of', 'his', 'stock', 'sales', '.']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#[['The_DT', 'Arizona_NNP', 'Corporations_NNPS', 'Commission_NNP', 'authorized_VBD', 'an_DT', '11.5_CD', '%_NN', 'rate_NN', 'increase_NN', 'at_IN', 'Tucson_NNP', 'Electric_NNP', 'Power_NNP', 'Co._NNP', ',_,', 'substantially_RB', 'lower_JJR', 'than_IN', 'recommended_VBN', 'last_JJ', 'month_NN', 'by_IN', 'a_DT', 'commission_NN', 'hearing_NN', 'officer_NN', 'and_CC', 'barely_RB', 'half_PDT', 'the_DT', 'rise_NN', 'sought_VBN', 'by_IN', 'the_DT', 'utility_NN', '._.'], 
# ['The_DT', 'ruling_NN', 'follows_VBZ', 'a_DT', 'host_NN', 'of_IN', 'problems_NNS', 'at_IN', 'Tucson_NNP', 'Electric_NNP', ',_,', 'including_VBG', 'major_JJ', 'write-downs_NNS', ',_,', 'a_DT', '60_CD', '%_NN', 'slash_NN', 'in_IN', 'the_DT', 'common_JJ', 'stock_NN', 'dividend_NN', 'and_CC', 'the_DT', 'departure_NN', 'of_IN', 'former_JJ', 'Chairman_NNP', 'Einar_NNP', 'Greve_NNP', 'during_IN', 'a_DT', 'company_NN', 'investigation_NN', 'of_IN', 'his_PRP$', 'stock_NN', 'sales_NNS', '._.']]
```
If the data is Chinese text, the input needs to be the result after word segmentation with traditional POS method, e.i., `joint_cws_pos=False`.
### Joint Segmentation and POS tagging for Chinese text
If you want to obatin the POS tagging result directly from Chinese text, you can use `joint_cws_pos` model and `use_memory` model like this
```python
#POS
from DeepNLP.model.DPOS import DPOS
deepnlp = DPOS.load_model(model_path='bert-base-chinese', language='zh', joint_cws_pos=True, use_memory=True, local_rank=-1, no_cuda=False)
sentence = [['法','正','研','究','从','波','黑','撤','军','计','划','新','华','社','巴','黎','９','月','１','日','电','（','记','者','张','有','浩','）'],
            ['法','国','国','防','部','长','莱','奥','塔','尔','１','日','说','，法','国','正','在','研','究','从','波','黑','撤','军','的','计','划','。']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#['法_NR', '正_AD', '研究_VV', '从_P', '波黑_NR', '撤军_VV', '计划_NN', '新华社_NR', '巴黎_NR', '９月_NT', '１日_NT', '电_NN', '（_PU', '记者_NN', '张有浩_NR', '）_PU'], 
#['法国_NR', '国防_NN', '部长_NN', '莱奥塔尔_NR', '１日_NT', '说_VV', '，法_PU', '国_NR', '正在_AD', '研究_VV', '从_P', '波黑_NR', '撤军_VV', '的_DEC', '计划_NN', '。_PU']]
```
### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this: 
```python
#POS
from DeepNLP.model.DPOS import DPOS
deepnlp = DPOS.load_model(model_path='./zh_POS_BERT_CTB5_sm_0.1.0',no_cuda=False)
```
Please refer to the table below for the specific performance and download links of each different model.

| Model name | Language | Size | CPU/GPU Predict | CPU/GPU Train | Memory Size | Performance |
| --- | --- | --- | --- |  --- |  --- |  --- |
| zh_POS_BERT_CTB5_sm_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_POS_BERT_CTB5_md_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_POS_BERT_CTB5_bs_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_POS_BERT_CTB5_ls_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| en_POS_BERT_CTB5_sm_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_POS_BERT_CTB5_md_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_POS_BERT_CTB5_bs_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_POS_BERT_CTB5_ls_0.1.0 | English | MB | it/s | it/s | GB | % |
| zh_POS_BERT_McASP_CTB5_sm_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_POS_BERT_McASP_CTB5_md_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_POS_BERT_McASP_CTB5_bs_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| zh_POS_BERT_McASP_CTB5_ls_0.1.0 | Chinese | MB | it/s | it/s | GB | % |
| en_POS_BERT_McASP_CTB5_sm_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_POS_BERT_McASP_CTB5_md_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_POS_BERT_McASP_CTB5_bs_0.1.0 | English | MB | it/s | it/s | GB | % |
| en_POS_BERT_McASP_CTB5_ls_0.1.0 | English | MB | it/s | it/s | GB | % |
With saved model, You can load it from local path like this

## Fine-tune own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own POS tagging model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
In details, you have to convert a sentence with POS tags into a column of words and a column of NER tags. Besides, line break is an instruction for dividing two sentences.
You can see two versions of training [demo datas](../../examples/DPOS/data_demo),e.i., `POS` for general POS tagging and `SP` for Joint Segmentation and POS tagging in Chinese text, here.

### Examples for Training and Testing

You can find [example](../../examples/DPOS/DPOS_train.py) here. 
We recommend using the command line in [`DPOS_train.sh`](./examples/DPOS/DPOS_train.sh) to fine-tune the model.

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
* `--use_memory`: use multi-channel attention.
* `--cat_type`: the categorization strategy to be used (can be either `freq` or `length`).
* `--ngram_length`: the max length of n-grams to be considered.
* `--cat_num`: the number of channels (categories) to use (this number needs to equal to `ngram_length` if `cat_type` is `length`).
* `--ngram_type`: use `av`, `dlg`, or `pmi` to construct the lexicon N.
* `--av_threshold`: when using `av` to construct the lexicon N, n-grams whose AV score is lower than the threshold will be excluded from the lexicon N.
* `--ngram_threshold`: n-grams whose frequency is lower than the threshold will be excluded from the lexicon N. Note that, when the threshold is set to 1, no n-gram is filtered out by its frequency. We therefore **DO NOT** recommend you to use 1 as the n-gram frequency threshold.
* `--joing_pos`:

