# DRel
This is the model designed for Relation Extraction task in Chinese and English. 
With a few lines of code, you can call a method based on deep learning to get the Rel tags of Chinese or English words in the text.


## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Downloading model for inference
For Relation Extraction, we provided pipeline to downloading model automatically with assigned name of pre-train model. For example, you can load `bert-base-uncased` to extract the relation between entities as below.
We support two methods of entering entities.
```python
#Rel
from DeepNLP.model.DRel import DRel
# deepnlp = DRel.load_model(model_path='bert-base-cased', no_cuda=False)
deepnlp = DRel.load_model(model_path='/data/Yangyang/D_project/saved_models/Rel_bilstm_en_2021-07-11-18-05-00/model', no_cuda=False)
sentence = ['The most common audits were about waste and recycling.',
            'The company fabricates plastic chairs.']
'''First method to input entity'''
e1_list = [(6, 7), (1, 2)]
e2_list = [(1, 2), (4, 5)]
'''Second method to input entity'''
# e1_list = ['audits', 'company']
# e2_list = ['waste', 'chairs']
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
# ['Message-Topic(e1,e2)', 'Product-Producer(e2,e1)']
```
Considering that a word may appear more than once in the text, we recommend using the first method of inputting entities.

### Models comparison
Besides load model automatically, you can also download the model in advance and load it through the local path like this: 
```python
#Rel
from DeepNLP.model.DRel import DRel
deepnlp = DRel.load_model(model_path='./zh_Rel_BERT_CTB5_sm_0.1.0',no_cuda=False)
```
Please refer to the table below for the specific performance and download links of each different model.

| Model name                       | Language | Size   | CPU/GPU Predict   | CPU/GPU Train   | Memory Size | Performance          |
|----------------------------------|----------|--------|-------------------|-----------------|-------------|----------------------|
| en_Rel_BiLSTM_ace2005en_sm_0.1.0 | English  | 959 MB | sentence/s        | sentence/s      | MiB         | 50.49% (54.31%)      |
| en_Rel_BERT_ace2005en_md_0.1.0   | English  | 263 MB | 185.89 sentence/s | ~160 sentence/s | 3905 MiB    | 66.70% (66.70%) same |
| en_Rel_BERT_ace2005en_bs_0.1.0   | English  | 425 MB | 165.44 sentence/s | ~92 sentence/s  | 6197 MiB    | 72.82% (72.82%) same |
| en_Rel_BERT_ace2005en_ls_0.1.0   | English  | 1.3GB  | 113.24 sentence/s | ~37 sentence/s  | 15795 MiB   | 73.81% (73.81%) same |
| en_Rel_BiLSTM_semeval_sm_0.1.0   | English  | 955 MB | sentence/s        | sentence/s      | MiB         | 72.38% (72.38%) same |
| en_Rel_BERT_semeval_md_0.1.0     | English  | 259 MB | 185.03 sentence/s | ~190 sentence/s | 4233 MiB    | 85.91% (85.91%) same |
| en_Rel_BERT_semeval_bs_0.1.0     | English  | 421 MB | 162.02 sentence/s | ~105 sentence/s | 6737 MiB    | 87.74% (87.74%) same |
| en_Rel_BERT_semeval_ls_0.1.0     | English  | 4.5G   | 112.35 sentence/s | ~45 sentence/s  | 15977 MiB   | 88.93% (88.93%) same |


## Fine-tune own model
Moreover, you are able to train models on your own data sets with BERT([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

[comment]: <> (or ZEN&#40;[paper]&#40;https://arxiv.org/abs/1911.00720&#41;&#41; as the encoder.)

### Downloading BERT, ZEN for train
For BERT, please download pre-trained model from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ZEN, you can download the pre-trained model from [here]&#40;https://github.com/sinovation/ZEN&#41;.)

### Datasets Requirements
In order to fine-tune your own Relation Extraction model, you need to divide the dataset into train, dev, test sets and save them with `.tsv` file format.
In details, you need to give entity 1, entity 2, the relationship between entity 1 and entity 2, and the sentence in turn, and use '\t' as the separator
You can see the [demo data](../../examples/DRel/data_demo) here.

### Examples for Training and Testing

You can find [example](../../examples/DRel/DRel_train.py) here. 
We recommend using the command line in [`DRel_train.sh`](./examples/DRel/DRel_train.sh) to fine-tune the model.

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
* `--dataset_name`: Choose the evaluation method according to dataset