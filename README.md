# DeepNLP
DeepNLP is a library for Natural Language Processing in python. 
It provides a unified framework to perform NLP tasks such as Part-Of-Speech tagging, Named Entity Recognition, relation extraction and sentiment analysis in quite straigthforward way. 


This project was conducted under the supervision of Prof. Yan Song and coutesy of all team members' extensive efforts.

<br>
## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.4`

Use `pip install -r requirements.txt` to install the required packages.

## Installation

### pip ?
### conda ?
### python setup.py install




## Loading and using models 
For specific task, use `Dtasks.load()` with the model name or a path to load the model data directory.
You only need to specify the pre-training model that you want to use like `bert-base-uncase`, and then the model for the task can be downloaded and loaded automatically. Moreover, you can download the model locally through the link provided in [Model Documentation](#1), and then load it through the local path. Here are some examples for sequence labeling tasks:
```python
#Chinese Word Segmentation
from DeepNLP.model.DSeg import DSeg
deepnlp = DSeg.load_model(model_path='bert-base-chinese',no_cuda=False)
sentence = [['法'，'正','研','究','从','波','黑','撤','军','计','划','新','华','社','巴','黎','９','月','１','日','电','（','记','者','张','有','浩','）'],
            ['法','国','国','防','部','长','莱','奥','塔','尔','１','日','说','，法','国','正','在','研','究','从','波','黑','撤','军','的','计','划','。']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#[['法', '正', '研究', '从', '波黑', '撤军', '计划', '新华社', '巴黎', '９月', '１日', '电', '（', '记者', '张有浩', '）'], 
#['法国', '国防', '部长', '莱奥塔尔', '１日', '说', '计', '划。']]
```

```python
#POS tagging
from DeepNLP.model.DPOS import DPOS
deepnlp = DPOS.load_model(model_path='bert-base-cased',no_cuda=False)
sentence = [['The', 'Arizona', 'Corporations', 'Commission', 'authorized', 'an', '11.5', '%', 'rate', 'increase', 'at', 'Tucson', 'Electric', 'Power', 'Co.', ',', 'substantially', 'lower', 'than', 'recommended', 'last', 'month', 'by', 'a', 'commission', 'hearing', 'officer', 'and', 'barely', 'half', 'the', 'rise', 'sought', 'by', 'the', 'utility', '.'], 
            ['The', 'ruling', 'follows', 'a', 'host', 'of', 'problems', 'at', 'Tucson', 'Electric', ',', 'including', 'major', 'write-downs', ',', 'a', '60', '%', 'slash', 'in', 'the', 'common', 'stock', 'dividend', 'and', 'the', 'departure', 'of', 'former', 'Chairman', 'Einar', 'Greve', 'during', 'a', 'company', 'investigation', 'of', 'his', 'stock', 'sales', '.']]
predict_result = deepnlp.predict(sentence_list=sentence)
print(predict_result)
#[['The_DT', 'Arizona_NNP', 'Corporations_NNPS', 'Commission_NNP', 'authorized_VBD', 'an_DT', '11.5_CD', '%_NN', 'rate_NN', 'increase_NN', 'at_IN', 'Tucson_NNP', 'Electric_NNP', 'Power_NNP', 'Co._NNP', ',_,', 'substantially_RB', 'lower_JJR', 'than_IN', 'recommended_VBN', 'last_JJ', 'month_NN', 'by_IN', 'a_DT', 'commission_NN', 'hearing_NN', 'officer_NN', 'and_CC', 'barely_RB', 'half_PDT', 'the_DT', 'rise_NN', 'sought_VBN', 'by_IN', 'the_DT', 'utility_NN', '._.'], 
# ['The_DT', 'ruling_NN', 'follows_VBZ', 'a_DT', 'host_NN', 'of_IN', 'problems_NNS', 'at_IN', 'Tucson_NNP', 'Electric_NNP', ',_,', 'including_VBG', 'major_JJ', 'write-downs_NNS', ',_,', 'a_DT', '60_CD', '%_NN', 'slash_NN', 'in_IN', 'the_DT', 'common_JJ', 'stock_NN', 'dividend_NN', 'and_CC', 'the_DT', 'departure_NN', 'of_IN', 'former_JJ', 'Chairman_NNP', 'Einar_NNP', 'Greve_NNP', 'during_IN', 'a_DT', 'company_NN', 'investigation_NN', 'of_IN', 'his_PRP$', 'stock_NN', 'sales_NNS', '._.']]
```
A demo for dependency parsing.
```python
#Dependency Parsing
from DeepNLP.model.DPar import DPar
deepnlp = DPar.load_model(model_path='bert-base-cased',no_cuda=False)
sentence = [['分布', '于', '西达', '印度', '洋', '塞席尔', '群岛', '及', '马尔地夫', '群岛', '以及', '海南', '省', '中沙', '群岛', '等', '，', '属', '于', '热带', '浅', '海', '底层', '鱼', '。']]
sentence_list, head_list, label_list= deepnlp.predict(sentence_list=sentence)
print(sentence_list, head_list, label_list)

#
#([['分布', '于', '西达', '印度', '洋', '塞席尔', '群岛', '及', '马尔地夫', '群岛', '以及', '海南', '省', '中沙', '群岛', '等', '，', '属', '于', '热带', '浅', '海', '底层', '鱼', '。']],
#[[18, 7, 7, 5, 7, 7, 1, 10, 10, 7, 15, 13, 15, 15, 7, 7, 18, 0, 18, 24, 24, 24, 24, 18, 18]],
#[['advcl', 'case', 'nmod', 'compound', 'nmod', 'nmod', 'obl', 'cc', 'nmod', 'conj', 'cc', 'compound', 'nmod', 'nmod', 'conj', 'acl', 'punct', 'root', 'mark', 'nmod', 'nmod', 'nmod', 'compound', 'obj', 'punct']])
```
### Pre-train model and train your own model
Even though the model for each task provided, you can download the pre-train model or word embedding to train your own model and get the result of the corresponding task.

<h2 id="1">Model Documentation</h2>

| **Packages**        | **Descriptions**                                                               |
| ------------------- | -------------------------------------------------------------- |
| [Dseg](./doc/Dseg/README.md) | Word segmentation for simplified standard Chinese or ancient Chinese.    |
| [DPOS](./doc/DPOS/README.md) | POS tagging in chinese and englishand joint task of Chinese word segmentation and POS tagging.            |
| [DPar](./doc/DPar/README.md) | Dependency Parsing in chinese and english.             |
| [DNER](./doc/DNER/README.md) | Named Entity Recognition in chinese and english.       |
| [DSRL](./doc/DSRL/README.md) | Semantic Role Labeling in chinese and english.         |
| [DRel](./doc/DRel/README.md) | Relation Extraction in chinese and english.            |
| [DSnt](./doc/DSnt/README.md) | Aspected-based Sentiment Analysis and general Sentiment Analysis in chinese and english. |


## To-do List

* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.



Except the CWS and CWS-POS joint tagging which assigned for chinese, all other tasks provide english and chinese models.


[comment]: <> (| Model name | Task\Language | Chinese | English | Chinese | Eng-Uncased | Eng-Uncased |)

[comment]: <> (| --- | --- | --- | --- |  --- |  --- |  --- |)

[comment]: <> (| DSeg | Chinese Word Segmentation &#40;CWS&#41; | [LSTM]&#40;http://nlp.&#41; |  | Bert-base-chinese |  |  | )

[comment]: <> (| DSP | CWS-POS joint tagging  | [LSTM]&#40;http://nlp.&#41; |  | Bert-base-chinese |  |  | )

[comment]: <> (| DPOS | Part-Of-Speech tagging | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )

[comment]: <> (| DPar | Dependency Parsing | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )

[comment]: <> (| DNER | Named Entity Recognition&#40;NER&#41; | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )

[comment]: <> (| DSRL | Semantic Role Labeling&#40;SRL&#41; | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )

[comment]: <> (| DSRel | Relation Extraction &#40;RE&#41; | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )

[comment]: <> (| DSA | Sentiment Analysis | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )

[comment]: <> (| DSnt | Aspected-based Sentiment Analysis &#40;ASA&#41; | [LSTM]&#40;http://nlp.&#41; | LSTM | Bert-base-chinese | Bert-base-uncased | Bert-large-uncased | )



[comment]: <> (Now we )

[comment]: <> (For senquence labeling tasks like CWS, NER and CWS-POS joint tagging, )

[comment]: <> (, SRL and the input sentences should be lists of word.)

[comment]: <> (Here is how to quickly use DeepNLP to obtain the chinese word segmentation:)

[comment]: <> (## Downloading model)

[comment]: <> (In DeepNLP, you can use the models provided by us directly to obtain the result of tasks.)

[comment]: <> (Moreover, you can finetune your own model for some tasks with commonly used pre-training model.)

[comment]: <> (- \#\#\# DeepNLP model)

[comment]: <> (- \#\#\# pre-train model &#40;BERT, ZEN, XLNet&#41;)
