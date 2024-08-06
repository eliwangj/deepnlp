import sys
sys.path.append("../..")
from DeepNLP.model.DNER import DNER

if __name__ == '__main__':
    # inference
    sentence = [
        ['常', '建', '良', '，', '男', '，', '1', '9', '6', '3', '年', '出', '生', '，', '工', '科', '学', '士', '，', '高', '级', '工', '程', '师', '，', '北', '京', '物', '资', '学', '院', '客', '座', '副', '教', '授', '。'],
        ['陈', '宝', '杰', '，', '男', '，', '汉', '族', '。', '毕', '业', '于', '解', '放', '军', '南', '京', '政', '治', '学', '院', '，', '大', '学', '学', '历', '。']
    ]

    label_list = [
        ['B-NAME', 'M-NAME', 'E-NAME', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PRO', 'E-PRO', 'B-EDU', 'E-EDU', 'O', 'B-TITLE', 'M-TITLE', 'M-TITLE', 'M-TITLE', 'E-TITLE', 'O', 'B-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'E-ORG', 'B-TITLE', 'M-TITLE', 'M-TITLE', 'M-TITLE', 'E-TITLE', 'O'],
        ['B-NAME', 'M-NAME', 'E-NAME', 'O', 'O', 'O', 'B-RACE', 'E-RACE', 'O', 'O', 'O', 'O', 'B-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'M-ORG', 'E-ORG','O', 'B-EDU', 'M-EDU', 'M-EDU', 'E-EDU', 'O']
    ]

    deepnlp = DNER.load_model(
        # model_path='//data/Yangyang/D_project/saved_models/NEWTEST_NER_baseline_cn_2021-07-13-10-52-36/model',
        model_path='/data1/qinhan/107/deepnlp_models/DNER/saved_models/zh_NER_BERT_RE_bs_0.1.0/model',
        no_cuda=False
    )

    predict_result = deepnlp.predict(sentence_list=sentence)
    print(predict_result)
    # [['常建良_NAME', '，_', '男_', '，_', '1_', '9_', '6_', '3_', '年_', '出_', '生_', '，_', '工科_PRO', '学士_EDU', '，_', '高级工程师_TITLE', '，_', '北京物资学院_ORG', '客座副教授_TITLE', '。_'],
    #  ['陈宝杰_NAME', '，_', '男_', '，_', '汉族_RACE', '。_', '毕_', '业_', '于_', '解放军南京政治学院_ORG', '，_', '大学学历_EDU', '。_']]

    # test 在自己的数据集上test模型表现
    # P, R, F = deepnlp.test(sentence_list=sentence, gold_list=label_list)
    # print(P, R, F)
