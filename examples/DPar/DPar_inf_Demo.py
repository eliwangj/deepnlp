import sys
sys.path.append("../..")
from DeepNLP.model.DPar import DPar

if __name__ == '__name__':
    # English
    deepnlp = DPar.load_model(model_path='bert-base-uncased', no_cuda=False)
    sentence = [['From', 'the', 'AP', 'comes', 'this', 'story', ':'],
                ['The', 'sheikh', 'in', 'wheel', '-', 'chair', 'has', 'been', 'attacked', 'with', 'a', 'F', '-', '16',
                 '-', 'launched', 'bomb', '.']]
    predict_result = deepnlp.predict(sentence_list=sentence)
    print(predict_result)
    # [(['From', 'the', 'AP', 'comes', 'this', 'story', ':'],
    #  [3, 3, 4, 0, 6, 4, 4],
    #  ['case', 'det', 'obl', 'root', 'det', 'nsubj', 'punct']),
    # (['The', 'sheikh', 'in', 'wheel', '-', 'chair', 'has', 'been', 'attacked', 'with', 'a', 'F', '-', '16', '-', 'launched', 'bomb', '.'],
    #  [2, 9, 6, 6, 6, 2, 9, 9, 0, 17, 17, 16, 12, 12, 16, 17, 9, 9],
    #  ['det', 'nsubj:pass', 'case', 'compound', 'punct', 'nmod', 'aux', 'aux:pass', 'root', 'case', 'det', 'obl:npmod', 'punct', 'nummod', 'punct', 'amod', 'obl', 'punct'])
    # ]

    #Chinese
    deepnlp = DPar.load_model(model_path='bert-base-chinese', no_cuda=False)
    sentence = [
        ['大', '多数', '的', '加长', '型', '礼车', '则是', '租车', '公司', '的', '财产', '。'],
        ['1355', '年', '，', '勃兰登堡', '被', '神圣', '罗马', '帝国', '皇帝', '查理', '四世', '升', '为', '选侯', '国', '。']
    ]
    result_list = deepnlp.predict(sentence_list=sentence)
    print(result_list)
    # [(['大', '多数', '的', '加长', '型', '礼车', '则是', '租车', '公司', '的', '财产', '。'],
    #   [2, 6, 2, 5, 6, 11, 11, 9, 11, 9, 0, 11],
    #   ['advmod', 'amod', 'case', 'compound', 'p', 'nmod', 'nmod', 'case', 'root', 'punct']),
    #  (['1355', '年', '，', '勃兰登堡', '被', '神圣', '罗马', '帝国', '皇帝', '查理', '四世', '升', '为', '选侯', '国', '。'],
    #   [2, 12, 9, 10, 0, 12, 15, 12, 12],
    #   ['nummod', 'nmod:tmod', 'punct', 'nsubj:pass', 'aux:pass', 'amod', 'nmod', 'nmod', 'nsubj', 'appos', 'flat:name', 'root', 'mark', 'compound', 'obj', 'punct'])
    #  ]