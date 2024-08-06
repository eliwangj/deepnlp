import sys
sys.path.append("../..")
from DeepNLP.model.DRel_wjq import DRel, readfile
import datetime  # for speed recording


if __name__ == '__main__':
    # Rel in English
    model_path = '/data1/junqiang/dnlptk-main-sep2/examples/DRel/saved_models/en_Rel_BERT_semeval_ls_0.1.0/model'
    deepnlp = DRel.load_model(model_path=model_path, no_cuda=False)
    
    # sentence = ['The most common audits were about waste and recycling.',
    #             'The company fabricates plastic chairs.']

    # # support two type e1 list or e2 list
    # e1_list = ['audits', 'company']
    # e2_list = ['waste', 'chairs']
    # e1_list = [(6, 7), (1, 2)]
    # e2_list = [(1, 2), (4, 5)]

    # result_list = deepnlp.predict(sentence_list=sentence, e1_list=e1_list, e2_list=e2_list)
    # print(result_list)
    # # ['Message-Topic(e1,e2)', 'Product-Producer(e2,e1)']


    # to test predict speed using more example sentences
    num_sentences = int(sys.argv[1])

    test_datapath = '/data1/junqiang/resources/DRel/data/ace2005en/test.tsv'
    lines = readfile(test_datapath)
    sentence_list = []
    e1_list = []
    e2_list = []
    for i in range(num_sentences):
        e1 = lines[i][0]
        e2 = lines[i][1]
        e1_list.append(e1)
        e2_list.append(e2)
        sentence = lines[i][3].replace("<e1> %s </e1>" % e1, e1).replace("<e2> %s </e2>" % e2, e2)
        sentence_list.append(sentence)

    start_time = datetime.datetime.now()
    result_list = deepnlp.predict(sentence_list=sentence_list, e1_list=e1_list, e2_list=e2_list)
    end_time = datetime.datetime.now()
    time_spent = (end_time - start_time).total_seconds()


    print("#sentences: ", num_sentences)
    print("manual speed: %.2f sentence/s" % (num_sentences/time_spent))
    print("model: ", model_path)
    print("Rel result example: \n", result_list[:2])
