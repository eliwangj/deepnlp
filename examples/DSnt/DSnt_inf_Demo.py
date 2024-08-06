import sys
sys.path.append("../..")
from DeepNLP.model.DSnt import DSnt, readfile

if __name__ == '__main__':
    
    model_path = '/data/wangjunqiang/dnlptk-main-sep2/examples/DSnt/saved_models/en_ABSA_BERT_rest16_ls_0.1.0/model'
    deepnlp = DSnt.load_model(model_path=model_path, ABSA=True, dataset='MAMS', TGCN=False, local_rank=-1, no_cuda=False)

    num_sentences = int(sys.argv[1])

    test_datapath = '/data/wangjunqiang/resources/DSnt/data/MAMS/test.txt'
    lines = readfile(test_datapath, 'testing')
    sentence_list = []
    aspect_list = []
    for i in range(num_sentences):
        sentence_list.append(lines[i][0])  # data format of elements in 'lines' : (sentence, aspect_list, aspect_index, label, None, None))
        aspect_list.append(lines[i][1])

    result_list, time_spent = deepnlp.predict(sentence_list=sentence_list, aspect_list=aspect_list)

    print("#sentences: ", num_sentences)
    print("manual speed: %.2f sentence/s" % (num_sentences/time_spent))
    print("model: ", model_path)
    print("Snt result examples: \n", result_list[:10])


    # # [Aspect Based Sentiment Analysis] in English
    # model_path = '/data/Yangyang/D_project/saved_models/ASA_baseline_en_2021-07-09-21-17-13/model'
    # deepnlp = DSnt.load_model(model_path=model_path, ABSA=False, dataset='MAMS', TGCN=False, local_rank=-1, no_cuda=False)
    # sentence = [['i', 'charge', 'it', 'at', 'night', 'and', 'skip', 'taking', 'the', 'cord', 'with', 'me', 'because', 'of', 'the', 'good', 'battery', 'life', '.'],
    #             ['the', 'tech', 'guy', 'then', 'said', 'the', 'service', 'center', 'does', 'not', 'do', '1-to-1', 'exchange', 'and', 'i', 'have', 'to', 'direct', 'my', 'concern', 'to', 'the', "''", 'sales', "''", 'team', ',', 'which', 'is', 'the', 'retail', 'shop', 'which', 'i', 'bought', 'my', 'netbook', 'from', '.']]
    # # The first way to enter aspect term（recommend）
    # aspect_list = [(16, 17), (23, 25)]
    # # The second way to enter aspect term
    # aspect_list = [['battery', 'life'], ['sales', 'team']]

    # result_list, time_spent = deepnlp.predict(sentence_list=sentence, aspect_list=aspect_list)
    # print("result_list: ", result_list)
    # print("time_spent: {}s ".format(time_spent))
    # # ['1', '0']
    # # positive: '1'
    # # negative: '-1'
    # # neutral: '0'



    # # [Sentiment Analysis] in Chinese
    # model_path = '/data/Yangyang/D_project/saved_models/SA_baseline_cn_2021-07-14-00-26-57/model'
    # deepnlp = DSnt.load_model(model_path=model_path, no_cuda=False)
    # sentence = [
    #     ['不错的上网本，外形很漂亮，操作系统应该是个很大的 卖点，电池还可以。整体上讲，作为一个上网本的定位，还是不错的'],
    #     ['XP的驱动不好找！我的17号提的货，现在就降价了100元，而且还送杀毒软件！']
    # ]
    # result_list = deepnlp.predict(sentence_list=sentence)
    # print(result_list)
    # # ['1', '0']
    # # positive: '1'
    # # negative: '0'

