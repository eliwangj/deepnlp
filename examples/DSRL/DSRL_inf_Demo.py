import sys
sys.path.append("../..")
from DeepNLP.model.DSRL import DSRL, readfile

if __name__ == '__main__':
    # Semantic Role labeling in English
    # model_path = 'bert-base-uncased'
    # model_path = '/data1/junqiang/dnlptk-main-sep2/examples/DSRL/saved_models/en_SRL_BERT_CoNLL12_ls_0.1.0/model'
    model_path = '/data1/qinhan/107/deepnlp_models/DSRL/saved_models/en_SRL_BiLSTM_CoNLL05_sm_0.1.0/model'
    deepnlp = DSRL.load_model(model_path=model_path, language='en', dataset='CoNLL05', local_rank=-1, no_cuda=False)

    ## handmade example sentences

    sentence_list = [['The', 'economy', "'s", 'temperature', 'will', 'be', 'taken', 'from', 'several', 'vantage', 'points', 'this', 'week', ',', 'with', 'readings', 'on', 'trade', ',', 'output', ',', 'housing', 'and', 'inflation', '.'],
                ['Exports', 'are', 'thought', 'to', 'have', 'risen', 'strongly', 'in', 'August', ',', 'but', 'probably', 'not', 'enough', 'to', 'offset', 'the', 'jump', 'in', 'imports', ',', 'economists', 'said', '.']]
    verb_index_list = [[6], [2]]
    # result_list = deepnlp.predict(sentence_list=sentence, verb_index_list=verb_list)
    # print("SRL results: \n", result_list)

    
    ## to test predict speed using more example sentences
    num_sentences = int(sys.argv[1])

    test_datapath = '/data1/junqiang/resources/DSRL/data/CoNLL05/test.tsv'
    lines = readfile(test_datapath)
    sentence_list = []
    verb_index_list = []
    for i in range(num_sentences):
        sentence_list.append(lines[i][0])
        verb_index_list.append(lines[i][2])

    # start_time = datetime.datetime.now()
    result_list, time_spent = deepnlp.predict(sentence_list=sentence_list, verb_index_list=verb_index_list)
    # end_time = datetime.datetime.now()
    # time_spent = (end_time - start_time).total_seconds()


    print("#sentences: ", num_sentences)
    print("manual speed: %.2f sentence/s" % (num_sentences/time_spent))
    print("model: ", model_path)
    print("SRL results: \n", result_list[0])

    
    # 100: 59.18sentence/s; 3.70it/s            | 52.26 sentence/s
    # 200:                                      | 62.23 sentence/s
    # 1000: 158.03sentence/s; 9.87it/s          | 156.57 sentence/s
    # 2000: 226.87sentence/s; 14.18it/s         | 229.33 sentence/s
    # (all test data) 5267: 323.18sentence/s


    # data file: ./DSRL/data/CoNLL05/test.tsv
    # number of sentences   | speed   
    # -------------------   | -----
    # 100                   | 52.26  sentence/s
    # 200                   | 62.23  sentence/s
    # 1000:                 | 156.57 sentence/s
    # 2000:                 | 229.33 sentence/s
    # (all test data) 5267  | 323.18 sentence/s


    # data file: ./DSRL/data/CoNLL05/test.tsv
    # start_time & end_time positions revised
    # number of sentences   | speed   
    # -------------------   | -----
    # 100                   | 819.85 sentence/s
    # 200                   | 938.55 sentence/s
    # 500                   | 948.33 sentence/s
    # 1000:                 | 989.73 sentence/s
    # 2000:                 | 968.88 sentence/s
    # (all test data) 5267  | 967.42 sentence/s
    

        

