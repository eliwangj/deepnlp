import sys, os
import csv
sys.path.append("../..")
from DeepNLP.model.DPOS import DPOS

### to handle _csv.Error: field larger than field limit (131072), we increase the field_size_limit
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


if __name__ == '__main__':
    # Declare paths
    model_path = '/data1/junqiang/dnlptk-main-sep2/examples/DPOS/saved_models/en_POS_BERTlarge_supertager_2021-09-13-15-26-42/model'
    # srl_data_path = '/data1/junqiang/resources/DSRL/data/CoNLL05/'
    srl_data_path = '/data1/junqiang/resources/DSRL/data/ontonotes5/'

    

    # data_names = ['dev.tsv', 'test.tsv', 'train.tsv', 'brown.tsv']
    # out_names = ['dev_supertag.txt', 'test_supertag.txt', 'train_supertag.txt', 'brown_supertag.txt']

    data_names = ['dev.tsv', 'test.tsv', 'train.tsv']
    out_names = ['dev_supertag.txt', 'test_supertag.txt', 'train_supertag.txt']



    for i in range(len(data_names)):
        # Set up data paths
        data_path = os.path.join(srl_data_path, data_names[i])
        outfile_path = os.path.join(srl_data_path, out_names[i])

        # Supertag for SRL
        deepnlp = DPOS.load_model(model_path=model_path, no_cuda=False, joint_cws_pos=False, use_memory=True)
        sentences = []
        with open(data_path, 'r') as infile, open(outfile_path, 'w') as outfile:
            writer = csv.writer(outfile)
            sentence = []
            for line in csv.reader(infile, delimiter='\t', quoting=csv.QUOTE_NONE): # shouldn't recognize any as quotes since we need them in our sentence
                if line == []:
                    sentences.append(sentence)
                    sentence = []

                else:
                    word = line[0] 
                    sentence.append(word)
                    
            print(sentences[:5])
                    
            result_list = deepnlp.predict(sentence_list=sentences)

            print(result_list[:5])

            for outsentence in result_list:
                # print(label_list)
                outrow = ' '.join(outsentence) + '\n'
                outfile.write(outrow)
        

# 结果文件
# The_(S/S)/NP $T$_NP[nb]/N Oct._N/N 19_N/N
# ...
# ...
# Howard_N/N Mosher_N ,_, president_N and_conj