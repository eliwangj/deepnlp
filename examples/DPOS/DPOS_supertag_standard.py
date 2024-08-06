import sys, os
# import csv
sys.path.append("../..")
from DeepNLP.model.DPOS import DPOS


if __name__ == '__main__':
    # Declare paths
    model_path = '/data1/junqiang/dnlptk-main-sep2/examples/DPOS/saved_models/en_POS_BERTlarge_supertag_2021-09-15-22-07-30/model'
    
    
    ## SRL data
    # dir_path = '/data1/junqiang/resources/DSRL/data/'
    # dataset_names = ['ontonotes5/', 'CoNLL09/conll09en_dep/']

    ## Rel data
    dir_path = '/data1/junqiang/resources/DRel/data/'
    dataset_names = ['ace2005en/', 'semeval/']
    
    data_names = ['dev_processed.txt', 'test_processed.txt', 'train_processed.txt']
    out_names = ['dev_supertag.txt', 'test_supertag.txt', 'train_supertag.txt']
    
    ## Rel testing
    # dir_path = '/data1/junqiang/resources/DRel/data/'
    # # dir_path = '/Users/eliwang/NLP_Work/NLP_Lab/DeepnlpToolkit/resources/DRel/data/'
    # dataset_names = ['ace2005en/']
    # data_names = ['dev_processed.txt']
    # out_names = ['dev_supertag.txt']



    for k in range(len(dataset_names)):
        dataset_path = os.path.join(dir_path, dataset_names[k])

        for i in range(len(data_names)):
            data_path = os.path.join(dataset_path, data_names[i])
            outfile_path = os.path.join(dataset_path, out_names[i])

            # Supertag
            deepnlp = DPOS.load_model(model_path=model_path, no_cuda=False, joint_cws_pos=False, use_memory=True)
            sentences = []
            with open(data_path, 'r') as infile, open(outfile_path, 'w') as outfile:
                ## using csv.reader to read file
                # for line in csv.reader(infile, delimiter=' '):
                #     sentence = []
                #     for word in line:
                #         if word != '': # except the empty word '', otherwise the number of words will exceed the number of labels
                #             sentence.append(word)
                #     sentences.append(sentence)
                
                ## using readlines() to read file
                lines = infile.readlines()

                for line in lines:
                    sentence = []
                    line = line.strip()
                    if line == '':
                        continue
                    splits = line.split()
                    for word in splits:
                        if word != '':     # except the empty word '' if any, otherwise the number of words will exceed the number of labels
                            sentence.append(word)
                    sentences.append(sentence)
                    
                result_lists = deepnlp.predict(sentence_list=sentences)

                for label_list in result_lists:
                    outrow = ' '.join(label_list) + '\n'
                    outfile.write(outrow)
