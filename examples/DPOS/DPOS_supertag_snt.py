import sys, os
import csv
sys.path.append("../..")
from DeepNLP.model.DPOS import DPOS

### to handle _csv.Error: field larger than field limit (131072), we increase the field_size_limit
# maxInt = sys.maxsize
# while True:
#     # decrease the maxInt value by factor 10 
#     # as long as the OverflowError occurs.
#     try:
#         csv.field_size_limit(maxInt)
#         break
#     except OverflowError:
#         maxInt = int(maxInt/10)


if __name__ == '__main__':
    # Declare paths
    model_path = '/data1/junqiang/dnlptk-main-sep2/examples/DPOS/saved_models/en_POS_BERTlarge_supertager_2021-09-13-15-26-42/model'
    snt_dir_path = '/data1/junqiang/resources/DSnt/data/'


    dataset_names = ['MAMS/', 'laptop/', 'rest14/', 'rest15/', 'rest16/', 'twitter/']

    # dataset_names = ['MAMS/']
    # data_names = ['val_processed.txt']
    # out_names = ['val_supertag.txt']
    data_names = ['test_processed.txt', 'train_processed.txt']
    out_names = ['test_supertag.txt', 'train_supertag.txt']

    # data_path = os.path.join(snt_data_path, 'val_processed.txt')
    # outfile_path = os.path.join(snt_data_path, 'val_supertag.txt')

    # data_path = os.path.join(snt_data_path, 'train_processed.txt')
    # outfile_path = os.path.join(snt_data_path, 'train_supertag.txt')

    for k in range(len(dataset_names)):
        snt_data_path = os.path.join(snt_dir_path, dataset_names[k])

        for i in range(len(data_names)):
            data_path = os.path.join(snt_data_path, data_names[i])
            outfile_path = os.path.join(snt_data_path, out_names[i])

            # Supertag for Snt
            deepnlp = DPOS.load_model(model_path=model_path, no_cuda=False, joint_cws_pos=False, use_memory=True)
            sentences = []
            with open(data_path, 'r') as infile, open(outfile_path, 'w') as outfile:
                writer = csv.writer(outfile)
                for line in csv.reader(infile, delimiter=' '):
                    sentence = []
                    for word in line:
                        if word != '': # except the empty word '', otherwise the number of words will exceed the number of labels
                            sentence.append(word)
                    sentences.append(sentence)
                # print(sentences)
                result_list = deepnlp.predict(sentence_list=sentences)

                for label_list in result_list:
                    # print(label_list)
                    outrow = ' '.join(label_list) + '\n'
                    outfile.write(outrow)
        

# 结果文件
# In_(S/S)/NP an_NP[nb]/N was_N/N 19_N/N
# ...
# ...
# In_(S/S)/NP an_NP[nb]/N was_N/N 19_N/N