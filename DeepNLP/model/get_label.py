def get_label(train_data_path): # get a list of all types of labels 
    label_list = ['<UNK>']  # initialize the list with an unknown token

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split()
            NUM_PRED = len(splits) - 14
            for v in range(NUM_PRED):
                srl_label = splits[14+v] # go over the argument positions horizontally
                if srl_label not in label_list and srl_label != '_':  # new label (exclude placeholder)
                    label_list.append(srl_label)    # add this label if it hasn't been recorded

    label_list.extend(['[CLS]', '[SEP]']) # beginning and seperating tokens
    return label_list

if __name__ == '__main__':
    filepath = '/Users/eliwang/NLP_Work/NLP_Lab/DeepnlpToolkit/resources/DSRL/data/CoNLL09/conll09cn_dep/double.txt'
    labels = get_label(filepath)
    print(labels)