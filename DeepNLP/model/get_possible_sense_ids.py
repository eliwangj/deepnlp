def get_verb2sense(train_data_path): #每个predicates对应的可能的senses
    verb2sense = {}

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        splits = line.split()

        if splits[12] == 'Y':  # when reading the line of predicates
            verb = splits[1]    # the 2nd column, this column has the word itself
            sense = splits[13]  # the 14th column. this column has the sense of predicate
            if verb not in verb2sense:     # if this verb has been recorded
                verb2sense[verb] = [sense]
            elif sense not in verb2sense[verb]: # if this verb has been recorded but the sense has been linked with this verb
                verb2sense[verb].append(sense)
    return verb2sense

def get_sense2id(train_data_path): # get a dictionary where keys are unique ids and values are corresponding predicate senses
    sense2id = {}
    index = 0

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    for line in lines:  # Indenting back ensures that I don't hold the file open longer than necessary.
        line = line.strip()
        if line == '':
            continue
        splits = line.split()
        sense = splits[13]  # the 14th column. this column has the sense of predicate
        # if sense != '_' and splits[12] == 'Y':  # the second condition is for double check
        if splits[12] == 'Y':  # '_' should also have an index
            if sense not in sense2id:
                sense2id[index] = sense
                index += 1
    return sense2id     # {0: 'predict.01', 1: 'predict.02', ...}

def get_sense_id(sense, sense2id_dic): # given a sense, return its index
    position = list(sense2id_dic.values()).index(sense) # get the position of the given sense in the dictionary
    sense_id = list(sense2id_dic.keys())[position]      # get the id of the given sense
    return sense_id

def get_possible_sense_ids(target_verb, sense2id_dic, verb2sense_dic): # given the target verb, we want to find the possible sense ids it corresponds to
    sense_ids = []
    for sense in verb2sense_dic[target_verb]:
        sense_id = get_sense_id(sense, sense2id_dic)
        sense_ids.append(sense_id)
    return sense_ids
    
if __name__ == '__main__':
    filepath = '/Users/eliwang/NLP_Work/NLP_Lab/DeepnlpToolkit/resources/DSRL/data/CoNLL09/conll09cn_dep/trial.txt'
    verb2sense_dic = get_verb2sense(filepath)
    sense2id_dic = get_sense2id(filepath)
    print(verb2sense_dic)
    print(sense2id_dic)
    target_verb = '输'
    sense_ids = get_possible_sense_ids(target_verb, sense2id_dic, verb2sense_dic)
    print(sense_ids)