import csv

def readfile(filepath): # anything related to file/data reading is here, no need to modify other places
    data = []  

    I_FORM = 1
    I_FILLPRED = 12
    I_PoneA = 14 # Pone = the first predicate; A = argument

    data_dic = {} # {'predicate1': (sentence, label, verb_index), 'predicate2': (sentence, label, verb_index), ...}
    pred_count = -1

    with open(filepath, 'r', encoding='utf8') as f:
        lines = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in lines:
            ## if this is an empty line, which indicates the end of a sentence, reset the dictionary and all sentence info
            if line == []:  
                for v in range(NUM_PRED):
                    # predicate = list(data_dic)[v] # the key of the data_dic
                    # data.append(data_dic[predicate]) # add the data entry
                    data.append(data_dic[v]) # add the data entry, v is the index and also the key of a corresponding predicate
                
                data_dic = {}   # reset the dictionary
                pred_count = -1 # reset the predicate count
                continue
            
            ## info of a data item
            NUM_PRED = len(line) - I_PoneA
            fillv = line[I_FILLPRED]

            ## Initialize the data_dic dictionary when reading the first line
            if line[0] == '1':
                for v in range(NUM_PRED):
                    data_dic[v] = ([], [], [])

            ## record the predicates (in index: 0 if there's one, 1 if there's two, etc.)
            if fillv == 'Y':
                pred_count += 1  

            ## Construct the data_dic dictionary
            for v in range(NUM_PRED): # if NUM_PRED = 2: v = 0, 1
                current_sentence = data_dic[v][0]
                current_label = data_dic[v][1]
                current_verb_index = data_dic[v][2]

                sr = line[I_PoneA + v] # the column of arguments regarding the current predicate
                word = line[I_FORM]

                current_sentence.append(word) # append the word into the sentence list

                if fillv == 'Y':
                    if pred_count == v: # if this is the current verb
                        current_label.append('V')  # record label V
                        current_verb_index.append(len(current_label))  # record the index of this predicate
                    else:   # if it's a verb but not the current verb
                        current_label.append('O')

                if sr == '_' and fillv != 'Y':
                    current_label.append('O')  # record label O
                elif sr != '_' and fillv != 'Y':
                    current_label.append(sr)  # record argument label
            
                assert len(current_sentence) == len(current_label)  # check if the number of words and number of labels match

    return data # a list of sentences of the structure (sentence, label, verb_index)


if __name__ == '__main__':
    filepath = '/Users/eliwang/NLP_Work/NLP_Lab/DeepnlpToolkit/resources/DSRL/data/CoNLL09/conll09cn_dep/trial.txt'
    lines = readfile(filepath)
    print(lines[0])
