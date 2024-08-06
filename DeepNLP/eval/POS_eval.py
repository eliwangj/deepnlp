from seqeval.metrics import f1_score, precision_score, recall_score


def pos_eval_sentence(y_pred, y, sentence, word2id):
    words = sentence.split(' ')
    seg_true = []
    seg_pred = []
    for i in range(len(words)):
        seg_true.append(words[i]+'_'+y[i])
        seg_pred.append(words[i]+'_'+y_pred[i])
    seg_true_str = ' '.join(seg_true)
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def pos_evaluate_word_accuracy(y_pred_list, y_true_list):
    score = 0
    length = 0
    for y_pred, y_true in zip(y_pred_list,y_true_list):
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                score += 1
        length += len(y_true)
    accuracy=score/length

    return accuracy

def pos_evaluate_word_PRF(y_pred_list, y_true_list):
    pP = precision_score(y_true_list,y_pred_list)
    pR = recall_score(y_true_list,y_pred_list)
    pF = f1_score(y_true_list,y_pred_list)

    return  (100 * pP, 100 * pR, 100 * pF)

def pos_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
    pos_cor_num = 0
    yt_wordnum = 0

    for y_p, y_p_p, sentence in zip(y_pred_list, y_list, sentence_list):
        for i in range(len(sentence)):
            word = sentence[i]
            if word in word2id:
                continue
            yt_wordnum += 1
            if y_p[i] != y_p_p[i]:
                pos_cor_num += 1

    pos_OOV = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1

    return 100 * pos_OOV



def pos_pred_result(y_pred, sentence, seperated_type='list'):
    if type(sentence) == list:
        words = sentence
    elif type(sentence) == str:
        words = sentence.split(' ')

    seg_pred = []
    for i in range(len(words)):
        try:
            seg_pred.append(words[i]+'_'+y_pred[i])
        except: 
            print("words:", words)
            print("#words:", len(words))
            print("y_pred:", y_pred)
            print("#y_pred:", len(y_pred))
            print("i: ", i)
            # store the problematic sentences into a seperate file
            problem_path = "/data1/junqiang/resources/DSRL/data/ontonotes5/problem_sentences.txt"
            with open(problem_path, 'w') as outfile:
                outrow = ' '.join(words) + '\n'
                outfile.write(outrow)
        # import pdb; pdb.set_trace()
   

    if seperated_type == 'list':
        return seg_pred
    elif seperated_type == 'blank':
        seg_pred_str = ' '.join(seg_pred)
        return seg_pred_str
    else:
        raise ValueError()
