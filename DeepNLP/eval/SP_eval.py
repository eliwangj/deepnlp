from seqeval.metrics import f1_score, precision_score, recall_score


def joint_pos_eval_sentence(y_pred, y, sentence, word2id):
    words = sentence.split(' ')
    seg_true = []
    seg_pred = []
    word_true = ''
    word_pred = ''

    y_word = []
    y_pos = []
    y_pred_word = []
    y_pred_pos = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_pos.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    for i in range(len(y_word)):
        word_true += words[i]
        word_pred += words[i]
        if y_word[i] in ['S', 'E']:
            pos_tag_true = y_pos[i]
            word_pos_true = word_true + '_' + pos_tag_true
            if word_true not in word2id:
                word_pos_true = '*' + word_pos_true + '*'
            seg_true.append(word_pos_true)
            word_true = ''
        if y_pred_word[i] in ['S', 'E']:
            pos_tag_pred = y_pred_pos[i]
            word_pos_pred = word_pred + '_' + pos_tag_pred
            seg_pred.append(word_pos_pred)
            word_pred = ''

    seg_true_str = ' '.join(seg_true)
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def joint_pos_evaluate_word_PRF(y_pred_list, y_true_list):
    #dict = {'E': 2, 'S': 3, 'B':0, 'I':1}

    # pP = pos_cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    # pR = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    # pF = 2 * pP * pR / (pP + pR)

    pP = precision_score(y_true_list,y_pred_list)
    pR = recall_score(y_true_list,y_pred_list)
    pF = f1_score(y_true_list,y_pred_list)

    return  (100 * pP, 100 * pR, 100 * pF)


def joint_pos_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
    word_cor_num = 0
    pos_cor_num = 0
    yt_wordnum = 0

    y_word_list = []
    y_pos_list = []
    y_pred_word_list = []
    y_pred_pos_list = []
    for y_label, y_pred_label in zip(y_list, y_pred_list):
        y_word = []
        y_pos = []
        y_pred_word = []
        y_pred_pos = []
        for y_l in y_label:
            y_word.append(y_l[0])
            y_pos.append(y_l[2:])
        for y_pred_l in y_pred_label:
            y_pred_word.append(y_pred_l[0])
            y_pred_pos.append(y_pred_l[2:])
        y_word_list.append(y_word)
        y_pos_list.append(y_pos)
        y_pred_word_list.append(y_pred_word)
        y_pred_pos_list.append(y_pred_pos)

    for y_w, y_p, y_p_w, y_p_p, sentence in zip(y_word_list, y_pos_list, y_pred_word_list, y_pred_pos_list, sentence_list):
        start = 0
        for i in range(len(y_w)):
            if y_w[i] == 'E' or y_w[i] == 'S':
                word = ''.join(sentence[start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                word_flag = True
                pos_flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y_w[j] != y_p_w[j]:
                        word_flag = False
                        pos_flag = False
                        break
                    if y_p[j] != y_p_p[j]:
                        pos_flag = False
                if word_flag:
                    word_cor_num += 1
                if pos_flag:
                    pos_cor_num += 1
                start = i + 1

    # word_OOV = word_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    pos_OOV = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1

    return 100 * pos_OOV



def joint_pos_pred_result(y_pred, sentence, seperated_type='list'):
    if type(sentence) == list:
        words = sentence
    elif type(sentence) == str:
        words = list(sentence)

    seg_pred = []
    word_pred = ''

    y_pred_word = []
    y_pred_pos = []
    for y_pred_label in y_pred:
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    for i in range(len(words)):
        word_pred += words[i]
        if y_pred_word[i] in ['S', 'E']:
            pos_tag_pred = y_pred_pos[i]
            word_pos_pred = word_pred + '_' + pos_tag_pred
            seg_pred.append(word_pos_pred)
            word_pred = ''

    if seperated_type == 'list':
        return seg_pred
    elif seperated_type == 'blank':
        seg_pred_str = ' '.join(seg_pred)
        return seg_pred_str
    else:
        raise ValueError()
