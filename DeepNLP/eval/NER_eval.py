from seqeval.metrics import f1_score, precision_score, recall_score


def eval_sentence(y_pred, y, sentence, NE2id, language):
    words = sentence.split(' ')
    seg_true = []
    seg_pred = []
    word_true = ''
    word_pred = ''

    y_word = []
    y_ner = []
    y_pred_word = []
    y_pred_ner = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_ner.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_ner.append(y_pred_label[2:])

    for i in range(len(y_word)):
        if word_true != '' and y_word[i] in ['B', 'O']:
            ner_tag_true = y_ner[i-1]
            word_ner_true = word_true.strip() + '_' + ner_tag_true
            if word_true not in NE2id:
                word_ner_true = '*' + word_ner_true + '*'
            seg_true.append(word_ner_true)
            word_true = ''
        if word_pred != '' and y_pred_word[i] in ['B', 'O']:
            ner_tag_pred = y_pred_ner[i-1]
            word_ner_pred = word_pred.strip() + '_' + ner_tag_pred
            seg_pred.append(word_ner_pred)
            word_pred = ''
        if language == 'en':
            word_true += ' ' + words[i]
            word_pred += ' ' + words[i]
        elif language == 'zh':
            word_true += words[i]
            word_pred += words[i]

    if word_true != '' and y_word[i] in ['B', 'O']:
        ner_tag_true = y_ner[i]
        word_ner_true = word_true.strip() + '_' + ner_tag_true
        if word_true not in NE2id:
            word_ner_true = '*' + word_ner_true + '*'
        seg_true.append(word_ner_true)
    if word_pred != '' and y_pred_word[i] in ['B', 'O']:
        ner_tag_pred = y_pred_ner[i]
        word_ner_pred = word_pred.strip() + '_' + ner_tag_pred
        seg_pred.append(word_ner_pred)


    seg_true_str = ' '.join(seg_true)
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def evaluate_word_PRF(y_pred_list, y_true_list):
    pP = precision_score(y_true_list,y_pred_list)
    pR = recall_score(y_true_list,y_pred_list)
    pF = f1_score(y_true_list,y_pred_list)
    return (100 * pP, 100 * pR, 100 * pF)



def evaluate_OOV(y_pred_list, y_list, sentence_list, NE2id, language):
    word_cor_num = 0
    ner_cor_num = 0
    yt_wordnum = 0

    y_word_list = []
    y_ner_list = []
    y_pred_word_list = []
    y_pred_ner_list = []
    for y_label, y_pred_label in zip(y_list, y_pred_list):
        y_word = []
        y_ner = []
        y_pred_word = []
        y_pred_ner = []
        for y_l in y_label:
            y_word.append(y_l[0])
            y_ner.append(y_l[2:])
        for y_pred_l in y_pred_label:
            y_pred_word.append(y_pred_l[0])
            y_pred_ner.append(y_pred_l[2:])
        y_word_list.append(y_word)
        y_ner_list.append(y_ner)
        y_pred_word_list.append(y_pred_word)
        y_pred_ner_list.append(y_pred_ner)

    for y_w, y_p, y_p_w, y_p_p, sentence in zip(y_word_list, y_ner_list, y_pred_word_list, y_pred_ner_list,
                                                sentence_list):
        start = 0
        word = ''
        for i in range(len(y_w)):
            if y_w[i] == 'O':
                if word:
                    word = ''.join(sentence[start:i])
                    if word in NE2id:
                        word = ''
                        continue
                    word_flag = True
                    ner_flag = True
                    yt_wordnum += 1
                    for j in range(start, i + 1):
                        if y_w[j] != y_p_w[j]:
                            word_flag = False
                            ner_flag = False
                            break
                        if y_p[j] != y_p_p[j]:
                            ner_flag = False
                    if word_flag:
                        word_cor_num += 1
                    if ner_flag:
                        ner_cor_num += 1
                    word = ''
            elif y_w[i] == 'B':
                if word:
                    word = ''.join(sentence[start:i])
                    if word in NE2id:
                        start = i
                        if language == 'zh':
                            word = ''.join(sentence[start:i + 1])
                        elif language == 'en':
                            word = ' '.join(sentence[start:i + 1])
                        continue
                    word_flag = True
                    ner_flag = True
                    yt_wordnum += 1
                    for j in range(start, i + 1):
                        if y_w[j] != y_p_w[j]:
                            word_flag = False
                            ner_flag = False
                            break
                        if y_p[j] != y_p_p[j]:
                            ner_flag = False
                    if word_flag:
                        word_cor_num += 1
                    if ner_flag:
                        ner_cor_num += 1
                    start = i
                    if language == 'zh':
                        word = ''.join(sentence[start:i + 1])
                    elif language == 'en':
                        word = ' '.join(sentence[start:i + 1])
                else:
                    start = i
                    if language == 'zh':
                        word = ''.join(sentence[start:i + 1])
                    elif language == 'en':
                        word = ' '.join(sentence[start:i + 1])
    ner_OOV = ner_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    return 100 * ner_OOV



def pred_result(y_pred, sentence, language, seperated_type='list'):
    if type(sentence) == list:
        words = sentence
    elif type(sentence) == str:
        words = list(sentence)
#
    seg_pred = []
    word_pred = ''
#
    y_pred_word = []
    y_pred_ner = []
    for y_pred_label in y_pred:
        y_pred_word.append(y_pred_label[0])
        y_pred_ner.append(y_pred_label[2:])
#
    for i in range(len(words)):
        if word_pred != '' and y_pred_word[i] in ['B', 'O']:
            ner_tag_pred = y_pred_ner[i-1]
            word_ner_pred = word_pred.strip() + '_' + ner_tag_pred
            seg_pred.append(word_ner_pred)
            word_pred = ''

        if language == 'en':
            word_pred += ' ' + words[i]
        elif language == 'zh':
            word_pred += words[i]
    if word_pred != '' and y_pred_word[i] in ['B', 'O']:
        ner_tag_pred = y_pred_ner[i]
        word_ner_pred = word_pred.strip() + '_' + ner_tag_pred
        seg_pred.append(word_ner_pred)
#
    if seperated_type == 'list':
        return seg_pred
    elif seperated_type == 'blank':
        seg_pred_str = ' '.join(seg_pred)
        return seg_pred_str
    else:
        raise ValueError()
