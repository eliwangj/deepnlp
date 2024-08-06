import re
import json
import numpy as np


def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    sentence = []
    labels = []
    for line in lines:
        line = line.strip()
        if line == '':
            if len(sentence) > 0:
                sentence_list.append(sentence)
                label_list.append(labels)
                sentence = []
                labels = []
            continue
        items = re.split('\\s+', line)
        character = items[0]
        label = items[-1]
        sentence.append(character)
        labels.append(label)
    #
    if len(sentence) > 0:
        sentence_list.append(sentence)
        label_list.append(labels)
    #
    return sentence_list, label_list


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f)
        f.write('\n')


def read_embedding(embedding_file):
    word2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}
    index = 4
    with open(embedding_file, 'r', encoding='utf8') as f:
        #读取embedding的默认设置
        line = f.readline().strip()
        splits = line.split()
        token_number = int(splits[0])
        dimension = int(splits[1])
        embedding = np.zeros((token_number+index, dimension), dtype=float)
        embedding[1:index] = np.random.standard_normal((index-1, dimension))
        line = f.readline()
        while not line == '' and index < token_number + 4:
            split = line.strip().split()
            if not len(split) == dimension + 1:
                line = f.readline()
                continue
            word = split[0]
            if word == '<s>':
                embedding[2] = np.array([float(i) for i in split[1:]])
            elif word == '</s>':
                embedding[3] = np.array([float(i) for i in split[1:]])
            elif word in ['<OOV>', '<unk>', '<UNK>', 'UNK_W']:
                embedding[1] = np.array([float(i) for i in split[1:]])
            elif word not in word2id:
                word2id[word] = index
                embedding[index] = np.array([float(i) for i in split[1:]])
                index += 1
            line = f.readline()
    embedding = embedding[:index]
    assert len(word2id) == embedding.shape[0]
    return word2id, embedding


def get_language(c):
    while (c >= u'\u0030' and c <= u'\u0040'):
        c = c[1:]
    #删除特殊字符和标点
    while (c < u'\u4e00' or c > u'\u9fa5') and (c < u'\u0041' or c > u'\u005a') and (c < u'\u0061' or c > u'\u007a'):
        c = c[1:]
    if c >= u'\u4e00' and c <= u'\u9fa5':
        return 'zh'
    if (c >= u'\u0041' and c <= u'\u005a') or (c >= u'\u0061' and c <= u'\u007a'):
        return 'en'
