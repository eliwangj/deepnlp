## DPOS

**DPOS_supertag_standard.py**

```sentences[:10]```
```
[['One', 'of', 'the', 'toughest', 'jobs', 'in', 'Iraq', 'right', 'now', 'is', 'that', 'of', 'the', 'military', 'doctor', '.'], ['CNN', 'medical', 'correspondent', ',', 'Dr', '.'], ['CNN', 'medical', 'correspondent', ',', 'Dr', '.'], ['CNN', 'medical', 'correspondent', ',', 'Dr', '.'], ['Sanjay', 'Gupta', ',', 'has', 'been', 'traveling', 'with', 'the', 'Devil', 'Docs', 'throughout', 'the', 'war', 'and', 'now', 'they', 'are', 'close', 'to', 'Baghdad', '.'], ['Sanjay', 'Gupta', ',', 'has', 'been', 'traveling', 'with', 'the', 'Devil', 'Docs', 'throughout', 'the', 'war', 'and', 'now', 'they', 'are', 'close', 'to', 'Baghdad', '.'], ['Sanjay', 'Gupta', ',', 'has', 'been', 'traveling', 'with', 'the', 'Devil', 'Docs', 'throughout', 'the', 'war', 'and', 'now', 'they', 'are', 'close', 'to', 'Baghdad', '.'], ['Sanjay', 'Gupta', ',', 'has', 'been', 'traveling', 'with', 'the', 'Devil', 'Docs', 'throughout', 'the', 'war', 'and', 'now', 'they', 'are', 'close', 'to', 'Baghdad', '.'], ['Sanjay', 'Gupta', ',', 'has', 'been', 'traveling', 'with', 'the', 'Devil', 'Docs', 'throughout', 'the', 'war', 'and', 'now', 'they', 'are', 'close', 'to', 'Baghdad', '.'], ['Sanjay', 'Gupta', ',', 'has', 'been', 'traveling', 'with', 'the', 'Devil', 'Docs', 'throughout', 'the', 'war', 'and', 'now', 'they', 'are', 'close', 'to', 'Baghdad', '.']]
```



## DRel - Inside the code

#### readfile()
```python
def readfile(filename):
```

```
(Pdb) p lines[0]
['scott peterson', 'attorney', 'PER-SOC', "moving onto the laci peterson case bill , an unusual request from <e1> scott peterson </e1> 's <e2> attorney </e2> ."]

(Pdb) p lines[:5]
[['scott peterson', 'attorney', 'PER-SOC', "moving onto the laci peterson case bill , an unusual request from <e1> scott peterson </e1> 's <e2> attorney </e2> ."], ['laci peterson', 'bill', 'Other', "moving onto the <e1> laci peterson </e1> case <e2> bill </e2> , an unusual request from scott peterson 's attorney ."], ['laci peterson', 'scott peterson', 'Other', "moving onto the <e1> laci peterson </e1> case bill , an unusual request from <e2> scott peterson </e2> 's attorney ."], ['laci peterson', 'attorney', 'Other', "moving onto the <e1> laci peterson </e1> case bill , an unusual request from scott peterson 's <e2> attorney </e2> ."], ['bill', 'scott peterson', 'Other', "moving onto the laci peterson case <e1> bill </e1> , an unusual request from <e2> scott peterson </e2> 's attorney ."]]
```





## DSRL - Inside the code

### Dependency-base SRL




----
### Span-base SRL

#### 1.

```python
def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            valid_ids=None, attention_mask_label=None,
            verb_index=None, labels=None,
            input_ngram_ids=None, ngram_position_matrix=None,
            ):
```

```
(Pdb) p feat_dim
768

(Pdb) p max_len
73

(Pdb) p attention_mask_label.shape
torch.Size([16, 73])

# 每个句子所有词的embeddings
(Pdb) p valid_output.shape
torch.Size([16, 73, 768])

(Pdb) p valid_output
tensor([[[-0.0511,  0.5648,  0.0000,  ..., -0.5214, -0.5995,  0.0000],
         [-0.2334,  0.0000, -1.1179,  ...,  1.4579,  0.2439,  0.1733],
         [ 1.6641,  0.1123,  0.6462,  ..., -0.0230,  0.7214, -0.1229],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        [[-0.2007,  0.6344,  0.9781,  ...,  0.6690, -1.0709, -0.3669],
         [ 0.8554, -0.1193, -0.3903,  ...,  0.0900,  0.0000, -0.2878],
         [ 0.8927,  0.0000,  0.0000,  ..., -0.0371,  1.2229,  0.0095],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        [[-0.1637, -0.1055,  0.1855,  ..., -0.8615, -0.1831, -0.1489],
         [-0.5427,  0.0142, -1.1840,  ..., -0.2070, -0.2576,  0.0475],
         [ 0.0000, -0.4253, -1.2415,  ...,  1.5599,  0.2867,  0.1941],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        ...,

        [[-0.0609, -0.1342,  0.5048,  ..., -1.4818, -1.1081, -0.2368],
         [ 0.0000,  0.0000, -0.0953,  ...,  0.9282, -0.2497,  0.6585],
         [ 0.8978,  0.4548, -1.0045,  ...,  1.6655, -0.0381,  0.4105],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        [[-0.0357,  0.2414,  0.4207,  ..., -0.9046, -0.4051, -0.1459],
         [ 1.2936,  0.0000,  0.0000,  ...,  0.3723,  0.1364,  0.4431],
         [-0.8381, -0.7913, -0.2240,  ...,  0.6048,  0.8336, -0.9033],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        [[ 0.7586,  0.3674, -0.0589,  ..., -0.0612, -0.2658,  0.3096],
         [ 0.8057,  0.7780,  0.4619,  ..., -0.1471,  0.0000, -0.5858],
         [ 0.7972, -0.3270,  0.5871,  ..., -0.5218,  2.2686, -0.5419],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]],
       device='cuda:0', grad_fn=<FusedDropoutBackward>)


# 16个predicates，每个predicate一个embedding，#feat_dim = 768
(Pdb) p predicates.shape
torch.Size([16, 768])

(Pdb) p predicates
tensor([[ 1.4607e+00,  2.3861e-01,  2.9378e-01,  ...,  0.0000e+00,
         -3.0211e-01,  0.0000e+00],
        [ 1.1313e+00, -8.3085e-02, -3.9418e-01,  ...,  5.9986e-01,
         -1.0336e+00, -6.3338e-02],
        [ 2.6463e-01, -4.6240e-01, -3.6809e-01,  ...,  1.4504e-01,
          5.7497e-01, -2.3439e-01],
        ...,
        [ 4.8378e-01, -2.5291e-01,  2.6477e-01,  ...,  2.5564e-01,
          9.0657e-05, -3.4169e-01],
        [-8.0370e-01,  6.7440e-02,  7.1969e-01,  ...,  2.2262e-01,
          1.5133e-03,  2.6120e-01],
        [ 1.8626e-01, -1.1761e-01, -6.2783e-01,  ...,  7.2028e-01,
          8.3923e-01,  1.7880e-01]], device='cuda:0', grad_fn=<CopySlices>)

(Pdb) p pre_h.shape
torch.Size([16, 400])

(Pdb) p arg_h.shape
torch.Size([16, 73, 400])

(Pdb) p labels.shape
torch.Size([16, 73])

(Pdb) p s_labels.shape
torch.Size([16, 73, 14])

(Pdb) p labels[attention_mask_label].shape
torch.Size([647])

(Pdb) p s_labels[attention_mask_label].shape
torch.Size([647, 14])

(Pdb) p labels[attention_mask_label]
tensor([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  7, 10,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         6,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, ......, 1,  1,  1,  1,  1,  1,  4],
       device='cuda:0')

(Pdb) p s_labels[attention_mask_label]
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
       [0., 0., 0.,  ..., 0., 0., 0.],
       [0., 0., 0.,  ..., 0., 0., 0.],
       ...,
       [0., 0., 0.,  ..., 0., 0., 0.],
       [0., 0., 0.,  ..., 0., 0., 0.],
       [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0',
      grad_fn=<IndexBackward>)

(Pdb) p predictions[3]
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0], device='cuda:0')

(Pdb) p predictions.shape
torch.Size([16, 54])
```


----
#### 2.

```InputExample```
```
(Pdb) p examples[2].guid
'trial-2'
(Pdb) p examples[2].text_a
['中国', '陆上', '石油', '工业', '在', '过去', '一', '年', '中', '取得', '重大', '成绩', '：', '全', '年', '发现', '十', '个', '亿', '吨', '级', '储量', '规模', '的', '油气区', '。']
(Pdb) p examples[2].text_b
None
(Pdb) p examples[2].label
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ADV', 'O', 'V', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'A1', 'O']
(Pdb) p examples[2].verb_index
[16]
```

----
#### 3.

```InputFeatures```
```
(Pdb) p train_features[0].input_ids
[101, 1912, 5307, 6588, 6956, 712, 5052, 2190, 3949, 5307, 6588, 769, 3837, 4638, 2135, 1447, 792, 5305, 6432, 8024, 1079, 1765, 2190, 7676, 3949, 6822, 1139, 1366, 1872, 7270, 6862, 2428, 1469, 1139, 1366, 1872, 7270, 1, 6862, 2428, 2, 6823, 7770, 754, 1398, 3309, 1079, 1765, 6822, 1139, 1366, 1469, 1139, 1366, 1872, 7270, 6862, 2428, 8020, 4636, 1146, 722, 1282, 1724, 1469, 4636, 1146, 722, 753, 1282, 676, 4157, 753, 4638, 6862, 2428, 8021, 8024, 3221, 1079, 1765, 1139, 1366, 1872, 7270, 3297, 2571, 4638, 1765, 1277, 722, 671, 511, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

(Pdb) p train_features[0].verb_index
[20]

(Pdb) p train_features[0].input_mask
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# true labels (in idx).
(Pdb) p train_features[0].label_id
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

(Pdb) p label_map
{'<UNK>': 1, 'LOC': 2, 'TMP': 3, 'A1': 4, 'A2': 5, 'ADV': 6, 'A0': 7, 'MNR': 8, 'C-A1': 9, 'C-A0': 10, 'DIR': 11, '[CLS]': 12, '[SEP]': 13}

# 1 for the positions that do have label. 0 for padding.
(Pdb) p train_features[0].label_mask
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 1 for the positions that do have label and need to be evaluated, which excludes the predicate. 0 for padding.
(Pdb) p train_features[0].eval_mask
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

----
#### 4.

```one of the items in data```

(['中', '国', '进', '出', '口', '银', '行', '与', '中', '国', '银', '行', '加', '强', '合', '作'], ['B-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'I-A0', 'V', 'V', 'B-A1', 'I-A1'], [12, 13])

data consists of many items like the above.


----
```label_map```

```
{1: '<UNK>', 2: 'B-LOC', 3: 'I-LOC', 4: 'B-TMP', 5: 'I-TMP', 6: 'V', 7: 'B-A1', 8: 'I-A1', 9: 'B-A2', 10: 'I-A2', 11: 'O', 12: 'B-ADV', 13: 'I-ADV', 14: 'B-A0', 15: 'I-A0', 16: 'B-MNR', 17: 'I-MNR', 18: 'B-C-A1', 19: 'I-C-A1', 20: 'B-C-A0', 21: 'I-C-A0', 22: 'B-DIR', 23: 'I-DIR', 24: '[CLS]', 25: '[SEP]', 0: 'O'}
```

----
```all_pred``` #786

转化前是tensor

tensor([11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11], device='cuda:0')

* each number is a key of label_map. In this case, all are 'O'.
* all_pred is a bunch of tensors. each tensor corresponds a sentence

转化后是a list of lists
```
(Pdb) p len(all_pred)
116
```
116是总共predicate的数量，也是分句后句子的数量。
这是116个list, 每个list里面是这个句子从第一个词到最后一个词的labels。
例如all_pred[0]是以下的形式

```
['_', '_', '_', 'A1', '_', 'V', '_', 'A2', '_', '_', '_']

```

----
```emb_word2id```
