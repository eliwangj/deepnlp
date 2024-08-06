from torch import nn
import torch

class KVMN(nn.Module):
    def __init__(self, hidden_size, key_size, value_size):
        super(KVMN, self).__init__()
        self.temper = hidden_size ** 0.5
        self.word_embedding_key = nn.Embedding(key_size, hidden_size)
        self.word_embedding_value = nn.Embedding(value_size, hidden_size)

    def forward(self, word_seq, hidden_state, label_value_matrix, word_mask_metrix):
        # word_seq: (batch_size,  word_seq_len)
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        embedding_a = self.word_embedding_key(word_seq) # (batch_size,  word_seq_len,hidden_size)
        embedding_c = self.word_embedding_value(label_value_matrix)#(batch_size, character_seq_len, word_seq_len,hidden_size)

        embedding_a = embedding_a.permute(0, 2, 1) #进行纬度变换。(batch_size, hidden_size ，word_seq_len)
        u = torch.matmul(hidden_state, embedding_a) / self.temper  #  (batch_size, character_seq_len, word_seq_len)

        tmp_word_mask_metrix = torch.clamp(word_mask_metrix, 0, 1) #压缩到1，0之内#  (batch_size, character_seq_len, word_seq_len)

        exp_u = torch.exp(u) #做指数变换
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)#  (batch_size, character_seq_len, word_seq_len)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        #stack 吧list变成tensor，变成50个一样的矩阵，去对前面的矩阵做归一化。softmax

        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)  #softmax？(batch_size, character_seq_len, word_seq_len)

        embedding_c = embedding_c.permute(3, 0, 1, 2)  #(hidden_size,batch_size,character_seq_len, word_seq_len)
        o = torch.mul(p, embedding_c)#(hidden_size,batch_size,character_seq_len, word_seq_len)

        o = o.permute(1, 2, 3, 0)#还原 (batch_size,character_seq_len, word_seq_len,hidden_size)
        o = torch.sum(o, 2)#最后word_seq_len上求和

        return o