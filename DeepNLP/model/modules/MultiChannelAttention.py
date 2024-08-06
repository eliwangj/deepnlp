from torch import nn
import torch

class MultiChannelAttention(nn.Module):
    def __init__(self, ngram_size, hidden_size, cat_num):
        super(MultiChannelAttention, self).__init__()
        self.word_embedding = nn.Embedding(ngram_size, hidden_size, padding_idx=0)
        self.channel_weight = nn.Embedding(cat_num, 1)
        self.temper = hidden_size ** 0.5

    def forward(self, word_seq, hidden_state, char_word_mask_matrix, channel_ids):
        # word_seq: (batch_size, channel, word_seq_len)
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        # mask_matrix: (batch_size, channel, character_seq_len, word_seq_len)

        # embedding (batch_size, channel, word_seq_len, word_embedding_dim)
        batch_size, character_seq_len, hidden_size = hidden_state.shape
        channel = char_word_mask_matrix.shape[1] # attention (batch_size, channel, character_seq_len, word_seq_len)
        word_seq_length = word_seq.shape[2]

        embedding = self.word_embedding(word_seq)  #(batch_size, channel, word_seq_len，hidden_size)

        tmp = embedding.permute(0, 1, 3, 2)   #(batch_size, channel，hidden_size，word_seq_len)

        tmp_hidden_state = torch.stack([hidden_state] * channel, 1) #(batch_size, channel， character_seq_len, hidden_size)

        # u (batch_size, channel, character_seq_len, word_seq_len)
        u = torch.matmul(tmp_hidden_state, tmp) / self.temper

        # attention (batch_size, channel, character_seq_len, word_seq_len)
        tmp_word_mask_metrix = torch.clamp(char_word_mask_matrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)#将mask的结果改为0
        # 在第三个纬度上求和，并重新拼接为(batch_size, channel, character_seq_len, word_seq_len)的矩阵
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 3)] * delta_exp_u.shape[3], 3)

        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        attention = attention.view(batch_size * channel, character_seq_len, word_seq_length)
        embedding = embedding.view(batch_size * channel, word_seq_length, hidden_size)
        #三维的矩阵相乘
        character_attention = torch.bmm(attention, embedding)#(batch_size*channel, character_seq_len, hidden_size)

        character_attention = character_attention.view(batch_size, channel, character_seq_len, hidden_size)

        channel_w = self.channel_weight(channel_ids)
        channel_w = nn.Softmax(dim=1)(channel_w)#分配权重

        channel_w = channel_w.view(batch_size, -1, 1, 1)

        character_attention = torch.mul(character_attention, channel_w)#(batch_size, channel, character_seq_len, hidden_size)

        character_attention = character_attention.permute(0, 2, 1, 3)#(batch_size, character_seq_len, channel, hidden_size)
        character_attention = character_attention.flatten(start_dim=2)#(batch_size, character_seq_len, channel * hidden_size)

        return character_attention #(batch_size, character_seq_len, channel *  hidden_size)