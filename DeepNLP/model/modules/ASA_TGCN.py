import torch
import torch.nn as nn
import numpy as np


class TypeGraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(TypeGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, text, adj, dep_embed):
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)
        val_sum = val_us + dep_embed
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)
        # hidden = torch.matmul(val_sum, self.weight)
        hidden = self.linear(val_sum)
        output = hidden.transpose(1,2) * adj_us

        output = torch.sum(output, dim=2)
        return output

    @staticmethod
    def get_attention(val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, dep_embed), -1).float()
        atten_expand = (val_cat * val_cat.transpose(1,2))

        attention_score = torch.sum(atten_expand, dim=-1)
        attention_score = attention_score / np.power(feat_dim, 0.5)
        # LayerNorm
        norm = nn.LayerNorm(attention_score.size()[1:], elementwise_affine=True).to(attention_score.device)
        attention_score = norm(attention_score)

        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score, adj.float()) # mask
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)

        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        if 'HalfTensor' in val_out.type():
            attention_score = attention_score.half()

        return attention_score

    @staticmethod
    def get_avarage(aspect_indices, x):
        aspect_indices_us = torch.unsqueeze(aspect_indices, 2)
        x_mask = x * aspect_indices_us
        aspect_len = (aspect_indices_us != 0).sum(dim=1)
        x_sum = x_mask.sum(dim=1)
        x_av = torch.div(x_sum, aspect_len)

        return x_av
