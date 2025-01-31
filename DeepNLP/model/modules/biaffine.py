# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        # x: dep, y: head
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        if len(y.shape) == 2:
            s = torch.einsum('bxi,oij,bj->box', x, self.weight, y)
        elif len(y.shape) == 3:
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        else:
            raise ValueError()
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class KnowledgeBiaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(KnowledgeBiaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, m):
        # x: dep, y: head
        batch_size, sent_len = x.shape[0], x.shape[1]
        y = torch.stack([y] * sent_len, 1)
        y = torch.add(y, m)

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        x_tmp = torch.einsum('bxi,oij->boxj', x, self.weight)
        x_tmp = torch.stack([x_tmp] * 1, 3)

        y_tmp = y.permute(0, 1, 3, 2)
        y_tmp = torch.stack([y_tmp] * 1, 1)

        s = torch.matmul(x_tmp, y_tmp)

        s = s.flatten(start_dim=3)

        # [batch_size, n_out, seq_len, seq_len]
        # s = torch.einsum('bxi,oij,bxyj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)


        # if self.bias_x:
        #     x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        # if self.bias_y:
        #     y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # # [batch_size, n_out, seq_len, seq_len]
        # s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # # remove dim 1 if n_out == 1
        # s = s.squeeze(1)

        return s

