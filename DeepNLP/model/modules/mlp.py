# -*- coding: utf-8 -*-

from .dropout import SharedDropout

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, bias=True):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden, bias=bias)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
