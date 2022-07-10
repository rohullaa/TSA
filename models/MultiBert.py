# This file contains implementation of multilingual bert
# For details see here https://github.com/google-research/bert/blob/master/multilingual.md

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiBert(nn.Module):
    def __init__(self, freeze, dropout_, char_hidden_dim=50, use_char=None, output_dim=5):
        super().__init__()

        self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(dropout_)
        self.use_char = use_char
        self.output_dim = output_dim
        if self.use_char:
            self.linear = nn.Linear(self.bert.config.hidden_size + char_hidden_dim, self.output_dim)
        else:
            self.linear = nn.Linear(self.bert.config.hidden_size, self.output_dim)

        if freeze:
            self.bert.requires_grad_(False)

    def forward(self, encoding, mask, char_o=None):
        output = self.bert(**encoding).last_hidden_state  # shape: [B, N_subwords, D]
        pooled = torch.einsum("bsd,bws->bwd", output, mask)
        pooled = pooled / mask.sum(-1, keepdim=True).clamp(min=1.0)
        if self.use_char:
            pooled = torch.cat((pooled, char_o), 2)

        output = self.dropout(pooled)
        output = self.linear(output)
        return output