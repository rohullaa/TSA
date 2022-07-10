import torch
import torch.nn as nn
from transformers import AutoModel


class BERT(nn.Module):
    def __init__(self,
                 model_path,
                 freeze,
                 encoding='BIO',
                 char_hidden_dim=50,
                 use_char=None
                 ):
        super().__init__()

        self.output_dim = 5 if encoding == 'BIO' else 9
        self.use_char = use_char

        # set up the Bert model
        self.bert = AutoModel.from_pretrained(model_path)
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
        output = self.linear(pooled)
        return output