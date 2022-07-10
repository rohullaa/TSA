import torch
import torch.nn as nn
from simple_elmo import ElmoModel

class ELMO(nn.Module):
    def __init__(self,
                 model_path,
                 dropout_,
                 output_dim=5,
                 ):
        super().__init__()

        self.output_dim = output_dim

        # set up the Bert model
        self.elmo = ElmoModel()
        self.elmo.load(model_path)

        self.dropout = nn.Dropout(dropout_)
        self.linear = nn.Linear(1024, self.output_dim)

    def get_elmo_vectors(self, x):
      word_embs = self.elmo.get_elmo_vectors(x, layers="top")  # shape: [B, N_subwords, D]
      word_embs = torch.from_numpy(word_embs).float()

      return word_embs

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output.to('cuda'))
        return output
