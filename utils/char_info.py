import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import torch.nn.functional as F
import numpy as np


class CharInfo(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharInfo, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(CharInfo.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

    @staticmethod
    def random_embedding(vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        pretrain_emb[0, :] = np.zeros((1, embedding_dim))
        return pretrain_emb

    def forward(self, input):
        batch_size, seq_len, word_len = input.size()
        input = torch.reshape(input, (batch_size*seq_len, word_len))
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).contiguous().view(batch_size, -1)
        return char_cnn_out