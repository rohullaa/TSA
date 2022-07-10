import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids


class Classifier(nn.Module):
    def __init__(self, args, classifier, input_size, hidden_size, out_size):
        self.classifier = classifier

        super(Classifier, self).__init__()
        if self.classifier == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size // 2, batch_first=True, bidirectional=True,
                              num_layers=args.num_layers)
            f_dim = hidden_size
        elif self.classifier == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size // 2, batch_first=True, bidirectional=True,
                               num_layers=args.num_layers)
            f_dim = hidden_size

        else:
            f_dim = input_size

        self.fc = nn.Linear(f_dim, out_size)
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        nn.init.uniform_(self.fc.bias, -0.1, 0.1)

    def forward(self, inputs):
        if self.classifier == 'gru' or self.classifier == 'lstm':
            out, _ = self.rnn(inputs)
        else:
            out = inputs
        return self.fc(out)


class ELMO(nn.Module):
    def __init__(self, args, device):
        super(ELMO, self).__init__()
        self.args = args
        self.device = device

        self.init_elmo()

        self.classifier = Classifier(args, args.classifier, self.word_dim, args.hidden_size, args.out_size)
        self.cls = nn.Linear(args.out_size, args.num_labels)
        nn.init.uniform_(self.cls.weight, -0.1, 0.1)
        nn.init.uniform_(self.cls.bias, -0.1, 0.1)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, x):
        word_embs = self.get_elmo(x)

        x = self.classifier(word_embs)
        x = self.dropout(x)
        x = self.cls(x)
        return x

    def init_elmo(self):
        self.elmo = Elmo(self.args.options_file, self.args.weight_file, 1)
        for param in self.elmo.parameters():
            param.requires_grad = False
        self.word_dim = self.args.input_dim

    def get_elmo(self, sentence_lists):
        character_ids = batch_to_ids(sentence_lists)
        character_ids = character_ids.to(self.device)
        embeddings = self.elmo(character_ids)
        return embeddings['elmo_representations'][0]