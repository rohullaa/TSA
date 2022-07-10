from collections import defaultdict
import os

import torch
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Vocab(defaultdict):
    """
    This function creates the vocabulary dinamically. As you call ws2ids, it updates the vocabulary with any new tokens.
    """

    def __init__(self, train=True):
        super().__init__(lambda: len(self))
        self.train = train
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        # set UNK token to 1 index
        self[self.PAD]
        self[self.UNK]
        #

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 1 for w in ws]

    #
    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if int(i) in idx2w else "<UNK>" for i in ids]


class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)

        raws = [raw for raw, word, target, char, idx in batch]
        words = pack_sequence([word for raw, word, target, char, idx in batch])
        targets = pack_sequence([target for raw, word, target, char, idx in batch])
        chars = self.pad_char([char for raw, word, target, char, idx in batch])
        seq_len_for_char = [len(word) for raw, word, target, char, idx in batch]
        idxs = [idx for raw, word, target, char, idx in batch]
        return raws, words, targets, chars, seq_len_for_char, idxs

    def pad_char(self, chars):
        batch_size = len(chars)
        max_seq_len = max(map(len, chars))
        pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        return char_seq_tensor


class ConllDataset(object):
    def __init__(self, vocab, encoding, alphabet=None):

        self.vocab = vocab
        self.encoding = encoding

        if self.encoding == 'BIOUL':
            self.label2idx = {"O": 0, "B-TARG-POSITIVE": 1, "I-TARG-POSITIVE": 2,
                              "U-TARG-POSITIVE": 3, "L-TARG-POSITIVE": 4, "B-TARG-NEGATIVE": 5,
                              "I-TARG-NEGATIVE": 6, "U-TARG-NEGATIVE": 7, "L-TARG-NEGATIVE": 8}
        elif self.encoding == 'BIO':
            self.label2idx = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                              "B-targ-Negative": 3, "I-targ-Negative": 4}
        else:
            raise 'Please input correct encoding!'

        self.alphabet = alphabet

    def load_conll(self, data_file):
        sents, all_labels = [], []
        sent, labels = [], []
        for line in open(data_file, encoding="utf8"):
            if line.strip() == "":
                sents.append(sent)
                all_labels.append(labels)
                sent, labels = [], []
            else:
                if self.encoding == 'BIOUL':
                    token, label = line.strip().split(" ")
                elif self.encoding == 'BIO':
                    token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)

        seq_char_list = list()
        for sent in sents:
            char_sent = list()
            for word in sent:
                char_list = list(word)
                char_id = list()
                for char in char_list:
                    char_id.append(self.alphabet.char_to_id(char))
                char_sent.append(char_id)
            seq_char_list.append(char_sent)

        return sents, all_labels, seq_char_list

    def get_split(self, data_file):

        sents, labels, seq_char_list = self.load_conll(data_file)
        data_split = [(text,
                       torch.LongTensor(self.vocab.ws2ids(text)),
                       torch.LongTensor([self.label2idx[l] for l in label]),
                       char_info,
                       idx) for idx, (text, label, char_info) in enumerate(zip(sents, labels, seq_char_list))]

        return Split(data_split)


class BERT_DATA(torch.utils.data.Dataset):
    def __init__(self, data_file, alphabet, encoding, vocab=None):
        self.sents, self.labels = self.load_conll(data_file, encoding)
        self.alphabet = alphabet
        self.encoding = encoding

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = list(set([item for sublist in self.labels for item in sublist]))

        if self.encoding == 'BIOUL':
            self.label2idx = {"O": 0, "B-TARG-POSITIVE": 1, "I-TARG-POSITIVE": 2,
                              "U-TARG-POSITIVE": 3, "L-TARG-POSITIVE": 4, "B-TARG-NEGATIVE": 5,
                              "I-TARG-NEGATIVE": 6, "U-TARG-NEGATIVE": 7, "L-TARG-NEGATIVE": 8}
        elif self.encoding == 'BIO':
            self.label2idx = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                              "B-targ-Negative": 3, "I-targ-Negative": 4}
        else:
            raise 'Please input correct encoding!'

    def load_conll(self, data_file, encoding):
        sents, all_labels = [], []
        sent, labels = [], []
        for line in open(data_file, encoding='utf8'):
            if line.strip() == "":
                sents.append(sent)
                all_labels.append(labels)
                sent, labels = [], []
            else:
                if encoding == 'BIOUL':
                    token, label = line.strip().split(" ")
                elif encoding == 'BIO':
                    token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)

        # if data_file=="data/train.conll":
        #     return sents[:1000], all_labels[:1000]
        # else:
        #     return sents[:100], all_labels[:100]

        return sents, all_labels

    def __getitem__(self, index):
        forms = self.sents[index]
        lengths = [len(form) for form in forms]
        labels = torch.LongTensor([self.label2idx.get(label, -1) for label in self.labels[index]])

        seq_char_list = list()
        for word in forms:
            char_list = list(word)
            char_id = list()
            for char in char_list:
                char_id.append(self.alphabet.char_to_id(char))
            seq_char_list.append(char_id)

        return forms, lengths, labels, seq_char_list

    def __len__(self):
        return len(self.sents)

    def collate_fn(self, batch):
        sentences, lengths, labels, char_lists = zip(*batch)
        longest = max(label.size(0) for label in labels)
        labels = torch.stack([F.pad(label, (0, longest - label.size(0)), value=-1) for label in labels])
        return list(sentences), list(lengths), labels, self.pad_char(char_lists)

    def pad_char(self, chars):
        batch_size = len(chars)
        max_seq_len = max(map(len, chars))
        pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        return char_seq_tensor


class ELMO_DATA(torch.utils.data.Dataset):
    def __init__(self, data_file, encoding):
        self.encoding = encoding
        self.sents, self.labels = self.load_conll(data_file, encoding)
        self.vocab = list(set([item for sublist in self.labels for item in sublist]))
        self.label2idx = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                          "B-targ-Negative": 3, "I-targ-Negative": 4}

        if encoding == "BIO":
            self.label2idx = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                              "B-targ-Negative": 3, "I-targ-Negative": 4}
        elif encoding == "BIOUL":
            self.label2idx = {"O": 0, "B-TARG-POSITIVE": 1, "I-TARG-POSITIVE": 2,
                              "U-TARG-POSITIVE": 3, "L-TARG-POSITIVE": 4, "B-TARG-NEGATIVE": 5,
                              "I-TARG-NEGATIVE": 6, "U-TARG-NEGATIVE": 7, "L-TARG-NEGATIVE": 8}

        self.labels = self.convert_labels_to_int(self.labels)

    def load_conll(self, data_file, encoding):
        sents, all_labels = [], []
        sent, labels = [], []
        for line in open(data_file, encoding='utf8'):
            if line.strip() == "":
                sents.append(sent)
                all_labels.append(labels)
                sent, labels = [], []
            else:
                if encoding == 'BIOUL':
                    token, label = line.strip().split(" ")
                elif encoding == 'BIO':
                    token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)

        return sents, all_labels

    def convert_labels_to_int(self, all_labels):
        for i, labels in enumerate(all_labels):
            for j, label in enumerate(labels):
                all_labels[i][j] = self.label2idx.get(label)

        return all_labels

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]

    def __len__(self):
        return len(self.sents)

    def collate_fn(self, batch):
        data, labels = zip(*batch)
        labels = [torch.tensor(l) for l in labels]

        longest = max(label.size(0) for label in labels)
        labels = torch.stack([F.pad(label, (0, longest - label.size(0)), value=-1) for label in labels])
        return data, labels

class Alphabet(object):
    def __init__(self, train_path, dev_path, test_path, encoding):
        self.encoding = encoding
        self._id_to_char = []
        self._char_to_id = {}
        self._pad = -1
        self._unk = -1
        self.index = 0

        self._id_to_char.append('<PAD>')
        self._char_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1

        self._id_to_char.append('<UNK>')
        self._char_to_id['<UNK>'] = self.index
        self._unk = self.index
        self.index += 1
        for line in open(train_path, 'r', encoding='utf8'):
            if not line.strip() == "":
                if self.encoding == 'BIOUL':
                    token, label = line.strip().split(" ")
                elif self.encoding == 'BIO':
                    token, label = line.strip().split("\t")
                chars = list(token)
                for char in chars:
                    if char not in self._char_to_id:
                        self._id_to_char.append(char)
                        self._char_to_id[char] = self.index
                        self.index += 1

        for line in open(dev_path, 'r', encoding='latin-1'):
            if not line.strip() == "":
                if self.encoding == 'BIOUL':
                    token, label = line.strip().split(" ")
                elif self.encoding == 'BIO':
                    token, label = line.strip().split("\t")
                chars = list(token)
                for char in chars:
                    if char not in self._char_to_id:
                        self._id_to_char.append(char)
                        self._char_to_id[char] = self.index
                        self.index += 1

        for line in open(test_path, 'r', encoding='latin-1'):
            if not line.strip() == "":
                if self.encoding == 'BIOUL':
                    token, label = line.strip().split(" ")
                elif self.encoding == 'BIO':
                    token, label = line.strip().split("\t")
                chars = list(token)
                for char in chars:
                    if char not in self._char_to_id:
                        self._id_to_char.append(char)
                        self._char_to_id[char] = self.index
                        self.index += 1

    def pad(self):
        return self._pad

    def unk(self):
        return self._unk

    def size(self):
        return len(self._id_to_char)

    def char_to_id(self, char):
        if char in self._char_to_id:
            return self._char_to_id[char]
        return self.unk()

    def id_to_char(self, cur_id):
        return self._id_to_char[cur_id]

    def items(self):
        return self._char_to_id.items()

