import argparse
import logging
import os
import random
import tqdm

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from models.MultiBert import MultiBert as BERT
from utils.datasets import BERT_DATA, Alphabet
from utils.char_info import CharInfo
from utils.metrics import binary_analysis, proportional_analysis, get_analysis


def seed_everything(seed_value=5550):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logs(save_dir, run_name):
    # initialize logger
    logger = logging.getLogger("TSA_bert_char")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_mask(offset_mapping, lengths, n_subwords: int, n_words: int):
    offset_mapping = offset_mapping.tolist()
    mask = torch.zeros(len(lengths), n_words, n_subwords)

    for i_batch in range(len(lengths)):
        current_word, remaining_len = 0, lengths[i_batch][0]

        for i, (start, end) in enumerate(offset_mapping[i_batch]):
            if start == end:
                continue

            mask[i_batch, current_word, i] = 1
            remaining_len -= end - start

            if remaining_len <= 0 and current_word < len(lengths[i_batch]) - 1:
                current_word += 1
                remaining_len = lengths[i_batch][current_word]

    return mask


def test():
    # Testing the model with test data
    predicted_tags = []
    gold_tags = []
    raw_sentences = []

    test_iter = tqdm.tqdm(test_loader)
    for sentences, lengths, labels, char_list in test_iter:
        encoding = tokenizer(sentences, is_split_into_words=True, return_tensors="pt", padding=True,
                             return_offsets_mapping=True)
        batch_mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

        if args.USE_CHAR:
            # generate embedding of char information and concatenate with the pooled output
            # after transformer and feed to linear layer
            batch_size, seq_len, _ = char_list.shape
            char_extract = CharInfo(alphabet.size(), args.CHAR_EMBEDDING_DIM, args.CHAR_HIDDEN_DIM, 0.5)
            char_o = char_extract(char_list).contiguous().view(batch_size, seq_len, -1)

        # move to GPU
        del encoding["offset_mapping"]
        encoding = {key: value.to(device) for key, value in encoding.items()}
        batch_mask = batch_mask.to(device)
        labels = labels.to(device)

        if args.USE_CHAR:
            prediction = model(encoding, batch_mask, char_o.to(device))
        else:
            prediction = model(encoding, batch_mask)

        new_labels = []
        new_preds = []
        for i in range(len(sentences)):
            new_labels.append(labels[i][:len(sentences[i])])
            new_preds.append(prediction.argmax(-1)[i][:len(sentences[i])])

        predicted_tags += new_preds
        gold_tags += new_labels
        raw_sentences += sentences

    binary_f1, propor_f1 = compute_score(sents=raw_sentences, golds=gold_tags, preds=predicted_tags)
    return binary_f1, propor_f1


def evaluate():
    predicted_tags, gold_tags, raw_sentences = [], [], []

    model.eval()
    dev_iter = tqdm.tqdm(dev_loader)

    for sentences, lengths, labels, char_list in dev_iter:
        encoding = tokenizer(sentences, is_split_into_words=True, return_tensors="pt", padding=True,
                             return_offsets_mapping=True)
        batch_mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

        if args.USE_CHAR:
            # generate embedding of char information and concatenate with the pooled output
            # after transformer and feed to linear layer
            batch_size, seq_len, _ = char_list.shape
            char_extract = CharInfo(alphabet.size(), args.CHAR_EMBEDDING_DIM, args.CHAR_HIDDEN_DIM, 0.5)
            char_o = char_extract(char_list).contiguous().view(batch_size, seq_len, -1)

        # move to GPU
        del encoding["offset_mapping"]
        encoding = {key: value.to(device) for key, value in encoding.items()}
        batch_mask = batch_mask.to(device)
        labels = labels.to(device)

        if args.USE_CHAR:
            prediction = model(encoding, batch_mask, char_o.to(device))
        else:
            prediction = model(encoding, batch_mask)

        new_labels = []
        new_preds = []
        for i in range(len(sentences)):
            new_labels.append(labels[i][:len(sentences[i])])
            new_preds.append(prediction.argmax(-1)[i][:len(sentences[i])])

        predicted_tags += new_preds
        gold_tags += new_labels
        raw_sentences += sentences

    binary_f1, propor_f1 = compute_score(sents=raw_sentences, golds=gold_tags, preds=predicted_tags)
    return binary_f1, propor_f1


def train():
    best_binary_f1 = 0
    best_epoch = 0
    for epoch in range(args.EPOCHS):
        model.train()
        logger.info(f"Epoch {epoch}")
        train_iter = tqdm.tqdm(train_loader)
        predicted_tags, gold_tags, raw_sentences = [], [], []

        for sentences, lengths, labels, char_list in train_iter:
            encoding = tokenizer(sentences, is_split_into_words=True, return_tensors="pt", padding=True,
                                 return_offsets_mapping=True)
            batch_mask = get_mask(encoding["offset_mapping"], lengths, encoding["input_ids"].size(1), labels.size(1))

            if args.USE_CHAR:
                # generate embedding of char information and concatenate with the pooled output
                # after transformer and feed to linear layer
                batch_size, seq_len, _ = char_list.shape
                char_extract = CharInfo(alphabet.size(), args.CHAR_EMBEDDING_DIM, args.CHAR_HIDDEN_DIM, 0.5)
                char_o = char_extract(char_list).contiguous().view(batch_size, seq_len, -1)

            # move to GPU
            del encoding["offset_mapping"]
            encoding = {key: value.to(device) for key, value in encoding.items()}
            batch_mask = batch_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if args.USE_CHAR:
                prediction = model(encoding, batch_mask, char_o.to(device))
            else:
                prediction = model(encoding, batch_mask)
            loss = criterion(prediction.flatten(0, 1), labels.flatten())

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_iter.set_postfix_str(f"loss: {loss.item()}")

            new_labels = []
            new_preds = []
            for i in range(len(sentences)):
                new_labels.append(labels[i][:len(sentences[i])])
                new_preds.append(prediction.argmax(-1)[i][:len(sentences[i])])

            predicted_tags += new_preds
            gold_tags += new_labels
            raw_sentences += sentences

        compute_score(sents=raw_sentences, golds=gold_tags, preds=predicted_tags)

        binary_f1, propor_f1 = evaluate()
        if binary_f1 > best_binary_f1:
            best_binary_f1 = binary_f1
            best_epoch = epoch
            logger.info(f'Current best epoch is {best_epoch}, and the best binary f1 is {best_binary_f1}.')
            torch.save(model.state_dict(), "model_{}.pt".format(args.RUN_NAME))
        else:
            logger.info(f'Current best epoch is {best_epoch}, and the best binary f1 is {best_binary_f1}.')


def compute_score(sents, golds, preds):
    analysis = get_analysis(sents, preds, golds)
    binary_f1 = binary_analysis(analysis)

    flat_preds = [int(i) for l in preds for i in l]
    flat_golds = [int(i) for l in golds for i in l]
    propor_f1 = proportional_analysis(flat_golds, flat_preds)
    return binary_f1, propor_f1


def test_stuff():
    train_iter = tqdm.tqdm(train_loader)
    gold_tags, raw_sentences = [], []

    for sentences, lengths, labels in train_iter:
        new_labels = []
        new_preds = []
        for i in range(len(sentences)):
            new_labels.append(labels[i][:len(sentences[i])])
        gold_tags += new_labels
        raw_sentences += sentences

    compute_score(sents=raw_sentences, golds=gold_tags, preds=gold_tags)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--LEARNING_RATE", "-lr", default=0.01, type=int)
    parser.add_argument("--EPOCHS", "-e", default=50, type=int)
    parser.add_argument("--FREEZE", "-freeze", default=False, type=bool)
    parser.add_argument("--TEST", "-t", default=False, type=bool)
    parser.add_argument("--DROPOUT", "-p", default=0.2, type=float)
    parser.add_argument("--CHAR_EMBEDDING_DIM", "-ced", default=30, type=int)
    parser.add_argument("--CHAR_HIDDEN_DIM", "-chd", default=50, type=int)
    parser.add_argument("--USE_CHAR", "-uc", action='store_true')
    parser.add_argument("--RUN_NAME", "-rn", default='MBert', type=str)

    args = parser.parse_args()
    run_name = args.RUN_NAME
    save_dir = './results'
    # Setup logs
    logger = setup_logs(save_dir, run_name)
    logger.info(args)

    alphabet = Alphabet("data/train.conll", "data/dev.conll", "data/test.conll")
    train_data = BERT_DATA(data_file="data/train.conll", alphabet=alphabet)
    dev_data = BERT_DATA(data_file="data/dev.conll", alphabet=alphabet, vocab=train_data.vocab)
    test_data = BERT_DATA(data_file="data/test.conll", alphabet=alphabet, vocab=train_data.vocab)

    train_loader = DataLoader(train_data,
                              batch_size=args.BATCH_SIZE,
                              collate_fn=train_data.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev_data,
                            batch_size=1,
                            collate_fn=train_data.collate_fn,
                            shuffle=False)

    test_loader = DataLoader(test_data,
                             batch_size=1,
                             collate_fn=train_data.collate_fn,
                             shuffle=False)

    logger.info("Data loaded")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERT(args.FREEZE, args.DROPOUT).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if args.FREEZE else 2e-5)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=200,
        num_training_steps=args.EPOCHS * len(train_loader)
    )

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    train()

    model = BERT(args.FREEZE, args.DROPOUT).to(device)
    model.load_state_dict(torch.load("model_{}.pt".format(args.RUN_NAME)))
    model.eval()

    test()