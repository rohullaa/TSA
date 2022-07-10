import torch
import torch.nn as nn
import torch.optim as optim

from utils.datasets import ELMO_DATA
from utils.helper import setup_logs
from torch.utils.data import DataLoader
from models.ELMO import ELMO
from utils.metrics import binary_analysis, proportional_analysis, get_analysis

import os, random, tqdm, logging, argparse
import numpy as np


def seed_everything(seed_value=5550):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(encoding):
    if encoding == "BIO":
        train_data = ELMO_DATA(data_file="data/train.conll", encoding=encoding)
        dev_data = ELMO_DATA(data_file="data/dev.conll", encoding=encoding)
        test_data = ELMO_DATA(data_file="data/test.conll", encoding=encoding)
    elif encoding == "BIOUL":
        train_data = ELMO_DATA(data_file="data/train_bioul.conll", encoding=encoding)
        dev_data = ELMO_DATA(data_file="data/dev_bioul.conll", encoding=encoding)
        test_data = ELMO_DATA(data_file="data/test_bioul.conll", encoding=encoding)
    else:
        print("Wrong encoding types.")

    return train_data, dev_data, test_data


def compute_score(sents, golds, preds):
    analysis = get_analysis(sents, preds, golds)
    binary_f1 = binary_analysis(analysis)

    flat_preds = [int(i) for l in preds for i in l]
    flat_golds = [int(i) for l in golds for i in l]
    propor_f1 = proportional_analysis(flat_golds, flat_preds, 'BIO')
    return binary_f1, propor_f1


def train():
    best_binary_f1 = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}")
        train_iter = tqdm.tqdm(train_loader)
        predicted_tags, gold_tags, raw_sentences = [], [], []
        model.train()

        for sentences, labels in train_iter:
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(sentences)

            loss = criterion(output.flatten(0, 1), labels.flatten())
            loss.backward()
            optimizer.step()

            train_iter.set_postfix_str(f"loss: {loss.item()}")

            new_labels = []
            new_preds = []
            for i in range(len(sentences)):
                new_labels.append(labels[i][:len(sentences[i])])
                new_preds.append(output.argmax(-1)[i][:len(sentences[i])])

            predicted_tags += new_preds
            gold_tags += new_labels
            raw_sentences += sentences

        compute_score(sents=raw_sentences, golds=gold_tags, preds=predicted_tags)
        lr_sheduler.step()

        binary_f1, propor_f1 = evaluate()
        if binary_f1 > best_binary_f1:
            best_binary_f1 = binary_f1
            best_epoch = epoch
            logger.info(f'Current best epoch is {best_epoch}, and the best binary f1 is {best_binary_f1}.')
            torch.save(model.state_dict(), "model_{}.pt".format(args.run_name))
        else:
            logger.info(f'Current best epoch is {best_epoch}, and the best binary f1 is {best_binary_f1}.')


def evaluate():
    predicted_tags, gold_tags, raw_sentences = [], [], []

    model.eval()
    dev_iter = tqdm.tqdm(dev_loader)
    for sentences, labels in dev_iter:
        labels = labels.to(device)
        output = model(sentences)

        new_labels = []
        new_preds = []
        for i in range(len(sentences)):
            new_labels.append(labels[i][:len(sentences[i])])
            new_preds.append(output.argmax(-1)[i][:len(sentences[i])])

        predicted_tags += new_preds
        gold_tags += new_labels
        raw_sentences += sentences

    binary_f1, propor_f1 = compute_score(sents=raw_sentences, golds=gold_tags, preds=predicted_tags)
    return binary_f1, propor_f1


def test():
    predicted_tags, gold_tags, raw_sentences = [], [], []

    model.eval()
    test_iter = tqdm.tqdm(test_loader)
    for sentences, labels in test_iter:
        labels = labels.to(device)
        output = model(sentences)

        new_labels = []
        new_preds = []
        for i in range(len(sentences)):
            new_labels.append(labels[i][:len(sentences[i])])
            new_preds.append(output.argmax(-1)[i][:len(sentences[i])])

        predicted_tags += new_preds
        gold_tags += new_labels
        raw_sentences += sentences

    binary_f1, propor_f1 = compute_score(sents=raw_sentences, golds=gold_tags, preds=predicted_tags)
    return binary_f1, propor_f1


if __name__ == "__main__":
    seed_everything(seed_value=5550)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_labels", default=5, type=int)
    parser.add_argument("--hidden_size", default=300, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--out_size", default=64, type=int)
    parser.add_argument("--batch_size", "-bs", default=50, type=int)
    parser.add_argument("--step_size", default=3, type=int)

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epochs", "-e", default=20, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--classifier", default="gru")  # gru, or lstm
    parser.add_argument("--input_dim", default=1024, type=int)
    parser.add_argument("--options_file", default="/content/drive/MyDrive/217/options.json")
    parser.add_argument("--weight_file", default="/content/drive/MyDrive/217/model.hdf5")
    parser.add_argument("--run_name", "-name", default="elmo")

    parser.add_argument("--tune", default=False, type=bool)
    parser.add_argument("--encoding", default="BIO")  # BIO or BIOUL

    args = parser.parse_args()
    args.run_name = f"{args.run_name}_{args.encoding}"
    if args.encoding == "BIOUL":
        args.num_labels = 9
    print(args)

    # getting the data based on the encoding
    train_data, dev_data, test_data = get_data(args.encoding)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
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

    run_name = f"ELMO_{args.classifier}"
    save_dir = './results'
    logger = setup_logs(save_dir, run_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.tune:
        for num_layers in [1, 2]:
            args.num_layers = num_layers
            for hidden_size in [200, 250, 300]:
                args.hidden_size = hidden_size
                logger.info(args)

                model = ELMO(
                    args=args,
                    device=device
                ).to(device)

                criterion = nn.CrossEntropyLoss(ignore_index=-1)
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

                train()
                test()
    else:
        # Setup logs
        logger.info(args)

        model = ELMO(
            args=args,
            device=device
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        train()

        model = ELMO(
            args=args,
            device=device
        ).to(device)
        model.load_state_dict(torch.load("model_{}.pt".format(args.run_name)))
        test()