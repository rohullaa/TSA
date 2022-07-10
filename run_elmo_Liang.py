import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.elmo_Liang import ELMO
from utils.datasets import BERT_DATA, Alphabet
from utils.metrics import binary_analysis, proportional_analysis, get_analysis

import os, random, tqdm, logging, argparse
import numpy as np
import pickle


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


# it takes time to get elmo vectors using simple_elmo, especially for validation and test
# here I get pickled version of vectors and validate and test using these vectors
def get_elmo(model, train_loader, dev_loader, test_loader):
    # train_elmo_vectors = []
    dev_elmo_vectors = []
    test_elmo_vectors = []
    # for sentences, lengths, labels, char_list in train_loader:
    #    train_elmo_vectors.append(model.get_elmo_vectors(sentences))

    # with open('train_elmo_vectors.pkl', 'wb') as f:
    #    pickle.dump(train_elmo_vectors, f)

    for sentences, lengths, labels, char_list in dev_loader:
        dev_elmo_vectors.append(model.get_elmo_vectors(sentences))

    with open('dev_elmo_vectors.pkl', 'wb') as f:
        pickle.dump(dev_elmo_vectors, f)

    for sentences, lengths, labels, char_list in test_loader:
        test_elmo_vectors.append(model.get_elmo_vectors(sentences))

    with open('test_elmo_vectors.pkl', 'wb') as f:
        pickle.dump(test_elmo_vectors, f)

    return train_elmo_vectors, dev_elmo_vectors, test_elmo_vectors


def test():
    # Testing the model with test data
    predicted_tags = []
    gold_tags = []
    raw_sentences = []

    test_iter = tqdm.tqdm(test_loader)
    idx = 0
    for sentences, lengths, labels, char_list in test_iter:
        labels = labels.to(device)

        prediction = model(test_elmo_vectors[idx])
        idx += 1

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

    idx = 0

    for sentences, lengths, labels, char_list in dev_iter:

        labels = labels.to(device)

        prediction = model(dev_elmo_vectors[idx])
        idx += 1

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

        idx = 0
        for sentences, lengths, labels, char_list in train_iter:

            labels = labels.to(device)

            optimizer.zero_grad()
            prediction = model(train_elmo_vectors[idx])
            idx += 1
            loss = criterion(prediction.flatten(0, 1), labels.flatten())

            loss.backward()
            optimizer.step()
            # scheduler.step()

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
    propor_f1 = proportional_analysis(flat_golds, flat_preds, encoding='BIO')
    return binary_f1, propor_f1


if __name__ == '__main__':
    seed_everything(seed_value=5550)

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--DROPOUT", "-p", default=0.2, type=float)
    parser.add_argument("--EPOCHS", "-e", default=50, type=int)
    parser.add_argument("--MODEL_PATH", "-path", default="/cluster/shared/nlpl/data/vectors/latest/216/")
    parser.add_argument("--USE_PICKLE", "-up", action='store_true')
    parser.add_argument("--RUN_NAME", "-rn", default='elmo', type=str)

    args = parser.parse_args()
    run_name = args.RUN_NAME
    save_dir = './results'
    # Setup logs
    logger = setup_logs(save_dir, run_name)
    logger.info(args)

    alphabet = Alphabet("data/train.conll", "data/dev.conll", "data/test.conll")
    # I built elmo model in similar way to bert mode, so just use the data class for bert
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
    model = ELMO(
        model_path=args.MODEL_PATH,
        dropout_=args.DROPOUT
    ).to(device)

    if args.USE_PICKLE:
        # with open('/content/TSA/train_elmo_vectors.pkl', 'rb') as f:
        #    train_elmo_vectors = pickle.load(f)

        with open('./dev_elmo_vectors.pkl', 'rb') as f:
            dev_elmo_vectors = pickle.load(f)

        with open('./test_elmo_vectors.pkl', 'rb') as f:
            test_elmo_vectors = pickle.load(f)
    else:
        train_elmo_vectors, dev_elmo_vectors, test_elmo_vectors = get_elmo(
            model, train_loader, dev_loader, test_loader
        )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train()

    # torch.save(model.state_dict(), "model.pt")
    # model = BERT(*args).to(device)
    # model.load_state_dict(torch.load("model.pt"))
    # model.eval()

    test()

