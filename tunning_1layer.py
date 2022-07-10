from models.Model_LSTM_GRU import MODEL
from utils.datasets import Vocab, ConllDataset, Alphabet
from utils.wordembs import WordVecs

import numpy as np
import os
import logging
import random
import zipfile
import argparse

import torch
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--WORD_DROPOUT", "-wdr", default=0.3, type=int)
    parser.add_argument("--RNN_DROPOUT", "-rdr", default=0.2, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=100, type=int)
    # parser.add_argument("--EMBEDDINGS", "-emb", default="58.zip")
    parser.add_argument("--EMBEDDINGS", "-emb", default="/cluster/shared/nlpl/data/vectors/latest/58.zip")
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_true")
    parser.add_argument("--LEARNING_RATE", "-lr", default=0.01, type=int)
    parser.add_argument("--EPOCHS", "-e", default=50, type=int)
    parser.add_argument("--BIDIRECTIONAL", "-bi", default=False)
    parser.add_argument("--CHAR_EMBEDDING_DIM", "-ced", default=30, type=int)
    parser.add_argument("--CHAR_HIDDEN_DIM", "-chd", default=50, type=int)
    parser.add_argument("--USE_CHAR", "-uc", action="store_true")
    parser.add_argument("--RUN_NAME", "-rn", default='baseline_tunning', type=str)
    parser.add_argument('--MODEL_TYPE', "-mt", default='lstm', type=str)
    parser.add_argument("--ENCODING", "-en", default='BIO', type=str)

    args = parser.parse_args()
    run_name = args.RUN_NAME
    save_dir = './results'
    # Setup logs
    logger = setup_logs(save_dir, run_name)

    seed_everything()

    # load text-version model from /cluster/shared/nlpl/data/vectors/latest/
    # convert encoding, and create cache
    if args.EMBEDDINGS.endswith(".zip"):
        with zipfile.ZipFile(args.EMBEDDINGS, "r") as archive:
            archive.extract("model.txt", path="wordemb")
        embeddings = WordVecs("wordemb/model.txt", encoding="latin-1")
    else:
        # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
        embeddings = WordVecs(args.EMBEDDINGS, encoding="latin-1")
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by two to avoid overwriting the PAD and UNK
    # tokens at index 0 and 1
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 2
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    if args.ENCODING == 'BIOUL':
        train_path = "data/train_bioul.conll"
        dev_path = "data/dev_bioul.conll"
        test_path = "data/test_bioul.conll"
    elif args.ENCODING == 'BIO':
        train_path = "data/train.conll"
        dev_path = "data/dev.conll"
        test_path = "data/test.conll"
    else:
        raise 'Please input correct encoding!'

    alphabet = Alphabet(train_path, dev_path, test_path)
    dataset = ConllDataset(vocab, args.ENCODING, alphabet)
    train_iter = dataset.get_split(train_path)
    dev_iter = dataset.get_split(dev_path)
    test_iter = dataset.get_split(test_path)

    # Create a new embedding matrix which includes the pretrained embeddings
    # as well as new embeddings for PAD UNK and tokens not found in the
    # pretrained embeddings.
    diff = len(vocab) - embeddings.vocab_length - 2
    PAD_UNK_embeddings = np.zeros((2, args.EMBEDDING_DIM))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((PAD_UNK_embeddings,
                                 embeddings._matrix,
                                 new_embeddings))

    # Set up the data iterators for the LSTM model. The batch size for the dev
    # and test loader is set to 1 for the predict() and evaluate() methods
    train_loader = DataLoader(train_iter,
                              batch_size=args.BATCH_SIZE,
                              collate_fn=train_iter.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev_iter,
                            batch_size=1,
                            collate_fn=dev_iter.collate_fn,
                            shuffle=False)

    test_loader = DataLoader(test_iter,
                             batch_size=1,
                             collate_fn=test_iter.collate_fn,
                             shuffle=False)

    # Automatically determine whether to run on CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    best_b_f1 = 0
    best_p_f1 = 0
    best_args = None
    # tuning
    for hidden_dim in [100, 200, 300, 500]:
        args.HIDDEN_DIM = hidden_dim

        for w_dp in [0.3, 0.4, 0.5, 0.6]:
            args.WORD_DROPOUT = w_dp
            for bi in [True, False]:
                args.BIDIRECTIONAL = bi

                logger.info(args)

                model = MODEL(word2idx=vocab,
                              embedding_matrix=new_matrix,
                              embedding_dim=args.EMBEDDING_DIM,
                              char_embedding_dim=args.CHAR_EMBEDDING_DIM,
                              char_hidden_dim=args.CHAR_HIDDEN_DIM,
                              hidden_dim=args.HIDDEN_DIM,  # tuning 100,200, 300, 500
                              alphabet_size=alphabet.size(),
                              device=device,
                              encoding=args.ENCODING,
                              num_layers=args.NUM_LAYERS,  # tuning 1,2,3
                              rnn_dropout=args.RNN_DROPOUT,  # tuning 0.2, 0.3, 0.4
                              word_dropout=args.WORD_DROPOUT,  # tuning 0.3, 0.4, 0.5, 0.6
                              learning_rate=args.LEARNING_RATE,
                              train_embeddings=args.TRAIN_EMBEDDINGS,
                              bidirectional=args.BIDIRECTIONAL,  # tuning true, false
                              model_type=args.MODEL_TYPE,  # test lstm and gru
                              use_char=args.USE_CHAR  # test use char or not
                              )

                model.to(device)
                model.fit(train_loader, dev_loader, epochs=args.EPOCHS)
                binary_f1, propor_f1 = model.evaluate(test_loader)
                if binary_f1 > best_b_f1:
                    best_b_f1 = binary_f1
                    best_p_f1 = propor_f1
                    best_args = args
                    logger.info(f'The best args are {best_args}')
                    logger.info(f'The best fi scores are {best_b_f1} for binary and {best_p_f1} for propotional!')
                else:
                    logger.info(f'The best args are {best_args}')
                    logger.info(f'The best fi scores are {best_b_f1} for binary and {best_p_f1} for propotional!')




