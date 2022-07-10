#!/bin/env python3
# coding: utf-8

# To run this on Saga, load this module:
# NLPL-nlptools/2021.01-gomkl-2019b-Python-3.7.4

import sys
import gensim
import logging
import zipfile
import json
import random

# Simple toy script to get an idea of what one can do with word embedding models using Gensim
# Models can be found at http://vectors.nlpl.eu/explore/embeddings/models/
# or in the /cluster/shared/nlpl/data/vectors/latest/  directory on Saga
# (for example, /cluster/shared/nlpl/data/vectors/latest/200.zip)


def load_embedding(modelfile):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            # metafile = archive.open("meta.json")
            # metadata = json.loads(metafile.read())
            # for key in metadata:
            #    print(key, metadata[key])
            # print("============")
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    embeddings_file = sys.argv[1]  # File containing word embeddings

    logger.info("Loading the embedding model...")
    model = load_embedding(embeddings_file)
    logger.info("Finished loading the embedding model...")

    logger.info(f"Model vocabulary size: {len(model.vocab)}")

    logger.info(f"Random example of a word in the model: {random.choice(model.index2word)}")

    while True:
        query = input("Enter your word (type 'exit' to quit):")
        if query == "exit":
            exit()
        words = query.strip().split()
        # If there's only one query word, produce nearest associates
        if len(words) == 1:
            word = words[0]
            print(word)
            if word in model:
                print("=====")
                print("Associate\tCosine")
                for i in model.most_similar(positive=[word], topn=10):
                    print(f"{i[0]}\t{i[1]:.3f}")
                print("=====")
            else:
                print(f"{word} is not present in the model")

        # Else, find the word which doesn't belong here
        else:
            print("=====")
            print("This word looks strange among others:", model.doesnt_match(words))
            print("=====")