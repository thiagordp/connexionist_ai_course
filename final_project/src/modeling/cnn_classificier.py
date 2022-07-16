import logging
import os
import random

import tqdm
from gensim.models import KeyedVectors


class CNN_Classifier():

    def __init__(self, embeddings_model, max_nb_words, embeddings_length,
                 input_length):
        self.embeddings_model = embeddings_model
        self.max_nb_words = max_nb_words
        self.embeddings_length = embeddings_length
        self.input_length = input_length
        self.EMBEDDINGS_PATH = "Data/pre-trained_embeddings/"

    def get_vocab(self, vocab, text):
        splits = text.split()

        for token in splits:
            if token not in vocab.keys():
                vocab[token] = 1
            else:
                vocab[token] += 1

    def load_embeddings(self):
        ignore_list = [
            "pre-trained_legal/glove_3500000000_100"
        ]
        matches = []
        logging.info("Listing Embeddings")
        for root, dirnames, filenames in os.walk(self.EMBEDDINGS_PATH):
            for filename in filenames:
                if filename.endswith('.txt'):
                    path_emb = os.path.join(root, filename)

                    matches.append(os.path.join(root, filename))

        logging.info("Loading embeddings")
        dict_list_embeddings = {}
        random.shuffle(matches)

        for path_emb in tqdm.tqdm(matches):
            if path_emb.find("glove") != -1:

                temp = get_tmpfile("glove2word2vec.txt")
                glove2word2vec(path_emb, temp)

                word_vectors = KeyedVectors.load_word2vec_format(temp, binary=False)
            else:
                word_vectors = KeyedVectors.load_word2vec_format(path_emb, binary=False)

            key = path_emb.replace(self.EMBEDDINGS_PATH, "").replace(".txt", "")
            dict_list_embeddings[key] = word_vectors

        return dict_list_embeddings
