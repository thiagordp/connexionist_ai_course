from gensim.models import KeyedVectors
from keras import Input, Model, metrics, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Reshape, Conv2D, MaxPooling2D, concatenate, Flatten, Dropout, Dense
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CNN_Classifier():

    def __init__(self, X_train, y_train, X_test, y_test, embeddings_model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.embeddings_model = embeddings_model

    def train(self):
        pass
