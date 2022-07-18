"""
Text preprocessing for Shallow Machine Learning

@author Thiago Raulino Dal Pont
@date July 14, 2022
"""

#
# Imports
#
import glob
import os
import string

import nltk
import pandas as pd
from bs4 import BeautifulSoup


class PreProcessingShallowML:

    def __init__(self, corpora: pd.DataFrame = None):
        self.labels = None
        self.dataset_path = ""
        self.df_corpora = corpora
        self.metadata_corpora = {}  # Classes, vocab, document count

    def load_dataset(self, dataset_path: str = "", save_pickle: bool = False):
        print("Loading dataset")
        self.dataset_path = dataset_path
        data = list()

        labels = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

        self.labels = [label.replace(dataset_path, '') for label in labels]
        self.metadata_corpora["labels"] = dict.fromkeys(self.labels)
        print(self.metadata_corpora)
        for cur_path, labels, files in os.walk(dataset_path):
            for label in labels:
                final_path = os.path.join(dataset_path, label, "*.txt")
                files = glob.glob(final_path)
                print("  -> Found %d files inside %s" % (len(files), final_path))

                self.metadata_corpora["labels"][label] = {}

                self.metadata_corpora["labels"][label]["docs"] = len(files)

                for case in files:
                    file_name = case.split(os.path.sep)[-1].replace(".txt", "")
                    with open(case) as fp:
                        text = fp.read()

                    data_file = [file_name, case, label, text]
                    data.append(data_file)

        self.df_corpora = pd.DataFrame(data, columns=["file_Name", "file_path", "label", "raw_content"])

    def preprocess_corpus(self, lowercase: bool = False, stemming: bool = False, stemming_method="rslp",
                          remove_punct=False, remove_html=True, remove_stopwords=False, keep_raw=True):

        print("Preprocessing corpus")
        self.df_corpora["processed_content"] = self.df_corpora["raw_content"]

        if lowercase:
            print("  -> Converting to lowercase")
            self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(lambda doc: doc.lower())

        if remove_html:
            print("  -> Removing HTML")
            self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(
                lambda doc: BeautifulSoup(doc, "lxml").text
            )

        print("  -> Tokenizing")
        self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(
            lambda doc: nltk.word_tokenize(doc))

        if remove_punct:
            print("  -> Removing punctuation")
            punctuation_list = string.punctuation + "`'"
            self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(
                lambda doc: [token for token in doc if token not in punctuation_list]
            )

        if remove_stopwords:
            print("  -> Removing Stopwords")
            stopwords_list = nltk.corpus.stopwords.words('portuguese')
            self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(
                lambda doc: [token for token in doc if token not in stopwords_list])

        if stemming:
            print("  -> Stemming using %s" % stemming_method)
            stemmer = nltk.stem.RSLPStemmer()
            self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(
                lambda doc: [stemmer.stem(token) for token in doc])

        if not keep_raw:
            print("  -> Removing raw text")
            self.df_corpora.drop(columns="raw_content", inplace=True)

        print("  -> Joining tokens into string")
        self.df_corpora["processed_content"] = self.df_corpora["processed_content"].apply(lambda doc: " ".join(doc))

        #print("  -> A sample of the preprocessed data:")
        #print(self.df_corpora["processed_content"].sample().values)
