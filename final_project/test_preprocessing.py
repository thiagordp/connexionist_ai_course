"""

"""

import nltk

from src.preprocessing.preprocessing_shallow_ml import PreProcessingShallowML

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('rslp')
    preprocess = PreProcessingShallowML()
    preprocess.load_dataset(dataset_path="Data/final_dataset_2l_wo_result/", save_pickle=True)
    preprocess.preprocess_corpus(lowercase=True, stemming=False, remove_stopwords=True, remove_html=True, remove_punct=True)
