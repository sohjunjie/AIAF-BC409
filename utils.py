from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import numpy as np


def standardize_features(dataset, features):
    """ standardize feature of pandas dataframe to zero mean, unit standard deviation """
    for f in features:
        dataset[f] = (dataset[f] - dataset[f].mean()) / dataset[f].std()
    return dataset


def load_word2vec():
    model_gs = Word2Vec.load('model/text8_gs.bin')
    return model_gs


def create_vector(word, word2vec, word_vector_size=100):
    """ if the word is missing from word2vec, create some fake vector and store in word2vec """
    vector = np.random.uniform(-1.0, 1.0, (word_vector_size,))
    word2vec.wv.add(word, vector)
    return vector


def process_word(word, word2vec):
    """ return word embedding from word2vec model given a word token """
    if not word in word2vec.wv.vocab:
        create_vector(word, word2vec)
    return (word2vec.wv.get_vector(word)).tolist()


def process_sentence(sentence, word2vec):

    if not type(sentence) is str:
        return []

    sent = sentence.lower().split(' ')
    sent = [w for w in sent if len(w) > 0]
    sent_vector = [process_word(word=w,
                                word2vec=word2vec) for w in sent]
    return sent_vector
