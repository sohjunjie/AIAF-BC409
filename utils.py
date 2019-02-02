from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import numpy as np


def load_word2vec():
    model_gs = Word2Vec.load('model/text8_gs.bin')
    return model_gs


def create_vector(word, word2vec, word_vector_size=100):
    # if the word is missing from word2vec, create some fake vector and store in word2vec!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec.wv.add(word, vector)
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    """ return word embedding from word2vec model given a word token """
    if not word in word2vec.wv.vocab
        create_vector(word, word2vec, word_vector_size, silent)
    return word2vec.wv.get_vector(word)
