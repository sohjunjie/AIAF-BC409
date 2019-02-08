from config import WORD2VEC_DIM, DOC2VEC_DIM, NEWS_MAXSEQ_LEN
from gensim.models import Word2Vec,  Doc2Vec
from gensim.models.word2vec import Text8Corpus
from pymongo import MongoClient

import numpy as np


def load_doc2vec():
    model = Doc2Vec.load('model/model_dv.bin')
    return model


def load_word2vec():
    model = Word2Vec.load('model/text8_gs.bin')
    return model


def _get_news_maxseq_len():
    """ 
    find the max news sequence len to determine pad length
    > precomputed len = 64
    """
    client = MongoClient('localhost', 27017)
    db = client.djia_news_dataset
    db_tbl_price_news = db.price_news
    maxlen = 0
    news_col_name = ['Top'+str(x) for x in range (1, 26)]
    res = db_tbl_price_news.find()
    for r in res:
        maxlen = max(maxlen, max([len(r[cn]) for cn in news_col_name]))
    return maxlen

def _get_allnews_len():
    """ 
    find the max news sequence len to determine pad length
    > precomputed len = 64
    """
    client = MongoClient('localhost', 27017)
    db = client.djia_news_dataset
    db_tbl_price_news = db.price_news
    maxlen = 0
    news_col_name = ['Top'+str(x) for x in range (1, 26)]
    res = db_tbl_price_news.find()
    ls = []
    for r in res:
        ls.append([len(r[cn]) for cn in news_col_name])
    return ls


def standardize_features(dataset, features):
    """ standardize feature of pandas dataframe to zero mean, unit standard deviation """
    for f in features:
        dataset[f] = (dataset[f] - dataset[f].mean()) / dataset[f].std()
    return dataset


def create_vector(word, word2vec, word_vector_size):
    """ if the word is missing from word2vec, create some fake vector and store in word2vec """
    vector = np.random.uniform(-1.0, 1.0, (word_vector_size,))
    word2vec.wv.add(word, vector)
    return vector


def process_word(word, word2vec, word_vector_size):
    """ return word embedding from word2vec model given a word token """
    if not word in word2vec.wv.vocab:
        create_vector(word, word2vec, word_vector_size)
    return word2vec.wv.get_vector(word)


def process_sentence(sentence, word2vec, truncate=True, pad_length=NEWS_MAXSEQ_LEN, word_vector_size=WORD2VEC_DIM):
    """
    64 is the precomputed maxseq length for a given news
    """
    if not type(sentence) is str:
        sent_vector = np.zeros(shape=(pad_length, word_vector_size), dtype='float')
        return sent_vector.tolist()

    sent = sentence.lower().split(' ')
    sent = [w for w in sent if len(w) > 0]

    sent_vector = np.array([process_word(word=w,
                                        word2vec=word2vec,
                                        word_vector_size=word_vector_size) for w in sent])
    if truncate:
        sent_vector = sent_vector[:pad_length]

    a, b = sent_vector.shape

    if pad_length - a < 0 or word_vector_size - b < 0:
        raise Exception('sentence vector shape out of bound')

    sent_vector = np.pad(array=sent_vector,
                         pad_width=((0, pad_length-a), (0, word_vector_size-b)),
                         mode='constant')

    return sent_vector.tolist()


def process_sentence_unpad(sentence, word2vec, word_vector_size=WORD2VEC_DIM):
    if not type(sentence) is str:
        return []

    sent = sentence.lower().split(' ')
    sent = [w for w in sent if len(w) > 0]

    sent_vector = np.array([process_word(word=w,
                                         word2vec=word2vec,
                                         word_vector_size=word_vector_size) for w in sent])
    return sent_vector.tolist()


def process_document(document, doc2vec, truncate=True, vector_size=DOC2VEC_DIM):
    """
    64 is the precomputed maxseq length for a given news
    """
    if not type(document) is str:
        doc_vector = np.zeros(shape=(vector_size, ), dtype='float')
        return doc_vector.tolist()

    doc = document.lower().split(' ')
    doc = [w for w in doc if len(w) > 0]

    doc_vector = doc2vec.infer_vector(doc)

    return doc_vector.tolist()
