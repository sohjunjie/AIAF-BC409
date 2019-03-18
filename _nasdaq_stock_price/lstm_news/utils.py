from _config import DOC2VEC_DIM, DOC2VEC_MODEL_FILENAME, MODEL_FOLDER

from gensim.models import Doc2Vec
from sklearn.preprocessing import MinMaxScaler

import numpy as np


def load_doc2vec():
    model = Doc2Vec.load(MODEL_FOLDER + DOC2VEC_MODEL_FILENAME)
    return model


def scale_features(dataset, features, scaler):
    """ standardize feature of pandas dataframe to zero mean, unit standard deviation """
    for f in features:
        dataset[f] = scaler.transform(dataset[f].values.reshape(-1,1))
    return dataset


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
