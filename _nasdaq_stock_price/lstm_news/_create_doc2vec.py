import gensim
import os
import collections
import smart_open
import random

from _config import CORPUS_FILENAME
from _config import DATA_FOLDER, MODEL_FOLDER
from _config import DOC2VEC_DIM, DOC2VEC_MODEL_FILENAME


train_file = DATA_FOLDER + CORPUS_FILENAME


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_corpus = list(read_corpus(train_file))


model = gensim.models.doc2vec.Doc2Vec(vector_size=DOC2VEC_DIM, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.save(MODEL_FOLDER + DOC2VEC_MODEL_FILENAME)
