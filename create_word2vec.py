from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

import os


MODELS_DIR = 'model/'

lr = 0.05
dim = 100
ws = 5
epoch = 5
minCount = 5
neg = 5
loss = 'ns'
t = 1e-4

params = {
    'alpha': lr,
    'size': dim,
    'window': ws,
    'iter': epoch,
    'min_count': minCount,
    'sample': t,
    'sg': 1,
    'hs': 0,
    'negative': neg
}


def train_word2vec_model(corpus_file, output_name):
    """ 
    Train the word2vec model with a corpus and save the trained model
    to disk

    :param
        string corpus_file: corpus file name to train on
        string output_name: model name to save as

    :corpus
        text8: http://mattmahoney.net/dc/text8.zip

    """

    output_file = '{:s}_gs'.format(output_name)
    if not (os.path.isfile(os.path.join(MODELS_DIR, '{:s}.vec'.format(output_file))) and
            os.path.isfile(os.path.join(MODELS_DIR, '{:s}.bin'.format(output_file)))
            ):
        print('Training word2vec on {:s} corpus..'.format(corpus_file))
        
        # Text8Corpus class for reading space-separated words file
        gs_model = Word2Vec(Text8Corpus(os.path.join(MODELS_DIR, corpus_file)), **params)

        # save word2vec model
        gs_model.wv.save_word2vec_format(os.path.join(MODELS_DIR, '{:s}.vec'.format(output_file)))
        gs_model.save(os.path.join(MODELS_DIR, '{:s}.bin'.format(output_file)))

        print('Saved gensim model as {:s}.vec'.format(output_file))
    else:
        print('Using existing model file {:s}'.format(output_file))


train_word2vec_model(corpus_file='text8', output_name='text8')
