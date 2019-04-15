import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, GRU, Input, concatenate
from keras.layers.core import Reshape
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def draw_price_trend_plot(dataset, train_preds, valid_preds, test_preds, y_train, y_valid, y_test):

    close = dataset[['Close']]
    close = close.values.reshape(-1)
    x_axis = np.arange(0, len(close))

    train_preds = train_preds.reshape(-1)
    valid_preds = valid_preds.reshape(-1)
    test_preds = test_preds.reshape(-1)
    trend = np.concatenate([train_preds, valid_preds, test_preds])

    y_train = y_train.reshape(-1)
    y_valid = y_valid.reshape(-1)
    y_test = y_test.reshape(-1)
    y_trend = np.concatenate([y_train, y_valid, y_test])

    padded_trend = np.pad(trend, [len(close) - len(trend), 0], mode='constant', constant_values=-1)
    padded_y_trend = np.pad(y_trend, [len(close) - len(trend), 0], mode='constant', constant_values=-1)

    test_idx = len(y_train) + len(y_valid)

    plt.subplot(3, 1, 1)
    plt.plot(x_axis[test_idx:], close[test_idx:], '-')
    plt.title('prediction of nasdaq trend momentum on test set')
    plt.ylabel('standardized closing price')

    plt.subplot(3, 1, 2)
    plt.plot(x_axis[test_idx:], padded_y_trend[test_idx:], 'o')
    plt.xlabel('time (s)')
    plt.ylabel('true momentum')

    plt.subplot(3, 1, 3)
    plt.plot(x_axis[test_idx:], padded_trend[test_idx:], 'o')
    plt.xlabel('time')
    plt.ylabel('predicted momentum')

    plt.show()


def draw_precision_recall_curve(actuals, predict_probs, titlestr):
    average_precision = average_precision_score(actuals, predict_probs)
    precision, recall, threshold = precision_recall_curve(actuals, predict_probs, pos_label=1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f} for {1}'.format(average_precision, titlestr))
    plt.show()


def draw_roc_curve(actual, predict_probs, titlestr):
    fpr, tpr, thresholds = roc_curve(actual, predict_probs, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve'.format(titlestr))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {0}'.format(titlestr))
    plt.legend(loc="lower right")
    plt.show()


def standard_normalize_dataset(dataset, features):
    """
    normalize selected features in dataset
    :param dataset: pandas dataset
    :param features: list of features to normalize
    :return: normalized dataset
    """
    for f in features:
        scaler = StandardScaler()
        scaler.fit(dataset[f].values.reshape(-1, 1))
        dataset[f] = scaler.transform(dataset[f].values.reshape(-1, 1))
    return dataset


def minmax_normalize_dataset(dataset, features):
    """
    normalize selected features in dataset
    :param dataset: pandas dataset
    :param features: list of features to normalize
    :return: normalized dataset
    """
    for f in features:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(dataset[f].values.reshape(-1, 1))
        dataset[f] = scaler.transform(dataset[f].values.reshape(-1, 1))
    return dataset


def texts_to_sequence(x, maxlen, indexer):
    """
    generate padded sequence of text
    :param x: list of texts
    :param maxlen: maximum sequence length
    :param indexer: keras indexing tokenizer
    :return: array of sequences
    """
    return pad_sequences(sequences=indexer.texts_to_sequences(x), maxlen=maxlen, padding='post')


def preprocess_dataset_text(dataset, features, maxlen, indexer):
    for f in features:
        texts = np.array(dataset[:][f])
        texts_seq = np.array(texts_to_sequence(texts, maxlen, indexer))
        dataset[f] = texts_seq.tolist()
    return dataset


def get_pretrain_embedding_matrix(word_index, embed_dim):
    embeddings_index = {}
    f = open('model/glove.6B.50d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


class GRUModel:
    def __init__(self, corpus, dataset, max_seqlen, embed_dim, timestep=10, batch_size=32, epochs=20,
                 num_train=600, num_valid=200):
        self.dataset = dataset

        # Embedding layer parameters
        self.indexer = Tokenizer(lower=True)
        self.indexer.fit_on_texts([doc for doc in corpus[0]])
        self.vocabSize = len(self.indexer.word_counts) + 1
        self.maxSeqLen = max_seqlen
        self.embedDim = embed_dim

        # LSTM model parameters
        self.timeStep = timestep
        self.batchSize = batch_size
        self.epochs = epochs
        self.num_train = num_train
        self.num_valid = num_valid

        # Pre-processing step
        self._preprocess_dataset(standard_normal_features=['Close', 'Volume'], minmax_feaures=['Change'],
                                 texts_features=['Top' + str(i) for i in range(1, 26)])

    def _preprocess_dataset(self, standard_normal_features, minmax_feaures, texts_features):
        self.dataset = standard_normalize_dataset(self.dataset, standard_normal_features)
        self.dataset = minmax_normalize_dataset(self.dataset, minmax_feaures)
        self.dataset = preprocess_dataset_text(self.dataset, texts_features,
                                               self.maxSeqLen,
                                               self.indexer)

    def _get_sequential_dataset(self):
        timestep = self.timeStep
        dataset_x = self.dataset.drop(columns=['Top' + str(i) for i in range(1, 26)])
        dataset_x_ts = self.dataset[['Top' + str(i) for i in range(1, 26)]]
        dataset_y = self.dataset[['Forward_momentum_10', ]]

        sequence_len = timestep + 1
        result_x = []
        for index in range(len(dataset_x) - sequence_len):
            result_x.append(dataset_x[index: index + sequence_len].to_numpy())
        result_x = np.array(result_x)

        news_result_x = {}
        for i in range(1, 26):
            result_x_ts = []
            for index in range(len(dataset_x_ts) - sequence_len):
                tmp = np.array(dataset_x_ts[index: index + sequence_len]['Top' + str(i)]
                               .to_numpy()
                               .tolist())
                tmp = tmp.reshape(tmp.shape[0], -1)
                result_x_ts.append(tmp)
            result_x_ts = np.array(result_x_ts)
            news_result_x['Top' + str(i)] = result_x_ts

        result_y = []
        for index in range(len(dataset_y) - sequence_len):
            result_y.append(dataset_y[index: index + sequence_len].to_numpy())
        result_y = np.array(result_y)
        result_y = result_y.reshape(result_y.shape[0], -1)

        assert (result_x.shape[0] == result_y.shape[0])

        row = self.num_train
        # training set
        x_train = result_x[:int(row), :]
        x_train = x_train[:, :-1]
        y_train = result_y[:int(row), :]
        y_train = y_train[:, -1]
        x_train_news = {}
        for i in range(1, 26):
            x_train_news['Top' + str(i)] = news_result_x['Top' + str(i)][:int(row), :]
            x_train_news['Top' + str(i)] = x_train_news['Top' + str(i)][:, :-1]
            x_train_news['Top' + str(i)] = x_train_news['Top' + str(i)].reshape(
                x_train_news['Top' + str(i)].shape[0], -1)

        # test dataset
        x_test = result_x[int(row):, :-1]
        y_test = result_y[int(row):, -1]
        x_test_news = {}
        for i in range(1, 26):
            x_test_news['Top' + str(i)] = news_result_x['Top' + str(i)][int(row):, :]
            x_test_news['Top' + str(i)] = x_test_news['Top' + str(i)][:, :-1]
            x_test_news['Top' + str(i)] = x_test_news['Top' + str(i)].reshape(
                x_test_news['Top' + str(i)].shape[0], -1)

        return x_train, x_train_news, y_train, x_test, x_test_news, y_test

    def create_model(self):

        # news headlines inputs
        news_embedding_input = {}
        for i in range(1, 26):
            news_embedding_input['Top' + str(i)] = Input(shape=(self.maxSeqLen * self.timeStep, ),
                                                         name='input_top' + str(i))

        # embedding layer
        embedding_matrix = get_pretrain_embedding_matrix(self.indexer.word_index, self.embedDim)
        embedding_layer = Embedding(self.vocabSize, self.embedDim, weights=[embedding_matrix], trainable=False,
                                    name='embedding_layer')
        news_embedding_layer = {}
        for i in range(1, 26):
            news_embedding_layer['Top' + str(i)] = embedding_layer(news_embedding_input['Top' + str(i)])

        # inner gru layer
        embedding_gru = GRU(8, dropout=0.2, name='inner_timeseries_embedding_gru')
        news_gru = {}
        for i in range(1, 26):
            news_gru['Top' + str(i)] = embedding_gru(news_embedding_layer['Top' + str(i)])
            news_gru['Top' + str(i)] = BatchNormalization()(news_gru['Top' + str(i)])

        # outer gru layer - concatenate and reshape to shape (25, 8)
        merged_news = concatenate([news_gru['Top' + str(i)] for i in range(1, 26)], axis=-1)
        merged_news = Reshape([25, 8])(merged_news)
        gru_news = GRU(16, return_sequences=False, dropout=0.2, name='outer_news_summarizer_gru')(merged_news)
        gru_news = BatchNormalization()(gru_news)

        # stock market data input and gru
        stock_input = Input(shape=(self.timeStep, 4))
        gru_stock = GRU(16, return_sequences=False, dropout=0.2, name='market_data_gru')(stock_input)
        gru_stock = BatchNormalization()(gru_stock)

        # output layer
        merged = concatenate([gru_news, gru_stock])
        out = Dense(1, activation='sigmoid', name='output_layer')(merged)

        model = Model(inputs=[news_embedding_input['Top' + str(i)]
                              for i in range(1, 26)]+[stock_input],
                      outputs=[out])
        model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
        model.summary()
        plot_model(model, to_file='model_seq.png', show_shapes=True)
        return model

    def train_model(self):
        x_train, x_train_news, y_train, x_test, x_test_news, y_test = self._get_sequential_dataset()

        row = self.num_valid
        x_valid = x_test[:int(row), :]
        y_valid = y_test[:int(row)]
        x_valid_news = {}
        for i in range(1, 26):
            x_valid_news['Top' + str(i)] = x_test_news['Top' + str(i)][:int(row), :]

        x_test = x_test[int(row):, :]
        y_test = y_test[int(row):]
        for i in range(1, 26):
            x_test_news['Top' + str(i)] = x_test_news['Top' + str(i)][int(row):, :]

        model = self.create_model()
        model.fit([x_train_news['Top' + str(i)] for i in range(1, 26)] + [x_train], [y_train],
                  shuffle=False,
                  batch_size=self.batchSize,
                  epochs=self.epochs,
                  validation_data=([x_valid_news['Top' + str(i)] for i in range(1, 26)] + [x_valid], [y_valid]))

        train_probs = model.predict([x_train_news['Top' + str(i)] for i in range(1, 26)] + [x_train])
        train_preds = (np.array(train_probs) > 0.5) * 1

        valid_probs = model.predict([x_valid_news['Top' + str(i)] for i in range(1, 26)] + [x_valid])
        valid_preds = (np.array(valid_probs) > 0.5) * 1

        test_probs = model.predict([x_test_news['Top' + str(i)] for i in range(1, 26)] + [x_test])
        test_preds = (np.array(test_probs) > 0.5) * 1

        train_acc = accuracy_score(y_train, train_preds)
        train_pre = precision_score(y_train, train_preds, pos_label=1)
        train_rec = recall_score(y_train, train_preds, pos_label=1)
        train_f1 = f1_score(y_train, train_preds, pos_label=1)

        valid_acc = accuracy_score(y_valid, valid_preds)
        valid_pre = precision_score(y_valid, valid_preds, pos_label=1)
        valid_rec = recall_score(y_valid, valid_preds, pos_label=1)
        valid_f1 = f1_score(y_valid, valid_preds, pos_label=1)

        test_acc = accuracy_score(y_test, test_preds)
        test_pre = precision_score(y_test, test_preds, pos_label=1)
        test_rec = recall_score(y_test, test_preds, pos_label=1)
        test_f1 = f1_score(y_test, test_preds, pos_label=1)

        print('train accuracy: ', train_acc)
        print('train precision: ', train_pre)
        print('train recall: ', train_rec)
        print('train f1 score: ', train_f1)

        print('valid accuracy: ', valid_acc)
        print('valid precision: ', valid_pre)
        print('valid recall: ', valid_rec)
        print('valid f1 score: ', valid_f1)

        print('test accuracy: ', test_acc)
        print('test precision: ', test_pre)
        print('test recall: ', test_rec)
        print('test f1 score: ', test_f1)

        # draw_roc_curve(y_train, train_probs, 'training')
        # draw_roc_curve(y_valid, valid_probs, 'validation')
        # draw_roc_curve(y_test, test_probs, 'testing')
        #
        # draw_precision_recall_curve(y_train, train_probs, 'training')
        # draw_precision_recall_curve(y_valid, valid_probs, 'validation')
        # draw_precision_recall_curve(y_test, test_probs, 'testing')

        draw_price_trend_plot(self.dataset, train_preds, valid_preds, test_preds, y_train, y_valid, y_test)


if __name__ == '__main__':
    corpus = pd.read_csv('data/RedditNewsCorpus.csv', header=None)
    dataset = pd.read_csv('data/combined_nasdaq_news_v2.csv', index_col='Date')
    dataset = dataset[['Close', 'Volume', 'Change', 'Forward_momentum_10'] +
                      ['Top' + str(i) for i in range(1, 26)]]

    MAX_TEXT_SEQLEN = 25
    EMBEDDING_DIM = 50
    TIMESTEP = 10
    BATCHSIZE = 32
    EPOCHS = 10
    NUM_TRAIN = 600
    NUM_VALID = 200

    gruModel = GRUModel(corpus, dataset,
                        max_seqlen=MAX_TEXT_SEQLEN,
                        embed_dim=EMBEDDING_DIM,
                        timestep=TIMESTEP,
                        batch_size=BATCHSIZE,
                        epochs=EPOCHS,
                        num_train=NUM_TRAIN,
                        num_valid=NUM_VALID)
    # m = gruModel.create_model()
    gruModel.train_model()
