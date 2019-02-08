import random

from config import DOC2VEC_DIM, NEWS_MAXSEQ_LEN
import numpy as np
from pymongo import MongoClient

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, TimeDistributed, Flatten, concatenate
from keras.layers import LSTM

"""
: sample execution :
from lstm_basic_v2 import LSTMBasic2

d = LSTMBasic2()
m = d.create_model()
x1, x2, y = d._get_train_dataseq()
tx1, tx2, ty = d._get_dataseq()

d.train_model(m, x1, x2, y, tx1, tx2, ty)
"""

class LSTMBasic2:
    def __init__(self, timestep=5, batch_size=12, iteration=30):
        client = MongoClient('localhost', 27017)
        db = client.djia_news_dataset
        self.db_tbl_price_news = db.price_news_v2
        self.timestep = timestep
        self.batch_size = batch_size
        self.iteration = iteration
        self.data_num_row = 1989
        self.train_start_idx = [(x * 5) for x in range(self.batch_size * self.iteration)]
        self.valid_start_idx = self.batch_size * self.iteration * self.timestep
        self.evalu_start_idx = self.valid_start_idx + (self.data_num_row - (self.valid_start_idx)) // 2 + 1
        self.train_start_idx_list = self.train_start_idx.copy()
        self.valid_start_idx_list = [x for x in range(self.valid_start_idx, self.evalu_start_idx-timestep)]
        self.evalu_start_idx_list = [x for x in range(self.evalu_start_idx, self.data_num_row-timestep)]

    def _get_train_dataseq(self):
        """ retrieve a sequential dataset starting at a random index """
        timestep, batch_size = self.timestep, self.batch_size
        news_seq_colname = ['Top' + str(x) for x in range(1, 26)]
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        news_dataseq_x, ohl_dataseq_x, dataseq_y = [], [], []
        if len(self.train_start_idx_list) == 0:
            self.train_start_idx_list = self.train_start_idx.copy()
        random.shuffle(self.train_start_idx_list)
        for idx in self.train_start_idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_x_full = [x for x in res_x]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            news_seq = [[r[colname]
                            for colname in news_seq_colname]
                            for r in res_x_full]
            ohlcv_ac = [[r[colname]
                            for colname in ohlcv_ac_colname]
                            for r in res_x_full]
            targetpr = [res_y['Adj Close']]
            news_dataseq_x.append(news_seq)
            ohl_dataseq_x.append(ohlcv_ac)
            dataseq_y.append(targetpr)
        return news_dataseq_x, ohl_dataseq_x, dataseq_y


    def _get_dataseq(self, type='validation'):
        """ retrieve dataset for validation or evaluation """
        timestep = self.timestep
        news_seq_colname = ['Top' + str(x) for x in range(1, 26)]
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        news_dataseq_x, ohl_dataseq_x, dataseq_y = [], [], []
        if type == 'validation':
            idx_list = self.valid_start_idx_list
        elif type == 'evaluation':
            idx_list = self.evalu_start_idx_list
        else:
            raise Exception('unsupported dataseq type')
        for idx in idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_x_full = [x for x in res_x]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            news_seq = [[r[colname]
                            for colname in news_seq_colname]
                            for r in res_x_full]
            ohlcv_ac = [[r[colname]
                            for colname in ohlcv_ac_colname]
                            for r in res_x_full]
            targetpr = [res_y['Adj Close']]
            news_dataseq_x.append(news_seq)
            ohl_dataseq_x.append(ohlcv_ac)
            dataseq_y.append(targetpr)
        return news_dataseq_x, ohl_dataseq_x, dataseq_y

    def create_model(self):

        input_news_seq = Input(shape=(5, 25, DOC2VEC_DIM, ))     # encoded top 25 news
        input_djia_seq = Input(shape=(5, 6,), dtype='float32')                     # open high low close volume adj close

        # construct the model
        news_seq = Sequential()
        # require Dense layer to flatten 5 dimension to 3 dimension
        news_seq.add(Dense(100, batch_input_shape=(None, 5, 25, DOC2VEC_DIM)))
        news_seq.add(TimeDistributed(Flatten()))
        news_seq.add(Dense(64))

        news_seq.add(LSTM(32, batch_input_shape=(None, 5, 64), return_sequences=True))
        news_seq.add(LSTM(16, return_sequences=True))
        news_seq.add(LSTM(8, return_sequences=False))
        encoded_news = news_seq(input_news_seq)

        djia_seq = Sequential()
        djia_seq.add(LSTM(8, batch_input_shape=(None, 5, 6,), return_sequences=False))
        encoded_djia = djia_seq(input_djia_seq)

        merge = concatenate([encoded_news, encoded_djia])
        merge = Activation('tanh')(merge)

        target = Dense(1)(merge)

        model = Model([input_news_seq, input_djia_seq], target)
        model.compile(optimizer='adam', loss='mean_squared_error',
                    metrics=['mae'])

        return model


    def train_model(self, model, newsseq_train, priceseq_train, target_train,
        newsseq_test, priceseq_test, target_test):

        x1 = np.array(newsseq_train)
        x2 = np.array(priceseq_train)
        y = np.array(target_train)

        tx1 = np.array(newsseq_test)
        tx2 = np.array(priceseq_test)
        ty = np.array(target_test)

        model.fit([x1, x2], y,
                batch_size=self.batch_size,
                epochs=self.iteration,
                validation_data=([tx1, tx2], ty))
