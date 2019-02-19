import random

from config import DOC2VEC_DIM, NEWS_MAXSEQ_LEN, PRICE_SCALER_LOC
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, TimeDistributed, Flatten, concatenate
from keras.layers import LSTM
from keras.utils import plot_model
from pymongo import MongoClient
from sklearn.externals import joblib

"""
: sample execution :
from lstm_basic import LSTMbasic

d = LSTMbasic()
m = d.create_model()
x, y = d._get_train_dataseq()
vx, vy = d._get_dataseq(d.valid_start_idx_list)
ex, ey = d._get_dataseq(d.evalu_start_idx_list)

d.train_model(m, x, y, vx, vy, ex, ey)
"""

class LSTMbasic:
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
        self.price_scaler = self._get_price_scaler()

    def _get_price_scaler(self):
        return joblib.load(PRICE_SCALER_LOC)

    def _get_train_dataseq(self):
        """ retrieve a sequential dataset starting at a random index """
        timestep = self.timestep
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        ohl_dataseq_x, dataseq_y = [], []
        if len(self.train_start_idx_list) == 0:
            self.train_start_idx_list = self.train_start_idx.copy()
        # random.shuffle(self.train_start_idx_list)
        for idx in self.train_start_idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            ohlcv_ac = [[r[colname]
                            for colname in ohlcv_ac_colname]
                            for r in res_x]
            targetpr = [res_y['Adj Close']]
            ohl_dataseq_x.append(ohlcv_ac)
            dataseq_y.append(targetpr)
        return ohl_dataseq_x, dataseq_y


    def _get_dataseq(self, idx_list):
        """ retrieve dataset for validation or evaluation """
        timestep = self.timestep
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        ohl_dataseq_x, dataseq_y = [], []
        for idx in idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            ohlcv_ac = [[r[colname]
                            for colname in ohlcv_ac_colname]
                            for r in res_x]
            targetpr = [res_y['Adj Close']]
            ohl_dataseq_x.append(ohlcv_ac)
            dataseq_y.append(targetpr)
        return ohl_dataseq_x, dataseq_y

    def create_model(self):

        input_djia_seq = Input(shape=(5, 6,), dtype='float32')                     # open high low close volume adj close

        djia_seq = Sequential()
        djia_seq.add(LSTM(8, batch_input_shape=(None, 5, 6,), return_sequences=True))
        djia_seq.add(LSTM(8))

        encoded_djia = djia_seq(input_djia_seq)
        target = Dense(1)(encoded_djia)

        model = Model([input_djia_seq], target)
        model.compile(optimizer='adam', loss='mean_squared_error',
                    metrics=['mse'])

        return model


    def train_model(self, model, priceseq_train, target_train, priceseq_valid,
        target_valid, priceseq_eval, target_eval):

        x = np.array(priceseq_train)
        y = np.array(target_train)

        vx = np.array(priceseq_valid)
        vy = np.array(target_valid)

        ex = np.array(priceseq_eval)
        ey = np.array(target_eval)

        history = model.fit([x], y,
                    batch_size=self.batch_size,
                    epochs=20,
                    validation_data=([vx], vy))

        results = model.predict([ex])
        results = self.price_scaler.inverse_transform(results)
        ey = self.price_scaler.inverse_transform(ey)

        plot_model(model, to_file='model_seq.png', show_shapes=True)
        plt.scatter(range(len(results)), results, c='r')
        plt.scatter(range(len(results)), ey, c='b')
        plt.show()
