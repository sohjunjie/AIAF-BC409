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
from lstm_basic1 import LSTMbasic

d = LSTMbasic()
m = d.create_model()
x, y1, y2 = d._get_train_dataseq()
vx, vy1, vy2 = d._get_dataseq(d.valid_start_idx_list)
ex, ey1, ey2 = d._get_dataseq(d.evalu_start_idx_list)

d.train_model(m, x, y1, y2, vx, vy1, vy2, ex, ey1, ey2)
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
        ohl_dataseq_x, dataseq_y1, dataseq_y2 = [], [], []
        if len(self.train_start_idx_list) == 0:
            self.train_start_idx_list = self.train_start_idx.copy()
        # random.shuffle(self.train_start_idx_list)
        for idx in self.train_start_idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            ohlcv_ac = [[r[colname]
                            for colname in ohlcv_ac_colname]
                            for r in res_x]
            target1, target2 = res_y['Adj Close'], res_y['Label']
            ohl_dataseq_x.append(ohlcv_ac)
            dataseq_y1.append(target1)
            dataseq_y2.append(target2)
        return ohl_dataseq_x, dataseq_y1, dataseq_y2

    def _get_dataseq(self, idx_list):
        """ retrieve dataset for validation or evaluation """
        timestep = self.timestep
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        ohl_dataseq_x, dataseq_y1, dataseq_y2 = [], [], []
        for idx in idx_list:
            res_x = self.db_tbl_price_news.find()[idx:idx+timestep]
            res_y = self.db_tbl_price_news.find()[idx+timestep]
            ohlcv_ac = [[r[colname]
                            for colname in ohlcv_ac_colname]
                            for r in res_x]
            target1, target2 = res_y['Adj Close'], res_y['Label']
            ohl_dataseq_x.append(ohlcv_ac)
            dataseq_y1.append(target1)
            dataseq_y2.append(target2)
        return ohl_dataseq_x, dataseq_y1, dataseq_y2

    def create_model(self):

        input_djia_seq = Input(shape=(5, 6,), dtype='float32')                     # open high low close volume adj close

        djia_seq = Sequential()
        djia_seq.add(LSTM(8, batch_input_shape=(None, 5, 6,), return_sequences=True))
        djia_seq.add(LSTM(8))

        encoded_djia = djia_seq(input_djia_seq)
        target1 = Dense(1, name='target_price')(encoded_djia)

        target2 = Dense(1)(encoded_djia)
        target2 = Activation('sigmoid', name='target_label')(target2)

        model = Model([input_djia_seq], [target1, target2])
        model.compile(optimizer='adam',
                      loss={'target_price': 'mean_squared_error', 'target_label': 'binary_crossentropy'},
                      loss_weights={'target_price': 0.5, 'target_label': 0.5},
                      metrics={'target_price': 'mse', 'target_label': 'accuracy'}
                    )

        return model


    def train_model(self, model,
        priceseq_train, target_train1, target_train2,
        priceseq_valid, target_valid1, target_valid2,
        priceseq_eval, target_eval1, target_eval2):

        x = np.array(priceseq_train)
        y1 = np.array(target_train1)
        y2 = np.array(target_train2)

        vx = np.array(priceseq_valid)
        vy1 = np.array(target_valid1)
        vy2 = np.array(target_valid2)

        ex = np.array(priceseq_eval)
        ey1 = np.array(target_eval1)
        ey2 = np.array(target_eval2)

        history = model.fit([x], [y1, y2],
                    batch_size=12,
                    epochs=20,
                    validation_data=([vx], [vy1, vy2]))

        r1, r2 = model.predict([ex])
        results = self.price_scaler.inverse_transform(r1)
        ey1 = self.price_scaler.inverse_transform(ey1)

        plot_model(model, to_file='model_seq.png', show_shapes=True)
        plt.scatter(range(len(results)), results, c='r')
        plt.scatter(range(len(results)), ey1, c='b')
        plt.show()
