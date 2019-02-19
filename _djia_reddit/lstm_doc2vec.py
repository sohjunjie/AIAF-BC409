# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(1)

from config import DOC2VEC_DIM, NEWS_MAXSEQ_LEN
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.externals import joblib

from keras.models import Sequential, Model
from keras.layers import Input, Activation, TimeDistributed, Flatten, concatenate, add
from keras.layers import LSTM, Dense, Dropout
from keras.utils import plot_model

"""
: sample execution :
from lstm_doc2vec import LSTMdoc2vec

d = LSTMdoc2vec()
m = d.create_model()
x1, x2, y1, y2 = d._get_train_dataseq()
vx1, vx2, vy1, vy2  = d._get_dataseq(d.valid_start_idx_list)
ex1, ex2, ey1, ey2 = d._get_dataseq(d.evalu_start_idx_list)

d.train_model(m, x1, x2, y1, y2, vx1, vx2, vy1, vy2, ex1, ex2, ey1, ey2)
"""

class LSTMdoc2vec:
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
        self.minMaxScaler = self._get_minmax_scaler()

    def _get_minmax_scaler(self):
        return None
        # return joblib.load(MIXMAX_SCALER_LOC)

    def _get_train_dataseq(self):
        timestep, batch_size = self.timestep, self.batch_size
        news_seq_colname = ['Top' + str(x) for x in range(1, 26)]
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        news_dataseq_x, ohl_dataseq_x, pri_dataseq_y, lbl_dataseq_y = [], [], [], []
        if len(self.train_start_idx_list) == 0:
            self.train_start_idx_list = self.train_start_idx.copy()
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
            target_pri, target_lbl = res_y['Adj Close'], res_y['Label']
            news_dataseq_x.append(news_seq)
            ohl_dataseq_x.append(ohlcv_ac)
            pri_dataseq_y.append(target_pri)
            lbl_dataseq_y.append(target_lbl)
        return news_dataseq_x, ohl_dataseq_x, pri_dataseq_y, lbl_dataseq_y


    def _get_dataseq(self, idx_list):
        """ retrieve dataset for validation or evaluation """
        timestep = self.timestep
        news_seq_colname = ['Top' + str(x) for x in range(1, 26)]
        ohlcv_ac_colname = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        news_dataseq_x, ohl_dataseq_x, pri_dataseq_y, lbl_dataseq_y = [], [], [], []

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
            target_pri, target_lbl = res_y['Adj Close'], res_y['Label']
            news_dataseq_x.append(news_seq)
            ohl_dataseq_x.append(ohlcv_ac)
            pri_dataseq_y.append(target_pri)
            lbl_dataseq_y.append(target_lbl)
        return news_dataseq_x, ohl_dataseq_x, pri_dataseq_y, lbl_dataseq_y

    def create_model(self):

        input_news_seq = Input(shape=(5, 25, DOC2VEC_DIM, ), name='top25_news_headline')   # encoded top 25 news
        input_djia_seq = Input(shape=(5, 6,), dtype='float32', name='ohlc_volume_label')   # open high low close volume adj close

        # construct the model
        news_seq = Sequential(name='news_lstm_ts5')
        # require Dense layer to flatten 5 dimension to 3 dimension
        news_seq.add(Dense(50, batch_input_shape=(None, 5, 25, DOC2VEC_DIM)))
        news_seq.add(TimeDistributed(Flatten()))
        news_seq.add(LSTM(64, return_sequences=False))
        # news_seq.add(Dropout(0.2))
        encoded_news = news_seq(input_news_seq)

        djia_seq = Sequential(name='ohlc_lstm_ts5')
        djia_seq.add(LSTM(64, return_sequences=False))
        # djia_seq.add(Dropout(0.2))
        encoded_djia = djia_seq(input_djia_seq)

        encoded_news = Dense(32)(encoded_news)
        encoded_djia = Dense(16)(encoded_djia)

        merge = concatenate([encoded_news, encoded_djia], name='news_price_concat')

        label = Dense(1)(merge)
        label = Dropout(0.2)(label)
        target_label = Activation('sigmoid', name='label_output')(label)

        target_price = Dense(32)(merge)
        target_price = Dense(1, name='price_output')(target_price)

        model = Model(inputs=[input_news_seq, input_djia_seq], outputs=[target_price, target_label])

        model.compile(optimizer='adam',
                      loss={'price_output': 'mean_squared_error', 'label_output': 'binary_crossentropy'},
                      metrics={'price_output': 'mse', 'label_output': 'accuracy'},
                      loss_weights={'price_output': 0.5, 'label_output': 0.5}
                    )
        return model

    def train_model(self, model,
        newsseq_train, priceseq_train, tgt_price_train, tgt_label_train,
        newsseq_valid, priceseq_valid, tgt_price_valid, tgt_label_valid,
        newsseq_eval, priceseq_eval, tgt_price_eval, tgt_label_eval):

        x1 = np.array(newsseq_train)
        x2 = np.array(priceseq_train)
        y1 = np.array(tgt_price_train)
        y2 = np.array(tgt_label_train)

        vx1 = np.array(newsseq_valid)
        vx2 = np.array(priceseq_valid)
        vy1 = np.array(tgt_price_valid)
        vy2 = np.array(tgt_label_valid)

        ex1 = np.array(newsseq_eval)
        ex2 = np.array(priceseq_eval)
        ey1 = np.array(tgt_price_eval)
        ey2 = np.array(tgt_label_eval)

        history = model.fit([x1, x2], [y1, y2],
                    batch_size=12,
                    epochs=50,
                    validation_data=([vx1, vx2], [vy1, vy2]))

        plot_model(model, to_file='model_seq.png', show_shapes=True)

        r1, r2 = model.predict([x1, x2])

        plt.scatter(range(len(r1)), r1, c='r')
        plt.scatter(range(len(r1)), y1, c='b')
        plt.show()

        plt.scatter(range(len(r1)), r2, c='r')
        plt.scatter(range(len(r1)), y2, c='b')
        plt.show()

        r1, r2 = model.predict([ex1, ex2])

        plt.scatter(range(len(r1)), r1, c='r')
        plt.scatter(range(len(r1)), ey1, c='b')
        plt.show()

        plt.scatter(range(len(r1)), (r2 > 0.5).astype(int), c='r')
        plt.scatter(range(len(r1)), ey2, c='b')
        plt.show()
