# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(1)

from _config import DOC2VEC_DIM
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
    def __init__(self, timestep=5, batch_size=12):
        client = MongoClient('localhost', 27017)
        db = client.djia_news_dataset
        self.db_tbl_price_news = db.price_news_v1
        self.timestep = timestep
        self.batch_size = batch_size
        self.data_num_row = 1017

    def _get_train_test_dataseq(self):

        timestep = self.timestep
        price_col_ls = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
        coy_prefix_ls = ['', 'AAPL_', 'AMZN_', 'FB_', 'GOOG_', 'IBM_', 'MSFT_', 'NFLX_', 'ORCL_', 'INTC_']

        price_feature_col = []
        for coy in coy_prefix_ls:
            for pricecol in price_col_ls:
                price_feature_col.append(coy + pricecol)
        news_col = ['Top' + str(x) for x in range(1, 26)]
        target_col = 'Momentum'

        res_x = self.db_tbl_price_news.find()
        res_x_full = [x for x in res_x]
        news_seq = [[r[colname]
                        for colname in news_col]
                        for r in res_x_full]
        ohlcv_ch = [[r[colname]
                        for colname in price_feature_col]
                        for r in res_x_full]
        target = [r[target_col] 
                        for r in res_x_full]

        sequence_length = timestep + 1
        ts_news_seq = []
        for index in range(len(news_seq) - sequence_length):
            ts_news_seq.append(news_seq[index: index + sequence_length])
        ts_news_seq = np.array(ts_news_seq)

        ts_ohlcv_ch = []
        for index in range(len(ohlcv_ch) - sequence_length):
            ts_ohlcv_ch.append(ohlcv_ch[index: index + sequence_length])
        ts_ohlcv_ch = np.array(ts_ohlcv_ch)

        ts_target = []
        for index in range(len(target) - sequence_length):
            ts_target.append(target[index: index + sequence_length])
        ts_target = np.array(ts_target)

        # 90% train, 10% test
        row = round(0.9 * ts_target.shape[0])
        train_ts_news_seq = ts_news_seq[:int(row), :]
        train_ts_ohlcv_ch = ts_ohlcv_ch[:int(row), :]
        train_ts_target = ts_target[:int(row), :]

        # shuffle
        rand_indices = np.arange(row)
        np.random.shuffle(rand_indices)
        train_ts_news_seq = train_ts_news_seq[rand_indices]
        train_ts_ohlcv_ch = train_ts_ohlcv_ch[rand_indices]
        train_ts_target = train_ts_target[rand_indices]

        x1_train = train_ts_news_seq[:, :-1]
        x2_train = train_ts_ohlcv_ch[:, :-1]
        y1_train = train_ts_target[:, -1]

        x1_test = ts_news_seq[int(row):, :-1]
        x2_test = ts_ohlcv_ch[int(row):, :-1]
        y1_test = ts_target[int(row):, -1]

        return x1_train, x2_train, y1_train, x1_test, x2_test, y1_test

    # def create_model(self):

    #     input_news_seq = Input(shape=(5, 25, DOC2VEC_DIM, ), name='top25_news_headline')   # encoded top 25 news
    #     input_djia_seq = Input(shape=(5, 6,), dtype='float32', name='ohlc_volume_label')   # open high low close volume adj close

    #     # construct the model
    #     news_seq = Sequential(name='news_lstm_ts5')
    #     # require Dense layer to flatten 5 dimension to 3 dimension
    #     news_seq.add(Dense(50, batch_input_shape=(None, 5, 25, DOC2VEC_DIM)))
    #     news_seq.add(TimeDistributed(Flatten()))
    #     news_seq.add(LSTM(64, return_sequences=False))
    #     # news_seq.add(Dropout(0.2))
    #     encoded_news = news_seq(input_news_seq)

    #     djia_seq = Sequential(name='ohlc_lstm_ts5')
    #     djia_seq.add(LSTM(64, return_sequences=False))
    #     # djia_seq.add(Dropout(0.2))
    #     encoded_djia = djia_seq(input_djia_seq)

    #     encoded_news = Dense(32)(encoded_news)
    #     encoded_djia = Dense(16)(encoded_djia)

    #     merge = concatenate([encoded_news, encoded_djia], name='news_price_concat')

    #     label = Dense(1)(merge)
    #     label = Dropout(0.2)(label)
    #     target_label = Activation('sigmoid', name='label_output')(label)

    #     target_price = Dense(32)(merge)
    #     target_price = Dense(1, name='price_output')(target_price)

    #     model = Model(inputs=[input_news_seq, input_djia_seq], outputs=[target_price, target_label])

    #     model.compile(optimizer='adam',
    #                   loss={'price_output': 'mean_squared_error', 'label_output': 'binary_crossentropy'},
    #                   metrics={'price_output': 'mse', 'label_output': 'accuracy'},
    #                   loss_weights={'price_output': 0, 'label_output': 1}
    #                 )
    #     return model

    # def train_model(self, model,
    #     newsseq_train, priceseq_train, tgt_price_train, tgt_label_train,
    #     newsseq_valid, priceseq_valid, tgt_price_valid, tgt_label_valid,
    #     newsseq_eval, priceseq_eval, tgt_price_eval, tgt_label_eval):

    #     x1 = np.array(newsseq_train)
    #     x2 = np.array(priceseq_train)
    #     y1 = np.array(tgt_price_train)
    #     y2 = np.array(tgt_label_train)

    #     vx1 = np.array(newsseq_valid)
    #     vx2 = np.array(priceseq_valid)
    #     vy1 = np.array(tgt_price_valid)
    #     vy2 = np.array(tgt_label_valid)

    #     ex1 = np.array(newsseq_eval)
    #     ex2 = np.array(priceseq_eval)
    #     ey1 = np.array(tgt_price_eval)
    #     ey2 = np.array(tgt_label_eval)

    #     history = model.fit([x1, x2], [y1, y2],
    #                 batch_size=12,
    #                 epochs=50,
    #                 validation_data=([vx1, vx2], [vy1, vy2]))

    #     plot_model(model, to_file='model_seq.png', show_shapes=True)

    #     r1, r2 = model.predict([x1, x2])

    #     plt.scatter(range(len(r1)), r1, c='r')
    #     plt.scatter(range(len(r1)), y1, c='b')
    #     plt.show()

    #     plt.scatter(range(len(r1)), r2, c='r')
    #     plt.scatter(range(len(r1)), y2, c='b')
    #     plt.show()

    #     r1, r2 = model.predict([ex1, ex2])

    #     plt.scatter(range(len(r1)), r1, c='r')
    #     plt.scatter(range(len(r1)), ey1, c='b')
    #     plt.show()

    #     plt.scatter(range(len(r1)), (r2 > 0.5).astype(int), c='r')
    #     plt.scatter(range(len(r1)), ey2, c='b')
    #     plt.show()
