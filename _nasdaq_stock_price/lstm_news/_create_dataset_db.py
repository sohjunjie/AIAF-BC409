import pandas as pd
import utils

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from pymongo import MongoClient

from _config import DATA_FOLDER

from _config import PRICE_SCALER_LOC, AAPL_PRICE_SCALER_LOC, AMZN_PRICE_SCALER_LOC, \
    FB_PRICE_SCALER_LOC, GOOG_PRICE_SCALER_LOC, IBM_PRICE_SCALER_LOC, MSFT_PRICE_SCALER_LOC, \
    NFLX_PRICE_SCALER_LOC, ORCL_PRICE_SCALER_LOC, INTC_PRICE_SCALER_LOC

from _config import VOLUME_SCALER_LOC, AAPL_VOLUME_SCALER_LOC, AMZN_VOLUME_SCALER_LOC, \
    FB_VOLUME_SCALER_LOC, GOOG_VOLUME_SCALER_LOC, IBM_VOLUME_SCALER_LOC, MSFT_VOLUME_SCALER_LOC, \
    NFLX_VOLUME_SCALER_LOC, ORCL_VOLUME_SCALER_LOC, INTC_VOLUME_SCALER_LOC

from _config import CHANGE_SCALER_LOC, AAPL_CHANGE_SCALER_LOC, AMZN_CHANGE_SCALER_LOC, \
    FB_CHANGE_SCALER_LOC, GOOG_CHANGE_SCALER_LOC, IBM_CHANGE_SCALER_LOC, MSFT_CHANGE_SCALER_LOC, \
    NFLX_CHANGE_SCALER_LOC, ORCL_CHANGE_SCALER_LOC, INTC_CHANGE_SCALER_LOC


client = MongoClient('localhost', 27017)
db = client.djia_news_dataset
db_tbl_price_news = db.price_news_v1

PRICE_NEWS = pd.read_csv(DATA_FOLDER + 'combined_nasdaq_news.csv', index_col='Date')


def normalize_dataset_values(dataset):
    coy_prefix_ls = ['', 'AAPL_', 'AMZN_', 'FB_', 'GOOG_', 'IBM_', 'MSFT_', 'NFLX_', 'ORCL_', 'INTC_']
    coy_price_scaler_loc_ls = [PRICE_SCALER_LOC, AAPL_PRICE_SCALER_LOC, AMZN_PRICE_SCALER_LOC,
                               FB_PRICE_SCALER_LOC, GOOG_PRICE_SCALER_LOC, IBM_PRICE_SCALER_LOC,
                               MSFT_PRICE_SCALER_LOC, NFLX_PRICE_SCALER_LOC, ORCL_PRICE_SCALER_LOC,
                               INTC_PRICE_SCALER_LOC]
    coy_volume_scaler_loc_ls = [VOLUME_SCALER_LOC, AAPL_VOLUME_SCALER_LOC, AMZN_VOLUME_SCALER_LOC,
                                FB_VOLUME_SCALER_LOC, GOOG_VOLUME_SCALER_LOC, IBM_VOLUME_SCALER_LOC,
                                MSFT_VOLUME_SCALER_LOC, NFLX_VOLUME_SCALER_LOC, ORCL_VOLUME_SCALER_LOC,
                                INTC_VOLUME_SCALER_LOC]
    coy_change_scaler_loc_ls = [CHANGE_SCALER_LOC, AAPL_CHANGE_SCALER_LOC, AMZN_CHANGE_SCALER_LOC,
                                FB_CHANGE_SCALER_LOC, GOOG_CHANGE_SCALER_LOC, IBM_CHANGE_SCALER_LOC,
                                MSFT_CHANGE_SCALER_LOC, NFLX_CHANGE_SCALER_LOC, ORCL_CHANGE_SCALER_LOC,
                                INTC_CHANGE_SCALER_LOC]

    price_scaler_ls = []
    volume_scaler_ls = []
    change_scaler_ls = []

    for i in range(len(coy_prefix_ls)):
        priceScaler = StandardScaler()
        priceScaler.fit(dataset[coy_prefix_ls[i] + 'Close'].values.reshape(-1,1))
        joblib.dump(priceScaler, coy_price_scaler_loc_ls[i])
        price_scaler_ls.append(priceScaler)

    for i in range(len(coy_prefix_ls)):
        volumeScaler = StandardScaler()
        volumeScaler.fit(dataset[coy_prefix_ls[i] + 'Volume'].values.reshape(-1,1))
        joblib.dump(volumeScaler, coy_volume_scaler_loc_ls[i])
        volume_scaler_ls.append(volumeScaler)

    for i in range(len(coy_prefix_ls)):
        changeScaler = StandardScaler()
        changeScaler.fit(dataset[coy_prefix_ls[i] + 'Change'].values.reshape(-1,1))
        joblib.dump(changeScaler, coy_change_scaler_loc_ls[i])
        change_scaler_ls.append(changeScaler)

    for i in range(len(coy_prefix_ls)):
        dataset = utils.scale_features(dataset,
                                          [coy_prefix_ls[i] + 'Open',
                                           coy_prefix_ls[i] + 'High',
                                           coy_prefix_ls[i] + 'Low',
                                           coy_prefix_ls[i] + 'Close'],
                                          price_scaler_ls[i])

        dataset = utils.scale_features(dataset,
                                          [coy_prefix_ls[i] + 'Volume'],
                                          volume_scaler_ls[i])

        dataset = utils.scale_features(dataset,
                                          [coy_prefix_ls[i] + 'Change'],
                                          change_scaler_ls[i])


normalize_dataset_values(PRICE_NEWS)

doc2vec = utils.load_doc2vec()

# Generate the new dataset with word vector representation of news headline
dataset_topnews_colname = ['Top'+str(i) for i in range(1, 26)]
coy_prefix_ls = ['', 'AAPL_', 'AMZN_', 'FB_', 'GOOG_', 'IBM_', 'MSFT_', 'NFLX_', 'ORCL_', 'INTC_']
dataset_list = []

k = 1
for idx, item in PRICE_NEWS.iterrows():

    df_item = {}
    df_item['_id'] = idx
    df_item['Date'] = idx
    df_item['Momentum'] = item['Momentum']

    for coy_prefix in coy_prefix_ls:
        df_item[coy_prefix + 'Open'] = item[coy_prefix + 'Open']
        df_item[coy_prefix + 'High'] = item[coy_prefix + 'High']
        df_item[coy_prefix + 'Low'] = item[coy_prefix + 'Low']
        df_item[coy_prefix + 'Close'] = item[coy_prefix + 'Close']
        df_item[coy_prefix + 'Volume'] = item[coy_prefix + 'Volume']
        df_item[coy_prefix + 'Change'] = item[coy_prefix + 'Change']

    for topnews_colname in dataset_topnews_colname:
        df_item[topnews_colname] = utils.process_document(item[topnews_colname], doc2vec)

    dataset_list.append(df_item)

    if k % 50 == 0 or k == len(PRICE_NEWS):
        result = db_tbl_price_news.insert_many(dataset_list)
        print('{0} of {1}'.format(k, len(PRICE_NEWS)))
        dataset_list.clear()
    k += 1
