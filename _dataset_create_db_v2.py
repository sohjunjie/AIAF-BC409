import pandas as pd
from pymongo import MongoClient
import utils


client = MongoClient('localhost', 27017)
db = client.djia_news_dataset
db_tbl_price_news = db.price_news_v2


DJIA_PRICE_NEWS = pd.read_csv('data/DJIA_PRICE_NEWS.csv', index_col='Date')
DJIA_PRICE_NEWS = utils.standardize_features(DJIA_PRICE_NEWS, ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
doc2vec = utils.load_doc2vec()

# Generate the new dataset with word vector representation of news headline
dataset_topnews_colname = ['Top'+str(i) for i in range(1, 26)]
dataset_list = []

k = 1
for idx, item in DJIA_PRICE_NEWS.iterrows():

    df_item = {}
    df_item['_id'] = idx
    df_item['Date'] = idx
    df_item['Open'] = item['Open']
    df_item['High'] = item['High']
    df_item['Low'] = item['Low']
    df_item['Close'] = item['Close']
    df_item['Volume'] = item['Volume']
    df_item['Adj Close'] = item['Adj Close']
    for topnews_colname in dataset_topnews_colname:
        df_item[topnews_colname] = utils.process_document(item[topnews_colname], doc2vec)

    dataset_list.append(df_item)

    if k % 50 == 0 or k == len(DJIA_PRICE_NEWS):
        result = db_tbl_price_news.insert_many(dataset_list)
        print('{0} of {1}'.format(k, len(DJIA_PRICE_NEWS)))
        dataset_list.clear()
    k += 1
