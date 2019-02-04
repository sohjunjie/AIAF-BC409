import pandas as pd
from pymongo import MongoClient
import utils


client = MongoClient('localhost', 27017)
db = client.djia_news_dataset
db_tbl_price_news = db.price_news


DJIA_PRICE_NEWS = pd.read_csv('data/DJIA_PRICE_NEWS.csv', index_col='Date')
DJIA_PRICE_NEWS = utils.standardize_features(DJIA_PRICE_NEWS, ['Adj Close', 'Volume'])
word2vec = utils.load_word2vec()

# Generate the new dataset with word vector representation of news headline
dataset_topnews_colname = ['Top'+str(i) for i in range(1, 26)]
dataset_list = []

k = 1
for idx, item in DJIA_PRICE_NEWS.iterrows():

    df_item = {}
    df_item['_id'] = idx
    df_item['Date'] = idx
    df_item['Adj Close'] = item['Adj Close']
    df_item['Volume'] = item['Volume']
    for topnews_colname in dataset_topnews_colname:
        df_item[topnews_colname] = utils.process_sentence(item[topnews_colname], word2vec)

        tmp = len(df_item[topnews_colname])

    dataset_list.append(df_item)

    if k % 50 == 0 or k == len(DJIA_PRICE_NEWS):
        result = db_tbl_price_news.insert_many(dataset_list)
        print('{0} of {1}'.format(k, len(DJIA_PRICE_NEWS)))
        dataset_list.clear()
    k += 1

word2vec.save('model/text8_gs.bin')

# dataset = pd.DataFrame(data=dataset_list, columns=['Date', 'Adj Close','Volume'] + dataset_topnews_colname)
# dataset.set_index('Date', inplace=True)

# time_step = 10
# for x in range(len(1989) - time_step):
#     train_y = db_tbl_price_news.find()[x+10]['Adj Close']
#     train_seq = [x for x in db_tbl_price_news.find()[x:x+10]]
