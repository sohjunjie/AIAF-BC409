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

k = 0
for idx, item in DJIA_PRICE_NEWS.iterrows():

    df_item = {}
    df_item['Date'] = idx
    df_item['Adj Close'] = item['Adj Close']
    df_item['Volume'] = item['Volume']
    for topnews_colname in dataset_topnews_colname:
        df_item[topnews_colname] = utils.process_sentence(item[topnews_colname], word2vec)

    dataset_list.append(df_item)

    if k % 100 == 0 or k == len(DJIA_PRICE_NEWS)-1:
        result = db_tbl_price_news.insert_many(dataset_list)
        print('Multiple insert: {0}'.format(result.inserted_ids))
        dataset_list.clear()
    k += 1


# dataset = pd.DataFrame(data=dataset_list, columns=['Date', 'Adj Close','Volume'] + dataset_topnews_colname)
# dataset.set_index('Date', inplace=True)

# print(dataset_list)


# time_step = 10
# for x in range(len(dataset) - time_step):

#     train_seq = dataset[x:x+time_step]
#     train_y = dataset.iloc[x+time_step, 5]
#     train_x1 = train_seq.iloc[:, 4:6]               # volume, adj close
#     train_x2 = train_seq.iloc[:, 7:32]              # top 1 - top 25 news

#     print(train_x2)

#     break
