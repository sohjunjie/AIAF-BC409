import pandas as pd
import utils

time_step = 10

dataset = pd.read_csv('data/DJIA_WITH_NEWS.csv', index_col='Date')
dataset = utils.standardize_features(dataset, ['Adj Close', 'Volume'])


for x in range(len(dataset) - time_step):

    train_seq = dataset[x:x+time_step]
    train_y = dataset.iloc[x+time_step, 5]
    train_x1 = train_seq.iloc[:, 4:6]               # volume, adj close
    train_x2 = train_seq.iloc[:, 7:32]              # top 1 - top 25 news
