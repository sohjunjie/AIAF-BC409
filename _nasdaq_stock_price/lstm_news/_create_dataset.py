import pandas as pd
from _config import DATA_FOLDER


NASDAQ_PRICE = pd.read_csv(DATA_FOLDER + "combined_nasdaq.csv", index_col="Date")
NASDAQ_NEWS = pd.read_csv(DATA_FOLDER + 'RedditNews_proc.csv', index_col='Date')
NASDAQ_PRICE_NEWS = pd.merge(NASDAQ_NEWS, NASDAQ_PRICE, left_index=True, right_index=True)
NASDAQ_PRICE_NEWS.to_csv(DATA_FOLDER + 'combined_nasdaq_news.csv')
