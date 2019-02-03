"""
Join DJIA price dataset with reddit news by common date

(Pandas.DataFrame) 1989 rows x 32 cols
"""

import pandas as pd

DJIA_NEWS = pd.read_csv('data/Combined_News_DJIA.csv', index_col='Date')
DJIA_PRICE = pd.read_csv('data/DJIA_table.csv', index_col='Date')
DJIA_PRICE_NEWS = pd.merge(DJIA_NEWS, DJIA_PRICE, left_index=True, right_index=True)
DJIA_PRICE_NEWS.to_csv('data/DJIA_WITH_NEWS.csv')

DJIA_PRICE_NEWS = pd.read_csv('data/DJIA_WITH_NEWS.csv', index_col='Date')
