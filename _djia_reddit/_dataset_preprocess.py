"""
Join DJIA price dataset with reddit news by common date

(Pandas.DataFrame) 1989 rows x 32 cols
"""

import pandas as pd
import re


def clean_news_string(news_str):
    """ function to clean reddit news string """
    if not type(news_str) is str:
        return ''
    news_str = re.sub(r"[\r\n]+", " ", news_str)
    news_str = re.sub(r"^\s+", "", news_str)
    news_str = news_str.replace("\"b'", "\"'")
    news_str = news_str.replace("\"b\"", "\"\"")
    news_str = news_str.replace("b'", "'")
    news_str = news_str.replace("b\"", "\"")
    news_str = re.sub(r"[ \t]+", " ", news_str)
    news_str = re.sub(r"'s\s", " ", news_str)
    news_str = re.sub(r"\\r\\n", " ", news_str)
    news_str = re.sub(r"(\\n|\\t|\\r)", " ", news_str)
    news_str = re.sub(r"\\", " ", news_str)
    news_str = re.sub("(\"|')", " ", news_str)
    news_str = re.sub(r"[ ]+", " ", news_str)
    news_str = re.sub(r"&amp;", "&", news_str)
    news_str = re.sub(r"\s+$", "", news_str)
    http_ptrn = r"https?\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%\$#_=]*)?"
    news_str = re.sub(http_ptrn, "http_url", news_str)
    news_str = re.sub(r'\w+\.com(\.\w+)?', "site_url", news_str)
    news_str = re.sub(r"\.\.+", " ", news_str)

    news_str_arr = news_str.split("|")
    news_str = max(news_str_arr, key=len)
    news_str = re.sub(r"[\-]+", "-", news_str)
    news_str = re.sub(r"\[", " [ ", news_str)
    news_str = re.sub(r"\]", " ] ", news_str)
    news_str = re.sub(r"[ ]+", " ", news_str)
    news_str = re.sub(r"(^\s+|\s+$)", "", news_str)

    return news_str


# Clean DJIA price news
DJIA_PRICE_NEWS = pd.read_csv('data/DJIA_PRICE_NEWS.csv', index_col='Date')
news_seq_colname = ['Top' + str(x) for x in range(1, 26)]
for news_col in news_seq_colname:
    DJIA_PRICE_NEWS[news_col] = DJIA_PRICE_NEWS[news_col].apply(clean_news_string)
DJIA_PRICE_NEWS.to_csv('data/DJIA_PRICE_NEWS.csv')

# Clean reddit news
DJIA_NEWS = pd.read_csv('data/RedditNews.csv', index_col='Date')
DJIA_NEWS['News'] = DJIA_NEWS['News'].apply(clean_news_string)
DJIA_NEWS.to_csv('data/RedditNews_Proc.csv')
