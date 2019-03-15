import pandas as pd

data_foldername = "./data/"

# not using alibaba cause its only public in 2014
symbols = ['NDAQ','FB','AAPL','AMZN','NFLX','GOOG','MSFT','IBM','ORCL','INTC']
originalfile_postfix = ".csv"
cleanedfile_postfix = "_cleaned.csv"


def generate_change_momentum(symbol, foldername):
    target_file = pd.read_csv(foldername + symbol + originalfile_postfix)
    date = target_file['Date']
    close = target_file['Close']
    change = [0] * len(date)
    momentum = [0] * len(date)

    for i in range(1, len(date)):
        change[i] = (close[i] - close[i - 1]) / close[i - 1]

    for j in range(0, len(date)-1):
        if close[j] > close[j + 1]:
            momentum[j] = "0"
        else:
            momentum[j] = "1"



    dataframe = pd.DataFrame({
        'Date': target_file['Date'],
        'High': target_file['High'],
        'Low': target_file['Low'],
        'Open': target_file['Open'],
        'Close': target_file['Close'],
        'Volume': target_file['Volume'],
        'Change': change,
        'Momentum': momentum
    })

    dataframe.to_csv(foldername + symbol + cleanedfile_postfix, index=False, header=True)


def combine_cleaned_files_nasdaq(companies, foldername):
    dataframe = pd.read_csv(foldername + companies[0] + cleanedfile_postfix)

    i = 1
    while i < len(companies):
        print(companies[i])
        target_file = pd.read_csv(foldername + companies[i] + cleanedfile_postfix)
        headers = list(target_file)

        j = 1
        # -1 to remove the momentum from those classes
        while j < len(headers) - 1:
            dataframe[companies[i]+"_" + headers[j]] = target_file[headers[j]]
            j += 1

        i += 1

    dataframe.drop(dataframe.index[0], inplace=True)
    dataframe.drop(dataframe.index[-1:], inplace=True)

    dataframe.to_csv(foldername + "combined_nasdaq.csv", index=False, header=True)


def combine_cleaned_files_top9(companies, foldername):
    dataframe = pd.read_csv(foldername + companies[0] + cleanedfile_postfix)

    i = 1
    while i < len(companies):
        print(companies[i])
        target_file = pd.read_csv(foldername + companies[i] + cleanedfile_postfix)
        headers = list(target_file)

        j = 1
        # -1 to remove the momentum from those classes
        while j < len(headers):
            dataframe[companies[i]+"_" + headers[j]] = target_file[headers[j]]
            j += 1

        i += 1

    dataframe.drop(dataframe.index[0], inplace=True)
    dataframe.drop(dataframe.index[-1:], inplace=True)
    dataframe.drop(columns=["Momentum"], inplace=True)

    dataframe.to_csv(foldername + "combined_top9.csv", index=False, header=True)


def combine_cleaned_files_with_news(foldername):
    NASDAQ_PRICE = pd.read_csv(foldername + "combined_nasdaq.csv", index_col="Date")
    NASDAQ_NEWS = pd.read_csv(foldername + 'RedditNews.csv', index_col='Date')
    NASDAQ_PRICE_NEWS = pd.merge(NASDAQ_NEWS, NASDAQ_PRICE, left_index=True, right_index=True)
    NASDAQ_PRICE_NEWS.to_csv(foldername + 'combined_nasdaq_news.csv')


for company in symbols:
    generate_change_momentum(company, data_foldername)

combine_cleaned_files_nasdaq(symbols, data_foldername)
combine_cleaned_files_top9(symbols, data_foldername)
# combine_cleaned_files_with_news(data_foldername)