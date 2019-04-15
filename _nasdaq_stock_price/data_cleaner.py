import pandas as pd
from ta import *

data_foldername = "./data/"

# not using alibaba cause its only public in 2014
symbols = ['NDAQ','FB','AAPL','AMZN','NFLX','GOOG','MSFT','IBM','ORCL','INTC']
originalfile_postfix = ".csv"
cleanedfile_postfix = "_cleaned.csv"

# how many days ahead to predict
trend_days = 10



def generate_ta(symbol, foldername):
    target_file = pd.read_csv(foldername + symbol + originalfile_postfix)
    date = target_file['Date']
    close = target_file['Close']
    change = [0] * len(date)
    trend = [0] * len(date)

    for j in range(0, len(date) - trend_days):
        if close[j] > close[j + trend_days]:
            trend[j] = "0"
        else:
            trend[j] = "1"



    dataframe = pd.DataFrame({
        'Date': target_file['Date'],
        'High': target_file['High'],
        'Low': target_file['Low'],
        'Open': target_file['Open'],
        'Close': target_file['Close'],
        'Volume': target_file['Volume'],
        'Trend_10': trend
    })

    dataframe = add_all_ta_features(dataframe, "Open", "High", "Low", "Close", "Volume", fillna=True)

    # remove top 35 and last row as cleanup
    # top 35 for some of the variables generated in TA
    # last row for the target
    dataframe = dataframe[35:-trend_days]
    dataframe.to_csv(foldername + symbol + cleanedfile_postfix, index=False, header=True)


def combine_cleaned_files_nasdaq(companies, foldername):
    dataframe = pd.read_csv(foldername + companies[0] + cleanedfile_postfix)

    i = 1
    while i < len(companies):
        print(companies[i])
        target_file = pd.read_csv(foldername + companies[i] + cleanedfile_postfix)
        headers = list(target_file)

        j = 1
        # -1 to remove the Y from those classes
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
    dataframe.drop(columns=["Trend_10"], inplace=True)

    dataframe.to_csv(foldername + "combined_top9.csv", index=False, header=True)


for company in symbols:
    generate_ta(company, data_foldername)

combine_cleaned_files_nasdaq(symbols, data_foldername)
combine_cleaned_files_top9(symbols, data_foldername)
