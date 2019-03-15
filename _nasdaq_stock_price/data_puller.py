from pandas_datareader import data

# Define which online source one should use
data_source = 'yahoo'

start_date = '2012-06-06'
end_date = '2018-12-31'

symbols = ['NDAQ','FB','AAPL','AMZN','NFLX','GOOG','BABA','MSFT','IBM','ORCL','INTC']
# symbols = ['NVDA']

for i in symbols:
    data.DataReader(i,'yahoo',start_date,end_date).to_csv('./data/' + i+'.csv')