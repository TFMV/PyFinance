# Value at Risk implementation in python
# By Thomas McGeehan
# Uses yahoo finance to get historical stock data for a single ticker
# Variance-Covariance calculation of daily Value-at-Risk
import datetime
import numpy as np
from scipy.stats import norm
import pandas_datareader.data as web

ticker = 'WFC'
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2016, 1, 1)

stock = web.DataReader(ticker, 'yahoo', start, end) 
stock['rets'] = stock['Adj Close'].pct_change()

P = 1e6   # 1,000,000 USD
c = 0.99  # 99% confidence interval
mu = np.mean(stock['rets'])
sigma = np.std(stock['rets'])
var = P - P*(norm.ppf(1-c, mu, sigma) + 1)
print 'Ticker: ', ticker 
print "Value-at-Risk: $%0.3f" % var 
