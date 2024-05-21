# Value at Risk implementation in Python
# By Thomas McGeehan
# Uses yfinance to get historical stock data for a single ticker
# Variance-Covariance calculation of daily Value-at-Risk

import datetime
import numpy as np
import yfinance as yf
from scipy.stats import norm

def get_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock['rets'] = stock['Adj Close'].pct_change()
    return stock

def calculate_var(stock_data, portfolio_value, confidence_interval):
    """Calculate the Value-at-Risk (VaR) for a given stock data."""
    mu = np.mean(stock_data['rets'])
    sigma = np.std(stock_data['rets'])
    var = portfolio_value - portfolio_value * (norm.ppf(1 - confidence_interval, mu, sigma) + 1)
    return var

def main():
    # Define parameters
    ticker = 'WFC'
    start_date = '2010-01-01'
    end_date = '2016-01-01'
    portfolio_value = 1e6  # 1,000,000 USD
    confidence_interval = 0.99  # 99% confidence interval

    try:
        # Fetch stock data
        stock_data = get_stock_data(ticker, start_date, end_date)
        
        # Calculate VaR
        var = calculate_var(stock_data, portfolio_value, confidence_interval)
        
        # Output results
        print(f'Ticker: {ticker}')
        print(f"Value-at-Risk: ${var:,.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

