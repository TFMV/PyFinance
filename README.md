# PyFinance

PyFinance is a project that provides tools for financial analysis and forecasting. It includes implementations for predicting stock prices using deep learning and calculating the Value at Risk (VaR) of a financial portfolio.

![PyFinance](assets/PyFinance.webp)

## Features

### Stock Price Prediction

- **Historical Data Fetching:** Uses `yfinance` to fetch historical stock data from Yahoo Finance.
- **Data Preprocessing:** Applies EMA and FFT transformations to preprocess the data.
- **Hybrid CNN-LSTM Model:** Combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for time series forecasting.
- **Model Evaluation:** Evaluates the model using Mean Absolute Error (MAE).

### Value at Risk (VaR) Calculation

- **Historical Data Fetching:** Uses `yfinance` to fetch historical stock data from Yahoo Finance.
- **Variance-Covariance VaR Calculation:** Calculates daily VaR at a specified confidence interval.

## License

This project is licensed under the MIT License

## Author

Thomas F McGeehan V
