import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from numpy.fft import rfft
import sys
import logging
from numba import jit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@jit(nopython=True)
def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

def apply_fft(prices):
    fft_vals = rfft(prices)
    return np.vstack((np.real(fft_vals), np.imag(fft_vals))).T

def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def fetch_stock_data(ticker):
    logging.info("Fetching stock data...")
    return yf.download(ticker, start='2000-01-01', end=None)

def prepare_data(stock_data):
    logging.info("Calculating EMA...")
    stock_data['EMA'] = calculate_ema(stock_data['Close'].values, span=10)
    stock_data.dropna(inplace=True)

    stock_prices = stock_data['Close'].values.reshape(-1, 1)
    ema_prices = stock_data['EMA'].values.reshape(-1, 1)

    logging.info("Applying FFT...")
    fft_features = apply_fft(stock_prices.flatten())
    
    # Truncate stock_prices and ema_prices to match the length of fft_features
    min_len = min(len(stock_prices), len(fft_features))
    stock_prices = stock_prices[:min_len]
    ema_prices = ema_prices[:min_len]
    fft_features = fft_features[:min_len]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(np.hstack((stock_prices, ema_prices, fft_features)))

    return scaled_features, scaler

def split_data(scaled_features, time_step):
    train_size = int(len(scaled_features) * 0.8)
    train_data, test_data = scaled_features[:train_size, :], scaled_features[train_size:, :]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    return X_train, y_train, X_test, y_test

def build_and_train_model(X_train, y_train, X_test, y_test):
    logging.info("Building and training the model...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=50, batch_size=64, 
              validation_data=(X_test, y_test), verbose=1, 
              callbacks=[early_stopping])
    
    return model

def make_predictions(model, X_test, scaler, scaled_features):
    logging.info("Making predictions...")
    predictions = model.predict(X_test, batch_size=64)

    predictions_stock = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((predictions.shape[0], scaled_features.shape[1]-1))))
    )[:, 0]
    
    return predictions_stock

def evaluate_model(y_test, predictions_stock, scaler, scaled_features):
    y_test_stock = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_features.shape[1]-1))))
    )[:, 0]

    mae = mean_absolute_error(y_test_stock, predictions_stock)
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")

def main(ticker):
    stock_data = fetch_stock_data(ticker)
    scaled_features, scaler = prepare_data(stock_data)
    time_step = 100
    X_train, y_train, X_test, y_test = split_data(scaled_features, time_step)
    model = build_and_train_model(X_train, y_train, X_test, y_test)
    predictions_stock = make_predictions(model, X_test, scaler, scaled_features)
    evaluate_model(y_test, predictions_stock, scaler, scaled_features)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stocks.py <ticker>")
        sys.exit(1)
    ticker = sys.argv[1]
    main(ticker)
