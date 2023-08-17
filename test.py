"""
Script used for testing the model on other stocks
"""

import stock_API
import pickle
import numpy as np
import pandas as pd
from plots import tomorrow_pred_v_today_close

def predict_prices(data, model, N_lookback):
    """

    Args:
        data: Uncaled data to predict prices for
        model: Model to use for predictions
        N_lookback: Number of days to use for predictions

    Returns:

    """

    scaled_data = scale_data(data)

    X, y = extract_seqX_outcomeY(scaled_data, N_lookback)

    scaled_predictions = model.predict(X)
    unscaled_predictions = unscale_data(scaled_data = scaled_predictions)
    predicted_price = pd.DataFrame(unscaled_predictions, index=data.iloc[N_lookback:].index)

    tomorrow_pred_v_today_close(data.iloc[-200:], predicted_price.iloc[-200:])

    return predicted_price


def extract_seqX_outcomeY(data, N_lookback):
    """
    Split time-series into training sequence X and outcome values Y
    Args:
        data: dataset
        N_lookback: window size, e.g., 50 for 50 days of historical stock prices

    Returns:
        An n-dimensional array of past N_lookback days with all predictor vars and an array of all next-day closing prices
    """
    X, y = [], []

    for i in range(N_lookback, len(data)):
        X.append(data[i - N_lookback : i])
        y.append(data.iloc[i]['Close'])

    return np.array(X), np.array(y)


def scale_data(unscaled_data):
    with open('./scalers', 'rb') as file:
        scalers = pickle.load(file)

    scaled_data = pd.DataFrame()
    for col in unscaled_data:  # Consider scaling only to training data
        # Also consider scaling to an exponential distribution (for closing price)
        scaler = scalers[col]
        scaled_data[[col]] = scaler.transform(unscaled_data[[col]])

    scaled_data.index = unscaled_data.index
    return scaled_data


def unscale_data(scaled_data):
    with open('./scalers', 'rb') as file:
        scalers = pickle.load(file)

    return scalers['Close'].inverse_transform(scaled_data)


if __name__ == "__main__":

    ticker = "AAPL"

    # Load in the last trained model that was created in index.py
    with open('./LSTM_model', 'rb') as file:
        history = pickle.load(file)
        model = history.model

    predictor_vars = ['Close', 'volume', 'VIX', 'USDX', 'EFFR', 'UNRATE', 'UMCSENT', 'MACD', 'ATR', 'RSI']
    window_size = 50

    stockprices = stock_API.retreive_data(ticker, "01/01/2001")
    stockprices.rename(columns={'adjclose': 'Close'}, inplace=True)
    stockprices = stockprices[predictor_vars]

    predict_prices(stockprices, model, window_size)