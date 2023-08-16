"""
Author: Arthur Sweetman

Start Date: August 9, 2023

This program is meant to be a predictive robot for the stock market to aid
(and hopefully drive) financial gains.

example1 taken from: https://neptune.ai/blog/predicting-stock-prices-using-machine-learning

"""
import keras.optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import Input, Model, Sequential, optimizers
from keras.layers import LSTM, Dense, Dropout
import pickle
import stock_API


# ------------ Global variables ------------
predictor_vars = ['Close', 'volume', 'VIX', 'USDX', 'EFFR', 'UNRATE', 'UMCSENT']
window_size = 50
N_forecast = 20
USE_CACHED_MODEL = False

cur_LSTM_args = {
    "units": 150,
    "optimizer": "adagrad",
    "batch_size": 16,
    "epochs": 2
}

# ------------ Train-Test split for time-series ------------
stockprices = stock_API.data
stockprices.rename(columns = {'adjclose' : 'Close'}, inplace = True)
stockprices = stockprices[predictor_vars]

test_ratio = 0.01
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

train = stockprices[:train_size]
test = stockprices[train_size:]


# ------------ Functions to be used ------------
def extract_seqX_outcomeY(data, N_lookback):
    """
    Split time-series into training sequence X and outcome values Y
    Args:
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
        (discontinued) N_forecast - number of days to predict in the future
        (discontinued) offset - position to start the split
    """
    X, y = [], []

    for i in range(N_lookback, len(data)):
        X.append(data[i - N_lookback : i])
        y.append(data.iloc[i]['Close'])

    return np.array(X), np.array(y)


# ------------ Scale our dataset ------------
scalers = {}
scaled_data = pd.DataFrame()
for col in stockprices:  # Consider scaling only to training data
    scaler = MinMaxScaler().fit(stockprices[[col]])
    scaled_data[[col]] = scaler.transform(stockprices[[col]])
    scalers[col] = scaler

scaled_data_train = scaled_data[: train.shape[0]]

# We use past 50 days’ stock prices for our training to predict the 51st day's closing price.
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size)

# ------------ Build a LSTM model ------------
def Run_LSTM(X_train, layer_units=cur_LSTM_args['units']):

    inp = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # opt = optimizers.Adagrad(learning_rate=0.01)  # Adagrad optimizer based on published paper
    # Compile the LSTM neural net
    model.compile(loss="mean_squared_error", optimizer='adam')

    return model


if USE_CACHED_MODEL:
    with open('./LSTM_model', 'rb') as file:
        history = pickle.load(file)
        model = history.model
else:
    model = Run_LSTM(X_train)
    history = model.fit(
        X_train,
        y_train,
        epochs=cur_LSTM_args['epochs'],
        batch_size=cur_LSTM_args['batch_size'],
        verbose=1,
        validation_split=0.1,
    )
    with open('./LSTM_model', 'wb') as file:
        pickle.dump(history, file)

# ------------ predict stock prices using past window_size stock prices ------------
scaled_data_test = scaled_data[-test.shape[0]-window_size:]
X_test, y_test = extract_seqX_outcomeY(scaled_data_test, N_lookback=window_size)

predicted_price_ = model.predict(X_test)
predicted_price = pd.DataFrame()
predicted_price = scalers['Close'].inverse_transform(predicted_price_)

# ------------ Plot predicted prices ------------

plt.figure()
plt.plot(predicted_price, ':', label = 'LSTM')
plt.plot(test['Close'].values, '--', label = 'Actual')
plt.legend()
plt.savefig("predicted_prices_1d")

# ------------ Calculate and plot forecasted prices ------------

print("Calculating forecasts...")
X_forecast = np.array(scaled_data_test.copy())[:window_size+100]  # Create a copy of the scaled test data to start with
actual = np.array(scaled_data_test.copy())[:window_size+100]
for i in range(window_size, len(X_forecast)):  # Iterate through each day
    t = X_forecast[i-window_size:i]
    X_forecast_input = t.reshape((1, window_size, X_forecast.shape[1]))
    X_forecast[i] = model.predict(X_forecast_input)  # Assign the next day's predicted price in the corrresponding position

X_forecast = scalers['Close'].inverse_transform(X_forecast)
actual = scalers['Close'].inverse_transform(actual)
plt.figure()
plt.plot(X_forecast[:,0], ':', label = 'LSTM')
plt.plot(actual[:,0], '--', label = 'Actual')
plt.legend()
plt.savefig("predicted_prices_forecast")

# preds = pd.DataFrame({
#     'Close': test['Close'],
#     'volume': test['volume'],
#     'pred': predicted_price.reshape(-1)
# }).to_csv("predictions.csv")

print("Done!")