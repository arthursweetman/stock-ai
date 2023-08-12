"""
Author: Arthur Sweetman

Start Date: August 9, 2023

This program is meant to be a predictive robot for the stock market to aid
(and hopefully drive) financial gains.

example1 taken from: https://neptune.ai/blog/predicting-stock-prices-using-machine-learning

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neptune
from sklearn.preprocessing import StandardScaler
from neptune.integrations.tensorflow_keras import NeptuneCallback
from keras import Input, Model
from keras.layers import LSTM, Dense
import pickle

window_size = 100
N_forecast = 100
USE_CACHED_MODEL = False
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NzExZDczNS1hOGFhLTQ2NjItYjNjOS1hMjc4MzRlM2NmYmIifQ=="
project = neptune.init_project(project="arthursweetman/stock-ai", api_token=NEPTUNE_API_TOKEN)
myProject = "arthursweetman/stock-ai"
# run = neptune.init_run(project="arthursweetman/stock-ai", api_token=NEPTUNE_API_TOKEN)

# %% Train-Test split for time-series
stockprices = pd.read_csv("AAPL.csv", index_col="Date")

test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

train = stockprices[:train_size][["Close"]]
test = stockprices[train_size:][["Close"]]


def extract_seqX_outcomeY(data, N_lookback, N_forecast):
    """
    Split time-series into training sequence X and outcome values Y
    Args:
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
        N_forecast - number of days to predict in the future
        (discontinued) offset - position to start the split
    """
    X, y = [], []

    for i in range(N_lookback, len(data) - N_forecast + 1):
        X.append(data[i - N_lookback : i])
        y.append(data[i : i + N_forecast])

    return np.array(X), np.array(y)


# -------------- Calculate the metrics' RMSE and MAPE --------------


def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def calculate_perf_metrics(var):
    ### RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )
    ### MAPE
    mape = calculate_mape(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )

    ## Log to Neptune
    run["RMSE"] = rmse
    run["MAPE (%)"] = mape

    return rmse, mape


def plot_stock_trend(var, cur_title, stockprices=stockprices):
    ax = stockprices[["Close", var, "200day"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

    ## Log to Neptune
    run["Plot of Stock Predictions"].upload(
        neptune.types.File.as_image(ax.get_figure())
    )


################### # Initialize a Neptune run
# run = neptune.init_run(
#     project=myProject,
#     name="SMA",
#     description="stock-prediction-machine-learning",
#     tags=["stockprediction", "MA_Simple", "neptune"],
#     api_token=NEPTUNE_API_TOKEN
# )
#
# window_var = f"{window_size}day"
#
# stockprices[window_var] = stockprices["Close"].rolling(window_size).mean()
#
# ### Include a 200-day SMA for reference
# stockprices["200day"] = stockprices["Close"].rolling(200).mean()
#
# ### Plot and performance metrics for SMA model
# plot_stock_trend(var=window_var, cur_title="Simple Moving Averages")
# rmse_sma, mape_sma = calculate_perf_metrics(var=window_var)
#
# ### Stop the run
# run.stop()
#
# # Initialize a Neptune run
# run = neptune.init_run(
#     project=myProject,
#     name="EMA",
#     description="stock-prediction-machine-learning",
#     tags=["stockprediction", "MA_Exponential", "neptune"],
#     api_token=NEPTUNE_API_TOKEN
# )
#
# ###### Exponential MA
# window_ema_var = f"{window_var}_EMA"
#
# # Calculate the 50-day exponentially weighted moving average
# stockprices[window_ema_var] = (
#     stockprices["Close"].ewm(span=window_size, adjust=False).mean()
# )
# stockprices["200day"] = stockprices["Close"].rolling(200).mean()
#
# ### Plot and performance metrics for EMA model
# plot_stock_trend(
#     var=window_ema_var, cur_title="Exponential Moving Averages")
# rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var)
#
# ### Stop the run
# run.stop()

############################

layer_units = 50
optimizer = "adam"
cur_epochs = 10  # Previously set to 15
cur_batch_size = 32  # Previously set to 20

cur_LSTM_args = {
    "units": layer_units,
    "optimizer": optimizer,
    "batch_size": cur_batch_size,
    "epochs": cur_epochs,
}

# Initialize a Neptune run
run = neptune.init_run(
    project=myProject,
    name="LSTM",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "LSTM", "neptune"],
    api_token=NEPTUNE_API_TOKEN
)
run["LSTM_args"] = cur_LSTM_args

# Scale our dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[["Close"]])
scaled_data_train = scaled_data[: train.shape[0]]

# We use past 50 days’ stock prices for our training to predict the 51st day's closing price.
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, N_forecast)

### Build a LSTM model and log training progress to Neptune ###
neptune_callback = NeptuneCallback(run=run)

def Run_LSTM(X_train, layer_units=50):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(N_forecast, activation="linear")(x)  # Originally was Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


model = Run_LSTM(X_train, layer_units=layer_units)

if USE_CACHED_MODEL:
    with open('./LSTM_model', 'rb') as file:
        history = pickle.load(file)
else:
    history = model.fit(
        X_train,
        y_train,
        epochs=cur_epochs,
        batch_size=cur_batch_size,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
        callbacks=[neptune_callback],
    )
    with open('./LSTM_model', 'wb') as file:
        pickle.dump(history, file)

# predict stock prices using past window_size stock prices
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):
    raw = data["Close"][len(data) - len(test) - window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)

    X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

X_test = preprocess_testdat()

predicted_price_ = model.predict(X_test[0].reshape(1, X_test.shape[1], 1))
predicted_price = scaler.inverse_transform(predicted_price_)

# Plot predicted price vs actual closing price
# test["Predictions_lstm"] = predicted_price
index = test.index[:N_forecast]
# close = test['Close'][:N_forecast]

test = pd.DataFrame({
    'Close' : stockprices[train_size : train_size + N_forecast]["Close"],
    'Predictions_lstm' : predicted_price.reshape(-1)
}, index=index)

# Evaluate performance
rmse_lstm = calculate_rmse(np.array(test["Close"]), np.array(test["Predictions_lstm"]))
mape_lstm = calculate_mape(np.array(test["Close"]), np.array(test["Predictions_lstm"]))

### Log to Neptune
run["RMSE"] = rmse_lstm
run["MAPE (%)"] = mape_lstm

### Plot prediction and true trends and log to Neptune
# q = train["Close"][len(train) - window_size:]
def plot_stock_trend_lstm(train, test):
    fig = plt.figure(figsize = (20,10))
    plt.plot(np.asarray(train.index[len(train) - window_size:]), np.asarray(train["Close"][len(train) - window_size:]), label = "Train Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Close"]), label = "Test Closing Price")
    plt.plot(np.asarray(test.index), np.asarray(test["Predictions_lstm"]), label = "Predicted Closing Price")
    plt.title("LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper left")

    ## Log image to Neptune
    run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(fig))

plot_stock_trend_lstm(train, test)

### Stop the run after logging
run.stop()