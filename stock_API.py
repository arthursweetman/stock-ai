"""
Module to be used in other files to import stock data from Yahoo Finance API
"""

from yahoo_fin.stock_info import get_data
import pyfredapi as pf
import requests
import numpy as np
import pandas as pd

start_date = "01/01/2001"
ticker = "SPY"

FRED_API_KEY = "492c019400f8f7963ee131c745be7685"
# TAAPI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjRkZTQzNDk0OThkNzVkYTM2ZTcxMzljIiwiaWF0IjoxNjkyMjg5NTU2LCJleHAiOjMzMTk2NzUzNTU2fQ.zkDmgI6goChvasL7PEuUaL2RGdIEXE_I-FaowG0A13E"
ALPHA_VANTAGE_API_KEY = "OR59ZKH3MZH9S2GU"

rsi_params = {
    'function' : 'RSI',
    'interval' : 'daily',
    'time_period' : '14',
    'series_type' : 'close',
    'apikey' : ALPHA_VANTAGE_API_KEY
}
atr_params = {
    'function' : 'ATR',
    'interval' : 'daily',
    'time_period' : '14',
    'series_type' : 'close',
    'apikey' : ALPHA_VANTAGE_API_KEY
}
macd_params = {
    'function' : 'MACDEXT',
    'interval' : 'daily',
    'series_type' : 'close',
    'apikey' : ALPHA_VANTAGE_API_KEY,
    'fastmatype' : 1,
    'slowmatype' : 1,
    'signalmatype' : 1
}

def get_technical_indicator(params, symbol):
    params['symbol'] = symbol.upper()
    url = 'https://www.alphavantage.co/query?'
    r = requests.get(url, params)
    data = r.json()["Technical Analysis: " + params['function']]
    df = pd.DataFrame.from_dict(data, orient = 'index')
    return df


def fred_date_format(old_format):
    return old_format[6:10] + "-" + old_format[0:2] + "-" + old_format[3:5]

pricedat = get_data(ticker,
                  start_date = start_date,
                  index_as_date = True,
                  interval="1d")

VIX = get_data("^VIX",
                start_date = start_date,
                index_as_date = True,
                interval="1d")[['close']].rename(columns={'close':'VIX'})

USDX = get_data("DX-Y.NYB",
                start_date = start_date,
                index_as_date = True,
                interval="1d")[['close']].rename(columns={'close':'USDX'})

EFFR = pf.get_series(series_id = "EFFR",
                     api_key = FRED_API_KEY,
                     observation_start = fred_date_format(start_date))[['date','value']].set_index('date').rename(columns={'value':'EFFR'})

UNRATE = pf.get_series(series_id = "UNRATE",
                       api_key = FRED_API_KEY,
                       observation_start = fred_date_format(start_date))[['date','value']].set_index('date').rename(columns={'value':'UNRATE'})

UMCSENT = pf.get_series(series_id = "UMCSENT",
                        api_key = FRED_API_KEY,
                        observation_start = fred_date_format(start_date))[['date','value']].set_index('date').rename(columns={'value':'UMCSENT'})

MACD = get_technical_indicator(macd_params, ticker)[['MACD']]
ATR = get_technical_indicator(atr_params, ticker)
RSI = get_technical_indicator(rsi_params, ticker)

MACD.index = pd.to_datetime(MACD.index)
ATR.index = pd.to_datetime(ATR.index)
RSI.index = pd.to_datetime(RSI.index)

__monthly = pricedat.join([UNRATE, UMCSENT], how="outer")[['UNRATE', 'UMCSENT']].ffill()
data = pricedat.join([VIX, USDX, EFFR, __monthly, MACD, ATR, RSI]).dropna()


if __name__ == "__main__":
    pass
