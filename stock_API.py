"""
Module to be used in other files to import stock data from Yahoo Finance API
"""

from yahoo_fin.stock_info import get_data
import pyfredapi as pf
import numpy as np
import pandas as pd

FRED_API_KEY = "492c019400f8f7963ee131c745be7685"

# API key for the Alpha Vantage API
# __API_KEY = "OR59ZKH3MZH9S2GU"

start_date = "01/01/2001"

def fred_date_format(old_format):
    return old_format[6:10] + "-" + old_format[0:2] + "-" + old_format[3:5]

def rsi_calc(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

ticker = get_data("spy",
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

__monthly = ticker.join([UNRATE, UMCSENT], how="outer")[['UNRATE', 'UMCSENT']].ffill()
data = ticker.join([VIX, USDX, EFFR]).join(__monthly).dropna()


if __name__ == "__main__":
    print(data)
