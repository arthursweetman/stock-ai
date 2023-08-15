"""
Module to be used in other files to import stock data from Yahoo Finance API
"""

from yahoo_fin.stock_info import get_data

# API key for the Alpha Vantage API
# __API_KEY = "OR59ZKH3MZH9S2GU"

data = get_data("aapl",
                start_date="12/04/1999",
                # end_date="12/04/2023",
                index_as_date = True,
                interval="1d")

if __name__ == "__main__":
    print(data)
