"""
Given a time-series set of data, optimize the "uphill climb distance" and
return the tuples of dates that correspond to the start-end timestamps of the optimized values

@params:
    data (DataFrame) - simple timeseries data Dataframe

@return:
    timestamps (2-dimensional array) - array of start-end timestamps
"""

import numpy as np
import pandas as pd

def calculate_performance(data):
    """
    This is a method that simulates buying the stock when the slope of the model
    turns positive and sells when it turns negative

    Args:
        data: Tomorrow's Predictions on Today's Close (tptc)

    Returns:
        Void
    """

    # curr = 0
    data.index = pd.to_datetime(data.index)
    extrema = find_local_extrema(data[['next_day_pred']])
    extrema = extrema.loc[extrema['local_extrema'] != "---"].drop(columns='Close')
    data = data.join(extrema)
    data['temp'] = range(0,len(data))  # temporary index variable
    t=data.loc[~pd.isnull(data['local_extrema'])][['temp','local_extrema']]
    data = data.drop(columns = 'local_extrema')
    t['temp'] += 1
    index = data.index
    data = data.merge(t, how="left", on='temp')
    data.index = index

    data['local_extrema'].loc[data['local_extrema'] == "min"] = "buy"
    data['local_extrema'].loc[data['local_extrema'] == "max"] = "sell"
    data = data.rename(columns = {'local_extrema':'action'})

    # Filter for the closing prices on action days
    t = data.loc[~pd.isnull(data['action'])]
    transactions = t[['Close', 'action']]

    # Now calculate % change over time period
    sells = t.loc[t['action'] == 'sell'].reset_index()['Close']
    buys = t.loc[t['action'] == 'buy'].reset_index()['Close']
    both = pd.DataFrame({
        'buy': buys,
        'sell': sells
    }).dropna()  # This dropna removes edge cases where a min or a max do not have a partner
    both['perc_change'] = both['sell'] / both['buy']
    print(np.prod(both['perc_change']))


    print("")


def calc_over_under(data):
    """

    Args:
        data: Tomorrow's Predictions on Today's Close (tptc)

    Returns:
        Void
    """

    # Define thresholds (% price) above or below predicted price to trigger a buy/sell
    LT = 0.03
    UT = 0.014

    data['perc_diff'] = data['Close'] / data['next_day_pred']
    data['action'] = (data['perc_diff']-1 > UT) | (data['perc_diff']-1 < -LT)
    data = data.loc[data['action']]
    data['action'].loc[data['perc_diff'] - 1 < 0] = 'buy'
    data['action'].loc[data['perc_diff'] - 1 > 0] = 'sell'

    data = data[['Close','action']].reset_index()
    data['keep'] = True
    for row in data.index:
        if row == 0:
            continue
        if data['action'][row-1] == data['action'][row]:
            data['keep'][row] = False
    t = data.loc[data['keep']]

    sells = t.loc[t['action'] == 'sell'].reset_index()['Close']
    buys = t.loc[t['action'] == 'buy'].reset_index()['Close']

    data = pd.DataFrame({
        'buy': buys,
        'sell': sells
    }).dropna()
    data['perc_change'] = data['sell'] / data['buy']
    print(np.prod(data['perc_change']))

    print("pause")

def optimize_gains(data):
    """

    Args:
        data:

    Returns:

    """

    # Step 1: Calculate "Uphill climb distance"
    data = find_local_extrema(data)

    # Filter out any values that are not local extrema
    data = data.loc[data['local_extrema'] != "---"]

    if data.reset_index().loc[0]['local_extrema'] == 'max':
        data = data.reset_index().drop(0)

    maxs = data.loc[data['local_extrema'] == 'max'].reset_index()['Close']
    mins = data.loc[data['local_extrema'] == 'min'].reset_index()['Close']

    # t = maxs.loc[maxs.index % 2 != 0]

    data = pd.DataFrame({
        'local_min': mins,
        'local_max': maxs
    }).dropna()  # This dropna removes edge cases where a min or a max do not have a partner
    uphill_climbs = data['local_max'] - data['local_min']
    uphill_climb_distance = sum(uphill_climbs)

    print(data, '\n', uphill_climbs, '\n', uphill_climb_distance)


def find_local_extrema(data):
    """
    Finds and identifies the local maxima and minima for a given time-series data set
    Returns the same dataset with a new column identifying the local max and mins
    Args:
        data: Pandas dataframe with one timeseries column

    Returns:
        Original data with a new column identifying the local max and mins
    """

    index = np.copy(data.index)
    data.columns.values[0] = "Close"
    future = data.copy()
    past = data.copy()

    future.loc[len(future.index)] = [np.NaN]
    future['temp'] = range(0, len(future))
    future = future.loc[future['temp'] != 0]  # drop the first row
    future = future.drop(columns='temp')
    future.index = index
    # future = future.drop(0).reset_index().drop(columns = 'index')
    data['future'] = future

    past = past.reset_index().drop(columns='index')
    past.loc[-1] = [np.NaN]
    past = past.sort_index().reset_index(drop=True)
    # past['temp'] = range(0, len(past))
    past = past.loc[past.index != len(past)-1]  # drop the last row
    past.index = index
    data['past'] = past

    # Repeat points will mess up the local extrema calculations, giving consecutive mins or maxs
    # Solution: delete the rows with consecutive points
    repeat_points = ((data['Close'] == data['future']) & (data['Close'] == data['past']) |
                     (data['Close'] != data['future']) & (data['Close'] == data['past']))

    # If there are repeat points, we need to re-calculate the 'past' and 'future' values
    if sum((repeat_points) > 0):
        data = data.loc[~repeat_points].reset_index().drop(columns = 'index')
        data = data[['Close']]
        future = data.copy()
        past = data.copy()

        future.loc[len(future.index)] = [np.NaN]
        future = future.drop(0).reset_index().drop(columns='index')
        data['future'] = future

        past.loc[-1] = [np.NaN]
        past.index += 1
        past = past.drop(len(past) - 1)
        data['past'] = past

    local_extreme = np.full(len(data), '---')
    maxs = (data['Close'] > data['future']) & (data['Close'] > data['past'])
    local_extreme[maxs] = 'max'
    mins = (data['Close'] < data['future']) & (data['Close'] < data['past'])
    local_extreme[mins] = 'min'

    data['local_extrema'] = local_extreme
    data = data.drop(columns=['future','past'])

    return data




if __name__ == "__main__":
    # data = pd.DataFrame({
    #     'Close': [1, 2, 3, 4, 5, 6, 5, 4, 5, 6, 7,8,9,10,9,8,7,6,5,4,2,3,4,5,7,3]
    # })
    data = pd.read_csv("tptc.csv", index_col=0)  # , index_col="Date")
    # optimize_gains(data[['next_day_pred']])
    calc_over_under(data)
    print("")