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

def optimize_gains(data):

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




"""
Finds and identifies the local maxima and minima for a given time-series data set
Returns the same dataset with a new column identifying the local max and mins
"""
def find_local_extrema(data):
    future = data.copy()
    past = data.copy()

    future.loc[len(future.index)] = [np.NaN]
    future = future.drop(0).reset_index().drop(columns = 'index')
    data['future'] = future

    past.loc[-1] = [np.NaN]
    past.index += 1
    past = past.drop(len(past)-1)
    data['past'] = past

    # Repeat points will mess up the local extrema calculations, giving consecutive mins or maxs
    # Solution: delete the rows with consecutive points
    repeat_points = ~((data['Close'] == data['future']) & (data['Close'] == data['past']) |
                      (data['Close'] != data['future']) & (data['Close'] == data['past']))

    # If there are repeat points, we need to re-calculate the 'past' and 'future' values
    if sum((~repeat_points) > 0):
        data = data.loc[repeat_points].reset_index().drop(columns = 'index')
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
    data = pd.read_csv("AAPL.csv")[["Close"]]  # , index_col="Date")
    optimize_gains(data)