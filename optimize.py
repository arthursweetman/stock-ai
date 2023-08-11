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
    data = find_local_extrema(data)
    print(data)

"""
Finds and identified the local maxima and minima for a given time-series data set
Returns the same dataset with a new column identifying the local max and mins
"""
def find_local_extrema(data):

    # Step 1: Calculate "Uphill climb distance"

    future = data.copy()
    past = data.copy()

    future.loc[len(future.index)] = [np.NaN]
    future = future.drop(0).reset_index().drop(columns = 'index')
    data['future'] = future

    past.loc[-1] = [np.NaN]
    past.index += 1
    past = past.drop(len(past)-1)
    data['past'] = past

    local_extreme = np.full(len(data),'---')
    maxs = (data['Close'] > data['future']) & (data['Close'] > data['past'])
    local_extreme[maxs] = 'max'
    mins = (data['Close'] < data['future']) & (data['Close'] < data['past'])
    local_extreme[mins] = 'min'

    data['local_extrema'] = local_extreme
    data = data.drop(columns=['future','past'])

    return data





if __name__ == "__main__":
    data = pd.DataFrame({
        'Close': [1, 2, 3, 4, 5, 6, 5, 4, 5, 6, 7]
    })
    optimize_gains(data)