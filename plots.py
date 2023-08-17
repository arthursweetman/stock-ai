"""

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def stagger_down(data, staggered_column_name):
    """

        Args:
            data: DataFrame with one column
            staggered_column_name: Name of newly created (staggered) column

        Returns:
            data that is staggered-down

        """
    index = np.copy(data.index)
    newdat = data.copy()
    newdat = newdat.reset_index(drop=True)
    newdat.loc[-1] = np.nan
    newdat = newdat.sort_index().reset_index(drop=True)
    newdat = newdat.iloc[:len(newdat)-1]  # Drop the last row
    newdat.index = index
    newdat = newdat.rename(columns = {0: staggered_column_name})

    return newdat


def stagger_up(data, staggered_column_name):
    """

    Args:
        data: DataFrame with one column
        staggered_column_name: Name of newly created (staggered) column

    Returns:
        data that is staggered-up

    """
    index = np.copy(data.index)
    newdat = data.copy()
    newdat = newdat.reset_index(drop=True)
    newdat.loc[len(newdat)] = np.nan
    newdat = newdat.sort_index().reset_index(drop=True)
    newdat = newdat.iloc[1:]  # Drop the first row
    newdat.index = index
    newdat = newdat.rename(columns={0: staggered_column_name})

    return newdat


def tomorrow_pred_v_today_close(data, predictions):
    
    next_day_pred = stagger_up(predictions, "next_day_pred")
    predictions = predictions.rename(columns={0: 'pred_for_today'})
    data = data.join([next_day_pred, predictions])

    zero_line_on_plot = np.min(data['Close']) - 10
    # Calculate slope and add to data
    data['derivative'] = data['next_day_pred'] - data['pred_for_today'] + zero_line_on_plot

    plt.figure()
    plt.plot(data.index, data['next_day_pred'].values, '+-b', linewidth=.25, markersize=.5, label="Next Day's Predicted close")
    plt.plot(data.index, data['Close'].values, '.-r', linewidth=.25, markersize=.5, label="Today's close")
    plt.plot(data.index, data['derivative'].values, '.-g', linewidth=.25, markersize=.5, label="Prediction Derivative")
    plt.hlines(zero_line_on_plot, data.index[0], data.index[len(data.index)-1])
    plt.legend()
    plt.savefig("tomorrow_pred_v_today_close", dpi=1000)

    print('plot done')


if __name__ == "__main__":
    pass