"""

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def tomorrow_pred_v_today_close(data, predictions):
    index = np.copy(predictions.index)
    newdat = predictions.copy()
    newdat.loc[len(index)] = np.nan  # model.predict ?
    newdat['temp'] = range(0, len(newdat))
    newdat = newdat.loc[newdat['temp'] != 0]  # drop the first row
    newdat = newdat.drop(columns = 'temp')
    newdat.index = index

    # We now are interpreting newdat dates as "the predicted price for tomorrow that we receive today"

    newdat = newdat.rename(columns = {0 : 'next_day_pred'})
    data = data.join(newdat)

    plt.figure()
    plt.plot(data['next_day_pred'].values, '+-b', linewidth=.25, markersize=.5, label="Next Day's Predicted close")
    plt.plot(data['Close'].values, '.-r', linewidth=.25, markersize=.5, label="Today's close")
    plt.legend()
    plt.savefig("tomorrow_pred_v_today_close", dpi=1000)

    print('plot done')
