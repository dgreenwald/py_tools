import numpy as np
import pandas as pd

from . import config
default_dir = config.base_dir() + 'fama_bliss/'
# data_dir = '/home/dan/Dropbox/data/fama_bliss/'
DATASET_NAME = "fama_bliss"
DESCRIPTION = "Fama-Bliss bond yield dataset loader."
def load(data_dir=default_dir):
    """Load Fama-Bliss bond price data and compute log prices and yields.

    Reads ``fbdata.csv`` from *data_dir*, pivots by maturity term (1–5 years),
    resamples to monthly frequency, and computes log prices and continuously
    compounded yields for each maturity.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the ``fbdata.csv`` file.

    Returns
    -------
    pandas.DataFrame
        Monthly DataFrame with columns ``P1``–``P5`` (raw prices),
        ``p1``–``p5`` (log prices), and ``y1``–``y5`` (yields), indexed
        by date.
    """

    infile = data_dir + 'fbdata.csv'
    df = pd.read_csv(infile)

    df = df.rename(columns={
        'MCALDT' : 'date',
        'TTERMTYPE' : 'term',
        'TMNOMPRC' : 'price',
    })

    df['price'] *= 0.01

    df['date'] = pd.to_datetime(df['date'])
    df['term'] = df['term'] - 5000
    df['term'] = df['term'].astype(str)
    df['term'].replace({
        str(val) : 'P' + str(val) for val in range(1, 6)
    }, inplace=True)

    df_stack = df.pivot(index='date', columns='term', values='price')
    df_stack = df_stack.resample('MS').mean()

    for ii in range(1, 6):
        df_stack['p' + str(ii)] = np.log(df_stack['P' + str(ii)])
        df_stack['y' + str(ii)] = -df_stack['p' + str(ii)] / float(ii)

    return df_stack
