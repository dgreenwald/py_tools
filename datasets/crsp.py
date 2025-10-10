import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'crsp/'
# data_dir = '/home/dan/Dropbox/data/crsp/'

def load(data_dir=default_dir, vintage='2017', **kwargs):

    infile = 'crsp_{}.csv'.format(vintage)
    df = pd.read_csv(data_dir + infile)
    df = ts.date_index(df, '1/1/1926', freq='QS')

    df['P'] = (df['vwretx'] + 1.0).cumprod()
    df['D'] = np.hstack((np.nan, df['P'][:-1])) * (df['vwretd'] - df['vwretx'])

    df['D4'] = df['D']
    for jj in range(1, 4):
        df['D4'] += df['D'].shift(jj)
    df['D4'] *= 0.25

    df['p'] = np.log(df['P'])
    df['d4'] = np.log(df['D4'])
    df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
    df['re'] = np.log(df['Re'])
    df['pd'] = df['p'] - df['d4']
    df['pd_q'] = df['p'] - np.log(df['D'])

    return df
