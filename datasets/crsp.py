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

def load_new(data_dir=default_dir, vintage='2017', freq='Q', **kwargs):

    version = vintage
    if freq == 'M':
        version += '_' + freq
    elif freq != 'Q':
        raise Exception

    infile = 'crsp_{}.csv'.format(version)
    df = pd.read_csv(data_dir + infile, parse_dates=['caldt'])
    df['date'] = df['caldt'] + pd.tseries.frequencies.to_offset(freq + 'S')
    df = df.set_index('date')

    df['P'] = (df['vwretx'] + 1.0).cumprod()
    df['D'] = np.hstack((np.nan, df['P'][:-1])) * (df['vwretd'] - df['vwretx'])

    if freq == 'Q':
        window = 4
    elif freq == 'M':
        window = 12
        
    df['D_ann'] = df['D'].rolling(window).sum()

    df['p'] = np.log(df['P'])
    df['d_ann'] = np.log(df['D_ann'])
    
    df['Re'] = (df['P'] + df['D']) / df['P'].shift()
    
    df['re'] = np.log(df['Re'])
    
    df['PD'] = df['P'] / df['D_ann']
    df['pd'] = df['p'] - df['d_ann']
    df['pd_q'] = df['p'] - np.log(df['D'])

    return df