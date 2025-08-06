import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'crsp/'
# data_dir = '/home/dan/Dropbox/data/crsp/'

def load_quarterly(data_dir=default_dir, vintage='2017', **kwargs):
    
    print("need to check")
    raise Exception

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

def load_monthly(data_dir=default_dir, vintage='2017', dates='parse', 
                 resample=None, **kwargs):

    if dates == 'parse':
        parse_dates = ['caldt']
    else:
        parse_dates = []
        
    infile = 'crsp_{}.csv'.format(vintage)
    df = pd.read_csv(data_dir + infile, parse_dates=parse_dates)
    
    if dates != 'parse':
        raise Exception
        df = ts.date_index(df, '1/1/1926', freq='MS')

    df['P'] = (df['vwretx'] + 1.0).cumprod()
    if '1925-10-01' in df.index:
        df.loc['1925-10-01', 'P'] = 1.0
    df['D'] = np.hstack((np.nan, df['P'][:-1])) * (df['vwretd'] - df['vwretx'])

    df['D_ann'] = df['D'].rolling(12).sum()
    # for jj in range(1, 4):
    #     df['D4'] += df['D'].shift(jj)
    # df['D4'] *= 0.25

    df['p'] = np.log(df['P'])
    df['d_ann'] = np.log(df['D_ann'])
    df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
    df['re'] = np.log(df['Re'])
    df['pd_ann'] = df['p'] - df['d_ann']
    df['pd_raw'] = df['p'] - np.log(df['D'])
    
    df = df.set_index('caldt')
    
    if resample is not None:
        
        resample_dict = {
            'P' : 'last',
            'p' : 'last',
            'Re' : 'prod',
            're' : 'sum',
            'd_ann' : 'last',
            'D_ann' : 'last',
            'pd_ann' : 'last',
            'pd_raw' : 'last',
            }

        df = (df.resample(resample)
              .agg(resample_dict))

    return df

