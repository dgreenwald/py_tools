import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'fama_bliss/'
# data_dir = '/home/dan/Dropbox/data/fama_bliss/'

def load(data_dir=default_dir):

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
