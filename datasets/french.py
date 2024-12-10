import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'french/'
# data_dir = '/home/dan/Dropbox/data/french/'

def load_industry(data_dir=default_dir):

    infile = data_dir + '49_Industry_Portfolios.CSV'
    df = pd.read_csv(infile, skiprows=11, nrows=(1108-12))
    df = ts.date_index(df, '7/1/1926', freq='MS')
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.replace({-99.99 : np.nan})

    return df

def load(source, data_dir=default_dir):
    
    infile = data_dir + source + '.csv'
    df = pd.read_csv(infile)
    df = (df.rename(columns={'Unnamed: 0' : 'date'})
          .dropna(subset=['date']))
    df['date'] = pd.to_datetime(df['date'].astype(np.int64).astype(str), format='%Y%m')
    df = df.set_index('date')
    
    return df