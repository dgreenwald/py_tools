import numpy as np
import pandas as pd
import py_tools.time_series as ts

data_dir = '/home/dan/Dropbox/data/french/'

def load():

    infile = data_dir + '49_Industry_Portfolios.CSV'
    df = pd.read_csv(infile, skiprows=11, nrows=(1108-12))
    df = ts.date_index(df, '7/1/1926', freq='MS')
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.replace({-99.99 : np.nan})

    return df