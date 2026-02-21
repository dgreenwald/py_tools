import numpy as np
import pandas as pd
import py_tools.data as dt

from . import config
default_dir = config.base_dir() + 'qd/'
# data_dir = '/home/dan/Dropbox/data/qd/'
DATASET_NAME = "qd"
DESCRIPTION = "Dataset loader for qd."


def load(vintage='2015-11', data_dir=default_dir):

    infile = data_dir + vintage + '.csv'
    df_all = pd.read_csv(infile)

    factors = df_all.iloc[0].values[1:].astype(int)
    transform = df_all.iloc[1].values[1:].astype(int)

    df = dt.date_index(df_all.iloc[2:].copy(), '1/1/1959', freq='QS')
    del df['sasdate']

    return (df, factors, transform)

