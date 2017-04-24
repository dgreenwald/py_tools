import numpy as np
import os
import pandas as pd
from py_tools.data import date_index

def load(dataset, master_dirs={}):
    """Load data from one-off file"""

    dirs = master_dirs.copy()
    if 'base' not in dirs:
        home_dir = os.environ['HOME']
        dirs['base'] = home_dir + '/Dropbox/data/'

    data_dir = dirs['base'] + 'misc/'

    if dataset == 'cleveland_fed':
        infile = data_dir + 'cleveland_fed_inflation_expectations.xlsx'
        df = pd.read_excel(
            infile,
            skiprows=6,
            header=None,
            names=['Date'] + ['infl_{}y'.format(yr) for yr in range(1, 31)]
        )

        del df['Date']
        date_index(df, '1982/1/1', 'MS')

    return df
