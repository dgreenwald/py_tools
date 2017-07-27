# import numpy as np
import os
import pandas as pd
from py_tools.time_series import date_index

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
        date_index(df, '1982/1/1', freq='MS')

    elif dataset == 'gertler_karadi':

        infile = data_dir + 'gk_factors.csv'
        df = pd.read_csv(infile)
        datestr = df['year'].astype(str) + '/' + df['month'].astype(str) + '/1'
        df.set_index(pd.to_datetime(datestr), inplace=True)
        df.drop(['year', 'month'], axis=1, inplace=True)

    elif dataset == 'gz':

        infile = data_dir + 'gz.csv'
        df = pd.read_csv(infile)
        df = date_index(df, '1973-01-01', freq='MS')
        df.drop(['date'], axis=1, inplace=True)

    elif dataset == 'fernald':

        infile = data_dir + 'fernald_tfp.xls'
        df = pd.read_excel(
            infile,
            skiprows=1,
            sheetname='quarterly',
            skip_footer=6,
        )

        del df['date']
        date_index(df, '1947/1/1', freq='QS')

    elif dataset == 'price_rent':

        infile = data_dir + 'price_rent.pkl'
        df = pd.read_pickle(infile)

    return df
