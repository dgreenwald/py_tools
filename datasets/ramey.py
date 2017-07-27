import os
import pandas as pd
from py_tools.time_series import date_index

def load(dataset, master_dirs={}):
    """Load data from Ramey shocks file"""

    dirs = master_dirs.copy()
    if 'base' not in dirs:
        home_dir = os.environ['HOME']
        dirs['base'] = home_dir + '/Dropbox/data/'

    data_dir = dirs['base'] + 'ramey/'

    if dataset == 'technology':

        infile = data_dir + 'technology/Technology_data.xlsx'
        df = pd.read_excel(
            infile,
            sheetname='techdat',
        )

        df = date_index(df, '1947-01-01', freq='QS')

    return df
