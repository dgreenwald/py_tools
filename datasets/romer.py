import pandas as pd
from py_tools.time_series import date_index

from . import defaults
default_dir = defaults.base_dir() + 'ahs/'

def load(data_dir=default_dir):

    # data_dir = '/home/dan/Dropbox/data/romer/'

    df = pd.read_excel(
        data_dir + 'romer_romer_original.xls',
        sheetname='DATA BY MONTH',
    )

    df = date_index(df, '1/1/1966', freq='MS')

    return df
