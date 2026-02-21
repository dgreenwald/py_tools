import pandas as pd
from py_tools.time_series import date_index

from . import config
default_dir = config.base_dir() + 'romer/'
DATASET_NAME = "romer"
DESCRIPTION = "Romer-Romer monetary shock dataset loader."
def load(data_dir=default_dir):

    # data_dir = '/home/dan/Dropbox/data/romer/'

    df = pd.read_excel(
        data_dir + 'romer_romer_original.xls',
        sheet_name='DATA BY MONTH',
    )

    df = date_index(df, '1/1/1966', freq='MS')

    return df
