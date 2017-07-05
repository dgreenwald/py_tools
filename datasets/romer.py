import pandas as pd
from py_tools.data import date_index

def load():

    data_dir = '/home/dan/Dropbox/data/romer/'

    df = pd.read_excel(
        data_dir + 'romer_romer_original.xls',
        sheetname='DATA BY MONTH',
    )

    df = date_index(df, '1/1/1966', freq='MS')

    return df
