import pandas as pd
from py_tools.time_series import date_index

from . import config
default_dir = config.base_dir() + 'romer/'
DATASET_NAME = "romer"
DESCRIPTION = "Romer-Romer monetary shock dataset loader."
def load(data_dir=default_dir):
    """Load Romer-Romer monetary shock data from Excel file.

    Reads the 'DATA BY MONTH' sheet from the Romer-Romer original Excel
    file and assigns a monthly date index starting January 1966.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing the Romer-Romer Excel file.

    Returns
    -------
    pandas.DataFrame
        Monthly time-indexed DataFrame of Romer-Romer monetary shock data.
    """
    # data_dir = '/home/dan/Dropbox/data/romer/'

    df = pd.read_excel(
        data_dir + 'romer_romer_original.xls',
        sheet_name='DATA BY MONTH',
    )

    df = date_index(df, '1/1/1966', freq='MS')

    return df
