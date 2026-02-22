import pandas as pd
from py_tools.time_series import date_index

from . import config
default_dir = config.base_dir()
DATASET_NAME = "ramey"
DESCRIPTION = "Ramey macro shock datasets loader."
def load(dataset, master_dirs={}):
    """Load data from Ramey shocks file.

    Parameters
    ----------
    dataset : str
        Name of the Ramey shock dataset to load. Supported values are
        ``'technology'`` (quarterly, starting 1947-01-01) and
        ``'monetary'`` (monthly, starting 1959-01-01).
    master_dirs : dict, optional
        Directory overrides. May contain key ``'base'`` to override the
        default data root.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with a ``DatetimeIndex``.
    """

    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = default_dir
        # home_dir = os.environ['HOME']
        # dirs['base'] = home_dir + '/Dropbox/data/'

    data_dir = dirs['base'] + 'ramey/'

    if dataset == 'technology':

        infile = data_dir + 'technology/Technology_data.xlsx'
        df = pd.read_excel(
            infile,
            sheet_name='techdat',
        )

        df = date_index(df, '1947-01-01', freq='QS')

    elif dataset == 'monetary':

        infile = data_dir + 'monetary/Monetarydat.xlsx'
        df = pd.read_excel(
            infile,
            sheet_name='Monthly',
        )

        df = date_index(df, '1959-01-01', freq='MS')

    return df
