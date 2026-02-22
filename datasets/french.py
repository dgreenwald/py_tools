import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import config
default_dir = config.base_dir() + 'french/'
# data_dir = '/home/dan/Dropbox/data/french/'
DATASET_NAME = "french"
DESCRIPTION = "Kenneth French data library dataset loader."
def load_industry(data_dir=default_dir):
    """Load Ken French 49-industry portfolio monthly returns.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing the French data CSV files.

    Returns
    -------
    pandas.DataFrame
        Monthly equal-weighted returns for the 49 Fama-French industry
        portfolios, indexed by date starting July 1926.  Missing values
        (encoded as ``-99.99`` in the source) are replaced with ``NaN``.
    """

    infile = data_dir + '49_Industry_Portfolios.CSV'
    df = pd.read_csv(infile, skiprows=11, nrows=(1108-12))
    df = ts.date_index(df, '7/1/1926', freq='MS')
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.replace({-99.99 : np.nan})

    return df

def load(source, data_dir=default_dir):
    """Load a Ken French data file by source name.

    Parameters
    ----------
    source : str
        Filename (without the ``.csv`` extension) of the Ken French data
        file to load (e.g. ``'F-F_Research_Data_Factors'``).
    data_dir : str, optional
        Path to the directory containing the French data CSV files.

    Returns
    -------
    pandas.DataFrame
        Data from the requested file with a monthly ``datetime`` index
        parsed from the ``YYYYMM``-formatted first column.
    """

    infile = data_dir + source + '.csv'
    df = pd.read_csv(infile)
    df = (df.rename(columns={'Unnamed: 0' : 'date'})
          .dropna(subset=['date']))
    df['date'] = pd.to_datetime(df['date'].astype(np.int64).astype(str), format='%Y%m')
    df = df.set_index('date')

    return df
    
def load_table(table, data_dir=default_dir):
    """Load beta or book-to-market sorted portfolio data.

    Parameters
    ----------
    table : str
        Portfolio sort to load.  ``'beta'`` loads beta-sorted decile
        portfolios; ``'bm'`` loads book-to-market-sorted decile portfolios.
    data_dir : str, optional
        Path to the directory containing the French data CSV files.

    Returns
    -------
    pandas.DataFrame
        Monthly returns for decile portfolios (columns ``p1`` through
        ``p10``), indexed by date with a monthly ``datetime`` index.
    """

    if table == 'beta':
        infile = data_dir + 'beta_portfolios_monthly.csv'
    elif table == 'bm':
        infile = data_dir + 'bm_portfolios_monthly.csv'
        
    df = pd.read_csv(infile)
    df.columns = ['date'] + [f'p{ii:d}' for ii in range(1, 11)]
    df = df.dropna(subset=['date']).copy()
    df['date'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m')
    df = df.set_index('date')
    return df
