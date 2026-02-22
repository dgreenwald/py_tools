import pandas as pd
import py_tools.data as dt

from . import config

default_dir = config.base_dir() + "qd/"
# data_dir = '/home/dan/Dropbox/data/qd/'
DATASET_NAME = "qd"
DESCRIPTION = "FRED-QD dataset loader for Stock and Watson (2012)-style macro series."


def load(vintage="2015-11", data_dir=default_dir):
    """Load the FRED-QD dataset CSV for a given vintage.

    Parameters
    ----------
    vintage : str, optional
        Vintage date string identifying which CSV file to load
        (e.g., ``'2015-11'``).
    data_dir : str, optional
        Directory containing vintage CSV files.

    Returns
    -------
    tuple
        A three-element tuple ``(df, factors, transform)`` where

        * **df** (*pandas.DataFrame*) – macro series with a quarterly
          ``DatetimeIndex`` starting 1959-01-01.
        * **factors** (*numpy.ndarray*) – integer array of factor group
          codes, one per series.
        * **transform** (*numpy.ndarray*) – integer array of
          transformation codes, one per series.
    """

    infile = data_dir + vintage + ".csv"
    df_all = pd.read_csv(infile)

    factors = df_all.iloc[0].values[1:].astype(int)
    transform = df_all.iloc[1].values[1:].astype(int)

    df = dt.date_index(df_all.iloc[2:].copy(), "1/1/1959", freq="QS")
    del df["sasdate"]

    return (df, factors, transform)
