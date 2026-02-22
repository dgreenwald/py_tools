import os
import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import config
default_dir = config.base_dir() + 'jst/'
# data_dir = '/home/dan/Dropbox/data/jst/'
DATASET_NAME = "jst"
DESCRIPTION = "Jorda-Schularick-Taylor (JST) macro-financial dataset loader."
def load(data_dir=default_dir):
    """Load the Jorda-Schularick-Taylor macro-financial dataset.

    Reads the JST dataset from a Stata DTA file located in ``data_dir``.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing ``JSTdatasetR2.dta``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the full JST macro-financial dataset.
    """
    dta_file = data_dir + 'JSTdatasetR2.dta'
    df = pd.read_stata(dta_file)

    return df
