import pandas as pd

from . import config

default_dir = config.base_dir() + "bea_industry/"
DATASET_NAME = "bea_industry"
DESCRIPTION = "BEA industry accounts dataset loader."


def load(freq="q", data_dir=default_dir):
    """Load BEA industry accounts data from a pre-processed feather file.

    Reads the profits feather file for the requested frequency from
    ``data_dir`` and returns it as a DataFrame.

    Parameters
    ----------
    freq : str, optional
        Frequency of the data.  Use ``'q'`` for quarterly or ``'a'`` for
        annual.  This string is used to construct the feather filename
        (e.g. ``'qprofits.feather'``).  Default is ``'q'``.
    data_dir : str, optional
        Directory containing the feather and CSV source files.  Default is
        the package-level ``default_dir``.

    Returns
    -------
    pandas.DataFrame
        BEA industry accounts profits data for the requested frequency.
    """
    df = pd.read_feather(data_dir + freq + "profits.feather")
    return df
