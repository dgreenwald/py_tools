import datetime
import os
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data as web  # noqa: E402
import pickle  # noqa: E402

from . import config  # noqa: E402

default_dir = config.base_dir() + "fred/"
# data_dir = '/home/dan/Dropbox/data/fred/'
DATASET_NAME = "fred"
DESCRIPTION = "FRED macroeconomic time series dataset loader."


def load(
    codes=None,
    code_names=None,
    start=datetime.datetime(1900, 1, 1),
    end=datetime.datetime.today(),
    reimport=False,
    data_dir=default_dir,
):
    """Load time series data from FRED.

    Downloaded series are cached as individual pickle files so that
    subsequent calls with ``reimport=False`` skip the network request.
    Column names are replaced with friendlier labels when ``code_names``
    is provided.

    Parameters
    ----------
    codes : list of str or None, optional
        FRED series codes to download (e.g. ``['GDPC1', 'UNRATE']``).
        If ``None``, the keys of ``code_names`` are used.
    code_names : dict, optional
        Mapping from FRED series codes to desired output column names
        (e.g. ``{'GDPC1': 'gdp', 'UNRATE': 'unemployment'}``).
    start : datetime.datetime, optional
        Start date for the requested data range.
    end : datetime.datetime, optional
        End date for the requested data range.
    reimport : bool, optional
        If ``True``, re-download all requested series from FRED and
        overwrite any cached pickles.  If ``False``, only download series
        that are not already cached.
    data_dir : str, optional
        Path to the directory used for caching downloaded series.

    Returns
    -------
    pandas.DataFrame
        Time series data indexed by date, with one column per requested
        series (renamed according to ``code_names`` when provided).
    """

    if code_names is None:
        code_names = {}
    if codes is None:
        codes = list(code_names.keys())

    if isinstance(codes, str):
        codes = [codes]

    stored_series_file = data_dir + "series.pkl"
    if os.path.isfile(stored_series_file):
        stored_series = pickle.load(open(stored_series_file, "rb"))
    else:
        stored_series = []

    if reimport:
        download_series = codes
    else:
        download_series = list(set(codes) - set(stored_series))

    df = None

    if download_series:
        # df = web.DataReader(codes, "fred", start, end).rename(columns=code_names)
        df = web.DataReader(download_series, "fred", start, end)

        # Store
        for series in download_series:
            df[series].to_pickle(data_dir + series + ".pkl")

        stored_series = list(set(stored_series + download_series))
        pickle.dump(stored_series, open(stored_series_file, "wb"))

    # Now load stored series
    for series in set(codes) - set(download_series):
        infile = data_dir + series + ".pkl"
        df_in = pd.read_pickle(infile).to_frame()
        if df is None:
            df = df_in
        else:
            df = pd.merge(df, df_in, left_index=True, right_index=True, how="outer")

    df = df.rename(columns=code_names)

    return df
