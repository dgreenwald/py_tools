import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import config

default_dir = config.base_dir() + "crsp/"
# data_dir = '/home/dan/Dropbox/data/crsp/'
DATASET_NAME = "crsp"
DESCRIPTION = "CRSP market and return dataset loader."


def load(data_dir=default_dir, vintage="2017", **kwargs):
    """Load CRSP quarterly market data and compute price/dividend/return series.

    Reads the CRSP CSV file for the given vintage, assigns a quarterly
    date index starting 1926-01-01, and derives cumulative price, dividend,
    annualised dividend, log price-dividend ratio, and log equity return
    series.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing CRSP CSV files.  Defaults to the
        package-level ``default_dir``.
    vintage : str, optional
        Dataset vintage identifier appended to the filename stem
        (e.g. ``'2017'`` resolves to ``crsp_2017.csv``).  Defaults to
        ``'2017'``.
    **kwargs
        Accepted for API compatibility; currently unused.

    Returns
    -------
    pandas.DataFrame
        Quarterly CRSP data with the original columns plus derived series:
        ``P`` (cumulative price index), ``D`` (quarterly dividend),
        ``D4`` (four-quarter average dividend), ``p`` (log price),
        ``d4`` (log annualised dividend), ``Re`` (gross equity return),
        ``re`` (log equity return), ``pd`` (log price-dividend ratio
        using ``D4``), and ``pd_q`` (log price-dividend ratio using
        quarterly ``D``).
    """
    infile = "crsp_{}.csv".format(vintage)
    df = pd.read_csv(data_dir + infile)
    df = ts.date_index(df, "1/1/1926", freq="QS")

    df["P"] = (df["vwretx"] + 1.0).cumprod()
    df["D"] = np.hstack((np.nan, df["P"][:-1])) * (df["vwretd"] - df["vwretx"])

    df["D4"] = df["D"]
    for jj in range(1, 4):
        df["D4"] += df["D"].shift(jj)
    df["D4"] *= 0.25

    df["p"] = np.log(df["P"])
    df["d4"] = np.log(df["D4"])
    df["Re"] = np.hstack(
        (np.nan, (df["P"][1:] + df["D"][1:]).values / df["P"][:-1].values)
    )
    df["re"] = np.log(df["Re"])
    df["pd"] = df["p"] - df["d4"]
    df["pd_q"] = df["p"] - np.log(df["D"])

    return df


def load_new(data_dir=default_dir, vintage="2017", freq="Q", **kwargs):
    """Load CRSP market data at a specified frequency and compute derived series.

    Reads the CRSP CSV file for the given vintage and frequency,
    constructs a period-start date index, and derives cumulative price,
    dividend, rolling annualised dividend, log price-dividend ratio, and
    log equity return series.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing CRSP CSV files.  Defaults to the
        package-level ``default_dir``.
    vintage : str, optional
        Dataset vintage identifier (e.g. ``'2017'``).  For monthly data
        the resolved filename is ``crsp_2017_M.csv``; for quarterly it is
        ``crsp_2017.csv``.  Defaults to ``'2017'``.
    freq : {'Q', 'M'}, optional
        Sampling frequency.  ``'Q'`` for quarterly (default) or ``'M'``
        for monthly.  Any other value raises an ``Exception``.
    **kwargs
        Accepted for API compatibility; currently unused.

    Returns
    -------
    pandas.DataFrame
        CRSP data indexed by period-start dates with the original columns
        plus derived series: ``P`` (cumulative price index), ``D``
        (per-period dividend), ``D_ann`` (rolling annualised dividend sum),
        ``p`` (log price), ``d_ann`` (log annualised dividend), ``Re``
        (gross equity return), ``re`` (log equity return), ``PD``
        (price-to-annualised-dividend ratio), ``pd`` (log
        price-dividend ratio using ``D_ann``), and ``pd_q`` (log
        price-dividend ratio using per-period ``D``).

    Raises
    ------
    Exception
        If ``freq`` is not ``'Q'`` or ``'M'``.
    """
    version = vintage
    if freq == "M":
        version += "_" + freq
    elif freq != "Q":
        raise Exception

    infile = "crsp_{}.csv".format(version)
    df = pd.read_csv(data_dir + infile, parse_dates=["caldt"])
    df["date"] = df["caldt"] + pd.tseries.frequencies.to_offset(freq + "S")
    df = df.set_index("date")

    df["P"] = (df["vwretx"] + 1.0).cumprod()
    df["D"] = np.hstack((np.nan, df["P"][:-1])) * (df["vwretd"] - df["vwretx"])

    if freq == "Q":
        window = 4
    elif freq == "M":
        window = 12

    df["D_ann"] = df["D"].rolling(window).sum()

    df["p"] = np.log(df["P"])
    df["d_ann"] = np.log(df["D_ann"])

    df["Re"] = (df["P"] + df["D"]) / df["P"].shift()

    df["re"] = np.log(df["Re"])

    df["PD"] = df["P"] / df["D_ann"]
    df["pd"] = df["p"] - df["d_ann"]
    df["pd_q"] = df["p"] - np.log(df["D"])

    return df
