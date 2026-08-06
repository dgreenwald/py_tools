import pandas as pd

import os

default_dir = os.environ["PY_TOOLS_DATA_DIR"] + "cfpb/"
DATASET_NAME = "cfpb"
DESCRIPTION = "Consumer Financial Protection Bureau (CFPB) complaint dataset loader."


def process_data(raw_dir="/data/cfpb", data_dir=default_dir):
    """Process raw CFPB mortgage complaint CSV and save as a pickle file.

    Reads the raw CSV, drops unused columns, converts categorical string
    columns to :class:`pandas.Categorical` dtype, parses date columns, and
    writes the result to ``data_dir/mortgage_complaints.pkl``.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing the raw ``mortgage_complaints.csv`` source file.
        Default is ``'/data/cfpb'``.
    data_dir : str, optional
        Directory where the processed pickle file is written.  Default is
        the package-level ``default_dir``.

    Returns
    -------
    None
    """
    df = pd.read_csv(raw_dir + "mortgage_complaints.csv")

    df = df.drop(columns=["Product", "Consumer complaint narrative", "Complaint ID"])

    for var in [
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Company public response",
        "Company",
        "State",
        "Tags",
        "Consumer consent provided?",
        "Submitted via",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
    ]:
        print("categorizing " + var)
        df[var] = df[var].astype("category")

    for var in ["Date received", "Date sent to company"]:
        df[var] = pd.to_datetime(df[var], format="%m/%d/%y")

    df.to_pickle(data_dir + "mortgage_complaints.pkl")

    return None


def load(reimport=False, data_dir=default_dir):
    """Load CFPB mortgage complaints data from the pickle cache.

    Returns the processed complaints DataFrame, calling :func:`process_data`
    first when the cache does not yet exist or when ``reimport=True``.

    Parameters
    ----------
    reimport : bool, optional
        If ``True``, re-process the raw CSV even when a cached pickle already
        exists.  Default is ``False``.
    data_dir : str, optional
        Directory containing (or to contain) the ``mortgage_complaints.pkl``
        cache file.  Default is the package-level ``default_dir``.

    Returns
    -------
    pandas.DataFrame
        CFPB mortgage complaints with categorical and datetime columns as
        produced by :func:`process_data`.
    """
    pkl_file = data_dir + "mortgage_complaints.pkl"

    if reimport or not os.path.exists(pkl_file):
        process_data()

    return pd.read_pickle(pkl_file)
