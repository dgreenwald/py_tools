import numpy as np
import pandas as pd
from py_tools import time_series as ts

from . import config
default_dir = config.base_dir() + 'hmda/'
DATASET_NAME = "hmda"
DESCRIPTION = "Home Mortgage Disclosure Act (HMDA) dataset loader."
def load(data_dir=None, **kwargs):
    """Load HMDA data from the local HDF store.

    Backward-compatible wrapper around ``load_hmda``; expects ``yr`` in
    ``kwargs``.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing the HDF store.  When provided it is
        forwarded as the ``data_dir`` keyword argument to ``load_hmda``.
    **kwargs
        Additional keyword arguments passed directly to ``load_hmda``.
        Must include ``yr`` (int) – the HMDA survey year to load.

    Returns
    -------
    pandas.DataFrame
        HMDA loan-application records for the requested year.
    """
    if data_dir is not None:
        kwargs.setdefault('data_dir', data_dir)
    if 'save_dir' not in kwargs or kwargs.get('save_dir') is None:
        kwargs['save_dir'] = kwargs.get('data_dir', default_dir)
    return load_hmda(**kwargs)


def cat(num):
    """Return a list of integers from 1 to num inclusive.

    Parameters
    ----------
    num : int
        Upper bound of the integer range (inclusive).

    Returns
    -------
    list
        ``[1, 2, ..., num]``.
    """
    return list(range(1, num+1))

def to_float(df, var):
    """Convert a DataFrame column to float64 in-place.

    Non-numeric values are coerced to ``NaN``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose column will be converted.
    var : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        The same ``df`` with ``df[var]`` cast to ``numpy.float64``.
    """
    df[var] = pd.to_numeric(df[var], errors='coerce').astype(np.float64)
    return df

def load_chunk(df, drop_columns=None, obj_columns=None, categories=None):
    """Process one chunk of HMDA data by dropping, casting, or categorising columns.

    For each column in ``df``:

    * columns in ``drop_columns`` are removed;
    * columns in ``obj_columns`` are cast to ``object`` dtype;
    * columns present as keys in ``categories`` are converted to
      ``pandas.Categorical`` with the supplied category list;
    * all remaining columns are converted to ``float64`` via :func:`to_float`.

    Parameters
    ----------
    df : pandas.DataFrame
        A single chunk of raw HMDA data.
    drop_columns : list, optional
        Column names to drop entirely.  Defaults to an empty list.
    obj_columns : list, optional
        Column names to cast to ``object`` dtype.  Defaults to an empty list.
    categories : dict, optional
        Mapping of column name to list of valid category values.  Columns
        present in this mapping are converted to ``pandas.Categorical``.
        Defaults to an empty dict.

    Returns
    -------
    pandas.DataFrame
        The processed chunk with columns dropped, recast, or categorised as
        specified.
    """
    if drop_columns is None:
        drop_columns = []
    if obj_columns is None:
        obj_columns = []
    if categories is None:
        categories = {}

    for col in df.columns:
        if col in drop_columns:
            df.drop(col, axis=1, inplace=True)
        elif col in obj_columns:
            df[col] = df[col].astype('object')
        elif col in categories:
            df[col] = pd.Categorical(df[col], categories=categories[col])
        else:
            to_float(df, col)

    return df

def store(yr, data_dir=default_dir, save_dir=default_dir, nrows=None,
          usecols=None, reimport=False, chunksize=500000):
    """Read a raw HMDA fixed-width file in chunks and persist it to HDF5.

    The compressed fixed-width file for ``yr`` is read with
    ``pandas.read_fwf`` in chunks of ``chunksize`` rows.  Each chunk is
    processed by :func:`load_chunk` (columns dropped, cast, or categorised)
    and appended to the HDF5 store under the key ``hmda_{yr}``.

    Parameters
    ----------
    yr : int
        HMDA survey year.  Controls both the source filename and the HDF5
        key used for storage.
    data_dir : str, optional
        Directory that contains the raw compressed fixed-width source files.
        Defaults to the configured HMDA data directory.
    save_dir : str, optional
        Directory where the HDF5 store (``hmda.hd5``) will be written.
        Defaults to the configured HMDA data directory.
    nrows : int, optional
        Maximum number of rows to read from the source file.  ``None`` reads
        all rows.
    usecols : list, optional
        Subset of column names to read from the source file.  ``None`` reads
        all columns.
    reimport : bool, optional
        Currently unused; reserved for future use to force re-ingestion even
        when the store key already exists.
    chunksize : int, optional
        Number of rows per chunk passed to ``pandas.read_fwf``.  Defaults to
        500 000.

    Returns
    -------
    None
    """
    store_file = save_dir + 'hmda.hd5'
    key = 'hmda_{}'.format(yr)
    store = pd.HDFStore(store_file)

    if yr == 2001:
        filename = 'HMS.U2001.LARS.PUBLIC.DATA'
    elif yr == 2004:
        filename = 'u2004lar.public.dat'
    elif yr in [2005, 2006]:
        filename = 'LARS.ULTIMATE.{}.DAT'.format(yr)
    elif yr in [2007, 2008]:
        filename = 'lars.ultimate.{}.dat'.format(yr)
    elif yr == 2009:
        filename = '2009_Ultimate_PUBLIC_LAR.dat'
    elif yr > 2009:
        filename = 'Lars.ultimate.{0}.dat'.format(yr)
    else:
        filename = 'HMS.U{}.LARS'.format(yr)

    if yr < 2004:
        widths = [
                4, 10, 1, 1, 1,
                1, 5, 1, 4, 2,
                3, 7, 1, 1, 1,
                1, 4, 1, 1, 1,
                1, 1, 7,
                ]
        names = [
                'asof_date', 'resp_id', 'agency_code', 'loan_type', 'loan_purp',
                'occupancy', 'loan_amt', 'action_taken', 'prop_msa', 'state_code',
                'county_code', 'census_tract', 'app_race', 'co_app_race', 'app_sex',
                'co_app_sex', 'app_income', 'purchaser_type', 'denial_reason_1', 'denial_reason_2',
                'denial_reason_3', 'edit_status', 'seq_num',
                ]
    else:
        widths=[
                4, 10, 1, 1, 1,
                1, 5, 1, 5, 2,
                3, 7, 1, 1, 4,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 5,
                1, 1, 7,
                ]
        names = [
                'asof_date', 'resp_id', 'agency_code', 'loan_type', 'loan_purp',
                'occupancy', 'loan_amt', 'action_taken', 'prop_msa', 'state_code',
                'county_code', 'census_tract', 'app_sex', 'co_app_sex', 'app_income', 
                'purchaser_type', 'denial_reason_1', 'denial_reason_2', 'denial_reason_3', 'edit_status', 
                'prop_type', 'preapprovals', 'app_ethnicity', 'co_app_ethnicity', 'app_race_1', 
                'app_race_2', 'app_race_3', 'app_race_4', 'app_race_5', 'co_app_race_1', 
                'co_app_race_2', 'co_app_race_3', 'co_app_race_4', 'co_app_race_5', 'rate_spread',
                'hoepa_status', 'lien_status', 'seq_num',
                ]

    # filepath = data_dir + filename + '.zip?download=true'
    filepath = data_dir + filename + '.zip'
    reader = pd.read_fwf(filepath, widths=widths, names=names,
            usecols=usecols, nrows=nrows, compression='zip',
            chunksize=chunksize)

    data_columns = ['loan_type', 'loan_purp', 'occupancy',
            'action_taken', 'lien_status', 'purchaser_type']

    obj_columns = []
    drop_columns = ['resp_id', 'agency_code', 'app_sex', 'co_app_sex',
            'app_ethnicity', 'co_app_ethnicity', 'app_race', 'co_app_race',
            'app_race_1', 'app_race_2', 'app_race_3', 'app_race_4',
            'app_race_5', 'co_app_race_1', 'co_app_race_2', 'co_app_race_3',
            'co_app_race_4', 'co_app_race_5', 'hoepa_status', 'seq_num']

    # cat_vars = ['']
    categories = {
            'loan_type' : cat(4),
            'prop_type' : cat(3),
            'loan_purp' : cat(3),
            'occupancy' : cat(3),
            'preapprovals' : cat(3),
            'action_taken' : cat(8),
            'denial_reason_1' : cat(9),
            'denial_reason_2' : cat(9),
            'denial_reason_3' : cat(9),
            'edit_status' : list(range(5, 8)),
            'state_code' : list(range(1, 100)),
            'purchaser_type' : list(range(10)),
            'lien_status' : list(range(5)),
            }

    for ii, df in enumerate(reader):
        
        print("reading chunk {}".format(ii))

        for col in df.columns:
            if col in drop_columns:
                df.drop(col, axis=1, inplace=True)
            elif col in obj_columns:
                df[col] = df[col].astype('object')
            elif col in categories:
                df[col] = pd.Categorical(df[col], categories=categories[col])
            else:
                to_float(df, col)

        if ii == 0:
            store.append(key, df, append=False, data_columns=data_columns)
        else:
            store.append(key, df, data_columns=data_columns)

    store.close()

    return None 

def load_hmda(yr, data_dir=default_dir, save_dir=default_dir, query=None,
              columns=None):
    """Load HMDA data for a given year from the local HDF5 store.

    Opens the HDF5 store at ``save_dir/hmda.hd5``, selects the table stored
    under the key ``hmda_{yr}``, and returns the result as a DataFrame,
    optionally filtered by a query expression and/or a subset of columns.

    Parameters
    ----------
    yr : int
        HMDA survey year to load (e.g. ``2010``).
    data_dir : str, optional
        Directory containing the raw source files (not used during loading,
        but kept for API consistency with :func:`store`).  Defaults to the
        configured HMDA data directory.
    save_dir : str, optional
        Directory containing the HDF5 store (``hmda.hd5``).  Defaults to the
        configured HMDA data directory.
    query : str, optional
        PyTables query string passed to ``HDFStore.select`` for row filtering
        (e.g. ``"state_code == 6"``).  ``None`` returns all rows.
    columns : list, optional
        List of column names to return.  ``None`` returns all columns.

    Returns
    -------
    pandas.DataFrame
        HMDA loan-application records for the requested year, filtered
        according to ``query`` and ``columns``.
    """
    store_file = save_dir + 'hmda.hd5'
    key = 'hmda_{}'.format(yr)
    store = pd.HDFStore(store_file)

    store.open()
    df = store.select(key, query, columns=columns)
    store.close()

    return df
