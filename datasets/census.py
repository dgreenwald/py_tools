import glob, os
import numpy as np
import pandas as pd

from . import config, misc
default_dir = config.base_dir() + 'census/'
DATASET_NAME = "census"
DESCRIPTION = "U.S. Census population dataset loader."
def load(data_dir=None, **kwargs):
    """Load census population data.

    Backward-compatible wrapper around :func:`load_pop`.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the census data files.  When provided it is
        forwarded to :func:`load_pop` as the ``data_dir`` argument.  Default
        is ``None``, which causes :func:`load_pop` to use its own default.
    **kwargs
        Additional keyword arguments forwarded verbatim to :func:`load_pop`
        (e.g. ``level``, ``year``, ``infile``).

    Returns
    -------
    pandas.DataFrame
        Census population data as returned by :func:`load_pop`.
    """
    if data_dir is not None:
        kwargs.setdefault('data_dir', data_dir)
    return load_pop(**kwargs)


def load_pop(level, year, data_dir=default_dir, infile=None):
    """Load U.S. Census population data from an NHGIS CSV file.

    Reads a National Historical Geographic Information System (NHGIS) CSV for
    the requested geographic ``level`` and ``year``, standardises column names,
    and returns a tidy DataFrame.

    Parameters
    ----------
    level : str
        Geographic level of aggregation.  One of ``'msa'``, ``'zip'``,
        ``'zip3'``, ``'county'``, ``'fips'``, or ``'state'``.
    year : int
        Census year (e.g. ``2000``).  Used to locate the year-specific
        subdirectory inside ``data_dir``.
    data_dir : str, optional
        Root directory containing year-specific population subdirectories.
        Default is the package-level ``default_dir``.
    infile : str or None, optional
        Explicit path to the NHGIS CSV file.  When ``None`` (default), the
        path is constructed automatically from ``data_dir``, ``year``, and
        ``level``.

    Returns
    -------
    pandas.DataFrame
        Population data with columns depending on ``level``:

        - ``'fips'``, ``'pop'``, ``'hh'`` for ``'county'`` / ``'fips'``.
        - ``'zip'``, ``'pop'``, ``'hh'`` for ``'zip'`` / ``'zcta'``.
        - ``'zip3'``, ``'pop'``, ``'hh'`` for ``'zip3'``.
        - ``'state'``, ``'statea'``, ``'pop'``, ``'hh'`` for ``'state'``.
    """
    year_dir = data_dir + '{:d}_pop/'.format(year)

    level_str_map = {
        'msa' : 'msa_cmsa',
        'zip' : 'zcta',
        'zip3' : 'zcta',      
        'county' : 'county',
        'fips' : 'county',
    }

    level_str = level_str_map.get(level, level)

    if infile is None:
        infile = year_dir + 'nhgis0001_ds146_2000_' + level_str + '.csv'

    df = pd.read_csv(infile, encoding='latin1')
    df = df.rename(columns={name : name.lower() for name in df.columns})
    df = df.rename(columns={
        'fl5001' : 'pop',
        'fnh001' : 'hh',
    })
    
    if level in ['county', 'fips']:
        df['fips'] = 1000 * df['statea'] + df['countya']
        df = df[['fips', 'pop', 'hh']]
    elif level in ['zip', 'zcta']:
        df = df.rename(columns={'zctaa' : 'zip'})
        df = df[['zip', 'pop', 'hh']]
    elif level == 'zip3':
        df = df.rename(columns={'zip3a' : 'zip3'}).groupby('zip3')[['pop', 'hh']].sum().reset_index()
    elif level == 'state':
        state_codes = misc.load('state_codes').rename(columns={'state_abbr' : 'state'})
        df = df[['statea', 'pop', 'hh']]
        df = pd.merge(df, state_codes, left_on='statea', right_on='state_code')
        df = df[['state', 'statea', 'pop', 'hh']]

    return df
