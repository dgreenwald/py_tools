import pandas as pd
import os

from . import config
default_dir = config.base_dir() + 'crosswalk/'

data_dir = default_dir

def county_to_zip(year=2000, zip_level=5, data_dir=default_dir, reimport=False):
    """Build a county-to-ZIP crosswalk with population weights.

    Reads a raw CSV crosswalk for the given year, optionally truncates ZIP
    codes to fewer digits, and computes within-county population-share
    weights so that each county's weights sum to 1.  Results are cached as
    a pickle file and reused on subsequent calls unless ``reimport`` is
    ``True``.

    Parameters
    ----------
    year : int, optional
        Reference year of the crosswalk file.  Defaults to 2000.
    zip_level : int, optional
        Number of ZIP-code digits to retain (1–5).  When less than 5 the
        ZIP codes are truncated and population weights are re-aggregated.
        Defaults to 5.
    data_dir : str, optional
        Directory containing crosswalk CSV and pickle files.  Defaults to
        the package-level ``default_dir``.
    reimport : bool, optional
        If ``True``, re-read the raw CSV even if a cached pickle exists.
        Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Crosswalk with columns ``fips``, ``zip`` (or ``zip<N>`` for
        truncated codes), and ``factor`` (population-share weight).
    """
    assert zip_level in [1, 2, 3, 4, 5]

    pkl_file = data_dir + 'county_to_zip{0:d}_{1:d}.pkl'.format(zip_level, year)
    if reimport or (not os.path.exists(pkl_file)):

        names = ['fips', 'zip', 'county_name', 'zip_name', 'pop', 'factor']
        df = pd.read_csv(data_dir + 'county_to_zip_{:d}.csv'.format(year),
                         skiprows=2, header=None, names=names, 
                         usecols=['fips', 'zip', 'factor'])

        if zip_level < 5:
            to_drop = 5 - zip_level
            zip_var = 'zip' + str(zip_level)
            df[zip_var] = df['zip'].astype(str).str[:-to_drop].astype(int)
            df = df.groupby(['fips', zip_var])['factor'].sum().to_frame(name='factor')
            df = df.reset_index()
        else:
            zip_var = 'zip'
            
        df['total'] = df.groupby(['fips'])['factor'].transform(sum)
        df['factor'] /= df['total']
        df = df.drop(columns=['total'])
            
        df.to_pickle(pkl_file)

    else:

        df = pd.read_pickle(pkl_file)

    return df

def county_to_zip_hud(data_dir=default_dir, reimport=True):
    """Load the HUD COUNTY_ZIP Excel file as a county-to-ZIP crosswalk.

    Reads ``COUNTY_ZIP_122021.xlsx`` from ``data_dir``, caches the result
    as a Parquet file for faster subsequent loads, and returns the
    DataFrame.  When ``reimport`` is ``True`` (the default) the Excel file
    is always re-read; otherwise the cached Parquet file is used if it
    exists.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the HUD Excel and Parquet files.  Defaults to
        the package-level ``default_dir``.
    reimport : bool, optional
        If ``True``, re-read the raw Excel file even if a cached Parquet
        file exists.  Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        HUD county-to-ZIP crosswalk with all columns from the source Excel
        file.
    """
    parquet_file = data_dir + 'county_to_zip_hud.parquet'
    if reimport or (not os.path.exists(parquet_file)):
        df = pd.read_excel(data_dir + 'COUNTY_ZIP_122021.xlsx')
        df.to_parquet(parquet_file)
    else:
        df = pd.read_parquet(parquet_file)
        
    return df

def zip_to_county_hud(data_dir=default_dir, reimport=True):
    """Load the HUD ZIP_COUNTY Excel file as a ZIP-to-county crosswalk.

    Reads ``ZIP_COUNTY_122021.xlsx`` from ``data_dir``, caches the result
    as a Parquet file for faster subsequent loads, and returns the
    DataFrame.  When ``reimport`` is ``True`` (the default) the Excel file
    is always re-read; otherwise the cached Parquet file is used if it
    exists.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the HUD Excel and Parquet files.  Defaults to
        the package-level ``default_dir``.
    reimport : bool, optional
        If ``True``, re-read the raw Excel file even if a cached Parquet
        file exists.  Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        HUD ZIP-to-county crosswalk with all columns from the source Excel
        file.
    """
    parquet_file = data_dir + 'zip_to_county_hud.parquet'
    if reimport or (not os.path.exists(parquet_file)):
        df = pd.read_excel(data_dir + 'ZIP_COUNTY_122021.xlsx')
        df.to_parquet(parquet_file)
    else:
        df = pd.read_parquet(parquet_file)
        
    return df
        

#def load(origin='county', destination='zip', year=2000, data_dir=default_dir, reimport=False):
#
#    intermediate_dests = {
#        'zip3' :' zip',
#    }
#
#    pkl_file = data_dir + '{0}_to_{1}_{2:d}.pkl'.format(origin, destination, year)
#    if reimport or os.path.exists(pkl_file):
#
#        this_dest = intermediate_dests.get(destination, destination)
#        df = pd.read_csv(data_dir + '{0}_to_{1}_{2:d}'.format(origin, this_dest))
#
#    else:
#
#        df = pd.read_pickle(pkl_file)
#
#    return df
