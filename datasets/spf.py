import numpy as np
import os
import pandas as pd
import py_tools.time_series as ts

from . import config
default_dir = config.base_dir()

idx = pd.IndexSlice
DATASET_NAME = "spf"
DESCRIPTION = "Survey of Professional Forecasters (SPF) dataset loader."
def load(table, data_dir=default_dir+'/spf/', reimport=False):
    """Load SPF mean forecast level for a single variable from CSV.

    Reads the Philadelphia Fed SPF mean-level CSV file for the specified
    variable, constructs a quarterly date index, and caches the result as
    a pickle file.

    Parameters
    ----------
    table : str
        Variable name to load (e.g. ``'RGDP'``). Used to locate the file
        ``Mean_<TABLE>_Level.csv`` and to select the output column.
    data_dir : str, optional
        Path to the directory containing the SPF CSV files.
    reimport : bool, optional
        If ``True``, re-read from the source CSV even if a cached pickle
        exists.

    Returns
    -------
    pandas.DataFrame
        Quarterly date-indexed DataFrame with a single column named after
        ``table`` (uppercased).
    """
    filebase = data_dir + 'Mean_' + table.upper() + '_Level'
    pkl_file = filebase + '.pkl'
    
    if reimport or not os.path.exists(pkl_file):
        # excel_file = filebase + '.xlsx'
        # df = pd.read_excel(excel_file)
        df = pd.read_csv(filebase + '.csv')
        df['date'] = ts.date_from_qtr(df['YEAR'], df['QUARTER'])
        df = df.set_index('date')[[table.upper()]]
        df.to_pickle(pkl_file)
    else:
        df = pd.read_pickle(pkl_file)
    
    return df

def load_master(table, data_dir=default_dir + '/spf/', reimport=False, growth=False):
    """Load SPF forecasts for a table from the master Excel file.

    Reads the specified sheet from the SPF master mean-level or mean-growth
    Excel workbook, constructs a quarterly date index, and caches the result
    as a pickle file.

    Parameters
    ----------
    table : str
        Sheet name in the master Excel file corresponding to the variable
        or forecast horizon to load.
    data_dir : str, optional
        Path to the directory containing the SPF master Excel file.
    reimport : bool, optional
        If ``True``, re-read from the source Excel file even if a cached
        pickle exists.
    growth : bool, optional
        If ``True``, read from ``meanGrowth.xlsx``; if ``False`` (default),
        read from ``meanLevel.xlsx``.

    Returns
    -------
    pandas.DataFrame
        Quarterly date-indexed DataFrame of SPF forecast columns for the
        requested table.
    """
    if growth:
        base_name = 'meanGrowth'
    else:
        base_name = 'meanLevel'
    
    pkl_file = data_dir + base_name + '_' + table + '.pkl'
    
    if reimport or not os.path.exists(pkl_file):
        
        excel_file = data_dir + base_name + '.xlsx'
        df = pd.read_excel(excel_file, sheet_name=table)
        
        df = df.dropna(subset=['YEAR', 'QUARTER'])
        for var in ['YEAR', 'QUARTER']:
            df[var] = df[var].astype(int)
            
        df['date'] = ts.date_from_qtr(df['YEAR'], df['QUARTER'])
        df = df.set_index('date').drop(columns=['YEAR', 'QUARTER'])
        
        df.to_pickle(pkl_file)
        
    else:
        
        df = pd.read_pickle(pkl_file)
        
    return df