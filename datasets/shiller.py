#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:29:33 2023

@author: dan
"""

import pandas as pd

from py_tools import time_series as ts

from . import config
default_dir = config.base_dir()
DATASET_NAME = "shiller"
DESCRIPTION = "Shiller long-run asset price and valuation dataset loader."
def load(vintage='2310', user='DAN', master_dirs={}, freq='Q'):
    """Load Shiller long-run equity price/yield data from Excel.

    Reads the Shiller IE data Excel file for the specified vintage, selects
    relevant columns, and optionally resamples from monthly to quarterly
    frequency.

    Parameters
    ----------
    vintage : str, optional
        Date code identifying the data vintage (e.g. ``'2310'`` for October
        2023).
    user : str, optional
        User identifier used for directory resolution (currently unused
        internally but reserved for future multi-user path logic).
    master_dirs : dict, optional
        Override directory paths. If ``'base'`` key is absent, the default
        base directory from config is used.
    freq : str, optional
        Output frequency. ``'M'`` returns monthly data; ``'Q'`` (default)
        resamples to quarterly frequency, summing dividends/earnings and
        taking period-end values for prices/CAPE.

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame of Shiller equity price, dividend, earnings,
        CPI, GS10, and CAPE series at the requested frequency.
    """
    default_dir = config.base_dir()
    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = default_dir

    colnames = [
        'Date', 'P', 'D', 'E', 'CPI', 'date_frac', 'GS10', 
        'real_P', 'real_D', 'real_E', 'CAPE',
    ]

    df_m = pd.read_excel(
        dirs['base'] + f'shiller/ie_data_{vintage}.xls',
        sheet_name='Data',
        skiprows=7,
        # colnames=colnames,
        # usecols=usecols,
    )

    df_m = df_m.rename(columns={
        'Rate GS10' : 'GS10',
        'Fraction' : 'date_frac',
        'Price' : 'real_P',
        'Dividend' : 'real_D',
        'Earnings' : 'real_E',
    })

    df_m = df_m[colnames]
    df_m = ts.date_index(df_m, '01/01/1871', freq='MS')

    if freq == 'M':
        return df_m

    assert freq == 'Q'

    methods_vars = {
        # 'sum' : ['Dividend', 'Earnings'],
        # 'last' : ['Price', 'CAPE'],
        'sum' : ['real_D', 'real_E'],
        'last' : ['real_P', 'CAPE'],
    }
    
    df = ts.resample(df_m, methods_vars)

    return df