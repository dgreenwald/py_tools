#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:29:33 2023

@author: dan
"""

import numpy as np
import pandas as pd

from py_tools import time_series as ts

from . import defaults
default_dir = defaults.base_dir('DAN')

def load(vintage='2310', user='DAN', master_dirs={}, freq='Q'):
    
    default_dir = defaults.base_dir(user)
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