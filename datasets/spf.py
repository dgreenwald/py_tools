import numpy as np
import os
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir()

idx = pd.IndexSlice

def load(table, data_dir=default_dir+'/spf/', reimport=False):
    
    filebase = data_dir + 'Mean_' + table.upper() + '_Level'
    pkl_file = filebase + '.pkl'
    
    if reimport or not os.path.exists(pkl_file):
        excel_file = filebase + '.xlsx'
        df = pd.read_excel(excel_file)
        df['date'] = ts.date_from_qtr(df['YEAR'], df['QUARTER'])
        df = df.set_index('date')[[table.upper()]]
        df.to_pickle(pkl_file)
    else:
        df = pd.read_pickle(pkl_file)
    
    return df