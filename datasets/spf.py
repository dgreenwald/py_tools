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