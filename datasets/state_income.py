import numpy as np
import pandas as pd
import os

from . import defaults, fred, misc
default_dir = defaults.base_dir() + 'state_income/'

data_dir = default_dir

def load(data_dir=default_dir, reimport=False):

    pkl_file = data_dir + 'state_annual_income_long.pkl'

    if reimport or (not os.path.exists(pkl_file)):

        state_codes = misc.load('state_codes')
        state_codes['fips'] = state_codes['state_code'] * 1000

        state_incomes = pd.read_csv(data_dir+'state_annual_income.csv', skiprows=4).dropna(subset=['1998'])
        state_incomes['GeoFips'] = pd.to_numeric(state_incomes['GeoFips'], errors='coerce')
        state_incomes = pd.merge(state_incomes, state_codes, left_on='GeoFips', right_on='fips')

        si_long = pd.melt(state_incomes, id_vars=['state_abbr'], 
                          value_vars=[str(year) for year in range(1998, 2018)], 
                          var_name='year', value_name='pc_income')

        si_long['date'] = pd.to_datetime(si_long['year'] + '-01-01')
        si_long = si_long.drop(columns=['year'])
        si_long.to_pickle(pkl_file)

    else:

        si_long = pd.read_pickle(pkl_file)

    return si_long

def load_fred(data_dir=default_dir, reimport=False):
    
    pkl_file = data_dir + 'state_pc_income_fred.pkl'
    
    if reimport or (not os.path.exists(pkl_file)):
    
        state_codes = misc.load('state_codes')
        ix = ~state_codes['state_abbr'].isin(['AS', 'GU', 'PR', 'VI'])
        state_codes = state_codes.loc[ix, :]
        
        code_names = {
                '{}PCPI'.format(abbr) : abbr for abbr in state_codes['state_abbr'].values
                }
        
        df_wide = fred.load(code_names=code_names, reimport=reimport)
        df_wide = df_wide.reset_index().rename(columns={'DATE' : 'date'})
        df = pd.melt(df_wide, id_vars='date', value_vars=df_wide.columns[1:], 
                     var_name='state', value_name='pc_income')
        df = df.set_index(['state', 'date']).sort_index()
        
        df.to_pickle(pkl_file)
        
    else:
        
        df = pd.read_pickle(pkl_file)
        
    return df

def load_median(data_dir=default_dir, reimport=False, fred_reimport=True):
    
    pkl_file = data_dir + 'state_median_income.pkl'
    
    if reimport or (not os.path.exists(pkl_file)):
    
        state_codes = misc.load('state_codes')
        ix = ~state_codes['state_abbr'].isin(['AS', 'GU', 'PR', 'VI'])
        state_codes = state_codes.loc[ix, :]
        
        code_names = {
                'MEHOINUS{}A646N'.format(abbr) : abbr for abbr in state_codes['state_abbr'].values
                }
        
        df_wide = fred.load(code_names=code_names, reimport=fred_reimport)
        df_wide = df_wide.reset_index().rename(columns={'DATE' : 'date'})
        df = pd.melt(df_wide, id_vars='date', value_vars=df_wide.columns[1:], 
                     var_name='state', value_name='median_income')
        df = df.set_index(['state', 'date']).sort_index()
        
        df.to_pickle(pkl_file)
        
    else:
        
        df = pd.read_pickle(pkl_file)

    return df