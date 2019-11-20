import numpy as np
import pandas as pd
import os

from . import defaults, misc
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
