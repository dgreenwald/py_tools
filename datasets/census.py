import os
import numpy as np
import pandas as pd

from . import defaults, misc
default_dir = defaults.base_dir() + 'census/'

def load_pop(level, year, data_dir=default_dir):

    year_dir = data_dir + '{:d}_pop/'.format(year)

    level_str_map = {
        'msa' : 'msa_cmsa',
        'zip' : 'zcta',
        'zip3' : 'zcta',      
        'county' : 'county',
        'fips' : 'county',
    }

    level_str = level_str_map.get(level, level)

    infile = year_dir + 'nhgis0001_ds146_2000_' + level_str + '.csv'

    df = pd.read_csv(infile)
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
        df = df.rename(columns={'zip3a' : 'zip3'}).groupby('zip3')['pop', 'hh'].sum().reset_index()
    elif level == 'state':
        state_codes = misc.load('state_codes').rename(columns={'state_abbr' : 'state'})
        df = df[['statea', 'pop', 'hh']]
        df = pd.merge(df, state_codes, left_on='statea', right_on='state_code')
        df = df[['state', 'statea', 'pop', 'hh']]

    return df