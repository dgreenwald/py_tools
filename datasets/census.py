import os
import numpy as np
import pandas as pd

from . import defaults, misc
default_dir = defaults.base_dir() + 'census/'

def load_pop(level, year, data_dir=default_dir):

    year_dir = data_dir + '{:d}_pop/'.format(year)

    if level == 'msa':
        level_str = 'msa_cmsa'
    else:
        level_str = level

    infile = year_dir + 'nhgis0001_ds146_2000_' + level_str + '.csv'

    df = pd.read_csv(infile)
    df = df.rename(columns={name : name.lower() for name in df.columns})
    df = df.rename(columns={
        'fl5001' : 'pop',
        'fnh001' : 'hh',
    })

    return df
