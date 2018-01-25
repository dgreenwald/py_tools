import pandas as pd

from . import defaults
default_dir = defaults.base_dir() + 'irs/'
# data_dir = '/home/dan/data/irs/'

def load(year, data_dir=default_dir, **kwargs):

    if year >= 2009:

        yy = year - 2000
        infile = data_dir + '{:02d}zpallagi.csv'.format(yy)
        df = pd.read_csv(infile)
        
    else:

        raise Exception

    return df
