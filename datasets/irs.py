import pandas as pd

data_dir = '/home/dan/data/irs/'

def load(year, **kwargs):

    if year >= 2009:

        yy = year - 2000
        infile = data_dir + '{:02d}zpallagi.csv'.format(yy)
        df = pd.read_csv(infile)
        
    else:

        raise Exception

    return df
