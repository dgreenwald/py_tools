import pandas as pd

from . import defaults
default_dir = defaults.base_dir() + 'compustat/'

# def load(dataset='quarterly', data_dir=default_dir):

def read_single_chunk(name, data_dir=default_dir, keep_cols=None, chunksize=10000):

    filepath = data_dir + name + '.sas7bdat'
    
    print("NOTE: NOT LOADING FULL FILE")
    reader = pd.read_sas(filepath, chunksize=chunksize)
    for df in reader:
        break
    
    if keep_cols is not None:
        df = df[pd.Index.intersection(df.columns, keep_cols)] 
        
    return df

def read_by_chunk(name, data_dir=default_dir, keep_cols=None, chunksize=10000):

    filepath = data_dir + name + '.sas7bdat'

    reader = pd.read_sas(filepath, chunksize=chunksize)
    return pd.concat([chunk[pd.Index.intersection(chunk.columns, keep_cols)] 
                            for chunk in reader])
