import pandas as pd

from . import config
default_dir = config.base_dir() + 'compustat/'

# def load(dataset='quarterly', data_dir=default_dir):
DATASET_NAME = "compustat"
DESCRIPTION = "Compustat dataset loader (SAS chunk readers)."
def load(data_dir=None, **kwargs):
    """Load Compustat data by chunk.

    Parameters are passed to either `read_by_chunk` (default) or
    `read_single_chunk` if `single_chunk=True`.
    """
    if data_dir is not None:
        kwargs.setdefault('data_dir', data_dir)

    single_chunk = kwargs.pop('single_chunk', False)
    if single_chunk:
        return read_single_chunk(**kwargs)
    return read_by_chunk(**kwargs)


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
    if keep_cols is None:
        return pd.concat([chunk for chunk in reader])
    return pd.concat([
        chunk[pd.Index.intersection(chunk.columns, keep_cols)]
        for chunk in reader
    ])
