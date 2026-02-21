import pandas as pd

from . import config
default_dir = config.base_dir() + 'dfa/'
DATASET_NAME = "dfa"
DESCRIPTION = "Dataset loader for dfa."


def load(dataset, levels=True, data_dir=default_dir, **kwargs):
    """Load pre-packaged set of variables"""
    
    filename = 'dfa-' + dataset + '-'
    
    if levels:
        filename += 'levels'
    else:
        filename += 'shares'
        
    filename += '.csv'
    
    return pd.read_csv(data_dir + filename, **kwargs)