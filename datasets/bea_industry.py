import pandas as pd

from . import config
default_dir = config.base_dir() + 'bea_industry/'
DATASET_NAME = "bea_industry"
DESCRIPTION = "BEA industry accounts dataset loader."
def load(freq='q', data_dir=default_dir):

    df = pd.read_feather(data_dir + freq + 'profits.feather')
    return df
