import os
import numpy as np
import pandas as pd

from . import config, misc
default_dir = config.base_dir() + 'bea_industry/'
DATASET_NAME = "bea_industry"
DESCRIPTION = "BEA industry accounts dataset loader."
def load(freq='q', data_dir=default_dir):

    df = pd.read_feather(data_dir + freq + 'profits.feather')
    linker = pd.read_csv(data_dir + 'NAIC_{}profits.csv'.format(freq))[['Industry', 'NAIC']]

    return df
