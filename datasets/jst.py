import os
import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import config
default_dir = config.base_dir() + 'jst/'
# data_dir = '/home/dan/Dropbox/data/jst/'
DATASET_NAME = "jst"
DESCRIPTION = "Dataset loader for jst."


def load(data_dir=default_dir):

    dta_file = data_dir + 'JSTdatasetR2.dta'
    df = pd.read_stata(dta_file)

    return df
