import os
import numpy as np
import pandas as pd
import py_tools.time_series as ts

data_dir = '/home/dan/Dropbox/data/jst/'

def load():

    dta_file = data_dir + 'JSTdatasetR2.dta'
    df = pd.read_stata(dta_file)

    return df
