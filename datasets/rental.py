import numpy as np
import os
import pandas as pd

default_base = os.environ.get('DROP_DIR', '/home/dan/Dropbox/') + 'greenwald_guren/Data/Rents/'

def load(dataset, base_dir=default_base):

    if dataset == 'tw':

        data_dir = default_base + 'CBRE_EA/'
        df = pd.read_csv(data_dir + 'CBRE-EA.csv')
        df['year'] = np.floor(df['Time']).astype(int)
        df['q'] = np.round(10.0 * (df['Time'] - df['year'].astype(float))).astype(int)
        df['m'] = 3 * df['q'] - 2

    return df
