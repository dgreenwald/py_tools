import numpy as np
import pandas as pd
import py_tools.data as dt

data_dir = '/home/dan/Dropbox/data/qd/'

def load(vintage='2015-11', reimport=False, transform='var'):

    if reimport:

        infile = data_dir + vintage + '.csv'
        df = pd.read_csv(infile)

    return df

df_all = load(reimport=True)

factors = df_all.iloc[0].values[1:].astype(int)
transform = df_all.iloc[1].values[1:].astype(int)

df = dt.date_index(df_all.iloc[2:].copy(), '1/1/1959', freq='QS')
del df['sasdate']

df_orig = df.copy()

for ii, var in enumerate(df.columns):

    trans_code = transform[ii]
    if trans_code in [4, 5, 6]:
        df[var] = np.log(df[var])
