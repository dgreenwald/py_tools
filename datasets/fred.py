import datetime
import os
import pandas as pd

from pandas_datareader import data as web
import pickle

from . import defaults
default_dir = defaults.base_dir('DAN') + 'fred/'
# data_dir = '/home/dan/Dropbox/data/fred/'

def load(codes=None, code_names={}, 
         start=datetime.datetime(1900, 1, 1),
         end = datetime.datetime.today(),
         reimport=False, data_dir=default_dir):
    """Load data from FRED, will replace codes with names if code_names is passed as a dict"""

    if codes is None:
        codes = list(code_names.keys())

    stored_series_file = data_dir + 'series.pkl'
    if os.path.isfile(stored_series_file):
        stored_series = pickle.load(open(stored_series_file, 'rb'))
    else:
        stored_series = []

    if reimport:
        download_series = codes
    else:
        download_series = list(set(codes) - set(stored_series))

    df = None

    if download_series:
        # df = web.DataReader(codes, "fred", start, end).rename(columns=code_names)
        df = web.DataReader(download_series, "fred", start, end)

        # Store
        for series in download_series:
            df[series].to_pickle(data_dir + series + '.pkl')

        stored_series = list(set(stored_series + download_series))
        pickle.dump(stored_series, open(stored_series_file, 'wb'))

    # Now load stored series
    for series in set(codes) - set(download_series):

        infile = data_dir + series + '.pkl'
        df_in = pd.read_pickle(infile).to_frame()
        if df is None:
            df = df_in
        else:
            df = pd.merge(df, df_in, left_index=True, right_index=True, how='outer')

    df = df.rename(columns=code_names)

    return df
