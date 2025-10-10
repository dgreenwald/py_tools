import numpy as np
import pandas as pd
from py_tools import time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'hmda/'

def cat(num):
    return list(range(1, num+1))

def to_float(df, var):
    df[var] = pd.to_numeric(df[var], errors='coerce').astype(np.float64)
    return df

def load_chunk(df):

    for col in df.columns:
        if col in drop_columns:
            df.drop(col, axis=1, inplace=True)
        elif col in obj_columns:
            df[col] = df[col].astype('object')
        elif col in categories:
            df[col] = pd.Categorical(df[col], categories=categories[col])
        else:
            to_float(df, col)

    return df

def store(yr, data_dir=default_dir, save_dir=default_dir, nrows=None,
          usecols=None, reimport=False, chunksize=500000):

    store_file = save_dir + 'hmda.hd5'
    key = 'hmda_{}'.format(yr)
    store = pd.HDFStore(store_file)

    if yr == 2001:
        filename = 'HMS.U2001.LARS.PUBLIC.DATA'
    elif yr == 2004:
        filename = 'u2004lar.public.dat'
    elif yr in [2005, 2006]:
        filename = 'LARS.ULTIMATE.{}.DAT'.format(yr)
    elif yr in [2007, 2008]:
        filename = 'lars.ultimate.{}.dat'.format(yr)
    elif yr == 2009:
        filename = '2009_Ultimate_PUBLIC_LAR.dat'
    elif yr > 2009:
        filename = 'Lars.ultimate.{0}.dat'.format(yr)
    else:
        filename = 'HMS.U{}.LARS'.format(yr)

    if yr < 2004:
        widths = [
                4, 10, 1, 1, 1,
                1, 5, 1, 4, 2,
                3, 7, 1, 1, 1,
                1, 4, 1, 1, 1,
                1, 1, 7,
                ]
        names = [
                'asof_date', 'resp_id', 'agency_code', 'loan_type', 'loan_purp',
                'occupancy', 'loan_amt', 'action_taken', 'prop_msa', 'state_code',
                'county_code', 'census_tract', 'app_race', 'co_app_race', 'app_sex',
                'co_app_sex', 'app_income', 'purchaser_type', 'denial_reason_1', 'denial_reason_2',
                'denial_reason_3', 'edit_status', 'seq_num',
                ]
    else:
        widths=[
                4, 10, 1, 1, 1,
                1, 5, 1, 5, 2,
                3, 7, 1, 1, 4,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 5,
                1, 1, 7,
                ]
        names = [
                'asof_date', 'resp_id', 'agency_code', 'loan_type', 'loan_purp',
                'occupancy', 'loan_amt', 'action_taken', 'prop_msa', 'state_code',
                'county_code', 'census_tract', 'app_sex', 'co_app_sex', 'app_income', 
                'purchaser_type', 'denial_reason_1', 'denial_reason_2', 'denial_reason_3', 'edit_status', 
                'prop_type', 'preapprovals', 'app_ethnicity', 'co_app_ethnicity', 'app_race_1', 
                'app_race_2', 'app_race_3', 'app_race_4', 'app_race_5', 'co_app_race_1', 
                'co_app_race_2', 'co_app_race_3', 'co_app_race_4', 'co_app_race_5', 'rate_spread',
                'hoepa_status', 'lien_status', 'seq_num',
                ]

    # filepath = data_dir + filename + '.zip?download=true'
    filepath = data_dir + filename + '.zip'
    reader = pd.read_fwf(filepath, widths=widths, names=names,
            usecols=usecols, nrows=nrows, compression='zip',
            chunksize=chunksize)

    data_columns = ['loan_type', 'loan_purp', 'occupancy',
            'action_taken', 'lien_status', 'purchaser_type']

    obj_columns = []
    drop_columns = ['resp_id', 'agency_code', 'app_sex', 'co_app_sex',
            'app_ethnicity', 'co_app_ethnicity', 'app_race', 'co_app_race',
            'app_race_1', 'app_race_2', 'app_race_3', 'app_race_4',
            'app_race_5', 'co_app_race_1', 'co_app_race_2', 'co_app_race_3',
            'co_app_race_4', 'co_app_race_5', 'hoepa_status', 'seq_num']

    # cat_vars = ['']
    categories = {
            'loan_type' : cat(4),
            'prop_type' : cat(3),
            'loan_purp' : cat(3),
            'occupancy' : cat(3),
            'preapprovals' : cat(3),
            'action_taken' : cat(8),
            'denial_reason_1' : cat(9),
            'denial_reason_2' : cat(9),
            'denial_reason_3' : cat(9),
            'edit_status' : list(range(5, 8)),
            'state_code' : list(range(1, 100)),
            'purchaser_type' : list(range(10)),
            'lien_status' : list(range(5)),
            }

    for ii, df in enumerate(reader):
        
        print("reading chunk {}".format(ii))

        for col in df.columns:
            if col in drop_columns:
                df.drop(col, axis=1, inplace=True)
            elif col in obj_columns:
                df[col] = df[col].astype('object')
            elif col in categories:
                df[col] = pd.Categorical(df[col], categories=categories[col])
            else:
                to_float(df, col)

        if ii == 0:
            store.append(key, df, append=False, data_columns=data_columns)
        else:
            store.append(key, df, data_columns=data_columns)

    store.close()

    return None 

def load_hmda(yr, data_dir=default_dir, save_dir=default_dir, query=None,
              columns=None):

    store_file = save_dir + 'hmda.hd5'
    key = 'hmda_{}'.format(yr)
    store = pd.HDFStore(store_file)

    store.open()
    df = store.select(key, query, columns=columns)
    store.close()

    return df
