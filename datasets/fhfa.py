import pandas as pd
from py_tools import time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'fhfa/'

def load(dataset, all_transactions=True, data_dir=default_dir):

    if dataset == 'metro':

        if not all_transactions: raise Exception

        # df = pd.read_csv(data_dir + 'HPI_AT_metro.csv')
        df = pd.read_csv(data_dir + 'HPI_AT_metro.csv', header=None, 
                         names=['MSA', 'code', 'year', 'q', 'hpi', 'unknown'])

        df['month'] = df['q'] * 3 - 2
        df['day'] = 1
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

        df['hpi'] = pd.to_numeric(df['hpi'], errors='coerce')

        df = df.set_index(['date', 'MSA'])
        df = df.drop(columns=['year', 'q', 'month', 'day', 'unknown'])
        # df['date'] = df['yr'].astype('str') + '-' + (3*df['qtr'] - 2).astype('str') + '-01'
        # df['date'] = pd.to_datetime(df['date'])
        # df = df.drop(columns=['yr', 'qtr', 'Warning'])

    elif dataset == 'state':

        if all_transactions:
            df = pd.read_table(data_dir + 'HPI_AT_state.txt', names=['state', 'yr', 'qtr', 'hpi'])
        else:
            df = pd.read_table(data_dir + 'HPI_PO_state.txt')
            df = df.drop(columns=['Warning'])

        df['date'] = df['yr'].astype('str') + '-' + (3*df['qtr'] - 2).astype('str') + '-01'
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'state'])

    df = df.apply(pd.to_numeric, errors='coerce')

    return df
