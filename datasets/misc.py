
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir('DAN')

def load(dataset, user='DAN', master_dirs={}, **kwargs):
    """Load data from one-off file"""
    default_dir = defaults.base_dir(user)
    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = default_dir
        # home_dir = os.environ['HOME']
        # dirs['base'] = home_dir + '/Dropbox/data/'

    data_dir = dirs['base'] + 'misc/'

    if dataset == 'bls_labor_share':

        infile = data_dir + 'bls_labor_share.csv'
        df = pd.read_csv(infile, usecols=['Labor Share'])
        ts.date_index(df, '1947/1/1', freq='QS')

    elif dataset == 'canada_fof':
        
        # outsheet qdate cltv clti mortgages residentialstructures land income

        columns={
            # 'mortgages3' : 'debt',
            # 'residentialstructures1' : 'value',
            # 'householddisposableincome3' : 'income',
            'residentialstructures' : 'structures',
            'mortgages' : 'debt',
        }

        infile = data_dir + 'canada_accounts.csv'
        df = pd.read_csv(infile, sep='\t').rename(columns=columns)

        for var in ['debt', 'structures', 'land', 'income']:
            df[var] = df[var].astype(np.float64)

        df['value'] = df['structures'] + df['land']

        ts.date_index(df, '1990/1/1', freq='QS')
        
    elif dataset == 'canada_inflation_expectations':
        
        infile = data_dir + 'canada_inflation_expectations.csv'
        names = ['date', 'date2', 'prior_year_infl', 'current_infl', '1y_infl',
                 '2y_infl', '3y_infl', '4y_infl', '5y_infl', '6y_infl']
        
        df = pd.read_csv(infile, skiprows=1, names=names)
        
        df['date'] = pd.to_datetime(df['date'])
        
        df = df.set_index('date').drop(['date2'], axis=1)
        
        for var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
        
        return df

    elif dataset == 'canada_rates':

        infile = data_dir + 'canada_rates.csv'
        df = pd.read_csv(infile, sep=';', decimal=',', 
                         # names=['date', 'rate'], header=None
                         )

        ts.date_index(df, '1980/1/1', freq='MS')
    
    elif dataset == 'cleveland_fed':

        infile = data_dir + 'cleveland_fed_inflation_expectations.xlsx'
        df = pd.read_excel(
            infile,
            skiprows=6,
            header=None,
            names=['Date'] + ['infl_{}y'.format(yr) for yr in range(1, 31)]
        )

        del df['Date']
        ts.date_index(df, '1982/1/1', freq='MS')

    elif dataset == 'crsp_bianchi':

        infile = data_dir + 'crsp_bianchi.mat'
        mat = loadmat(infile)
        df = pd.DataFrame(data={
            'DP_yogo' : np.squeeze(mat['Adiv_me_yogo']), 
            'payout_to_equity_yogo' : np.squeeze(mat['payout_equity_ratio_Yogo']),
        }, index=ts.get_date_index('1954-10-01', len(mat['dates_l']), 'QS'))

    elif dataset == 'gertler_karadi':

        infile = data_dir + 'gk_factors.csv'
        df = pd.read_csv(infile)
        datestr = df['year'].astype(str) + '/' + df['month'].astype(str) + '/1'
        df.set_index(pd.to_datetime(datestr), inplace=True)
        df.drop(['year', 'month'], axis=1, inplace=True)

    elif dataset == 'gz':

        infile = data_dir + 'gz.csv'
        df = pd.read_csv(infile)
        df = ts.date_index(df, '1973-01-01', freq='MS')
        df.drop(['date'], axis=1, inplace=True)

    elif dataset == 'fernald':

        infile = data_dir + 'fernald_tfp.xls'
        df = pd.read_excel(
            infile,
            skiprows=1,
            sheet_name='quarterly',
            skip_footer=6,
        )

        del df['date']
        ts.date_index(df, '1947/1/1', freq='QS')

    elif dataset == 'nber_dates':

        infile = data_dir + 'nber_dates.csv'
        df = pd.read_csv(infile)

        df['peak_yr'] = 1800 + (df['Peak month number'].astype(np.int) - 1) // 12
        df['peak_cal_mo'] = ((df['Peak month number'].astype(np.int) - 1) % 12) + 1
        df['trough_yr'] = 1800 + (df['Trough month number'].astype(np.int) - 1) // 12
        df['trough_cal_mo'] = ((df['Trough month number'].astype(np.int) - 1) % 12) + 1

        df['peak_date'] = df['peak_yr'].astype(str) + '-' + df['peak_cal_mo'].astype(str) + '-' + '01'
        df['peak_date'] = pd.to_datetime(df['peak_date'])

        df['trough_date'] = df['trough_yr'].astype(str) + '-' + df['trough_cal_mo'].astype(str) + '-' + '01'
        df['trough_date'] = pd.to_datetime(df['trough_date'])

        df = df[['peak_date', 'trough_date']]

    elif dataset == 'price_rent':

        # infile = data_dir + 'price_rent.pkl'
        infile = data_dir + 'price_rent_new.pkl'
        df = pd.read_pickle(infile)

    elif dataset == 'shiller':

        infile = data_dir + 'shiller.csv'
        df = pd.read_csv(infile)
        df = df.loc[pd.notnull(df['Date']), :]
        df = ts.date_index(df, '1871-01-01', 'MS')
        del df['Date']
        del df['date_frac']

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    elif dataset == 'welch_goyal':
        """Include keyword argument freq to specify frequency of dataset"""

        infile = data_dir + 'welch_goyal_2016.xlsx'
        freq_name = kwargs.get('frequency', 'Monthly')
        df = pd.read_excel(
            infile, sheet_name=freq_name,
        )

        freq_code = freq_name[0] + 'S'
        df = ts.date_index(df, '1871-01-01', freq=freq_code)

    else:

        raise Exception

    return df
