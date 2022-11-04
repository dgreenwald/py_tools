
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir('DAN')

def load(dataset, user='DAN', master_dirs={}, reimport=False, 
         save_pickle=True, **kwargs):
    
    default_dir = defaults.base_dir(user)
    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = default_dir
        # home_dir = os.environ['HOME']
        # dirs['base'] = home_dir + '/Dropbox/data/'

    data_dir = dirs['base'] + 'misc/'
    
    pkl_file = data_dir + dataset + '.pkl'
    if reimport or (not os.path.exists(pkl_file)):
        df = load_from_source(dataset, data_dir, **kwargs)
        df.to_pickle(pkl_file)
    else:
        df = pd.read_pickle(pkl_file)
        
    return df

def load_from_source(dataset, data_dir, **kwargs):
    """Load data from one-off file"""

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
        df = pd.read_csv(infile).rename(columns=columns)

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

    elif dataset == 'cieslak':

        infile = data_dir + 'factors_CieslakPovala_RFS2015.xlsx'
        df = pd.read_excel(infile)
        df = df.drop(['date', 'year', 'month'], axis=1).rename(columns={
            'trend inflation (tau_cpi)' : 'trend_infl',
            'real rate factor (cyc1)' : 'real_rate',
            'term premium factor (cf_hat)' : 'term_premium',
        })
        ts.date_index(df, '1971/11/1', freq='MS')
        return df
    
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
        
    elif dataset == 'dlm_price_rent':
        
        pkl_file = data_dir + 'dlm_price_rent.pkl'
        
        if not os.path.exists(pkl_file):
            
            names=['date', 'rent', 'price_cs', 'rent_price_cs', 'price_fhfa', 'rent_price_fhfa']
            
            infile = data_dir + 'RENT-PRICE-RATIO.2018q2.xlsx'
            df = pd.read_excel(infile, sheet_name='rent-price data', skiprows=2,
                               header=None, names=names)
            
            df['date'] = (np.round(10 * df['date'])).astype(int)
            year = df['date'].astype(str).str[:4].astype(int)
            qtr = df['date'].astype(str).str[-1].astype(int)
            df['date'] = ts.date_from_qtr(year, qtr)
            df = df.set_index('date').sort_index()
            
            df['price_rent_cs'] = 1.0 / df['rent_price_cs']
            df['price_rent_fhfa'] = 1.0 / df['rent_price_fhfa']
            
            df.to_pickle(pkl_file)
            
        else:
            
            df = pd.read_pickle(pkl_file)
            
        return df

    elif dataset == 'direct_investment_income':

        sa = kwargs.get('sa', True)
        if sa:
            infile = data_dir + 'direct_investment_income_sa.csv'
        else:
            infile = data_dir + 'direct_investment_income.csv'
        df = pd.read_csv(infile)
        df['Date'] = pd.to_datetime(df['Date'])

    elif dataset == 'fhfa':

        infile = data_dir + 'HPI_AT_metro.csv'
        df = pd.read_csv(infile, header=None, 
                         names=['MSA', 'code', 'year', 'q', 'hpi', 'unknown'])
        
        df['month'] = df['q'] * 3 - 2
        df['day'] = 1
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

        df['hpi'] = pd.to_numeric(df['hpi'], errors='coerce')

        df = df.set_index(['date', 'MSA'])
        df = df.drop(columns=['year', 'q', 'month', 'day', 'unknown'])

    elif dataset == 'gertler_karadi':

        infile = data_dir + 'gk_factors.csv'
        df = pd.read_csv(infile)
        datestr = df['year'].astype(str) + '/' + df['month'].astype(str) + '/1'
        df.set_index(pd.to_datetime(datestr), inplace=True)
        df.drop(['year', 'month'], axis=1, inplace=True)

    elif dataset == 'gsw':

        infile = data_dir + 'gsw_feds200628_downloadAug2017.xlsx'
        names = ['date'] + ['y{:d}'.format(yr) for yr in range(1, 31)]
        df = pd.read_excel(infile, skiprows=10, header=None, names=names)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

    elif dataset == 'gz':

        infile = data_dir + 'gz.csv'
        df = pd.read_csv(infile)
        df = ts.date_index(df, '1973-01-01', freq='MS')
        df.drop(['date'], axis=1, inplace=True)

    elif dataset == 'fernald':

        infile = data_dir + 'fernald_tfp.xlsx'
        df = pd.read_excel(
            infile,
            skiprows=1,
            sheet_name='quarterly',
            skipfooter=6,
        )

        del df['date']
        ts.date_index(df, '1947/1/1', freq='QS')

    elif dataset == 'martin_svix':

        infile = data_dir + 'martin_svix2.xls'
        df = pd.read_excel(
            infile,
            skiprows=0,
            header=None,
            names=['date', 'svix_1mo', 'svix_2mo', 'svix_3mo', 'svix_6mo', 'svix_12mo'],
        )

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date')
        
    elif dataset == 'martin_epbound':
        
        infile = data_dir + 'martin_epbound.xls'
        dates = pd.read_excel(infile, skiprows=0, header=None, 
                              names=['year', 'month', 'day'], 
                              sheet_name='Sheet1')
        
        ep_names = ['epbound_' + suffix for suffix in ['1mo', '2mo', '3mo', '6mo', '12mo']]
        values = pd.read_excel(infile, skiprows=0, header=None, names=ep_names,
                               sheet_name='Sheet2')
        
        df = pd.concat((dates, values), axis=1)
        df['date'] = (df['year'].astype(str) + '-' + df['month'].astype(str) 
                      + '-' + df['day'].astype(str))
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop(columns=['year', 'month', 'day'])
        df = df.set_index('date')
        
    elif dataset == 'ns_mp_shocks':
        # Nakamura and Steinsson shocks
        
        infile = data_dir + 'ns_mp_shocks.pkl'
        if not os.path.exists(infile):
            rawfile = data_dir + 'PolicyNewsShocksWeb.xlsx'
            df = pd.read_excel(rawfile, sheet_name='PolicyNewsShocks1995')
            df.to_pickle(infile)
        else:
            df = pd.read_pickle(infile)

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

    elif dataset == 'ssa_life_expectancy':

        infile = data_dir + 'ssa_life_expectancy.csv'
        df = pd.read_csv(infile)

    elif dataset == 'shiller':

        infile = data_dir + 'shiller.csv'
        df = pd.read_csv(infile)
        df = df.loc[pd.notnull(df['Date']), :]
        df = ts.date_index(df, '1871-01-01', 'MS')
        del df['Date']
        del df['date_frac']

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    elif dataset == 'state_codes':

        infile = data_dir + 'state_codes.csv'
        df = pd.read_csv(infile, header=None, 
                         names=['state_abbr', 'state_code', 'state_name'])

    elif dataset == 'vrp':

        infile = data_dir + 'VRPtable.txt'
        df = pd.read_table(infile, sep='\s+')
        df['date'] = ts.date_from_month(df['Year'], df['Month'])
        df = df.drop(columns=['Year', 'Month']).set_index('date').sort_index()

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
