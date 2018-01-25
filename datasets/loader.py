# import ipdb
import numpy as np
import os
import pandas as pd

import py_tools.time_series as ts
import py_tools.utilities as ut

from py_tools.datasets import fred, nipa, origins

from . import defaults

def load(dataset_list, reimport=False, no_prefix=True, master_dirs={}, **kwargs):

    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = defaults.base_dir()
    if 'pkl' not in dirs:
        dirs['pkl'] = dirs['base'] + 'pkl/'

    df = None
    for dataset in dataset_list:

        suffix = get_suffix(dataset, **kwargs)
        pkl_file = dirs['pkl'] + dataset + suffix + '.pkl'
        
        if not os.path.exists(pkl_file) or reimport:

            df_new = load_dataset(dataset, **kwargs)
            if len(df_new.columns) > 1 or not no_prefix:
                columns = {col : dataset.upper() + '_' + col for col in df_new.columns}
                df_new.rename(columns=columns, inplace=True)

            df_new.to_pickle(pkl_file)

        else:

            df_new = pd.read_pickle(pkl_file)

        if df is None:
            df = df_new
        else:
            df = pd.merge(df, df_new, left_index=True, right_index=True, how='outer')

    return df

def load_dataset(dataset, master_dirs={}, **kwargs):

    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = defaults.base_dir()
    if 'pkl' not in dirs:
        dirs['pkl'] = dirs['base'] + 'pkl/'

    if dataset in ['stockw', 'crsp', 'crsp_q', 'cay', 'cay_current', 'cay_source', 'bls_ls',
                   'fernald', 'tb3ms', 'uc']:

        df = origins.load(dataset, dirs, **kwargs)
    
    elif dataset == 'payouts':

        df = load(['nipa_11400', 'fof'], reimport=True)
        df['net_payouts'] = (df['FOF_net_dividends'] + df['NIPA_11400_net_interest_corp_nonfin']
                             - df['FOF_net_new_equity'] - df['FOF_net_new_paper'] - df['FOF_net_new_bonds'])
        neg_ix = df['net_payouts'].values < 0.0
        if np.any(neg_ix):
            max_negative_ix = int(np.amax(np.arange(len(df))[neg_ix]))
            df = df.ix[max_negative_ix + 1:, :]

        df = df['net_payouts'].to_frame() 

    elif dataset in ['fof', 'fof_csv']:

        if dataset == 'fof':
            var_index = {
                'liabilities_book' : ('b103', 'FL104190005'),
                'net_worth_book' : ('b103', 'FL102090005'),
                'net_worth_market' : ('b103', 'FL102090005'),
                'equities_outstanding_market' : ('b103', 'LM103164103'),
                # 'net_dividends' : ('u103', 'FU106121075'),
                # 'net_new_equity' : ('u103', 'FU103164103'),
                # 'net_new_paper' : ('u103', 'FU103169100'),
                # 'net_new_bonds' : ('u103', 'FU103163003')
                'net_dividends' : ('a103', 'FA106121075'),
                'net_new_equity' : ('a103', 'FA103164103'),
                'net_new_paper' : ('a103', 'FA103169100'),
                'net_new_bonds' : ('a103', 'FA103163003'),
                'stock_wealth' : ('b101', 'LM153064105'),
    #            'noncorp_business_wealth' : ('b101', 'LM152090205'),
            }
        elif dataset == 'fof_csv':
        
            var_index = {
                'liabilities_book' : ('b103', 'FL104190005'),
                'net_worth_book' : ('b103', 'FL102090005'),
                'net_worth_market' : ('b103', 'FL102090005'),
                'equities_outstanding_market' : ('b103', 'LM103164103'),
                # 'net_dividends' : ('u103', 'FU106121075'),
                # 'net_new_equity' : ('u103', 'FU103164103'),
                # 'net_new_paper' : ('u103', 'FU103169100'),
                # 'net_new_bonds' : ('u103', 'FU103163003')
                'net_dividends' : ('f103', 'FA106121075'),
                'net_new_equity' : ('f103', 'FA103164103'),
                'net_new_paper' : ('f103', 'FA103169100'),
                'net_new_bonds' : ('f103', 'FA103163003'),
                'corp_equities_wealth' : ('b101', 'LM153064105'),
                'noncorp_business_wealth' : ('b101', 'LM152090205'),
                'mutual_fund_wealth' : ('b101', 'LM153064205'),
            }

        full_list = sorted(list(var_index.keys()))

        tables, codes = zip(*[var_index[var] for var in full_list])
        codes = [code + '.Q' for code in codes]
        
        code_index = {code : var for var, code in zip(full_list, codes)}

        df = None

        unique_tables = sorted(list(set(tables)))

        if dataset == 'fof':

            data_dir = dirs['base'] + 'fof/all_prn/'

            for table in unique_tables:
                prefix, suffix = ut.split_str(table, 1)
                infile = prefix + 'tab' + suffix + 'd.prn'

                these_codes = [this_code for this_table, this_code in
                               zip(tables, codes) if this_table == table]
                usecols = ['DATES'] + these_codes

                df_new = pd.read_table(
                    data_dir + infile,
                    delimiter=' ',
                    usecols=usecols,
                )
                df_new.rename(columns=code_index, inplace=True)

                yr, q = (int(string) for string in ut.split_str(df_new.ix[0, 'DATES'], 4))
                df_new = ts.quarter_index(df_new, yr, q)
                # mon = 3 * (q - 1) + 1
                # ts.date_index(df_new, '{0}/1/{1}'.format(mon, yr))
                del df_new['DATES']

                df_new = df_new.apply(pd.to_numeric, errors='coerce')
                # df_new = df_new.convert_objects(convert_dates=False, convert_numeric=True)

                if df is not None:
                    df = pd.merge(df, df_new, left_index=True, right_index=True)
                else:
                    df = df_new

        elif dataset == 'fof_csv':

            data_dir = dirs['base'] + 'fof/all_csv/csv/'

            for table  in unique_tables:

                these_codes = [this_code for this_table, this_code in
                               zip(tables, codes) if this_table == table]

                usecols = ['date'] + these_codes

                infile = table + '.csv'
                df_new = pd.read_csv(data_dir + infile, usecols=usecols)
                df_new.rename(columns=code_index, inplace=True)

                yr_q_str = df_new.ix[0, 'date']
                yr = yr_q_str[:4]
                q = yr_q_str[-1]
                df_new = ts.quarter_index(df_new, yr, q)
                # mon = 3 * (q - 1) + 1
                # ts.date_index(df_new, '{0}/1/{1}'.format(mon, yr))

                df_new.drop(['date'], axis=1, inplace=True)

                if df is not None:
                    df = pd.merge(df, df_new, left_index=True, right_index=True)
                else:
                    df = df_new

        # Drop missing observations
        df = df.ix['1952-01-01':, :]

        # Ensure 
        for var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce').astype(np.float64)

        # Convert to billions
        df /= 1000.0

    elif dataset == 'fred':

        var_index = {
            # 'cpi_deflator' : 'CPIAUCSL',
            'pce_deflator' : 'PCECTPI',
            'real_gdp' : 'GDPC1',
        }

        # var_list = sorted(list(var_index.keys()))
        # codes = [var_index[var] for var in var_list]

        # start = datetime.datetime(1900, 1, 1)
        # end = datetime.datetime.today()

        # df = web.DataReader(codes, "fred", start, end)
        # df.rename(columns = {code : var for var, code in zip(var_list, codes)}, inplace=True)

        code_names = { code : name for name, code in var_index.items() }
        df = fred.load(code_names=code_names)

    elif dataset == 'fred_m':

        var_index = {
            'tbill' : 'TB3MS',
            'mortg' : 'MORTG',
        }

        # var_list = sorted(list(var_index.keys()))
        # codes = [var_index[var] for var in var_list]

        # start = datetime.datetime(1900, 1, 1)
        # end = datetime.datetime.today()
        # df = web.DataReader(codes, "fred", start, end)
        # df.rename(columns = {code : var for var, code in zip(var_list, codes)}, inplace=True)

        code_names = { code : name for name, code in var_index.items() }
        df = fred.load(code_names=code_names)

    elif dataset[:4] == 'nipa':

        nipa_table = dataset[5:]
        df = nipa.load(nipa_table=nipa_table, **kwargs)

    elif dataset == 'shiller':

        # colnames = [
            # 'Date', 'P', 'D', 'E', 'CPI', 'Date Frac', 'GS10', 
            # 'Real P', 'Real D', 'Real E', 'CAPE',
        # ]

        colnames = [
            'Date', 'P', 'D', 'E', 'CPI', 'date_frac', 'GS10', 
            'real_P', 'real_D', 'real_E', 'CAPE',
        ]

        df_m = pd.read_excel(
            dirs['base'] + 'shiller/ie_data.xls',
            sheet_name='Data',
            skiprows=7,
            # colnames=colnames,
            # usecols=usecols,
        )

        df_m = df_m.rename(columns={
            'Rate GS10' : 'GS10',
            'Fraction' : 'date_frac',
            'Price' : 'real_P',
            'Dividend' : 'real_D',
            'Earnings' : 'real_E',
        })

        df_m = df_m[colnames]
        df_m = ts.date_index(df_m, '01/01/1871', freq='MS')

        methods_vars = {
            # 'sum' : ['Dividend', 'Earnings'],
            # 'last' : ['Price', 'CAPE'],
            'sum' : ['real_D', 'real_E'],
            'last' : ['real_P', 'CAPE'],
        }

        df = ts.resample(df_m, methods_vars)

    else:

        print("Invalid dataset specified")
        raise Exception

    return df

def clean_nipa(df_t, nipa_quarterly=True):

    df_t = df_t.ix[df_t.index != ' ', :]
    df_t = df_t.ix[pd.notnull(df_t.index), :]
    del df_t['Line']
    del df_t['Unnamed: 1']

    df = df_t.transpose()
    start_date = df.index[0]

    if nipa_quarterly:
        yr = int(np.floor(start_date))
        q = int(10 * (start_date - yr) + 1)
        mon = int(3 * (q - 1) + 1)

        ts.date_index(df, '{0}/1/{1}'.format(mon, yr))
    else:
        ts.date_index(df, '1/1/{0}'.format(int(start_date)), freq='AS')

    return df

def get_suffix(dataset, **kwargs):

    if dataset[:4] == 'nipa':

        nipa_vintage = kwargs.get('nipa_vintage', '1604')
        suffix = nipa_vintage

    else:
        
        suffix = ''

    if suffix != '':
        suffix = '_' + suffix

    return suffix
