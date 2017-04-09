import os

import numpy as np
import pandas as pd
import py_tools.data
# from py_tools.data import date_index

# home_dir = os.environ['HOME']
# drop_dir = home_dir + '/Dropbox/'
# base_dir = drop_dir + 'data/'
# pkl_dir = base_dir + 'pkl/'
def load(dataset, master_dirs={}, **kwargs):

    dirs = master_dirs.copy()
    if 'drop' not in dirs:
        home_dir = os.environ['HOME']
        dirs['drop'] = home_dir + '/Dropbox/'

    dirs['gll'] = dirs['drop'] + 'gll/Dan/data/'
    dirs['gll_pred'] = dirs['drop'] + 'gll/Predictability/data/'

    if dataset == 'stockw':

        data_dir = dirs['gll']
        infile = 'stockw.csv'
        df = pd.read_table(data_dir + infile, sep=',', 
                               names=['dates', 'stockw'], usecols=['stockw'])
        # df['stockw'] = np.log(df['stockw_level'])

        # del df['dates']
        # del df['stockw_level']

        df = py_tools.data.date_index(df, '1/1/1952')

    elif dataset == 'crsp':

        data_dir = dirs['gll']
        infile = 'crsp.csv' 

        df_m = pd.read_table(data_dir + infile, sep=',',
                               names=['date', 'vwretd', 'vwretx'],
                               usecols=['vwretd', 'vwretx'])
        df_m = py_tools.data.date_index(df_m, '12/1/1925', freq='MS')
        df_m['P'] = (df_m['vwretx'] + 1.0).cumprod()
        df_m['D'] = np.hstack((np.nan, df_m['P'][:-1])) * (df_m['vwretd'] - df_m['vwretx'])

# df = df_m['P'].resample('QS').last().to_frame()
# df = pd.merge(df, df_m['D'].resample('QS').sum().to_frame(),
                        # left_index=True, right_index=True)

        methods_vars = {
            'last' : ['P'],
            'sum' : ['D'],
        }
        df = resample(df_m, methods_vars)

        df['D4'] = df['D']
        for jj in range(1, 4):
            df['D4'] += df['D'].shift(jj)
        df['D4'] *= 0.25

        df['p'] = np.log(df['P'])
        df['d4'] = np.log(df['D4'])
# df['dd'] = df['d'].diff()
        df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
        df['re'] = np.log(df['Re'])
        df['pd'] = df['p'] - df['d4']

    elif dataset == 'crsp_q':

        data_dir = dirs['gll_pred']
        infile = 'crsp.csv'

        df = pd.read_table(
            data_dir + infile,
            sep=',',
            # names=['date', 'vwretd', 'vwretx'],
            usecols=['vwretd', 'vwretx'],
        )

        df = py_tools.data.date_index(df, '10/1/1925')

        df['P'] = (df['vwretx'] + 1.0).cumprod()
        df['D'] = np.hstack((np.nan, df['P'][:-1])) * (df['vwretd'] - df['vwretx'])

        df['D4'] = df['D']
        for jj in range(1, 4):
            df['D4'] += df['D'].shift(jj)
        df['D4'] *= 0.25

        df['p'] = np.log(df['P'])
        df['d4'] = np.log(df['D4'])
        # df['dd'] = df['d'].diff()
        df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
        df['re'] = np.log(df['Re'])
        df['pd'] = df['p'] - df['d4']

    elif dataset == 'cay':

        data_dir = dirs['gll']
        infile = 'caydata.txt'

        df = pd.read_table(data_dir + infile, delim_whitespace=True, 
                           names=['dates', 'c', 'a', 'y', 'cay'])
        df = py_tools.data.date_index(df, '1/1/1952')

    elif dataset == 'cay_current':

        data_dir = dirs['gll']
        infile = 'cay_current.csv'

        df = pd.read_table(data_dir + infile, sep=',', header=2,
                           names=['dates', 'c', 'a', 'y', 'cay'])
        df = py_tools.data.date_index(df, '1/1/1952')

    elif dataset == 'cay_source':

        cay_source_vintage = kwargs.get('cay_source_vintage', '1302')
        data_dir = dirs['drop'] + 'Dan Greenwald Files/CreateCAY/data_{}/'.format(cay_source_vintage)

        df = pd.read_excel(
            data_dir + 'source_{0}_{1}_rats.xlsx'.format(cay_source_vintage[:2], cay_source_vintage[2:]),
            sheetname='Sheet1',
        )

        df = py_tools.data.date_index(df, '1/1/1947')

    elif dataset == 'bls_ls':

        data_dir = dirs['gll']
        infile = 'bls_labor_share.csv'

        df = pd.read_table(data_dir + infile, sep=',',
                               names=['dates', 'bls_ls'],
                               usecols=['bls_ls'])
        df = py_tools.data.date_index(df, '1/1/1947')
        # start = datetime.datetime(1900, 1, 1)
        # end = datetime.date.today()
        # df_fred = web.DataReader(['GDPC1'], "fred", start, end)
        # df = pd.merge(df, df_fred, left_index=True, right_index=True) 
        # df['bls_cap_inc'] = np.log((1.0 - df['bls_ls']) * df['GDPC1'])
        # df['log_bls_ls'] = np.log(df['bls_ls'])

    elif dataset == 'fernald':

        df = pd.read_excel(
            dirs['gll'] + 'quarterly_tfp.xls',
            sheetname='quarterly',
            skiprows=1,
        )

        df['tfp_util'] = np.cumsum(df['dtfp_util'] / 400.0)
        df['tfp'] = np.cumsum(df['dtfp'] / 400.0)

        df = py_tools.data.date_index(df, '01/01/1947')

    elif dataset == 'tb3ms':

        df_m = pd.read_csv(
            dirs['gll'] + 'TB3MS.csv',
            names=['date', 'tb3ms'],
            skiprows=1,
            usecols=['tb3ms'],
        ) 

        df_m['rf'] = 0.25 * np.log(1.0 + 0.01 * df_m['tb3ms'])
        df_m = df_m['rf'].to_frame()

        df_m = py_tools.data.date_index(df_m, '01/01/1934', freq='MS')

        df = resample(df_m, {'first' : 'rf'}).to_frame()

    elif dataset == 'uc':

        # usecols = ['u{:02d}'.format(ii + 1) for ii in range(12)]

        # df_m = pd.read_excel(
            # dirs['gll_pred'] + 'macro_uncertainty.xlsx',
            # sheetname='data',
            # usecols=usecols,
        # )

        # df_m = py_tools.data.date_index(df_m, '7/01/1960', freq='MS')

        usecols = ['{:d}'.format(ii + 1) for ii in range(12)]

        df_m = pd.read_excel(
            dirs['gll_pred'] + 'ut_cons.xls',
            # sheetname='All',
            # skiprows=1,
            names=(['Dates'] + usecols),
            # usecols=usecols,
        )

        df_m = df_m[usecols]
        df_m = py_tools.data.date_index(df_m, '07/01/1960', freq='MS')

        methods_vars = {
            'first' : usecols,
        }

        df = resample(df_m, methods_vars)

    return df
