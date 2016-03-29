import datetime
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web

base_dir = '/home/dan/Dropbox/data/'
pkl_dir = base_dir + 'pkl/'
gll_dir = '/home/dan/Dropbox/gll/Dan/data/'

def deflate(df, var_list, index='cpi', log=False, diff=False, reimport=False):
    
    prefix = 'DEF' + index.upper() + '_'
    index_var = index + '_index'

    df_fred = load_fred(reimport=reimport)
    for var in var_list:

        scale = np.log(df_fred[index_var])

        if diff:
            scale = scale.diff()

        if log:
            df[prefix + var] = df[var] - scale
        else:
            scale = np.exp(scale)
            df[prefix + var] = df[var] / scale

    return df

def date_index(df, startdate, freq='QS'):
    df.set_index(pd.date_range(startdate, periods=len(df), freq=freq), inplace=True)
    return df

def splitstr(string, length):
    string = str(string)
    return (string[:length], string[length:])

# def transform(df, var, lag=0, diff=0, other=None, deflate=None):

    # new_var = var

    # if deflate is not None:
        # prefix = deflate.upper()
        # new_var = prefix + new_var

    # if other is not None:
        # prefix = other.upper()
        # new_var = prefix + new_var

    # if diff != 0:
        # if diff > 1:
            # prefix = 'D({})_'.format(diff)
        # else:
            # prefix = 'D_'
        # new_var = prefix + new_var

    # if lag != 0:
        # if lag > 1:
            # prefix = 'L({})_'.format(lag)
        # else:
            # prefix = 'L_'
        # new_var = prefix + new_var

    # df[new_var] = df[var]

    # if deflate is not None:

    # if other is not None:

    # if diff != 0:

    # if lag != 0:

# def add_var(var, var_set, dependencies):

    # # new_vars = set(dependencies.get(var, []))
    # # var_set |= new_vars
    # # for new_var in new_vars:
        # # var_set = add_var(new_var, var_set, dependencies)

    # if var in dependencies:
        # var_set |= set(dependencies[var])
    # else:
        # var_set.add(var)

    # return var_set

def load_datasets(dataset_list, reimport=False):

    df = None
    for dataset in dataset_list:

        df_new = eval('load_{}(reimport=reimport)'.format(dataset))
        columns = {col : dataset.upper() + '_' + col for col in df_new.columns}
        df_new.rename(columns=columns, inplace=True)

        if df is None:
            df = df_new
        else:
            df = pd.merge(df, df_new, left_index=True, right_index=True)

    return df

def load_fof(reimport=False):

    data_dir = base_dir + 'fof/'
    pkl_file = pkl_dir + 'fof.pkl'

    if not os.path.exists(pkl_file) or reimport:

        var_index = {
            # 'liabilities_book' : ('b103', 'FL104190005'),
            # 'net_worth_book' : ('b103', 'FL102090005'),
            # 'net_dividends' : ('u103', 'FU106121075'),
            # 'net_new_equity' : ('u103', 'FU103164103'),
            # 'net_new_paper' : ('u103', 'FU103169100'),
            # 'net_new_bonds' : ('u103', 'FU103163003')
            'net_dividends' : ('a103', 'FA106121075'),
            'net_new_equity' : ('a103', 'FA103164103'),
            'net_new_paper' : ('a103', 'FA103169100'),
            'net_new_bonds' : ('a103', 'FA103163003')
        }

        full_list = sorted(list(var_index.keys()))

        tables, codes = zip(*[var_index[var] for var in full_list])
        codes = [code + '.Q' for code in codes]

        df = None

        unique_tables = sorted(list(set(tables)))
        for table in unique_tables:
            prefix, suffix = splitstr(table, 1)
            infile = prefix + 'tab' + suffix + 'd.prn'

            these_codes = [this_code for this_table, this_code in zip(tables, codes) if this_table == table]
            usecols = ['DATES'] + these_codes

            df_new = pd.read_table(
                data_dir + infile,
                delimiter=' ',
                usecols=usecols,
            )
            df_new.rename(columns = {code : var for var, code in zip(full_list, codes)}, inplace=True)

            yr, q = (int(string) for string in splitstr(df_new.ix[0, 'DATES'], 4))
            mon = 3 * (q - 1) + 1
            date_index(df_new, '{0}/1/{1}'.format(mon, yr))
            del df_new['DATES']

            df_new = df_new.convert_objects(convert_dates=False, convert_numeric=True)

            if df is not None:
                df = pd.merge(df, df_new, left_index=True, right_index=True)
            else:
                df = df_new

        # Drop missing observations
        df = df.ix['1951-10-01':, :]
        
        # Save to pickle format
        df.to_pickle(pkl_file)

    df = pd.read_pickle(pkl_file)

    return df

def clean_nipa(df_t):

    df_t = df_t.ix[df_t.index != ' ', :]
    df_t = df_t.ix[pd.notnull(df_t.index), :]
    del df_t['Line']
    del df_t['Unnamed: 1']

    df = df_t.transpose()
    start_date = df.index[0]
    yr = int(np.floor(start_date))
    q = int(10 * (start_date - yr) + 1)
    mon = int(3 * (q - 1) + 1)

    date_index(df, '{0}/1/{1}'.format(mon, yr))

    return df

def load_nipa(reimport=False):

    data_dir = base_dir + 'nipa/'
    pkl_file = pkl_dir + 'nipa1.pkl'

    if not os.path.exists(pkl_file) or reimport:

        var_index = {
            'net_interest' : 'B471RC1',
        } 

        full_list = sorted(list(var_index.keys()))

        # Current file
        df_t = pd.read_excel(
            data_dir + 'Section1All_xls.xls',
            sheetname='11400 Qtr',
            skiprows=7,
            # header=[0, 1],
            index_col=2,
        )
        df_curr = clean_nipa(df_t)

        # Current file
        df_t = pd.read_excel(
            data_dir + 'Section1All_Hist.xls',
            sheetname='11400 Qtr',
            skiprows=7,
            # header=[0, 1],
            index_col=2,
        )
        df_hist = clean_nipa(df_t)

        start_date = df_curr.index[0]
        df = df_hist.ix[:start_date, :].append(df_curr)

        codes = [var_index[var] for var in full_list]
        df = df.ix[:, codes]
        df.rename(columns = {code : var for var, code in zip(var_list, codes)}, inplace=True)

        # Save to pickle format
        df.to_pickle(pkl_file)

    df = pd.read_pickle(pkl_file)
    # df = df.ix[:, var_list]

    return df

def load_stockw(reimport=False):

    data_dir = gll_dir
    infile = 'stockw.csv'
    df = pd.read_table(data_dir + infile, sep=',', 
                           names=['dates', 'stockw_level'], usecols=['stockw_level'])
    df['stockw'] = np.log(df['stockw_level'])

    del df['dates']
    del df['stockw_level']

    df = date_index(df, '1/1/1952')

    return df

def load_crsp(reimport=False):

    data_dir = gll_dir
    infile = 'crsp.csv' 

    df_m = pd.read_table(data_dir + infile, sep=',',
                           names=['date', 'vwretd', 'vwretx'],
                           usecols=['vwretd', 'vwretx'])
    df_m = date_index(df_m, '12/1/1925', freq='MS')
    df_m['P'] = (df_m['vwretx'] + 1.0).cumprod()
    df_m['D'] = np.hstack((np.nan, df_m['P'][:-1])) * (df_m['vwretd'] - df_m['vwretx'])

    df = df_m['P'].resample('QS').last().to_frame()
    df = pd.merge(df, df_m['D'].resample('QS').sum().to_frame(),
                    left_index=True, right_index=True)
    df['D4'] = df['D']
    for jj in range(1, 4):
        df['D4'] += df['D'].shift(jj)
    df['D4'] *= 0.25

    df['p'] = np.log(df['P'])
    df['d'] = np.log(df['D4'])
    # df['dd'] = df['d'].diff()
    df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
    df['re'] = np.log(df['Re'])
    df['pd'] = df['p'] - df['d']

    return df

def load_cay(reimport=False):

    data_dir = gll_dir
    infile = 'caydata.txt'

    df = pd.read_table(data_dir + infile, delim_whitespace=True, 
                       names=['dates', 'c', 'a', 'y', 'cay'])
    df = date_index(df, '1/1/1952')

def load_bls_ls(reimport=False):

    data_dir = gll_dir
    infile = 'bls_labor_share.csv'

    df = pd.read_table(data_dir + infile, sep=',',
                           names=['dates', 'bls_ls'],
                           usecols=['bls_ls'])
    df = date_index(df, '1/1/1947')
    # start = datetime.datetime(1900, 1, 1)
    # end = datetime.date.today()
    # df_fred = web.DataReader(['GDPC1'], "fred", start, end)
    # df = pd.merge(df, df_fred, left_index=True, right_index=True) 
    # df['bls_cap_inc'] = np.log((1.0 - df['bls_ls']) * df['GDPC1'])
    # df['log_bls_ls'] = np.log(df['bls_ls'])

def load_fernald(reimport=False):

    data_dir = gll_dir
    infile = 'quarterly_tfp.csv'

    df = pd.read_table(data_dir + infile, sep=',',
                           header=0, usecols=['dtfp_util'])
    df = set_index(df, '1/1/1947')
    df['tfp_util'] = df['dtfp_util'].cumsum()

def load_fred(reimport=False):

    data_dir = base_dir + 'fred/'
    pkl_file = pkl_dir + 'fred.pkl'

    var_index = {
        'cpi_index' : 'CPIAUCSL',
        'pce_index' : 'PCEPI',
    }

    var_list = sorted(list(var_index.keys()))
    codes = [var_index[var] for var in var_list]

    if not os.path.exists(pkl_file) or reimport:

        start = datetime.datetime(1900, 1, 1)
        end = datetime.datetime.today()
        df = web.DataReader(codes, "fred", start, end)
        df.rename(columns = {code : var for var, code in zip(var_list, codes)}, inplace=True)
        df.to_pickle(pkl_file) 

    df = pd.read_pickle(pkl_file)
    return df
