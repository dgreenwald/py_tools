import numpy as np
import os
import pandas as pd

base_dir = '/home/dan/Dropbox/data/'

def splitstr(string, length):
    string = str(string)
    return (string[:length], string[length:])

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

def load_fof(var_list):

    data_dir = base_dir + 'fof/'

    # dependencies = {
        # 'net_payouts' : ['book_liabilities', 'net_worth', 'net_dividends', 'net_new_equity', 'net_new_paper', 'net_new_bonds']
    # }

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

    # var_set = set()
    # for var in var_list:
        # var_set = add_var(var, var_set, dependencies)

    # full_list = sorted(list(var_set))

    tables, codes = zip(*[var_index[var] for var in var_list])
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
        df_new.rename(columns = {code : var for var, code in zip(var_list, codes)}, inplace=True)

        yr, q = (int(string) for string in splitstr(df_new.ix[0, 'DATES'], 4))
        mon = 3 * (q - 1) + 1
        df_new.set_index(pd.date_range('{0}/1/{1}'.format(mon, yr), periods=len(df_new), freq='QS'), inplace=True)
        del df_new['DATES']

        df_new = df_new.convert_objects(convert_dates=False, convert_numeric=True)

        if df is not None:
            df = pd.merge(df, df_new, left_index=True, right_index=True)
        else:
            df = df_new

    # Drop missing observations
    df = df.ix['1951-10-01':, :]

    return df

def clean_nipa(df_t):

    df_t = df_t.ix[df_t.index != ' ', :]
    df_t = df_t.ix[pd.notnull(df_t.index), :]
    del df_t['Line']
    del df_t['Unnamed: 1']

    df_new = df_t.transpose()
    start_date = df_new.index[0]
    yr = int(np.floor(start_date))
    q = int(10 * (start_date - yr) + 1)
    mon = int(3 * (q - 1) + 1)

    df_new.set_index(pd.date_range('{0}/1/{1}'.format(mon, yr), periods=len(df_new), freq='QS'), inplace=True)

    return df_new

def load_nipa(var_list, reimport=False):

    data_dir = base_dir + 'nipa/'
    pkl_file = data_dir + '/table1.pkl'

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

        df.to_pickle(pkl_file)

    df = pd.read_pickle(pkl_file)
    df = df.ix[:, var_list]

    return df
