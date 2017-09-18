import pandas as pd
from . import fred
from py_tools import time_series as ts

data_dir = '/home/dan/Dropbox/data/fof/'

def load():

    value_var = 'LM155035015.Q'
    debt_var = 'FL153165105.Q'

    df = pd.read_table(
            data_dir + 'btab101d.prn',
            delimiter=' ',
            usecols=[value_var, debt_var],
            )

    df.rename(columns={value_var : 'value', debt_var : 'debt'}, inplace=True)
    df.set_index(pd.date_range('1/1/1952', periods=len(df), freq='QS'), inplace=True)

    income_var = 'FA156012005.Q' # Disposable Personal Income
    df_a = pd.read_table(
            data_dir + 'atab101d.prn',
            delimiter=' ',
            usecols=[income_var],
            )
    df_a.rename(columns={income_var : 'income'}, inplace=True)
#ipdb.set_trace()
    df_a.set_index(pd.date_range('1/1/1952', periods=len(df_a), freq='QS'), inplace=True)

# Merge
    return pd.merge(df, df_a, left_index=True, right_index=True)

def load_fred():

    var_titles = {
        'HMLBSHNO' : 'debt',
        'HNOREMV' : 'value',
        'HNODPI' : 'income',
        'HHMSDODNS' : 'debt_sa',
        'DHUTRC1Q027SBEA' : 'housing_services',
    }
    
    var_list = var_titles.keys()
    df = fred.load(var_list).rename(columns=var_titles).loc['1952-01-01':, :]

    return df

def load_csv():
    """Load from CSV files"""

    infile = data_dir + 'csv/fof.csv'
    df = pd.read_csv(infile, skiprows=5)
    df = ts.date_index(df, '1945-10-01', freq='QS')
    df = df.loc['1951-10-01':, :]

# Update names
    raw_labels = {
            '156012005' : 'income',
            '155035005' : 'value',
            '153165105' : 'debt',
            }

    for prefix in ['LM', 'FA', 'FU', 'FL']:
        these_labels = {
                prefix + key + '.Q' : prefix + '_' + val for key, val in raw_labels.items()
                }
        df = df.rename(columns=these_labels)
        
    for col in df.columns:
        if col != 'Time Period':
            df[col] = pd.to_numeric(df[col])

    return df.rename(columns={'FA_income' : 'income', 'FL_debt' : 'debt',
                              'FL_value' : 'value'})[['income', 'debt', 'value']]
