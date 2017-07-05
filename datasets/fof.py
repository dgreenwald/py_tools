import pandas as pd

def load():

    data_dir = '/home/dan/Dropbox/data/frm/fof/'

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
