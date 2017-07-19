# TODO: data about type of series (log, real, units, etc)
# TODO: default index for deflation

# import datetime
import pandas as pd
# import pandas_datareader.data as web

# TODO: probably shouldn't be hard-coded
# dirs['gll'] = dirs['drop'] + 'gll/Dan/data/'
# dirs['gll_pred'] = dirs['drop'] + 'gll/Predictability/data/'

# def dirs['get_drop']():
    # return dirs['drop']

def merge_date(df1, df2, **kwargs):

    return pd.merge(df1, df2, left_index=True, right_index=True, **kwargs)

def date_index(df, startdate, freq='QS'):
    df.set_index(pd.date_range(startdate, periods=len(df), freq=freq), inplace=True)
    return df

def resample(df, methods_vars, freq='QS'):
    df_resamp = None

    for method, var_list in methods_vars.items():

        if var_list is not None:

            df_new = getattr(df[var_list].resample(freq), method)()

            # if len(var_list) == 1:
                # df_new = df_new.to_frame()
            if df_resamp is None:
                df_resamp = df_new
            else:
                df_resamp = pd.merge(df_resamp, df_new, left_index=True, right_index=True)

    return df_resamp

def get_suffix(dataset, **kwargs):

    if dataset[:4] == 'nipa':

        nipa_vintage = kwargs.get('nipa_vintage', '1604')
        suffix = nipa_vintage

    else:
        
        suffix = ''

    if suffix != '':
        suffix = '_' + suffix

    return suffix

