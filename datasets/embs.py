import numpy as np
import pandas as pd

from . import defaults
default_dir = defaults.base_dir() + 'frm/embs/by_coupon/'

# data_dir = '/home/dan/Dropbox/data/frm/'
# embs_dir = data_dir + 'embs/by_coupon/'

def load(embs_dir=default_dir):

    var_list = ['CPR', 'RPB', 'Issuance', 'Wac', 'Wam', 'Wala']

    for i_var, var in enumerate(var_list):
        df_new = pd.read_pickle(embs_dir + 'embs_{0}_clean.pkl'.format(var.lower()))
        if i_var == 0:
            df = df_new
        else:
            df = pd.merge(df, df_new, on = ['coupon', 'orig_date', 'date'])

    return df

def aggregate(df_in, weight='rpb'):

    var_list = ['cpr', 'wac', 'wam', 'wala']

    df = df_in.ix[pd.notnull(df_in[weight]), :].copy()
    
    for var in var_list:
        df['wtd_' + var] = df[weight] * df[var]

    wtd_list = ['wtd_' + var for var in var_list]
    keep_list = ['date', weight] + wtd_list

    columns = {var : var + '_total' for var in [weight] + wtd_list}

    df_agg = df[keep_list].groupby(['date']).sum().rename(columns=columns) 
    for var in var_list:
        df_agg[var] = df_agg['wtd_' + var + '_total'] / df_agg[weight + '_total']

    return df_agg[var_list] 

def resample_cpr(cpr, freq='QS'):

    log_survival = np.log((1.0 - 0.01 * cpr) ** (1.0 / 12.0))
    log_survival_resamp = log_survival.resample(freq).mean()
    return 100.0 * (1.0 - np.exp(12.0 * log_survival_resamp))
