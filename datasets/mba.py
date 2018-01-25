import pandas as pd
from py_tools.data import date_index

from . import defaults
default_dir = defaults.base_dir() + 'datastream/'
# data_dir = '/home/dan/Dropbox/data/frm/datastream/'

def load(data_dir=default_dir):
    
    col_names = {
        'Code': 'date',
        'USMACRA': 'conv_apps_ix_sa_refi',
        'USMACRU': 'conv_apps_ix_refi',
        'USMAGRA': 'govt_apps_ix_sa_refi',
        'USMAGRU': 'govt_apps_ix_refi',
        'USMAHR%': 'total_apps_ix_sa_refi',
        'USMAHRA': 'total_apps_ix_sa_refi',
        'USMAHRU': 'total_apps_ix_refi',
        'USMANCR': 'conv_apps_pct_refi',
        'USMANGR': 'govt_apps_pct_refi',
        'USMANHR': 'total_apps_pct_refi',
        'USMAVCR': 'conv_apps_wtd_pct_refi',
        'USMAVGR': 'govt_apps_wtd_pct_refi',
        'USMAVHR': 'total_apps_wtd_pct_refi',
        'USMLCRU': 'conv_apps_avg_loan_refi',
        'USMLHRU': 'total_apps_avg_loan_refi',
    }
    
    df = pd.read_excel(data_dir + 'refi_nomacro.xlsx', skiprows=1).rename(columns=col_names)
    df = date_index(df, '1990-01-01', 'W')
    return df

df = load()
