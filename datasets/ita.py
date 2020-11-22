import numpy as np
import os
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir()

idx = pd.IndexSlice

def load(table, data_dir=default_dir+'/ita/', quarterly=True, sa=True):
    
    df = pd.read_csv(data_dir + 'Ita_T' + table + '.csv', skiprows=4, 
                     header=None, encoding='latin1')
    
    df = df.drop(columns=[0]).transpose()
    
    var_dict = {
        0 : 'date_type',
        1 : 'year',
        2 : 'quarter',
        }
    
    var_lists = {
        '4.2' : [
            'receipts',
            'direct_inv_income_a',
            'equity_income_a',
            'div_withdraw_a',
            'reinvested_a', #
            'int_a',
            'int_parent_a',
            'int_affiliate_a',
            'adj_a',
            'adj_int_parent_a', # 10
            'adj_int_affiliate_a',
            'direct_inv_income_ao',
            'equity_income_ao',
            'div_withdraw_ao',
            'reinvested_ao', #
            'reinvested_wo_cca_ao',
            'cca_ao',
            'int_net_ao',
            'int_net_receipts_ao',
            'int_net_payments_ao', # 20
            'direct_inv_income_wo_cca_a',
            'dii_manufacturing_a',
            'dii_wholesale_a',
            'dii_finance_a',
            'dii_holding_a', #
            'dii_other_a',
            'equity_income_wo_cca_a'
            'equity_manufacturing_a',
            'equity_wholesale_a',
            'equity_finance_a', # 30
            'equity_holding_a', 
            'equity_other_a',
            'int_net_payments_a',
            'int_manufacturing_a',
            'int_wholesale_a', #
            'int_finance_a',
            'int_holding_a',
            'int_other_a',
            'payments',
            'direct_inv_income_l', # 40
            'equity_income_l',
            'div_withdraw_l',
            'reinvested_l',
            'int_l',
            'int_affiliate_l', #
            'int_parent_l',
            'adj_l',
            'adj_int_parent_l',
            'adj_int_affiliate_l',
            'direct_inv_income_li', # 50
            'equity_income_li',
            'div_withdraw_li',
            'reinvested_li',
            'reinvested_wo_cca_li',
            'cca_li', #
            'int_net_li',
            'int_net_receipts_li',
            'int_net_payments_li',
            'direct_inv_income_wo_cca_l',
            'dii_manufacturing_l', # 60
            'dii_wholesale_l',
            'dii_finance_l',
            'dii_other_l',
            'equity_income_wo_cca_l',
            'equity_manufacturing_l', # 65
            'equity_wholesale_l',
            'equity_finance_l',
            'equity_other_l',
            'int_net_payments_l',
            'int_manufacturing_l', # 70
            'int_wholesale_l',
            'int_finance_l',
            'int_other_l',
            ]
        }
    
    this_var_list = var_lists[table]
    var_dict.update({ii+3 : var for ii, var in enumerate(this_var_list)})
    
    drop_cols = [var for var in df.columns if var not in var_dict.keys()]
    df = df.drop(columns=drop_cols)
    df = df.rename(var_dict, axis=1)
    
    df = df.iloc[1:]
    
    if quarterly:
        if sa:
            ix = df['date_type'] == 'Seasonally adjusted'
        else:
            ix = df['date_type'] == 'Not seasonally adjusted'
    else:
        ix = ~df['date_type'].isin(['Seasonally adjusted', 'Not seasonally adjusted'])
        
    df = df.loc[ix]
    
    if quarterly:
        df['qtr_num'] = df['quarter'].str[1].astype(np.int64)
        df['date'] = ts.date_from_qtr(df['year'], df['qtr_num'])
    else:
        df['date'] = pd.to_datetime('year')
        
    df = df.drop(columns=['date_type', 'quarter'])
    df = df.set_index('date').sort_index()
    for var in df.columns:
        df[var] = pd.to_numeric(df[var], errors='coerce')
    
    # new_cols = df.iloc[0]
    # df = df.iloc[1:]
    # df.columns = new_cols
    
    # if quarterly:
    #     if sa:
    #         ix = 
    
    return df