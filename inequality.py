#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:27:08 2021

@author: dan
"""

import numpy as np
from scipy import integrate
from . import stats

def compute_gini(df, var, wvar=None):

    if var is None:
        df_by_var = df.groupby(var)[var].count()
    else:
        df_by_var = df.groupby(var)[wvar].sum()

    df_by_var = df_by_var.to_frame(name='weight').sort_index().reset_index()
    ix = df_by_var['weight'] > 1e-12
    df_by_var = df_by_var.loc[ix]
    df_by_var['w_share'] = df_by_var[var] * df_by_var['weight']

    df_by_var['w_share'] /= np.sum(df_by_var['w_share'])
    df_by_var['weight'] /= np.sum(df_by_var['weight'])

    c_shares = np.cumsum(df_by_var['w_share'].values)
    c_weight = np.cumsum(df_by_var['weight'].values)

    area_under = integrate.simps(c_shares, c_weight)
    gini_coeff = 1.0 - 2.0 * area_under

    return gini_coeff, c_weight, c_shares

def get_top_shares(df, var, shares=[10], wvar=None):
    
    shares = np.atleast_1d(shares)
    top_shares = []
    
    if wvar is None:
        weight = np.ones(len(df))
    else:
        weight = df[wvar].values
    
    for share in shares:
        
        q = 1.0 - (0.01 * share)
        cutoff = stats.weighted_quantile(df[var].values, weight, q)
        ix = df[var] >= cutoff
        top_share = np.sum(df.loc[ix, var] * weight[ix]) / np.sum(df[var] * weight)
        top_shares.append(top_share)
        
    return top_shares