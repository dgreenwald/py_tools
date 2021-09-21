#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:28:02 2021

@author: dan
"""

import patsy
import numpy as np
import pandas as pd
from py_tools import data as dt

def reghdfe(df, yvar, xvars, **kwargs):
    
    formula = '{0} ~ {1}'.format(yvar, ' + '.join(xvars))
    return reghdfe_formula(df, formula, **kwargs)

def group_mean_error():
    
    return None

def reghdfe_formula(df, formula, fes=[], weight_var=None, se_type='robust', 
                    tol=1e-8):
    
    unweighted = (weight_var is None)
    
    fe_vars = []
    for fe in fes:
        if isinstance(fe, list):
            fe_vars += fe
        else:
            fe_vars.append(fe)
            
    fe_vars = sorted(list(set(fe_vars)))
    
    y, X = patsy.dmatrices(formula, df, return_type='dataframe')
    
    ix = pd.notnull(y.iloc[:, 0]) & np.all(pd.notnull(X), axis=1) & np.all(pd.notnull(df[fe_vars]), axis=1)
    
    y = y.loc[ix]
    X = X.loc[ix]
    
    if fes:
        
        F = df.loc[ix, fe_vars].copy()
        
        nonsingle = np.ones(len(F), dtype=bool)
        # Remove singletons
        for fe in fes:
            this_count = F.groupby(fe).transform(len)
            nonsingle &= (this_count.values[:, 0] > 1)
            
        ix[ix] = nonsingle
        y = y.loc[nonsingle]
        X = X.loc[nonsingle]
        F = F.loc[nonsingle]
        
        # Count groups
        Ng = []
        for fe in fes:
            Ng.append(F.groupby(fe).ngroups)
        
        yvar = y.columns[0]
        xvars = X.columns
        
        data = pd.concat([y, X, F], axis=1)
        
        if weight_var is None:
            weight_var = '_w'
            data[weight_var] = 1.0
        else:
            data[weight_var] = df[weight_var]
        
        y = dt.absorb(data, fes, yvar, weight_var=weight_var, restore_mean=False, 
                      display=False, tol=tol)
        
        X_list = []
        for xvar in xvars:
            if xvar != 'Intercept':
                X_list.append(dt.absorb(data, fes, xvar, weight_var=weight_var, 
                                        restore_mean=False, display=False, tol=tol))
            # else:
            #     X_list.append(data[xvar])
            
        X = pd.concat(X_list, axis=1)
        
    y = y.values
    X = X.values
        
    Nt, Nx = X.shape
    Nt_adj = Nt - Nx - 1
    if fes:
        Nt_adj -= sum(Ng)
        Nt_adj += len(fes)
        
    if unweighted:
        XW = X.T
    else:
        XW = X.T * df.loc[ix, weight_var].values[np.newaxis, :]
        
    XWX = XW @ X
    coeffs = np.linalg.solve(XWX, XW @ y)
    
    e = (y - X @ coeffs)[:, np.newaxis]
    
    if se_type == 'robust':
        XWe = XW * e.T
        S_robust = (XWe @ XWe.T)
        XWX_inv = np.linalg.inv(XWX)
        V_ols_robust = Nt * (XWX_inv @ (S_robust @ XWX_inv))
        se = np.sqrt(np.diag(V_ols_robust) / Nt_adj)
    
    return coeffs, se