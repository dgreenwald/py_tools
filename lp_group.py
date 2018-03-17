#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:12:53 2018

@author: MaryGong
"""

import numpy as np
import pandas as pd

from py_tools import data as dt


def add_lags(df, var, max_lags, groupvars, sortvars):
    df=df.sort_values(by=sortvars)
    gp=df.groupby(by=groupvars)
    
    if max_lags>0:
        lag=1
        while lag<=max_lags:
            df['L'+str(lag)+'_'+var]=gp[var].shift(lag)
            lag=lag+1
    elif max_lags<0:
        lag=-1
        while lag>=max_lags:
            df['F'+str(abs(lag))+'_'+var]=gp[var].shift(lag)
            lag=lag-1
    df=df.set_index(np.arange(len(df)))
    return(df)

def add_leads(df, var, max_leads, group, sortvars):
    
    df=add_lags(df, var, -max_leads, group, sortvars)
    return(df)

def formula_lags(var, max_lags):

    formula = ''
    for lag in range(1, max_lags + 1):
        formula += '+ L{0}_{1}'.format(lag, var)

    return formula

def get_formula(horizon, y_var, shock_var, control_vars, fe_vars, shock_lags,
                y_lags, control_lags):

    ############################################################################
    # LHS
    ############################################################################

    if horizon > 0:
        formula = 'F{0}_{1} ~'.format(y_var, horizon)
    else:
        formula = '{0} ~'.format(y_var)

    ############################################################################
    # RHS
    ############################################################################

    # shocks
    formula += ' ' + shock_var
    formula += formula_lags(shock_var, shock_lags)

    # y_var
    formula += formula_lags(y_var, y_lags)

    # controls
    for var in control_vars:
        max_lags = control_lags.get(var, 1)
        formula += formula_lags(var, max_lags)

    for var in fe_vars:
        formula += ' C({0})'.format(var)

    return formula   

def estimate(df_in, y_var, shock_var, control_vars,groupvars, sortvars, timevar, fe_vars=[], shock_lags=2, y_lags=1,periods=20, control_lags={}):
    
    #copy in relevant variables 
    df={}
    all_vars = list(set([y_var] + [shock_var] + control_vars + fe_vars+groupvars+[timevar]))
    for var in all_vars:
        df[var]=df_in[var]
    
    df=pd.DataFrame(df)
    
    df=add_lags(df, shock_var, shock_lags, groupvars, sortvars)
    df=add_lags(df, y_var, y_lags, groupvars, sortvars)
    for var in control_vars:
        df=add_lags(df, var, control_lags.get(var,1), groupvars, sortvars)
    
    df=add_leads(df, y_var, periods, groupvars, sortvars)
    
    #run estimation
    x = np.zeros(periods)
    se = np.zeros(periods)
    fr_list = []    
    
    for jj in range(periods):

        formula = get_formula(jj, y_var, shock_var, control_vars, fe_vars,
                              shock_lags, y_lags, control_lags)

        fr_list.append(dt.formula_regression(df, formula, nw_lags=jj))

    for jj in range(periods):

        x[jj] = fr_list[jj].results.params[1]
        se[jj] = fr_list[jj].results.HC0_se[1]    
    return