#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:06:33 2024

@author: dan
"""

import numpy as np
from . import data as dt
from scipy.optimize import root_scalar

def ar1_bootstrap_inner(rho, mu, eps_boot, x_init):
    
    Nboot, Nt_eps = eps_boot.shape
    
    X = np.zeros((Nboot, Nt_eps+1))
    X[:, 0] = x_init
    for tt in range(1, Nt_eps+1):
        X[:, tt] = (1.0 - rho) * mu + rho * X[:, tt-1] + eps_boot[:, tt-1]
        
    # X_til = X - np.mean(axis=1)
    X_next = X[:, 1:]
    X_pre = X[:, :-1]
    
    X_next_til = X_next - np.mean(X_next, axis=1)[:, np.newaxis]
    X_pre_til = X_pre - np.mean(X_pre, axis=1)[:, np.newaxis]
    
    cov = np.sum(X_next_til * X_pre_til, axis=1) / (Nt_eps - 1)
    var = np.sum(X_pre_til ** 2, axis=1) / (Nt_eps - 1)
    
    rho_sim = cov / var
    rho_sim_avg = np.mean(rho_sim)
    return rho_sim_avg

def objfcn_ar1_bootstrap(rho, rho_ols, mu, eps_boot, x_init):
    
    rho_sim_avg = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
    return rho_ols - rho_sim_avg

def ar1_bootstrap_bias(df, obs_var, Nboot=100000, iterate=False):
    
    _df = df[[obs_var]].dropna()
    _df['L_' + obs_var] = _df[obs_var].shift()

    fr = dt.regression(_df, obs_var, ['L_' + obs_var])
    mu = fr.results.params[0]
    rho_ols = fr.results.params[1]
    print("rho_ols: " + repr(rho_ols))
    
    rho = rho_ols
    resid = fr.results.resid
    
    Nt_eps = len(resid)
    eps_boot = np.random.choice(resid, Nboot*Nt_eps, replace=True).reshape((Nboot, Nt_eps))
    x_init = _df.iloc[0][obs_var]
    
    if iterate:
        
        rho_guess = 0.5 * (rho_ols + 1.0)
        args = (rho_ols, mu, eps_boot, x_init)
        res = root_scalar(objfcn_ar1_bootstrap, args=args, x0=rho_ols, x1=rho_guess)
        rho = res.root
        
    else:
        
        rho_sim_avg = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        bias = rho_ols - rho_sim_avg
        rho = rho_ols + bias
        print("original bias: " + repr(bias))
    
    print("rho_new: " + repr(rho))
    
    return rho
        