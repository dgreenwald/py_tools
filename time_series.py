import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import solve_discrete_lyapunov

import data
from py_tools.debug import disp 
import py_tools.data as dt
import py_tools.utilities as ut

def sm_regression(df, lhs, rhs, match='inner', ix=None, nw_lags=0):
    """Regression using statsmodels"""

    if 'const' in rhs and 'const' not in df:
        df['const'] = 1.0

    X = df.ix[:, rhs].values
    z = df.ix[:, lhs].values

    ix, Xs, zs = match_sample(X, z, how=match, ix=ix)

    model = sm.OLS(zs, Xs)
    results = model.fit()
    if nw_lags > 0:
        results = results.get_robustcov_results('HAC', maxlags=nw_lags)
    else:
        results = results.get_robustcov_results('HC0')

    return results

def MA(df, lhs_var, rhs_vars, n_lags=16):

    lhs = [lhs_var]
    # lhs += dt.transform(df, [lhs_var], 
    # lhs.append(add_lag(df, lhs_var, lag=0, diff=0))

    rhs = ['const']
    for var in rhs_vars:
        for lag in range(n_lags):
            rhs += dt.transform(df, [var], lag=lag)
            # rhs.append(add_lag(df, var, lag=lag))

    # Get sample indices
    ix, _, _ = match_sample(df[rhs_vars].values, df[lhs_var].values)

    # Run regression
    return sm_regression(df, lhs, rhs, match='custom', ix=ix)

def VAR(df, var_list, n_var_lags=1, vecm=True, n_dls_lags=8):

    n_var = len(var_list)

    # LHS variables
    lhs = var_list

    # RHS variables
    rhs = ['const']

    if vecm:
        # Estimate cointegrating relationship
        dls_lhs = var_list[0]
        dls_rhs = var_list[1:]
        alp, dlt = run_dls(df, dls_lhs, dls_rhs, n_dls_lags)

        # Get cointegration term
        df['coint_resid'] = np.dot(df.ix[:, var_list], alp)
        rhs += dt.transform(df, ['coint_resid'], lag=1)
        # rhs.append(add_lag(df, var='coint_resid', lag=1))
    else:
        alp = np.zeros(n_var)
        dlt = 0.0

    for lag in range(1, n_var_lags + 1):
        for var in var_list:
            rhs += dt.transform(df, [var], lag=lag)
            # rhs.append(add_lag(df, var, lag))

    # Regression
    return sm_regression(df, lhs, rhs)

class LongHorizonMA:
    """Long Horizon Moving Average Regression"""

    def __init__(self, df, lhs_var, rhs_var, horizon, n_lags=16):

        # First stage: MA regression
        results = MA(df, lhs_var, [rhs_var], n_lags)

        # Second stage: compute LH coefficient
        ma_coeffs = results.params[1:]

        cov_term = 0.0
        for ii in range(horizon):
            start = max(0, ii - horizon + 1)
            end = ii + 1
            cov_term += np.sum(ma_coeffs[start : end])

        bet = cov_term / horizon

class LongHorizonVAR:
    """Long Horizon VAR Regression"""

    def __init__(self, df, lhs_var, rhs_vars, horizon, n_var_lags=1, vecm=False, diff=0):

        var_list = [lhs_var] + rhs_vars

        n_rhs = len(rhs_vars)
        n_var = len(var_list)

        results = VAR(df, var_list, n_var_lags, vecm=vecm, diff=diff)

        n_A = results.params.shape[0] - 1
        A = np.zeros((n_A, n_A))
        A[:n_var, :] = results.params[1:, :].T
        if n_var_lags > 1:
            A[n_var:, :-n_var] = np.eye(n_A - n_var)

        Q = np.zeros(A.shape)
        Q[:n_var, :n_var] = results.cov_HC0

        # C: unconditional covariances
        C = []
        C.append(solve_discrete_lyapunov(A, Q))
        C_sum = np.zeros(C[0].shape)
        for jj in range(1, horizon + 1):
            C.append(np.dot(A, C[jj-1]))
            C_sum += C[jj]

        Vk = horizon * C[0]
        for jj in range(1, horizon):
            Vk += (horizon - jj) * (C[jj] + C[jj].T)

        # Long-horizon regressions
        pick_lhs = np.zeros((n_A, 1))
        pick_lhs[0] = 1

        self.bet_lh = np.zeros(n_rhs)
        self.R2 = np.zeros(n_rhs)

        for ii in range(n_rhs):

            pick_rhs = np.zeros((n_A, 1))
            pick_rhs[ii] = 1

            bet_lh_num = np.dot(pick_lhs.T, np.dot(C_sum, pick_rhs))
            bet_lh_denom = quad_form(pick_rhs, C[0])
            self.bet_lh[ii] = bet_lh_num / bet_lh_denom

            R2_num = (self.bet_lh[ii] ** 2) * quad_form(pick_rhs, C[0])
            R2_denom = quad_form(pick_lhs, Vk)
            self.R2[ii] = R2_num / R2_denom

def orthogonalize_errors(u):
    """Cholesky decomposition"""
    Sig = np.dot(u.T, u) / u.shape[0]
    H = np.linalg.cholesky(Sig)
    e = (np.linalg.solve(H, u.T)).T
    return (e, H)

def match_sample(X, z, how='inner', ix=None):

    # TODO: should change so that originals not modified
    if len(z.shape) == 1:
        z = z[:, np.newaxis]
        
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    if how == 'inner':
        ix = np.all(pd.notnull(np.hstack((X, z))), axis=1)
    elif how == 'outer':
        ix = np.any(pd.notnull(np.hstack((X, z))), axis=1)
    elif how != 'custom':
        raise Exception
        
    Xs = X[ix, :]
    zs = z[ix, :]
    
    Xs[pd.isnull(Xs)] = 0.0
    zs[pd.isnull(zs)] = 0.0
    
    return (ix, Xs, zs)

def least_sq(X, z):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, z))

def quad_form(A, X):
    return np.dot(A.T, np.dot(X, A))

# Estimate cointegrating relationship using DLS
def run_dls(df, lhs_var, rhs_vars, n_lags=8):

    n_rhs = len(rhs_vars)
    rhs = ['const'] + rhs_vars

    lhs = [lhs_var]
    
    for lag in range(-n_lags, n_lags + 1):
        for var in rhs_vars:
            rhs += dt.transform(df, [var], lag=lag, diff=1)
            # rhs.append(add_lag(df, var, lag, diff=1))
            
    # Regression
    results = sm_regression(df, lhs, rhs)
    
    coint_vec = np.hstack([np.ones(1), -results.params[1 : n_rhs + 1]])
    const = results.params[0]

    return (coint_vec, const)
