import ipdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import solve_discrete_lyapunov
from tabulate import tabulate

from py_tools.debug import disp 
import py_tools.data as dt
import py_tools.utilities as ut

class FullResults:
    """Regression results with index and samples"""
    def __init__(self, results, ix, Xs, zs):
        self.results = results
        self.ix = ix
        self.Xs = Xs
        self.zs = zs

def deflate(df, var_list, index='cpi', log=False, diff=False, reimport=False):
    
    index_var = index + '_index'
    assert index_var not in df

    for var in var_list:

        new_var = 'DEF' + index.upper() + '_' + var
        assert new_var not in df
        new_var_list.append(new_var)

        df_fred = load_fred(reimport=reimport)
        df = pd.merge(df, df_fred[index_var].to_frame(), left_index=True, right_index=True)

        scale = np.log(df[index_var])
        if diff:
            scale = scale.diff()

        if log:
            df[new_var] = df[var] - scale
        else:
            df[new_var] = df[var] / scale

    return [new_var_list]

    # if diff:
        # df[index_var] /= df[index_var].shift()

    # if log:
        # df = 
        # df[index_var] = np.log(df[index_var])

    # if log:
        # df[index_var] = np.log(df[index_var])

    # # df_merge = pd.merge(df, df_fred[index_var].to_frame(), left_index=True, right_index=True)

    # # for var in var_list:

    # if new_var is None:
        # new_var = 'DEF' + index.upper() + '_' + var

    # df_merge['scale'] = np.log(df_fred[index_var])

    # if diff:
        # df_merge['scale'] = df_merge['scale'].diff()

    # if log:
        # df_merge[new_var] = df_merge[var] - df_merge['scale']
    # else:
        # df_merge['scale'] = np.exp(df_merge['scale'])
        # df_merge[new_var] = df_merge[var] / df_merge['scale']

    # df = pd.merge(df, df_merge[new_var].to_frame(), left_index=True, right_index=True)

    # return new_var

def transform(df, var_list, lag=0, diff=0, other=None,
              # , deflate=False, 
              # deflate_ix='cpi', deflate_log=False, deflate_diff=False,
              # deflate_reimport=False
              ):

    new_var_list = []
    for var in var_list:
        new_var = var

        if other is not None:
            prefix = other.upper() + '_'
            new_var = prefix + new_var

        if diff != 0:
            if diff != 1:
                prefix = 'D{}_'.format(diff)
            else:
                prefix = 'D_'
            new_var = prefix + new_var

        if lag != 0:
            if lag != 1:
                prefix = 'L{}_'.format(lag)
            else:
                prefix = 'L_'
            new_var = prefix + new_var

        new_var_list.append(new_var)

        if new_var not in df:
            df[new_var] = df[var]

            # if deflate:
                # _ = dt.deflate(df, new_var, index=deflate_ix, log=deflate_log, 
                               # diff=deflate_diff, reimport=deflate_reimport,
                               # new_var=new_var)

            if other is not None:
                df[new_var] = eval('np.{}(df[new_var])'.format(other))

            if diff != 0:
                df[new_var] = df[new_var].diff(diff)

            if lag != 0:
                df[new_var] = df[new_var].shift(lag)

    return new_var_list

def formula_regression(df, formula, var_list=None, match='inner', ix=None, nw_lags=0, display=False):

    if var_list is not None:
        ix, _ = match_sample(df[var_list].values, how=match, ix=ix)

    if ix is None:
        model = smf.ols(formula=formula, data=df)
    else:
        model = smf.ols(formula=formula, data=df.ix[ix, :])

    results = model.fit()

    if nw_lags > 0:
        results = results.get_robustcov_results('HAC', maxlags=nw_lags)
    else:
        results = results.get_robustcov_results('HC0')

    if display:
        print(results.summary())

    return FullResults(results, ix, Xs=None, zs=None)

def sm_regression(df, lhs, rhs, match='inner', ix=None, nw_lags=0, display=False):
    """Regression using statsmodels"""

    if 'const' in rhs and 'const' not in df:
        df['const'] = 1.0

    X = df.ix[:, rhs].values
    z = df.ix[:, lhs].values

    ix, Xs, zs = match_xy(X, z, how=match, ix=ix)

    model = sm.OLS(zs, Xs)
    results = model.fit()
    if nw_lags > 0:
        results = results.get_robustcov_results('HAC', maxlags=nw_lags)
    else:
        results = results.get_robustcov_results('HC0')

    if display:
        print(results.summary())

    return FullResults(results, ix, Xs, zs)

class MVOLSResults:
    """Regression object"""

    def __init__(self, nobs, params, fittedvalues, resid, cov_HC0):

        self.nobs = nobs
        self.params = params
        self.fittedvalues = fittedvalues
        self.resid = resid
        self.cov_HC0 = cov_HC0

    # def __init__(self, df, lhs, rhs, match='inner', ix=None, nw_lags=0):

        # if 'const' in rhs and 'const' not in df:
            # df['const'] = 1.0

        # X = df.ix[:, rhs].values
        # z = df.ix[:, lhs].values

        # self.match=match
        # self.ix, self.Xs, self.zs = match_xy(X, z, how=self.match, ix=ix)

        # self.nobs = X.shape[0]
        # self.params = least_sq(self.Xs, self.zs)
        # self.fittedvalues = np.dot(self.Xs, self.params)
        # self.resid = self.zs - self.fittedvalues
        # self.cov_HC0 = np.dot(self.resid.T, self.resid) / self.nobs

def mv_ols(df, lhs, rhs, match='inner', ix=None, nw_lags=0):

    if 'const' in rhs and 'const' not in df:
        df['const'] = 1.0

    X = df.ix[:, rhs].values
    z = df.ix[:, lhs].values

    match=match
    ix, Xs, zs = match_xy(X, z, how=match, ix=ix)

    nobs = X.shape[0]
    params = least_sq(Xs, zs)
    fittedvalues = np.dot(Xs, params)
    resid = zs - fittedvalues
    cov_HC0 = np.dot(resid.T, resid) / nobs

    results = MVOLSResults(nobs, params, fittedvalues, resid, cov_HC0)
    return FullResults(results, ix, Xs, zs)

def MA(df, lhs_var, rhs_vars, n_lags=16, display=False):

    lhs = [lhs_var]
    # lhs += transform(df, [lhs_var], 
    # lhs.append(add_lag(df, lhs_var, lag=0, diff=0))

    rhs = ['const']
    for var in rhs_vars:
        for lag in range(n_lags):
            rhs += transform(df, [var], lag=lag)
            # rhs.append(add_lag(df, var, lag=lag))

    # Get sample indices
    ix, _, _ = match_xy(df[rhs_vars].values, df[lhs_var].values)

    # Run regression
    return sm_regression(df, lhs, rhs, match='custom', ix=ix, display=display)

def VAR(df, var_list, n_var_lags=1):

    n_var = len(var_list)

    # LHS variables
    lhs = var_list

    # RHS variables
    rhs = ['const']

    for lag in range(1, n_var_lags + 1):
        for var in var_list:
            rhs += transform(df, [var], lag=lag)

    # Regression
    return mv_ols(df, lhs, rhs)

def VECM(df, var_list, n_var_lags=1, n_dls_lags=8):

    n_var = len(var_list)

    # LHS variables
    lhs = transform(df, var_list, diff=1)

    # RHS variables
    rhs = ['const']

    # Estimate cointegrating relationship
    dls_lhs = var_list[0]
    dls_rhs = var_list[1:]
    alp, dlt = run_dls(df, dls_lhs, dls_rhs, n_dls_lags)

    # Get cointegration term
    df['coint_resid'] = np.dot(df.ix[:, var_list], alp)
    rhs += transform(df, ['coint_resid'], lag=1)

    for lag in range(1, n_var_lags + 1):
        for var in var_list:
            rhs += transform(df, [var], diff=1, lag=lag)
            # rhs.append(add_lag(df, var, lag))

    # Regression
    return mv_ols(df, lhs, rhs)

class LongHorizonMA:
    """Long Horizon Moving Average Regression"""

    def __init__(self, df, lhs_var, rhs_var, horizon, n_lags=16):

        # First stage: MA regression
        fr = MA(df, lhs_var, [rhs_var], n_lags)

        # Second stage: compute LH coefficient
        ma_coeffs = fr.results.params[1:]

        cov_term = 0.0
        for ii in range(horizon):
            start = max(0, ii - horizon + 1)
            end = ii + 1
            cov_term += np.sum(ma_coeffs[start : end])

        bet = cov_term / horizon

class LongHorizonVAR:
    """Long Horizon VAR Regression"""

    def __init__(self, df, lhs_var, rhs_vars, horizon, n_var_lags=1):

        self.lhs_var = lhs_var
        self.rhs_vars = rhs_vars
        self.n_var_lags = n_var_lags
        self.horizon=horizon

        var_list = [self.lhs_var] + self.rhs_vars

        self.n_rhs = len(self.rhs_vars)
        self.n_var = len(var_list)

        self.fr = VAR(df, var_list, self.n_var_lags)

        n_A = self.fr.results.params.shape[0] - 1
        self.A = np.zeros((n_A, n_A))
        self.A[:self.n_var, :] = self.fr.results.params[1:, :].T
        if self.n_var_lags > 1:
            self.A[self.n_var:, :-self.n_var] = np.eye(n_A - self.n_var)

        self.Q = np.zeros(self.A.shape)
        self.Q[:self.n_var, :self.n_var] = self.fr.results.cov_HC0

        # C: unconditional covariances
        self.C = []
        self.C.append(solve_discrete_lyapunov(self.A, self.Q))

        # if predictive:
            # C_sum = np.zeros(self.C[0].shape)

        for jj in range(1, self.horizon + 1):
            self.C.append(np.dot(self.A, self.C[jj-1]))
            # if predictive:
                # C_sum += self.C[jj]

        self.Vk = self.horizon * self.C[0]
        for jj in range(1, self.horizon):
            self.Vk += (self.horizon - jj) * (self.C[jj] + self.C[jj].T)

        # Long-horizon regressions
        # TODO: right now lhs var must be ordered first
        self.lhs_ix = 0

    def predictive_reg(self):
        
        print("Need to check for lags > 1!")
        # Compute covariance of current with future sum
        C_sum = np.zeros(self.C[0].shape)

        for jj in range(1, horizon + 1):
            C_sum += self.C[jj]

        bet_lh = np.zeros(self.n_rhs)
        R2 = np.zeros(self.n_rhs)

        for ii in range(0, self.n_rhs):

            rhs_ix = ii + 1

            lh_rh_cov = C_sum[self.lhs_ix, rhs_ix]
            rh_var = self.C[0][rhs_ix, rhs_ix]

            bet_lh[ii] = lh_rh_cov / rh_var

            lh_var = self.Vk[self.lhs_ix, self.lhs_ix]
            R2[ii] = (lh_rh_cov ** 2) / (lh_var * rh_var)

        bet_lh = np.zeros(self.n_rhs)
        R2 = np.zeros(self.n_rhs)

        return (bet_lh, R2)

    def contemp_reg(self, display=False):

        # covariance terms
        var_y = self.Vk[0, 0]
        cov_xy = self.Vk[0, 1:self.n_var]
        cov_xx = self.Vk[1:self.n_var, 1:self.n_var]

        # OLS
        bet_lh = np.linalg.solve(cov_xx, cov_xy)
        R2 = quad_form(bet_lh, cov_xx) / var_y

        if display:
            # Print first table
            headers = ['LH VAR Regression', '']
            table = [
                ['Dep. Variable:', self.lhs_var],
                ['Lags:', self.n_var_lags],
                ['Horizon:', self.horizon],
                ['R2:', '{:4.3f}'.format(R2)],
            ]

            print('\n\n\n')
            print(tabulate(table, headers, tablefmt='rst', floatfmt='4.3f'))

            headers = ['', 'coef']
            table = []
            for ii in range(self.n_rhs):
                table.append([self.rhs_vars[ii], bet_lh[ii]])

            print('\n')
            print(tabulate(table, headers, tablefmt='rst', floatfmt='4.3f'))
            print('\n\n\n')

        return (bet_lh, R2)

    def orthogonal_contemp_reg(self, display=False):

        bet_lh = np.zeros(self.n_rhs)
        R2 = np.zeros(self.n_rhs)

        # covariance terms
        var_y = self.Vk[0, 0]
        cov_xy = self.Vk[0, 1:self.n_var]
        cov_xx = self.Vk[1:self.n_var, 1:self.n_var]

        # pre-allocate
        bet_xu = np.zeros((self.n_rhs, self.n_rhs))

        # initialize at un-orthogonalized values
        cov_yu = cov_xy.copy()
        cov_xu = cov_xx.copy()
        var_u = np.diagonal(cov_xx).copy()

        for jj in range(1, self.n_rhs):
            for kk in range(jj):
                bet_xu = cov_xu[jj, kk] / var_u[kk]
                cov_xu[:, jj] -= bet_xu * cov_xu[:, kk]
                cov_yu[jj] -= bet_xu * cov_yu[kk]
                var_u[jj] -= bet_xu * cov_xu[jj, kk]

        bet_lh = cov_yu / var_u
        R2 = ((bet_lh ** 2) * var_u) / var_y

        if display:
            # Print first table
            headers = ['LH VAR Orth. Regression', '']
            table = [
                ['Dep. Variable:', self.lhs_var],
                ['Lags:', self.n_var_lags],
                ['Horizon:', self.horizon],
                ['Total R2', '{:4.3f}'.format(np.sum(R2))],
            ]

            print('\n\n\n')
            print(tabulate(table, headers, tablefmt='rst', floatfmt='4.3f'))

            headers = ['', 'coef', 'R2']
            table = []
            for ii in range(self.n_rhs):
                table.append([self.rhs_vars[ii], bet_lh[ii], R2[ii]])

            print('\n')
            print(tabulate(table, headers, tablefmt='rst', floatfmt='4.3f'))
            print('\n\n\n')

        return (bet_lh, R2)

def orthogonalize_errors(u):
    """Cholesky decomposition"""
    Sig = np.dot(u.T, u) / u.shape[0]
    H = np.linalg.cholesky(Sig)
    e = (np.linalg.solve(H, u.T)).T
    return (e, H)

def match_sample(X, how='inner', ix=None):

    if how == 'inner':
        ix = np.all(pd.notnull(X), axis=1)
    elif how == 'outer':
        ix = np.any(pd.notnull(X), axis=1)
    elif how != 'custom':
        raise Exception
        
    Xs = X[ix, :]
    Xs[pd.isnull(Xs)] = 0.0
    
    return (ix, Xs)

def match_xy(X, z, how='inner', ix=None):

    # TODO: should change so that originals not modified
    if len(z.shape) == 1:
        z = z[:, np.newaxis]
        
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    Xall = np.hstack((X, z))
    ix, Xall_s = match_sample(Xall, how=how, ix=ix)

    Nx = X.shape[1]
    Nz = z.shape[1]

    Xs = Xall_s[:, :Nx]
    zs = Xall_s[:, Nx:]
    
    # if how == 'inner':
        # ix = np.all(pd.notnull(np.hstack((X, z))), axis=1)
    # elif how == 'outer':
        # ix = np.any(pd.notnull(np.hstack((X, z))), axis=1)
    # elif how != 'custom':
        # raise Exception
        
    # Xs = X[ix, :]
    # zs = z[ix, :]
    
    # Xs[pd.isnull(Xs)] = 0.0
    # zs[pd.isnull(zs)] = 0.0
    
    return (ix, Xs, zs)

def least_sq(X, z):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, z))

def quad_form(A, X):
    return np.dot(A.T, np.dot(X, A))

# Estimate cointegrating relationship using DLS
def run_dls(df, lhs_var, rhs_vars, n_lags=8, display=False):

    n_rhs = len(rhs_vars)
    rhs = ['const'] + rhs_vars

    lhs = [lhs_var]
    
    for lag in range(-n_lags, n_lags + 1):
        for var in rhs_vars:
            rhs += transform(df, [var], lag=lag, diff=1)
            # rhs.append(add_lag(df, var, lag, diff=1))
            
    # Regression
    fr = sm_regression(df, lhs, rhs, display=display)
    
    coint_vec = np.hstack([np.ones(1), -fr.results.params[1 : n_rhs + 1]])
    const = fr.results.params[0]

    return (coint_vec, const)
