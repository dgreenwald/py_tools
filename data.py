import numpy as np
import pandas as pd
import pdb
import pickle

import statsmodels.api as sm
import statsmodels.formula.api as smf

class FullResults:
    """Regression results with index and samples"""
    def __init__(self, results, ix, Xs, zs):
        self.results = results
        self.ix = ix
        self.Xs = Xs
        self.zs = zs

def winsorize(df_in, var_list, p_val=0.98):
    """Replace values of var_list outside the center p_val quantile mass with
    values at the edge of the mass"""

    # if p_val is not None:
        # assert (p_lo is None) and (p_hi is None)
    tail_prob = 0.5 * (1.0 - p_val)
    p_lo = tail_prob
    p_hi = 1.0 - tail_prob
    # else:
        # assert (p_lo is not None) and (p_hi is not None)

    df = df_in[var_list].copy()
    for var in var_list:
        lb, ub = df[var].quantile([p_lo, p_hi]).values
        df.loc[df[var] < lb, var] = lb
        df.loc[df[var] > ub, var] = ub

    return df 

def demean(df_in, var_list, by_var, weight_var=None):

    df = df_in[var_list + [by_var, weight_var]].copy()
    if weight_var is None:
        df_by = df[var_list].groupby(by_var).mean().rename(
            columns={var : var + '_mean' for var in var_list}
        )
    else:
        for var in var_list:
            df[var + '_wtd'] = df[var] * df[weight_var]

        df_by = df.groupby(by_var).sum()
        for var in var_list:
            df_by[var + '_mean'] = df_by[var + '_wtd'] / df_by[weight_var]

    mean_list = [var + '_mean' for var in var_list]
    df = pd.merge(df, df_by[mean_list], left_on=by_var, right_index=True)
    for var in var_list:
        df[var + '_demeaned'] = df[var] - df[var + '_mean']

    demean_list = [var + '_demeaned' for var in var_list]
    return (df[mean_list + demean_list], mean_list + demean_list)

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

    Xs = Xall_s[:, :Nx]
    zs = Xall_s[:, Nx:]
    
    return (ix, Xs, zs)


def regression(df_in, lhs, rhs, intercept=True, formula_extra=None, ix=None, 
               trend=None, **kwargs):
    """Run regression from pandas dataframe"""

    formula = '{0} ~ {1}'.format(lhs, ' + '.join(rhs))

    df = df_in[[lhs] + rhs].copy()

    if ix is None:
        ix, _ = match_sample(df.values, how='inner')

    if trend is not None:
        if trend in ['linear', 'quadratic']:
            df['t'] = np.arange(len(df))
            formula += ' + t '
        if trend == 'quadratic':
            df['t2'] = np.arange(len(df)) ** 2
            formula += ' + t2 '

    if formula_extra is not None:
        formula += ' + ' + formula_extra

    if not intercept:
        formula += ' -1'

    return formula_regression(df, formula, ix=ix, **kwargs)

def formula_regression(df, formula, var_list=None, match='inner', ix=None, 
                       nw_lags=0, display=False, trend=None):

    # if var_list is not None:
        # ix, Xs, zs = match_sample(df[var_list].values, how=match, ix=ix)
    # else:
        # Xs = None
        # zs = None

    if ix is None:
        model = smf.ols(formula=formula, data=df)
    else:
        model = smf.ols(formula=formula, data=df.loc[ix, :])

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

def clean(df_in, var_list):
    """Remove infinite and nan values from dataset"""

    good_list = [var for var in var_list if var in df_in]
    df = df_in[good_list].copy().replace([np.inf, -np.inf], np.nan)
    return df.dropna()

def dropna_ix(df):

    ix = np.all(pd.notnull(df).values, axis=1)
    return df.loc[ix, :], ix

class MVOLSResults:
    """Regression object"""

    def __init__(self, nobs, params, fittedvalues, resid, cov_e,
                 cov_HC0, HC0_se, HC0_tstat,
                 cov_HC1, HC1_se, HC1_tstat,
                 llf, aic, bic, hqc):

        self.nobs = nobs
        self.params = params
        self.fittedvalues = fittedvalues
        self.resid = resid
        self.cov_e = cov_e

        self.cov_HC0 = cov_HC0
        self.HC0_se = HC0_se
        self.HC0_tstat = HC0_tstat

        self.cov_HC1 = cov_HC1
        self.HC1_se = HC1_se
        self.HC1_tstat = HC1_tstat

        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.hqc = hqc

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
        # self.cov_e = np.dot(self.resid.T, self.resid) / self.nobs

def mv_ols(df, lhs, rhs, match='inner', ix=None, nw_lags=0):

    if 'const' in rhs and 'const' not in df:
        df['const'] = 1.0

    if nw_lags > 0:
        print("Need to code")
        raise Exception

    X = df.ix[:, rhs].values
    z = df.ix[:, lhs].values

    match=match
    ix, Xs, zs = match_xy(X, z, how=match, ix=ix)

    # Get sizes
    T, k = Xs.shape
    _, nz = zs.shape

    nobs = X.shape[0]
    params = least_sq(Xs, zs)
    fittedvalues = np.dot(Xs, params)
    resid = zs - fittedvalues
    # cov_HC0 = np.dot(resid.T, resid) / nobs

    # Homoskedastic covariance
    cov_HC0, cov_e = hc0(Xs, resid)
    # Note: reshape is transposed since params is (k x n) not (n x k)
    # otherwise would need to set order='F'

    HC0_se = standard_errors(cov_HC0, T).reshape(params.shape)
    HC0_tstat = params / HC0_se

    # Heteroskedastic covariance
    # cov_xeex = np.zeros((nz*k, nz*k))
    # for tt in range(T):
        # x_t = Xs[tt, :][:, np.newaxis]
        # e_t = resid[tt, :][:, np.newaxis]
        # cov_xeex += np.kron(np.dot(x_t, x_t.T), np.dot(e_t, e_t.T))

    # cov_xeex /= T
    # cov_HC1 = np.dot(cov_X_inv, np.dot(cov_xeex, cov_X_inv))

    # NOTE: for now, computing both HC0 and HC1
    cov_HC1, _ = hc1(Xs, resid)
    HC1_se = standard_errors(cov_HC1, T).reshape(params.shape)
    HC1_tstat = params / HC1_se

    # Compute likelihood
    n_free = np.prod(params.shape)

    log_det_cov_e = np.log(np.linalg.det(cov_e))
    llf = -0.5 * (T * log_det_cov_e + T * nz * (1.0 + np.log(2.0 * np.pi)))
    aic = log_det_cov_e + (2.0 / T) * n_free
    bic = log_det_cov_e + (np.log(T) / T) * n_free
    hqc = log_det_cov_e + (2.0 * np.log(np.log(T)) / T) * n_free

    results = MVOLSResults(nobs, params, fittedvalues, resid, cov_e,
                           cov_HC0, HC0_se, HC0_tstat,
                           cov_HC1, HC1_se, HC1_tstat,
                           llf, aic, bic, hqc)

    return FullResults(results, ix, Xs, zs)

def hc0(x, e):
    "Homoskedastic Covariance"
    cov_e, _, cov_x_inv, _, _, _ = init_cov(x, e)
    cov_HC0 = np.kron(cov_x_inv, cov_e)
    return (cov_HC0, cov_e)

def hc1(x, e):
    "Heteroskedastic Covariance"
    cov_e, cov_x, cov_x_inv, T, nz, k = init_cov(x, e)
    cov_xeex = np.zeros((nz*k, nz*k))
    for tt in range(T):
        x_t = x[tt, :][:, np.newaxis]
        e_t = e[tt, :][:, np.newaxis]
        cov_xeex += np.kron(np.dot(x_t, x_t.T), np.dot(e_t, e_t.T))
    cov_xeex /= T

    cov_X_inv = np.kron(cov_x_inv, np.eye(nz))
    cov_HC1 = np.dot(cov_X_inv, np.dot(cov_xeex, cov_X_inv))
    return (cov_HC1, cov_e)

def init_cov(x, e):

    T, k = x.shape
    Te, nz = e.shape
    assert(T == Te)

    cov_e = np.dot(e.T, e) / T
    cov_x = np.dot(x.T, x) / T
    cov_x_inv = np.linalg.inv(cov_x)

    return (cov_e, cov_x, cov_x_inv, T, nz, k)

def standard_errors(V, T):

    se = np.sqrt(np.diagonal(V) / T)
    return se

def least_sq(X, z):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, z))

def to_pickle(x, path):

    pickle.dump(x, open(path, "wb"))
    return None

def read_pickle(path):

    return pickle.load(open(path, "rb"))
