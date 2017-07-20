# import ipdb
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import solve_discrete_lyapunov
from tabulate import tabulate

# from py_tools.debug import disp 
# import py_tools.data as dt
from py_tools.datasets import loader
# import py_tools.utilities as ut

class FullResults:
    """Regression results with index and samples"""
    def __init__(self, results, ix, Xs, zs):
        self.results = results
        self.ix = ix
        self.Xs = Xs
        self.zs = zs

def deflate(df, var_list, index='cpi', log=False, diff=False, per_capita=False, 
            reimport=False, **kwargs):
    """Deflates series by price indexes
    
    Inputs:

        df: pandas dataframe containing data

        var_list: list of variables to be deflated

        index: price index to use, one of

            cpi: CPI deflator
            pce: PCE deflator
            pop: population (to put in per-capita terms)

        log: set to True if the variable is ALREADY in logs rather than levels
            (this will not take logs)

        diff: set to True if the variable is ALREADY in differences 
            (this will not take differences)

        per_capita: set to True if you ALSO want to put the variable in per-capita terms
            (unlike "log" and "diff" this WILL adjust the series)

        reimport: set to True if you want to re-import the raw data instead of using the
            saved pickle files

    Outputs:

        df: dataframe now including deflated series

        new_var_list: names of deflated variables
    
    """
    
    # index_var = 'FRED_' + index + '_index'
    new_var_list = []

    if per_capita:
        assert index != 'pop'
        df, var_list = deflate(df, var_list, index='pop', log=log, diff=diff, per_capita=False, 
                               reimport=reimport, **kwargs)

    for var in var_list:

        if index == 'pop':
            new_var = 'PC_' + var
        else:
            new_var = 'DEF' + index.upper() + '_' + var
        # if per_capita:
            # new_var += 'PC_'
        # new_var += var

        new_var_list.append(new_var)

        if new_var not in df:

            if index == 'cpi':
                dataset = 'fred'
                index_name = 'cpi_deflator'
            elif index == 'pce':
                dataset = 'nipa_10109'
                index_name = 'pce_deflator'
            elif index == 'pop':
                dataset = 'nipa_20100'

            if index == 'pop':
                index_var = 'pop'
            elif index == 'pce':
                index_var = index_name
            else:
                index_var = dataset.upper() + '_' + index_name 

            if index_var not in df:

                df_new = loader.load([dataset], reimport=reimport, **kwargs)

                if index == 'pop':
                    df_new[index_var] = df_new['NIPA_20100_real_disp_inc'] / df_new['NIPA_20100_real_pc_disp_inc']

                df = pd.merge(df, df_new[index_var].to_frame(), left_index=True, right_index=True)

            # if per_capita and 'pop' not in df:
                # df_new = dt.load(['nipa_20100'], reimport=reimport)
                # df_new['pop'] = df_new['NIPA_20100_real_disp_inc'] / df_new['NIPA_20100_real_pc_disp_inc']
                # df = pd.merge(df, df_new['pop'].to_frame(), left_index=True, right_index=True)
                
            scale = np.log(df[index_var])
            # if per_capita:
                # scale += np.log(df['pop'])

            if diff:
                scale = scale.diff()

            if log:
                df[new_var] = df[var] - scale
            else:
                df[new_var] = df[var] / np.exp(scale)

    return (df, new_var_list)

def add_lags(df, var, n_lags, init_lag=1):

    lag_list = []
    for lag in range(init_lag, n_lags + init_lag):
        lag_list += transform(df, [var], lag=lag)

    return lag_list

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

        new_var = re.sub('-', '_MINUS_', new_var)

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

def regression(df, lhs, rhs, intercept=True, formula_extra=None, ix=None, **kwargs):

    formula = '{0} ~ {1}'.format(lhs, ' + '.join(rhs))

    if ix is None:
        ix, _ = match_sample(df[[lhs] + rhs].values, how='inner')

    if formula_extra is not None:
        formula += ' + ' + formula_extra

    if not intercept:
        formula += ' -1'

    return formula_regression(df, formula, ix=ix, **kwargs)

def formula_regression(df, formula, var_list=None, match='inner', ix=None, 
                       nw_lags=0, display=False):

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

def long_horizon_contemp(df, lhs, rhs, horizon, **kwargs):

    raise Exception

    long_list = transform(df, [lhs] + rhs, diff=horizon, other='cumsum') 
    lhs_long = long_list[0]
    rhs_long = long_list[1:]

    return regression(df, lhs_long, rhs_long, **kwargs)

def long_horizon_predictive(df, lhs, rhs, horizon, **kwargs):

    # lhs_long = transform(df, [lhs], lag=-1, diff=horizon, other='cumsum')[0]

    df['lhs_long'] = 0
    for jj in range(1, horizon + 1):
        lhs_diff = transform(df, [lhs], lag=-jj)[0]
        df['lhs_long'] += df[lhs_diff]

    return regression(df, 'lhs_long', rhs, **kwargs)

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

def MA(df, lhs_var, rhs_vars, init_lag=1, default_lags=16, 
       lags_by_var={}, **kwargs):

    lhs = lhs_var
    # lhs += transform(df, [lhs_var], 
    # lhs.append(add_lag(df, lhs_var, lag=0, diff=0))

    # rhs = ['const']
    rhs = []
    for var in rhs_vars:
        this_lag = lags_by_var.get(var, default_lags)
        for lag in range(init_lag, this_lag + init_lag):
            rhs += transform(df, [var], lag=lag)
            # rhs.append(add_lag(df, var, lag=lag))

    # Get sample indices
    ix, _, _ = match_xy(df[rhs_vars].values, df[lhs_var].values)

    # Run regression
    return regression(df, lhs, rhs, match='custom', ix=ix, **kwargs)

def VAR(df_in, var_list, n_var_lags=1, use_const=True):
    """Estimate VAR using OLS"""

    df = df_in.copy()

    # LHS variables
    lhs = var_list
    rhs = []

    for lag in range(1, n_var_lags + 1):
        for var in var_list:
            rhs += transform(df, [var], lag=lag)

    # RHS variables
    if use_const:
        rhs += ['const']

    # Regression
    return mv_ols(df, lhs, rhs)

def VECM(df, var_list, n_var_lags=1, n_dls_lags=8):
    """Estimate VECM using DLS"""

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

        print("Need to fix!")
        raise Exception
        # # First stage: MA regression
        # fr = MA(df, lhs_var, [rhs_var], n_lags)

        # # Second stage: compute LH coefficient
        # ma_coeffs = fr.results.params[1:]

        # cov_term = 0.0
        # for ii in range(horizon):
            # start = max(0, ii - horizon + 1)
            # end = ii + 1
            # cov_term += np.sum(ma_coeffs[start : end])

        # bet = cov_term / horizon

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

        for jj in range(1, self.horizon + 1):
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

    Xs = Xall_s[:, :Nx]
    zs = Xall_s[:, Nx:]
    
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

def lagged_reg(df_in, lhs, rhs_list, n_lags, use_const=True, copy_df=True):
    """Regression of lhs on 1 through n_lags lags of rhs_list."""

    if copy_df:
        df = df_in.copy()
    else:
        df = df_in

    rhs = []
    for lag in range(1, n_lags + 1):
        for var in rhs_list:
            rhs += transform(df, [var], lag=lag)

    if use_const:
        rhs += ['const']

    return mv_ols(df, lhs, rhs)

def detrend_hamilton(df_full, varlist, p=4, h=8):
    """Apply Hamilton's recommended detrending procedure (instead of HP filter)"""

    fr_list = []
    for var in varlist:

        df = df_full[[var]].copy()

        rhs = []
        for ii in range(p):
            lag = h + ii
            rhs += transform(df, [var], lag=lag)

        fr = regression(df, var, rhs)
        fr_list.append(fr)

        df_full[var + '_detrend'] = np.nan
        df_full.loc[fr.ix, var + '_detrend'] = fr.results.resid

    varlist_detrended = [var + '_detrend' for var in varlist]

    return df_full, varlist_detrended, fr_list

def detrend_time(df_full, varlist):

    for var in varlist:

        df = df_full[[var]].copy() 
        df['diff'] = df[var].diff()
        df['diff'] -= df['diff'].mean()
        df_full[var + '_detrend'] = np.cumsum(df['diff'])

    varlist_detrended = [var + '_detrend' for var in varlist]

    return df_full, varlist_detrended
