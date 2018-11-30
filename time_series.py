# import ipdb
import numpy as np
import pandas as pd
import re
from scipy.linalg import solve_discrete_lyapunov
from tabulate import tabulate

# from py_tools.debug import disp 
# import py_tools.data as dt
import py_tools.data as dt, py_tools.numerical as nm
# from py_tools.datasets import loader
# import py_tools.utilities as ut

def merge_date(df1, df2, how='outer', **kwargs):

    return pd.merge(df1, df2, left_index=True, right_index=True, how=how, **kwargs)

def merge_date_many(df_list, how='outer', **kwargs):

    df = df_list[0]
    for ii in range(1, len(df_list)):
        df = pd.merge(df, df_list[ii], left_index=True, right_index=True,
                        how=how, **kwargs)

    return df

def get_date_index(startdate, periods, freq='QS'):
    return pd.date_range(startdate, periods=periods, freq=freq)

def date_index(df, startdate, freq='QS'):
    df.set_index(get_date_index(startdate, len(df), freq=freq), inplace=True)
    return df

def quarter_index(df, yr, q):
    mon = 3 * (int(q) - 1) + 1
    return month_index(df, yr, mon)

def month_index(df, yr, mon):
    return date_index(df, '{0}/1/{1}'.format(mon, yr))

def date_from_qtr(yr, q):
    mon = 3 * (q - 1) + 1
    return date_from_month(yr, mon)

def date_from_month(yr, mon):
    date = yr.astype(str) + '-' + mon.astype(str) + '-01'
    return pd.to_datetime(date)

def resample(df, methods_vars, freq='QS'):
    df_resamp = None

    for method, var_list in methods_vars.items():

        if var_list is not None:

            df_new = getattr(df[var_list].resample(freq), method)()

            # if len(var_list) == 1:
                # df_new = df_new.to_frame()
            if df_resamp is None:
                df_resamp = df_new
            else:
                df_resamp = pd.merge(df_resamp, df_new, left_index=True, right_index=True)

    return df_resamp

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

    from py_tools.datasets.loader import load
    
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

                df_new = load([dataset], reimport=reimport, **kwargs)

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

def long_horizon_contemp(df, lhs, rhs, horizon, **kwargs):

    raise Exception

    long_list = transform(df, [lhs] + rhs, diff=horizon, other='cumsum') 
    lhs_long = long_list[0]
    rhs_long = long_list[1:]

    return dt.regression(df, lhs_long, rhs_long, **kwargs)

def long_horizon_predictive(df_in, lhs, rhs, horizon, norm_lhs=False, **kwargs):

    # lhs_long = transform(df, [lhs], lag=-1, diff=horizon, other='cumsum')[0]

    df = df_in[[lhs] + rhs].copy()
    lhs_long = '{0}_{1}_Per_Diff'.format(lhs, horizon)
    df[lhs_long] = 0
    for jj in range(1, horizon + 1):
        lhs_diff = transform(df, [lhs], lag=-jj)[0]
        df[lhs_long] += df[lhs_diff]

    if norm_lhs:
        df[lhs_long] /= horizon

    return dt.regression(df, lhs_long, rhs, **kwargs)

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
    ix, _, _ = dt.match_xy(df[rhs_vars].values, df[lhs_var].values)

    # Run dt.regression
    return dt.regression(df, lhs, rhs, ix=ix, **kwargs)

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
    return dt.mv_ols(df, lhs, rhs)

class VECMResults(dt.FullResults):
    """VECM results object"""

    def __init__(self, fr, alp):
        dt.FullResults.__init__(self, fr.results, fr.ix, fr.Xs, fr.zs)
        self.alp = alp

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
    return VECMResults(dt.mv_ols(df, lhs, rhs), alp)

def VECM_companion_form(vr, H=None):

    if len(vr.alp.shape) == 1:
        n_coint = 1
    else:
        raise Exception
        
    n_var = vr.results.params.shape[1]
    
    At = vr.results.params[1:, :].T
    gam = At[:, :n_coint]
    Gam = At[:, n_coint:]
    
    nx, nz = At.shape

    Phi = np.zeros((nz, nz))
    Phi[:nx, :nx] = Gam
    Phi[:nx, nx:] = gam
    Phi[nx:, :nx] = np.dot(vr.alp, Gam)
    Phi[nx:, nx:] = 1.0 + np.dot(vr.alp, gam)

    if H is not None:
        C = np.zeros((nz, n_var))
        C[:nx, :] = H
        C[nx:, :] = np.dot(vr.alp, H)
    else:
        C = None

    return (Phi, C)

def VECM_irfs(vr, H, nt):

    # Update to companion form
    Phi, C = VECM_companion_form(vr, H)

    # Set sizes
    ny = len(vr.alp)
    nx = Phi.shape[0]
    ne = C.shape[1]

    # Run IRFs in companion form
    irfs = np.zeros((nx, ne, nt))
    irfs[:, :, 0] = C
    for tt in range(1, nt):
        irfs[:, :, tt] = np.dot(Phi, irfs[:, :, tt-1])

    # Drop lagged terms
    return np.cumsum(irfs[:ny, :, :], axis=2)

class LongHorizonMA:
    """Long Horizon Moving Average Regression"""

    def __init__(self, df, lhs_var, rhs_var, horizon, n_lags=16):

        print("Need to fix!")
        raise Exception
        # # First stage: MA dt.regression
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
        R2 = nm.quad_form(bet_lh, cov_xx) / var_y

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
    fr = dt.sm_regression(df, lhs, rhs, display=display)
    
    coint_vec = np.hstack([np.ones(1), -fr.results.params[1 : n_rhs + 1]])
    const = fr.results.params[0]

    return (coint_vec, const)

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

    return dt.mv_ols(df, lhs, rhs)

def detrend_hamilton(df_full, varlist, p=4, h=8):
    """Apply Hamilton's recommended detrending procedure (instead of HP filter)
    Returns a dataframe including detrended variables with suffix '_detrend'
    """

    fr_list = []
    for var in varlist:

        df = df_full[[var]].copy()

        rhs = []
        for ii in range(p):
            lag = h + ii
            rhs += transform(df, [var], lag=lag)

        fr = dt.regression(df, var, rhs)
        fr_list.append(fr)

        df_full[var + '_detrend'] = np.nan
        df_full.loc[fr.ix, var + '_detrend'] = fr.results.resid

        df_full[var + '_trend'] = np.nan
        df_full.loc[fr.ix, var + '_trend'] = fr.results.fittedvalues

    varlist_detrended = [var + '_detrend' for var in varlist]

    return df_full, varlist_detrended, fr_list

def detrend_time(df_full, varlist):
    """Remove a linear time trend from the variables in var_list.  

    Returns (df_full, varlist_detrend), where df_full is a dataframe including
    detrended variables with suffix '_detrend', and varlist_detrend is the list
    of detrended variables.
    """

    for var in varlist:

        df = df_full[[var]].copy() 
        df['diff'] = df[var].diff()
        df['diff'] -= df['diff'].mean()
        df_full[var + '_detrend'] = np.cumsum(df['diff'])

    varlist_detrended = [var + '_detrend' for var in varlist]

    return df_full, varlist_detrended

def lead_lag_correlations(df_in, var1, var2, lags=None,
                          max_leads=8, max_lags=8, make_plot=False,
                          **kwargs):
    """Compute the correlation between var1 and leads/lags of var2.  
    Setting make_plot to True will make two-axis plot under lag yielding 
    highest magnitude
    """

    from py_tools.plot import two_axis

    df = df_in[[var1, var2]].copy() 

    if lags is None:
        assert(max_lags >= 0)
        assert(max_leads >= 0)
        lags = range(-max_lags, max_leads + 1)

    table = np.zeros((len(lags), 2))

    for ii, lag in enumerate(lags):  

        lag_var = 'L({0})_{1}'.format(lag, var2)
        df[lag_var] = df[var2].shift(lag)

        corrs = df[[var1, lag_var]].corr()
        this_corr = corrs[var1][lag_var]

        table[ii, 0] = lag
        table[ii, 1] = this_corr

    best_lag = int(table[np.argmax(np.abs(table[:, 1])), 0])
    best_lag_var2 = 'L({0})_{1}'.format(best_lag, var2)

    if make_plot:
        two_axis(df, var1, best_lag_var2, **kwargs)

    return table 

def rolling_forecast(df_in, lhs, rhs=[], use_const=True, **kwargs):

    df, ix = dt.dropna_ix(df_in[lhs + rhs])
    
    y = df[lhs].values
    
    if use_const:
        X = np.ones((len(df), len(rhs) + 1))
        if rhs:
            X[:, 1:] = df[rhs].values
    else:
        X = df[rhs].values

    return rolling_forecast_internal(y, X, **kwargs), ix

def rolling_forecast_internal(y, X, t_min=None):
    """y should be Nt x Ny, X should be Nt x Nx"""

    if len(y.shape) == 1: y = y[:, np.newaxis]
    if len(X.shape) == 1: X = X[:, np.newaxis]

    Nt, Ny = y.shape
    _, Nx = X.shape

    if t_min is None:
        t_min = Nx + 2
    forecast = np.nan * np.ones((Nt, Ny))
    
    XX_t = np.dot(X[:t_min, :].T, X[:t_min, :])
    Xy_t = np.dot(X[:t_min, :].T, y[:t_min, :])

    for tt in range(t_min, Nt):

        x_t = X[tt, :][:, np.newaxis]
        y_t = y[tt, :][:, np.newaxis]

        # Forecast this period
        bet_t = np.linalg.solve(XX_t, Xy_t)
        forecast[tt, :] = np.dot(bet_t.T, x_t)

        # Testing
        # XX_check = np.dot(X[:tt, :].T, X[:tt, :])
        # Xy_check = np.dot(X[:tt, :].T, y[:tt, :])
        # bet_check = np.linalg.solve(XX_check, Xy_check)

        # Update forecast
        XX_t += np.dot(x_t, x_t.T)
        Xy_t += np.dot(x_t, y_t.T) 

    return forecast

def local_projection(df_in, y_var, shock_var, controls=[], periods=16, shock_lags=2, **kwargs):
    
    df = df_in[[y_var, shock_var] + controls].copy()
    
    rhs = [shock_var]
    
    for jj in range(1, shock_lags+1):
        var_name = 'L{0:d}_{1}'.format(jj, shock_var)
        df[var_name] = df[shock_var].shift(jj)
        rhs.append(var_name)
        
    for var in [y_var] + controls:
        var_name = 'L_' + var
        df[var_name] = df[var].shift()
        rhs.append(var_name)
    
    fr_list = []
    for tt in range(periods+1):
        var_name = 'F{0:d}_{1}'.format(tt, y_var)
        df[var_name] = df[y_var].shift(-tt)
        fr_list.append(dt.regression(df, var_name, rhs, nw_lags=tt, **kwargs))
        
    coeffs = np.array([fr.results.params[1] for fr in fr_list])
    se = np.array([fr.results.HC0_se[1] for fr in fr_list])
    
    return (coeffs, se)

def add_lags_by_group(df, x_var, date_var, lag_list, group_id=None, group_vars=None, freq='Q', 
                lag_str='lag', lead_str='lead', leads=False, zero_type=None):
    """Add lags of a variable by group.
    
    Args:
        df: pandas dataframe, should contain relevant series
        var: string, name of series to lag
        lag_list: list-like, use negative numbers for leads
        grouped: pandas groupby object, for pre-grouping if data is multiindexed
        group_levels: list-like, does grouping if not already grouped for multiindex
    
    Returns:
        df: pandas dataframe, now containing leads and lags.
    """

    assert (group_id is not None) or (group_vars is not None)

    if group_id is None:
        assert 'group_no' not in df
        group_id = 'group_no'
        df[group_id] = df.groupby(group_vars).ngroup()

    tmp = (df[[x_var, date_var, group_id]].set_index([group_id, date_var])\
           .unstack(group_id).resample(freq).asfreq())

    for lag in lag_list:
        
        if leads:
            lag *= -1
        
        if lag == 0:
            
            if zero_type == None:
                lag_var = x_var
            elif zero_type == 'lead':
                lag_var = '{0}{1:d}_{2}'.format(lead_str, -lag, x_var)
            elif zero_type == 'lag':
                lag_var = '{0}{1:d}_{2}'.format(lag_str, lag, x_var)
                
            if lag_var not in df:
                df[lag_var] = df[x_var]
                
        else:
            
            if lag < 0:
                lag_var = '{0}{1:d}_{2}'.format(lead_str, -lag, x_var)
            else:
                lag_var = '{0}{1:d}_{2}'.format(lag_str, lag, x_var)
                
            if lag_var not in df:
                shifted = tmp.shift(lag).stack(group_id)[x_var].rename(lag_var)
                df = df.join(shifted, on=[date_var, group_id])
                
    return df
