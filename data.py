import numpy as np
import pandas as pd
# import pdb
import pickle
import patsy
import pyhdfe

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa import stattools

from . import stats

def lowercase(df):
    
    return df.rename(columns={var : var.lower() for var in df.columns})

def absorb(df, groups, value_var, weight_var=None, restore_mean=True, tol=1e-8,
           display=False):
    """Remove the mean from a variable by group

    Arguments:
    df -- a dataframe containing the value and group variables.
    groups -- an iterable of variable names or iterables of names
    value_var -- the variable to be demeaned
    weight_var -- the weights to be used for demeaning

    Returns:
    x -- a Pandas Series containing the demeaned variable
    """

    # Check that internal variables are not in the dataframe
    # for var in ['_weight', '_x', '_x_weight']:
        # assert var not in df

    var_list = [value_var]
    if weight_var is not None:
        var_list.append(weight_var)
    for group in groups:
        if isinstance(group, str):
            var_list.append(group)
        else:
            var_list += group

    var_list = [var for var in var_list if var in df]
    _df = df[var_list].copy()

    # Set a default weight variable if none is specified
    if weight_var is None:
        # assert '_weight' not in _df
        _df['_weight'] = 1.0
    else:
        _df['_weight'] = _df[weight_var].copy()

    # Make sure we cut down to overlapping sample of values and weights
    _df['_x'] = _df[value_var].copy()
    ix = np.any(pd.isnull(_df[[value_var, '_weight']]), axis=1) | (_df['_weight'] == 0.0)
    _df.loc[ix, ['_weight', '_x']] = np.nan
    
    # Weighted variable
    _df['_x_weight'] = _df['_x'] * _df['_weight']
    
    # Weighted means error
    def get_err():
        
        err = 0.0
        for ii, group in enumerate(groups):
            group_means = gbfe_list[ii]['_res_weight'].transform(np.nansum) / sum_weight_list[ii]
            err += np.sqrt((group_means @ group_means) / len(group_means))
            
        return err
            
    # Compute weights by cell
    gb = _df.groupby(groups)
    sum_weights = gb['_weight'].transform(np.nansum)
    
    fe_list = ['_fe_weight_' + group for group in groups]
    gbfe_list = [_df.groupby(group) for group in groups]
    sum_weight_list = [gbfe['_weight'].transform(np.nansum) for gbfe in gbfe_list]
    for group in groups:
        _df['_fe_weight_' + group] = 0.0
        
    _df['_res_weight'] = _df['_x_weight'].copy()
    
    err = get_err()
        
    count = 0
    while err > tol:
        
        if False:
            
            for ii, group in enumerate(groups):
                fe_var = fe_list[ii]
                _df['_temp'] = _df['_res_weight'] + _df[fe_var]
                _df[fe_var] = gbfe_list[ii]['_temp'].transform(np.nansum) / sum_weight_list[ii]
                
            _df['_res_weight'] = _df['_x_weight'] - np.nansum(_df[fe_list], axis=1)
            
        else:
            
            for ii, group in enumerate(groups):
                fe_var = fe_list[ii]
                _df['_res_weight'] += _df[fe_var]
                _df[fe_var] = gbfe_list[ii]['_res_weight'].transform(np.nansum) / sum_weight_list[ii]
                _df['_res_weight'] -= _df[fe_var]
                
        err = get_err()
        count += 1
        if display: print("Iteration {0:d}, rmse = {1:g}".format(count, err))

    # # Compute overall mean and remove
    # if restore_mean:

    #     _df['_x_weight'] = _df['_x'] * _df['_weight']
    #     x_mean = np.sum(_df['_x_weight']) / np.sum(_df['_weight'])
    #     _df['_x'] -= x_mean

    # # Loop through groups demeaning each time
    # for group in groups:
        
    #     gb = _df.groupby(group)
    #     _df['_x_weight'] = _df['_x'] * _df['_weight']
        
    #     for var in ['_weight', '_x_weight']:
    #         _df[var + '_sum'] = gb[var].transform(sum)
            
    #     _df['_x'] -= (_df['_x_weight_sum']) / _df['_weight_sum']

    # Restore original mean and output
    fe_means = np.sum(_df[fe_list], axis=1) / _df['_weight']
    x = _df['_x'] - fe_means
    if restore_mean:
        x_mean = np.sum(_df['_x_weight']) / np.sum(_df['_weight'])
        x += x_mean
        
    # _df = _df.drop(columns=['_weight', '_x', '_x_weight'])

    return x

def compute_binscatter(df_in, yvar, xvar, wvar=None, n_bins=10, bins=None, median=False):
    
    if median:
        assert wvar is None

    df = df_in[[xvar, yvar]].copy()
    if wvar is not None:
        df[wvar] = df_in[wvar]
    else:
        wvar='weight'
        df[wvar] = 1.0
        
    df = df.dropna()
        
    if bins is None:
        df['x_bin'] = bin_data(df[xvar], n_bins, weights=df[wvar])
    else:
        df['x_bin'] = pd.cut(df[xvar], bins, labels=np.arange(len(bins)-1))
        
    if median:
        
        by_bin = df.groupby('x_bin')[[xvar, yvar]].median()
    
    else:

        df[xvar] *= df[wvar]
        df[yvar] *= df[wvar]
        
        by_bin = df.groupby('x_bin')[[xvar, yvar, wvar]].sum()
        by_bin[xvar] /= by_bin[wvar]
        by_bin[yvar] /= by_bin[wvar]
        
        weight_adj = n_bins / len(df)
        by_bin[wvar] *= weight_adj

    return by_bin

def bin_data(series, n_bins, weights=None):
    """Group data into bins based on quantile.
    
    series: pandas series
    n_bins: int
    
    returns series with bin indicators"""

    quantiles = np.linspace(0.0, 1.0, n_bins+1)
    if weights is None:
        bins = series.quantile(np.linspace(0.0, 1.0, n_bins+1)).values
    else:
        bins = stats.weighted_quantile(series.values, weights.values, quantiles)
        
    bins = np.unique(bins)
    bins = bins[np.isfinite(bins)]

    bins[0] = -np.inf
    bins[-1] = np.inf

    return pd.cut(series, bins, labels=np.arange(len(bins) - 1))

class FullResults:
    """Regression results with index and samples"""
    def __init__(self, results, ix, Xs, zs):
        self.results = results
        self.ix = ix
        self.Xs = Xs
        self.zs = zs

def winsorize(df_in, var_list, wvar=None, p_val=0.98):
    """Replace values of var_list outside the center p_val quantile mass with
    values at the edge of the mass"""

    # if p_val is not None:
        # assert (p_lo is None) and (p_hi is None)
    tail_prob = 0.5 * (1.0 - p_val)
    p_lo = tail_prob
    p_hi = 1.0 - tail_prob
    # else:
        # assert (p_lo is not None) and (p_hi is not None)

    keep_vars = var_list
    if wvar is not None:
        keep_vars += [wvar]

    df = df_in[keep_vars].copy()
    for var in var_list:

        if wvar is None:
            lb, ub = df_in[var].quantile([p_lo, p_hi]).values
        else:
            lb, ub = stats.weighted_quantile(
                df[var].values, df[wvar].values,
                [p_lo, p_hi],
            )

        df.loc[df[var] < lb, var] = lb
        df.loc[df[var] > ub, var] = ub

    return df 

def add_bin_dummies(df, var_list, n_bins):

    print("NEED TO TEST")
    raise Exception

    cutoffs = np.linspace(0.0, 1.0, n_bins+1)
    dummy_list = []

    for var in var_list:
        for ii in range(n_bins):

            bin_var = var + '_bin{:d}'.format(ii+1)
            dummy_list.append(bin_var)

            df[bin_var] = 0.0

            if ii == 0:
                lb = -np.inf
            else:
                lb = df[var].quantile(cutoffs[ii])

            if ii == n_bins - 1:
                ub = np.inf
            else:
                ub = df[var].quantile(cutoffs[ii+1])

            ix = (df[var] >= lb) & (df[var] < ub)
            df.loc[ix, bin_var] = 1.0

    return df, dummy_list

def demean(df_in, var_list, by_var, weight_var=None):

    keep_list = var_list + [by_var]
    if weight_var is not None:
        keep_list += [weight_var]

    df = df_in[keep_list].copy()

    if weight_var is None:
        df_by = df[keep_list].groupby(by_var).mean().rename(
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


def regression(df, lhs, rhs, fes=[], absorb_vars=[], intercept=True, formula_extra=None, ix=None, 
               trend=None, cluster_var=None, cluster_groups=None, weight_var=None, **kwargs):
    """Run regression from pandas dataframe"""

    if isinstance(rhs, str):
        rhs = [rhs]

    formula = '{0} ~ {1}'.format(lhs, ' + '.join(rhs))
    if fes:
        formula += ' + '.join([''] + ['C({})'.format(fe) for fe in fes])

    var_list = [lhs] + rhs + fes
    if weight_var is not None:
        var_list += [weight_var]
    if cluster_var is not None:
        var_list += [cluster_var]
    for group in absorb_vars:
        if isinstance(group, str):
            var_list.append(group)
        elif isinstance(group, list):
            var_list += group
        else:
            raise Exception
        
    var_list = list(set(var_list))
        
    this_list = []
    for var in var_list:
        if var in df:
            this_list.append(var)
        else:
            assert var in df.index.names

    _df = df[this_list].copy()
    
    ix_samp, _ = match_sample(_df.values, how='inner')
    if ix is None:
#        ix, _ = match_sample(_df.values, how='inner')
        ix = ix_samp.copy()
    
    ix_both = np.logical_and(ix, ix_samp)

    if absorb_vars:
        for var in [lhs] + rhs:
            _df.loc[ix_both, var] = absorb(_df.loc[ix_both, :], absorb_vars, var, weight_var=weight_var, restore_mean=True)
        
    Xs = _df.loc[ix_both, rhs].values
    zs = _df.loc[ix_both, lhs].values

    if trend is not None:
        if trend in ['linear', 'quadratic']:
            _df['t'] = np.arange(len(_df))
            formula += ' + t '
        if trend == 'quadratic':
            _df['t2'] = np.arange(len(_df)) ** 2
            formula += ' + t2 '

    if formula_extra is not None:
        formula += ' + ' + formula_extra

    if not intercept:
        formula += ' -1'
    else:
        Xs = np.hstack((np.ones((Xs.shape[0], 1)), Xs))

    if cluster_var is not None:
        cluster_groups = get_cluster_groups(_df, cluster_var)

    if cluster_groups is not None:
        these_groups = cluster_groups[ix_both]
    else:
        these_groups = None

    if weight_var is None:
        fr = formula_regression(_df, formula, ix=ix, cluster_groups=these_groups, **kwargs)
    else:
        fr = wls_formula(_df, formula, weight_var=weight_var, ix=ix, cluster_groups=these_groups, **kwargs)

    return FullResults(fr.results, ix=ix, Xs=Xs, zs=zs)

def wls_formula(df, formula, weight_var=None, weights=None, ix=None, nw_lags=0,
                cluster_groups=None, display=False):

    if ix is None:
        ix = np.ones(len(df), dtype=bool)
        
    if weight_var is not None:
        assert weights is None
        weights = df.loc[ix, weight_var]

    weights /= np.sum(weights)
    
    y, X = patsy.dmatrices(formula, df.loc[ix, :], return_type='dataframe')
    results = sm.WLS(y, X, weights=weights).fit()
    results = results.get_robustcov_results('HC0')

    results = update_results_cov(results, nw_lags=nw_lags, cluster_groups=cluster_groups)

    if display:
        print(results.summary())

    return FullResults(results, ix=ix, Xs=None, zs=None)

def compute_histogram(series, name='bin', **kwargs):
   
    this_hist, _ = np.histogram(series, **kwargs)
    return pd.Series(this_hist, index=[name + str(ii) for ii in range(len(this_hist))])

def update_results_cov(results, nw_lags=0, cluster_groups=None):

    if cluster_groups is not None:
        assert(nw_lags == 0)
        results = results.get_robustcov_results('cluster', groups=cluster_groups)
    elif nw_lags > 0:
        results = results.get_robustcov_results('HAC', maxlags=nw_lags)
    else:
        results = results.get_robustcov_results('HC0')

    return results

def get_cluster_groups(df, cluster_var):

    index_dict = {val : ii for ii, val in enumerate(df[cluster_var].unique())}
    return df[cluster_var].map(index_dict).values

def formula_regression(df, formula, ix=None, nw_lags=0, cluster_groups=None, display=False):

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

    results = update_results_cov(results, nw_lags=nw_lags, cluster_groups=cluster_groups)

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

def demean_separate(df, var_list, group_list, **kwargs):
    """This function demeans by each variable one at at a time"""
    
    for group in group_list:
        df = demean2(group, var_list, df, **kwargs)
        
    return df

def demean2(group_list, var_list, df, prefix=None):
    """Set prefix to None to overwrite existing variables with demeaned
    versions, otherwise demeaned versions will have specified prefix"""

    if prefix is None:
        full_prefix = ''
    else:
        full_prefix = prefix + '_'

    # Precautionarily drop means
    for var in var_list:
        if 'MEAN_' + var in df:
            df = df.drop(['MEAN_' + var], axis=1)

    df_mean = df.groupby(group_list)[var_list].mean()
    df_mean = df_mean.rename(columns={var : 'MEAN_' + var for var in var_list})
    df = pd.merge(df, df_mean, left_on=group_list, right_index=True)

    for var in var_list:
        df[full_prefix + var] = df[var] - df['MEAN_' + var]

    df = df.drop(['MEAN_' + var for var in var_list], axis=1)
        
    return df

def weight_regression_params(weights, params=None, cov=None, results=None):
    """Produce point estimate and standard error for weighted coefficient.
    
    Args:
        weights: vector of weights, x = w'c where c are coefficients
        params: vector of coefficients (if inputting directly)
        cov: covariance matrix of coefficients (if inputting directly)
        results: statsmodels results object (alternative input)

    Returns:
        x: summed coefficient
        se: standard error of summed coefficient
    
    """

    assert ((params is not None) and (cov is not None)) or (results is not None)

    if results is not None:
        params = results.params
        cov = results.cov_HC0

    x = np.dot(params, weights)
    se = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    return x, se

def sum_regression_params(positions, *args, **kwargs):
    """Produce point estimate and standard error for summed coefficient.
    
    Args:
        positions: indices of coefficients to sum
        *args, **kwargs: to be passed to weight_regression_params

    Returns:
        x: summed coefficient
        se: standard error of summed coefficient
    
    """

    e_vec = np.zeros(len(params))
    e_vec[positions] = 1.0

    return weight_regression_params(e_vec, *args, **kwargs)

def collapse(df, method='mean', var_list=None, by=[], wvar=None):
    
    assert by
    assert method in ['mean', 'sum']
    assert wvar is not None # just do groupby otherwise
    
    if var_list is None:
        var_list = [var for var in df.columns 
                    if (var not in (by + [wvar])) and pd.api.types.is_numeric_dtype(df[var])]
        
    assert wvar not in var_list
        
    keep_vars = list(set(var_list + by + [wvar]))
    _df = df[keep_vars].copy()

    for var in var_list:
        if (var not in by) and (var != wvar):
            _df[var] *= _df[wvar]
            
    collapsed = _df.groupby(by)[var_list + [wvar]].sum()
    if method == 'mean':
        for var in var_list:
            collapsed[var] /= collapsed[wvar]
        
    return collapsed
    
def safe_sum(x):
    
    return x.sum(skipna=False)

def chow_lin_V_default(a, N):
    """Default V matrix: AR(1) correlation structure"""
    
    V = np.zeros((N, N))
    for tt in range(N):
        V[tt, :] = a ** np.abs(np.arange(N) - tt)
        
    return V

def chow_lin(Y, Z, B, Vfcn=chow_lin_V_default, a0=0.9, tol=1e-4):
    """Use the Chow-Lin method to approximate the target series X using a
    coarser series Y and a proxy series Z. The inputs should be:
        
        Y: the Nt_coarse x 1 target series with limited availability
        Z: the Nt_fine x k target series
        B: the Nx_fine x Nt_coarse matrix relating Y and X
        Vfcn: a function for computing the error matrix V given the correlation parameter a
        a0: a scalar guess for the correlation parameter
        tol: the scalar tolerance for the iterative process to converge
        
    The function returns:
        
        X_hat: the approximated series for X
    """
    
    a = a0
    done = False
    
    N = Z.shape[0]
    
    while not done:
        
        V = Vfcn(a, N)
        
        ZB = Z.T @ B
        BV = B.T @ V
        BVB = BV @ B
        
        ZB_BVB_inv = np.linalg.solve(BVB.T, ZB.T).T
        
        bet_hat = np.linalg.solve(ZB_BVB_inv @ ZB.T,
                                  ZB_BVB_inv @ Y)
        
        X_hat = Z @ bet_hat + BV.T @ np.linalg.solve(
            BVB, Y - B.T @ (Z @ bet_hat)
            )
        
        u_hat = X_hat - Z @ bet_hat
        a_new = stattools.acf(u_hat, nlags=1, fft=False)[1]
        
        a_err = np.abs(a_new - a)
        if a_err < tol:
            done = True
            
        a = a_new
            
    return X_hat
        