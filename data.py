import numpy as np
import pandas as pd

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

