import numpy as np
import pandas as pd
import pickle

import statsmodels.api as sm
import statsmodels.formula.api as smf

from py_tools import stats


def pivot_no_hierarchical_columns(df, *args, **kwargs):
    """Pivot a DataFrame and flatten the resulting MultiIndex columns.

    Wraps :func:`pandas.pivot` and immediately drops the top level of the
    resulting hierarchical column index, returning a DataFrame with a plain
    single-level column index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data to pivot.
    *args
        Positional arguments forwarded to :func:`pandas.pivot`.
    **kwargs
        Keyword arguments forwarded to :func:`pandas.pivot`.

    Returns
    -------
    pandas.DataFrame
        Pivoted DataFrame with single-level columns taken from the second
        level of the pivot's MultiIndex.
    """
    df = pd.pivot(df, *args, **kwargs)
    df.columns = df.columns.get_level_values(1)
    return df


def lowercase(df):
    """Rename all DataFrame columns to lowercase.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all column names converted to lowercase.
    """
    return df.rename(columns={var: var.lower() for var in df.columns})


def absorb(
    df, groups, value_var, weight_var=None, restore_mean=True, tol=1e-12, display=False
):
    """Remove the mean from a variable by group (alternating-projections FWL).

    Implements a Frisch-Waugh-Lovell style within-group demeaning via the
    alternating-projections (Gauss-Seidel) algorithm. Supports multiple
    group dimensions and optional observation weights.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the value and group variables.
    groups : list
        List of group variable names (str) or lists of names for
        multi-dimensional group effects. Each element defines one set of
        fixed effects.
    value_var : str
        Name of the variable to be demeaned.
    weight_var : str or None, optional
        Name of the weight variable. If ``None``, equal weights are used.
    restore_mean : bool, optional
        If ``True`` (default), add back the (weighted) grand mean so that
        the output has the same mean as the input.
    tol : float, optional
        Convergence tolerance on the root-mean-square of group means of the
        residuals. Defaults to ``1e-12``.
    display : bool, optional
        If ``True``, print the RMSE at each iteration. Defaults to ``False``.

    Returns
    -------
    pandas.Series
        Series containing the demeaned variable, aligned with *df*.
    """

    Ng = len(groups)

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
        _df["_weight"] = 1.0
    else:
        _df["_weight"] = _df[weight_var].copy()

    # Make sure we cut down to overlapping sample of values and weights
    _df["_x"] = _df[value_var].copy()
    ix = np.any(pd.isnull(_df[[value_var, "_weight"]]), axis=1) | (
        _df["_weight"] == 0.0
    )
    _df.loc[ix, ["_weight", "_x"]] = np.nan

    # Weighted variable
    _df["_x_weight"] = _df["_x"] * _df["_weight"]

    # Weighted means error
    def get_err():

        err = 0.0
        for ii, group in enumerate(groups):
            group_means = (
                gbfe_list[ii]["_res_weight"].transform("sum") / sum_weight_list[ii]
            ).fillna(0.0)
            err += np.sqrt((group_means @ group_means) / len(group_means))

        return err

    # Prep across groups
    fe_list = ["_fe" + str(ii) for ii in range(Ng)]
    gbfe_list = [_df.groupby(group) for group in groups]
    sum_weight_list = [gbfe["_weight"].transform("sum") for gbfe in gbfe_list]
    for fe_var in fe_list:
        _df[fe_var] = 0.0
        _df[fe_var + "_weight"] = 0.0

    _df["_res_weight"] = _df["_x_weight"].copy()

    err = get_err()

    count = 0
    while err > tol:
        for ii, group in enumerate(groups):
            fe_var = fe_list[ii]
            _df["_res_weight"] += _df[fe_var + "_weight"]
            _df[fe_var] = (
                gbfe_list[ii]["_res_weight"].transform("sum") / sum_weight_list[ii]
            )
            _df[fe_var + "_weight"] = _df[fe_var] * _df["_weight"]
            _df["_res_weight"] -= _df[fe_var + "_weight"]

        err = get_err()
        count += 1
        if display:
            print("Iteration {0:d}, rmse = {1:g}".format(count, err))

    # Restore original mean and output
    fe_means = np.sum(_df[fe_list], axis=1)
    x = _df["_x"] - fe_means
    if restore_mean:
        x_mean = np.sum(_df["_x_weight"]) / np.sum(_df["_weight"])
        x += x_mean

    return x


def compute_binscatter(
    df_in,
    yvar,
    xvar,
    wvar=None,
    n_bins=10,
    bins=None,
    median=False,
    control=None,
    absorb=None,
):
    """Compute binscatter statistics for a pair of variables.

    Partitions *xvar* into quantile bins and computes the (weighted) mean or
    median of both *xvar* and *yvar* within each bin. Optionally residualises
    both variables against controls and/or absorbed fixed effects before
    binning.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input data.
    yvar : str
        Outcome variable name.
    xvar : str
        Running (x-axis) variable name.
    wvar : str or None, optional
        Weight variable name. Cannot be combined with ``median=True``.
    n_bins : int, optional
        Number of quantile bins. Defaults to ``10``. Ignored when *bins* is
        provided.
    bins : array-like or None, optional
        Explicit bin cut-points passed to :func:`pandas.cut`. If ``None``
        (default), quantile bins are constructed via :func:`bin_data`.
    median : bool, optional
        If ``True``, compute the unweighted median within each bin instead of
        the weighted mean. Defaults to ``False``.
    control : list of str or None, optional
        Additional RHS variables to partial out before binning. Currently not
        implemented; passing a non-empty value raises ``NotImplementedError``.
    absorb : list or None, optional
        Group variables to absorb (fixed effects) before binning. Currently not
        implemented; passing a non-empty value raises ``NotImplementedError``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by bin with columns *xvar* and *yvar* containing
        within-bin averages (or medians), and *wvar* (normalized weights) when
        not using the median.
    """
    if control is None:
        control = []
    if absorb is None:
        absorb = []
    if control or absorb:
        raise NotImplementedError(
            "compute_binscatter control/absorb residualization is not implemented."
        )

    if median:
        assert wvar is None

    df = df_in[[xvar, yvar]].copy()
    if wvar is not None:
        df[wvar] = df_in[wvar]
    else:
        wvar = "weight"
        df[wvar] = 1.0

    df = df.dropna()

    if bins is None:
        df["x_bin"] = bin_data(df[xvar], n_bins, weights=df[wvar])
    else:
        df["x_bin"] = pd.cut(df[xvar], bins, labels=np.arange(len(bins) - 1))

    if median:
        by_bin = df.groupby("x_bin", observed=False)[[xvar, yvar]].median()

    else:
        df[xvar] *= df[wvar]
        df[yvar] *= df[wvar]

        by_bin = df.groupby("x_bin", observed=False)[[xvar, yvar, wvar]].sum()
        by_bin[xvar] /= by_bin[wvar]
        by_bin[yvar] /= by_bin[wvar]

        weight_adj = n_bins / len(df)
        by_bin[wvar] *= weight_adj

    return by_bin


def bin_data(series, n_bins, weights=None):
    """Group data into bins based on quantile boundaries.

    Parameters
    ----------
    series : pandas.Series
        Data to bin.
    n_bins : int
        Number of bins to create.
    weights : array-like or None, optional
        Observation weights used to compute weighted quantile boundaries.
        If ``None``, unweighted quantiles are used.

    Returns
    -------
    pandas.Categorical
        Series of integer bin labels (0-indexed) aligned with *series*.
    """

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    if weights is None:
        bins = series.quantile(np.linspace(0.0, 1.0, n_bins + 1)).values
    else:
        bins = stats.weighted_quantile(series.values, weights.values, quantiles)

    bins = np.unique(bins)
    bins = bins[np.isfinite(bins)]

    bins[0] = -np.inf
    bins[-1] = np.inf

    return pd.cut(series, bins, labels=np.arange(len(bins) - 1))


class FullResults:
    """Regression results bundled with the sample index and design matrices.

    Parameters
    ----------
    results : statsmodels results object
        Fitted model results.
    ix : array-like of bool
        Boolean index indicating which rows of the original DataFrame were
        included in the regression sample.
    Xs : numpy.ndarray or None
        Design matrix used in the regression (rows correspond to *ix*).
    zs : numpy.ndarray or None
        Dependent variable matrix used in the regression.

    Attributes
    ----------
    results : statsmodels results object
        Fitted model results.
    ix : array-like of bool
        Sample inclusion index.
    Xs : numpy.ndarray or None
        Design matrix.
    zs : numpy.ndarray or None
        Dependent variable array/matrix.
    """

    def __init__(self, results, ix, Xs, zs):
        """Initialise a FullResults container.

        Parameters
        ----------
        results : statsmodels results object
            Fitted model results.
        ix : array-like of bool
            Boolean sample-inclusion index.
        Xs : numpy.ndarray or None
            Design matrix aligned to *ix*.
        zs : numpy.ndarray or None
            Dependent variable array aligned to *ix*.
        """
        self.results = results
        self.ix = ix
        self.Xs = Xs
        self.zs = zs


def winsorize(df_in, var_list, wvar=None, p_val=0.98):
    """Winsorize variables to the central *p_val* quantile mass.

    Replaces values below the lower tail and above the upper tail with the
    corresponding tail boundary values. Tail boundaries are computed as the
    ``(1 - p_val) / 2`` and ``1 - (1 - p_val) / 2`` quantiles.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input data.
    var_list : list of str
        Column names to winsorize.
    wvar : str or None, optional
        Weight variable for computing weighted quantile boundaries.
        If ``None``, unweighted quantiles are used.
    p_val : float, optional
        Central quantile mass to retain. For example, ``0.98`` (default)
        trims the bottom and top 1 % of values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with only the columns in *var_list* (and *wvar* if provided)
        where extreme values have been replaced by the tail boundaries.
    """

    tail_prob = 0.5 * (1.0 - p_val)
    p_lo = tail_prob
    p_hi = 1.0 - tail_prob

    keep_vars = var_list
    if wvar is not None:
        keep_vars += [wvar]

    df = df_in[keep_vars].copy()
    for var in var_list:
        if wvar is None:
            lb, ub = df_in[var].quantile([p_lo, p_hi]).values
        else:
            lb, ub = stats.weighted_quantile(
                df[var].values,
                df[wvar].values,
                [p_lo, p_hi],
            )

        df.loc[df[var] < lb, var] = lb
        df.loc[df[var] > ub, var] = ub

    return df


def add_bin_dummies(df, var_list, n_bins):
    """Add quantile bin dummy variables to a DataFrame.

    For each variable in *var_list*, creates *n_bins* binary indicator columns
    named ``'<var>_bin1'``, ``'<var>_bin2'``, …, ``'<var>_bin<n_bins>'``.
    Bin boundaries are determined by equal-probability quantile cut-points.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to augment (modified in place).
    var_list : list of str
        Variable names for which to create bin dummies.
    n_bins : int
        Number of quantile bins (and hence dummy columns) per variable.

    Returns
    -------
    df : pandas.DataFrame
        The input DataFrame with the new dummy columns appended.
    dummy_list : list of str
        Names of the newly created dummy columns.
    """
    cutoffs = np.linspace(0.0, 1.0, n_bins + 1)
    dummy_list = []

    for var in var_list:
        for ii in range(n_bins):
            bin_var = var + "_bin{:d}".format(ii + 1)
            dummy_list.append(bin_var)

            df[bin_var] = 0.0

            if ii == 0:
                lb = -np.inf
            else:
                lb = df[var].quantile(cutoffs[ii])

            if ii == n_bins - 1:
                ub = np.inf
            else:
                ub = df[var].quantile(cutoffs[ii + 1])

            ix = (df[var] >= lb) & (df[var] < ub)
            df.loc[ix, bin_var] = 1.0

    return df, dummy_list


def demean(df_in, var_list, by_var, weight_var=None):
    """Subtract within-group means from variables, returning means and residuals.

    Computes (optionally weighted) group means and subtracts them, returning
    both the mean and demeaned series for each variable.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input data.
    var_list : list of str
        Variables to demean.
    by_var : str
        Group variable used to compute means.
    weight_var : str or None, optional
        Weight variable. If ``None``, simple (unweighted) means are used.

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame with columns ``'<var>_mean'`` and ``'<var>_demeaned'`` for
        each variable in *var_list*.
    col_names : list of str
        List of column names in *result_df* (means first, then demeaned).
    """
    keep_list = var_list + [by_var]
    if weight_var is not None:
        keep_list += [weight_var]

    df = df_in[keep_list].copy()

    if weight_var is None:
        df_by = (
            df[keep_list]
            .groupby(by_var)
            .mean()
            .rename(columns={var: var + "_mean" for var in var_list})
        )
    else:
        for var in var_list:
            df[var + "_wtd"] = df[var] * df[weight_var]

        df_by = df.groupby(by_var).sum()
        for var in var_list:
            df_by[var + "_mean"] = df_by[var + "_wtd"] / df_by[weight_var]

    mean_list = [var + "_mean" for var in var_list]
    df = pd.merge(df, df_by[mean_list], left_on=by_var, right_index=True)
    for var in var_list:
        df[var + "_demeaned"] = df[var] - df[var + "_mean"]

    demean_list = [var + "_demeaned" for var in var_list]
    return (df[mean_list + demean_list], mean_list + demean_list)


def match_sample(X, how="inner", ix=None):
    """Select a sample from a matrix by handling missing values.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, p)
        Input matrix potentially containing ``NaN`` values.
    how : {'inner', 'outer', 'custom'}, optional
        Sample selection rule:

        - ``'inner'`` (default): keep rows where *all* columns are non-null.
        - ``'outer'``: keep rows where *any* column is non-null.
        - ``'custom'``: use the boolean array supplied via *ix*.
    ix : array-like of bool or None, optional
        Custom inclusion index. Required when ``how='custom'``.

    Returns
    -------
    ix : numpy.ndarray of bool, shape (n,)
        Boolean mask indicating which rows were selected.
    Xs : numpy.ndarray
        Sub-matrix of *X* for the selected rows, with remaining ``NaN``
        values replaced by ``0.0``.

    Raises
    ------
    Exception
        If *how* is not one of ``'inner'``, ``'outer'``, or ``'custom'``.
    """
    if how == "inner":
        ix = np.all(pd.notnull(X), axis=1)
    elif how == "outer":
        ix = np.any(pd.notnull(X), axis=1)
    elif how != "custom":
        raise Exception

    Xs = X[ix, :]
    Xs[pd.isnull(Xs)] = 0.0

    return (ix, Xs)


def match_xy(X, z, how="inner", ix=None):
    """Select a common sample for design matrix *X* and outcome *z*.

    Stacks *X* and *z* horizontally, calls :func:`match_sample`, then
    splits the result back into design matrix and outcome.

    Parameters
    ----------
    X : numpy.ndarray of shape (n,) or (n, p)
        Regressor matrix.
    z : numpy.ndarray of shape (n,) or (n, q)
        Outcome matrix or vector.
    how : {'inner', 'outer', 'custom'}, optional
        Passed to :func:`match_sample`. Defaults to ``'inner'``.
    ix : array-like of bool or None, optional
        Custom inclusion index. Required when ``how='custom'``.

    Returns
    -------
    ix : numpy.ndarray of bool, shape (n,)
        Boolean sample inclusion mask.
    Xs : numpy.ndarray
        Selected rows of *X* with NaNs zeroed.
    zs : numpy.ndarray
        Selected rows of *z* with NaNs zeroed.
    """
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


def regression(
    df,
    lhs,
    rhs,
    fes=None,
    absorb_vars=None,
    intercept=True,
    formula_extra=None,
    ix=None,
    trend=None,
    cluster_var=None,
    cluster_groups=None,
    weight_var=None,
    **kwargs,
):
    """Run an OLS or WLS regression from a pandas DataFrame.

    Supports categorical fixed effects, absorbed fixed effects (via
    :func:`absorb`), time trends, clustered or Newey-West standard errors,
    and observation weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    lhs : str
        Dependent variable name.
    rhs : str or list of str
        Independent variable name(s).
    fes : list of str or None, optional
        Categorical fixed-effect variable names added as ``C(<var>)`` in the
        formula. Defaults to ``[]``.
    absorb_vars : list or None, optional
        Group variables to absorb via :func:`absorb` before fitting. Supports
        the same syntax as :func:`absorb`'s *groups* argument.
        Defaults to ``[]``.
    intercept : bool, optional
        If ``True`` (default), include a constant term.
    formula_extra : str or None, optional
        Additional formula string appended to the base formula.
    ix : array-like of bool or None, optional
        Pre-computed sample-inclusion boolean index. If ``None``, the sample
        is determined by complete cases.
    trend : {None, 'linear', 'quadratic'}, optional
        Add a linear or quadratic time trend. Defaults to ``None``.
    cluster_var : str or None, optional
        Variable whose values define clustering groups for cluster-robust SE.
    cluster_groups : array-like or None, optional
        Pre-computed cluster group labels (alternative to *cluster_var*).
    weight_var : str or None, optional
        Weight variable for WLS estimation.
    **kwargs
        Additional keyword arguments forwarded to :func:`formula_regression`
        or :func:`wls_formula`.

    Returns
    -------
    FullResults
        Object containing the statsmodels results, the sample index, the
        design matrix, and the outcome vector.
    """

    if fes is None:
        fes = []
    if absorb_vars is None:
        absorb_vars = []

    if isinstance(rhs, str):
        rhs = [rhs]

    formula = "{0} ~ {1}".format(lhs, " + ".join(rhs))
    if fes:
        formula += " + ".join([""] + ["C({})".format(fe) for fe in fes])

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

    ix_samp, _ = match_sample(_df.values, how="inner")
    if ix is None:
        ix = ix_samp.copy()

    ix_both = np.logical_and(ix, ix_samp)

    if absorb_vars:
        for var in [lhs] + rhs:
            _df.loc[ix_both, var] = absorb(
                _df.loc[ix_both, :],
                absorb_vars,
                var,
                weight_var=weight_var,
                restore_mean=True,
            )

    Xs = _df.loc[ix_both, rhs].values
    zs = _df.loc[ix_both, lhs].values

    if trend is not None:
        if trend in ["linear", "quadratic"]:
            _df["t"] = np.arange(len(_df))
            formula += " + t "
        if trend == "quadratic":
            _df["t2"] = np.arange(len(_df)) ** 2
            formula += " + t2 "

    if formula_extra is not None:
        formula += " + " + formula_extra

    if not intercept:
        formula += " -1"
    else:
        Xs = np.hstack((np.ones((Xs.shape[0], 1)), Xs))

    if cluster_var is not None:
        cluster_groups = get_cluster_groups(_df, cluster_var)

    if cluster_groups is not None:
        these_groups = cluster_groups[ix_both]
    else:
        these_groups = None

    if weight_var is None:
        fr = formula_regression(
            _df, formula, ix=ix, cluster_groups=these_groups, **kwargs
        )
    else:
        fr = wls_formula(
            _df,
            formula,
            weight_var=weight_var,
            ix=ix,
            cluster_groups=these_groups,
            **kwargs,
        )

    return FullResults(fr.results, ix=ix, Xs=Xs, zs=zs)


def wls_formula(
    df,
    formula,
    weight_var=None,
    weights=None,
    ix=None,
    nw_lags=0,
    cluster_groups=None,
    display=False,
):
    """Fit a Weighted Least Squares model from a patsy formula string.

    Normalises weights to sum to 1, fits a WLS model via statsmodels, and
    applies robust covariance adjustments.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    formula : str
        Patsy formula string (e.g. ``'y ~ x1 + x2'``).
    weight_var : str or None, optional
        Column name of the weight variable. Mutually exclusive with *weights*.
    weights : array-like or None, optional
        Pre-computed weight array. Mutually exclusive with *weight_var*.
    ix : array-like of bool or None, optional
        Sample-inclusion boolean index. Defaults to all rows.
    nw_lags : int, optional
        Number of lags for Newey-West HAC standard errors. Defaults to ``0``
        (HC3 errors).
    cluster_groups : array-like or None, optional
        Cluster group labels for cluster-robust standard errors.
    display : bool, optional
        If ``True``, print the regression summary. Defaults to ``False``.

    Returns
    -------
    FullResults
        Object containing statsmodels results, the sample index, and ``None``
        for design matrix and outcome (formula-based fit).
    """
    if ix is None:
        ix = np.ones(len(df), dtype=bool)

    try:
        import patsy
    except ImportError as e:
        raise ImportError(
            "patsy is required for wls_formula. "
            "Install it with: pip install py_tools[ml]"
        ) from e

    if weight_var is not None:
        assert weights is None
        weights = df.loc[ix, weight_var].copy()

    weights /= np.sum(weights)

    y, X = patsy.dmatrices(formula, df.loc[ix, :], return_type="dataframe")
    results = sm.WLS(y, X, weights=weights).fit()
    results = results.get_robustcov_results("HC3")

    results = update_results_cov(
        results, nw_lags=nw_lags, cluster_groups=cluster_groups
    )

    if display:
        print(results.summary())

    return FullResults(results, ix=ix, Xs=None, zs=None)


def compute_histogram(series, name="bin", **kwargs):
    """Compute a histogram and return it as a named pandas Series.

    Parameters
    ----------
    series : array-like
        Data to histogram.
    name : str, optional
        Prefix for the bin labels in the output index. Defaults to
        ``'bin'``, producing labels ``'bin0'``, ``'bin1'``, etc.
    **kwargs
        Additional keyword arguments forwarded to :func:`numpy.histogram`
        (e.g. ``bins``, ``range``).

    Returns
    -------
    pandas.Series
        Histogram counts with string bin labels as the index.
    """
    this_hist, _ = np.histogram(series, **kwargs)
    return pd.Series(this_hist, index=[name + str(ii) for ii in range(len(this_hist))])


def update_results_cov(results, nw_lags=0, cluster_groups=None):
    """Apply robust covariance correction to a statsmodels results object.

    Selects between cluster-robust, Newey-West HAC, and HC3 standard errors
    depending on the provided arguments.

    Parameters
    ----------
    results : statsmodels results object
        Fitted model results to update.
    nw_lags : int, optional
        Number of lags for Newey-West HAC correction. Used only when
        *cluster_groups* is ``None`` and ``nw_lags > 0``. Defaults to ``0``.
    cluster_groups : array-like or None, optional
        Group labels for cluster-robust standard errors. When provided,
        ``nw_lags`` must be ``0``.

    Returns
    -------
    statsmodels results object
        Results object with updated covariance matrix.

    Raises
    ------
    AssertionError
        If both *cluster_groups* and *nw_lags* > 0 are supplied.
    """
    if cluster_groups is not None:
        assert nw_lags == 0
        results = results.get_robustcov_results("cluster", groups=cluster_groups)
    elif nw_lags > 0:
        results = results.get_robustcov_results("HAC", maxlags=nw_lags)
    else:
        results = results.get_robustcov_results("HC3")

    return results


def get_cluster_groups(df, cluster_var):
    """Map cluster variable values to contiguous integer group labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing *cluster_var*.
    cluster_var : str
        Column name whose unique values define the clusters.

    Returns
    -------
    numpy.ndarray of int
        Array of integer group labels aligned with *df*'s row order.
    """
    index_dict = {val: ii for ii, val in enumerate(df[cluster_var].unique())}
    return df[cluster_var].map(index_dict).values


def formula_regression(
    df, formula, ix=None, nw_lags=0, cluster_groups=None, display=False
):
    """Fit an OLS model from a patsy formula string with robust standard errors.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    formula : str
        Patsy formula string (e.g. ``'y ~ x1 + x2'``).
    ix : array-like of bool or None, optional
        Sample-inclusion boolean index. If ``None``, all rows are used.
    nw_lags : int, optional
        Number of lags for Newey-West HAC standard errors. Defaults to ``0``.
    cluster_groups : array-like or None, optional
        Group labels for cluster-robust standard errors.
    display : bool, optional
        If ``True``, print the regression summary. Defaults to ``False``.

    Returns
    -------
    FullResults
        Object containing statsmodels results, the sample index, and ``None``
        for design matrix and outcome (formula-based fit).
    """
    if ix is None:
        model = smf.ols(formula=formula, data=df)
    else:
        model = smf.ols(formula=formula, data=df.loc[ix, :])

    results = model.fit()

    results = update_results_cov(
        results, nw_lags=nw_lags, cluster_groups=cluster_groups
    )

    if display:
        print(results.summary())

    return FullResults(results, ix, Xs=None, zs=None)


def sm_regression(df, lhs, rhs, match="inner", ix=None, nw_lags=0, display=False):
    """Fit an OLS regression directly from numpy arrays via statsmodels.

    Extracts regressor and outcome arrays from *df*, handles missing values
    via :func:`match_xy`, and applies robust covariance correction.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    lhs : str
        Dependent variable column name.
    rhs : list of str
        Independent variable column names. Include ``'const'`` to add a
        constant column automatically.
    match : {'inner', 'outer', 'custom'}, optional
        Sample selection rule forwarded to :func:`match_xy`. Defaults to
        ``'inner'``.
    ix : array-like of bool or None, optional
        Custom sample-inclusion index (used when ``match='custom'``).
    nw_lags : int, optional
        Number of lags for Newey-West HAC standard errors. Defaults to ``0``
        (HC0 errors).
    display : bool, optional
        If ``True``, print the regression summary. Defaults to ``False``.

    Returns
    -------
    FullResults
        Object containing statsmodels results, the sample index, the design
        matrix, and the outcome vector.
    """

    if "const" in rhs and "const" not in df:
        df["const"] = 1.0

    X = df.loc[:, rhs].values
    z = df.loc[:, lhs].values

    ix, Xs, zs = match_xy(X, z, how=match, ix=ix)

    model = sm.OLS(zs, Xs)
    results = model.fit()
    if nw_lags > 0:
        results = results.get_robustcov_results("HAC", maxlags=nw_lags)
    else:
        results = results.get_robustcov_results("HC0")

    if display:
        print(results.summary())

    return FullResults(results, ix, Xs, zs)


def clean(df_in, var_list):
    """Remove rows with infinite or NaN values from a DataFrame subset.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input data.
    var_list : list of str
        Variables to include; columns not present in *df_in* are silently
        skipped. Infinite values are treated as NaN before dropping.

    Returns
    -------
    pandas.DataFrame
        Subset of *df_in* restricted to *var_list* (minus missing columns)
        with all rows containing NaN or ±Inf removed.
    """

    good_list = [var for var in var_list if var in df_in]
    df = df_in[good_list].copy().replace([np.inf, -np.inf], np.nan)
    return df.dropna()


def dropna_ix(df):
    """Drop rows with any NaN value and return both the cleaned DataFrame and a mask.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.

    Returns
    -------
    df_clean : pandas.DataFrame
        Subset of *df* with all NaN-containing rows removed.
    ix : numpy.ndarray of bool
        Boolean mask of shape ``(len(df),)`` indicating retained rows.
    """
    ix = np.all(pd.notnull(df).values, axis=1)
    return df.loc[ix, :], ix


class MVOLSResults:
    """Results container for multivariate OLS regressions.

    Stores fitted parameters, residuals, covariance matrices, and information
    criteria for a multivariate OLS model (multiple dependent variables).

    Parameters
    ----------
    nobs : int
        Number of observations in the original (unfiltered) data.
    params : numpy.ndarray of shape (k, nz)
        Estimated coefficient matrix.
    fittedvalues : numpy.ndarray of shape (T, nz)
        In-sample fitted values.
    resid : numpy.ndarray of shape (T, nz)
        Residuals.
    cov_e : numpy.ndarray of shape (nz, nz)
        Homoskedastic residual covariance matrix.
    cov_HC0 : numpy.ndarray of shape (k*nz, k*nz)
        HC0 (homoskedastic) parameter covariance matrix.
    HC0_se : numpy.ndarray of shape (k, nz)
        HC0 standard errors for each parameter.
    HC0_tstat : numpy.ndarray of shape (k, nz)
        HC0 t-statistics for each parameter.
    cov_HC1 : numpy.ndarray of shape (k*nz, k*nz)
        HC1 (heteroskedasticity-robust) parameter covariance matrix.
    HC1_se : numpy.ndarray of shape (k, nz)
        HC1 standard errors for each parameter.
    HC1_tstat : numpy.ndarray of shape (k, nz)
        HC1 t-statistics for each parameter.
    llf : float
        Log-likelihood of the fitted model.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    hqc : float
        Hannan-Quinn information criterion.

    Attributes
    ----------
    nobs, params, fittedvalues, resid, cov_e : see Parameters
    cov_HC0, HC0_se, HC0_tstat : see Parameters
    cov_HC1, HC1_se, HC1_tstat : see Parameters
    llf, aic, bic, hqc : see Parameters
    """

    def __init__(
        self,
        nobs,
        params,
        fittedvalues,
        resid,
        cov_e,
        cov_HC0,
        HC0_se,
        HC0_tstat,
        cov_HC1,
        HC1_se,
        HC1_tstat,
        llf,
        aic,
        bic,
        hqc,
    ):
        """Initialise a MVOLSResults container.

        Parameters
        ----------
        nobs : int
            Number of observations.
        params : numpy.ndarray
            Coefficient matrix of shape (k, nz).
        fittedvalues : numpy.ndarray
            Fitted values of shape (T, nz).
        resid : numpy.ndarray
            Residuals of shape (T, nz).
        cov_e : numpy.ndarray
            Residual covariance of shape (nz, nz).
        cov_HC0 : numpy.ndarray
            HC0 covariance matrix of shape (k*nz, k*nz).
        HC0_se : numpy.ndarray
            HC0 standard errors of shape (k, nz).
        HC0_tstat : numpy.ndarray
            HC0 t-statistics of shape (k, nz).
        cov_HC1 : numpy.ndarray
            HC1 covariance matrix of shape (k*nz, k*nz).
        HC1_se : numpy.ndarray
            HC1 standard errors of shape (k, nz).
        HC1_tstat : numpy.ndarray
            HC1 t-statistics of shape (k, nz).
        llf : float
            Log-likelihood.
        aic : float
            Akaike information criterion.
        bic : float
            Bayesian information criterion.
        hqc : float
            Hannan-Quinn information criterion.
        """

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


def mv_ols(df, lhs, rhs, match="inner", ix=None, nw_lags=0):
    """Fit a multivariate OLS model with multiple dependent variables.

    Estimates a system of equations sharing the same regressor matrix.
    Computes both HC0 (homoskedastic) and HC1 (heteroskedastic-robust)
    covariance matrices as well as log-likelihood and information criteria.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    lhs : list of str
        Dependent variable column names.
    rhs : list of str
        Independent variable column names. Include ``'const'`` to add a
        constant column automatically.
    match : {'inner', 'outer', 'custom'}, optional
        Sample selection rule. Defaults to ``'inner'``.
    ix : array-like of bool or None, optional
        Custom inclusion index (used when ``match='custom'``).
    nw_lags : int, optional
        Newey-West HAC lag order. Currently not implemented; must be ``0``.

    Returns
    -------
    FullResults
        Object whose ``results`` attribute is an :class:`MVOLSResults`
        instance, together with the sample index and design matrices.

    Raises
    ------
    NotImplementedError
        If ``nw_lags > 0`` (not yet implemented).
    """
    if "const" in rhs and "const" not in df:
        df["const"] = 1.0

    if nw_lags > 0:
        raise NotImplementedError("mv_ols with nw_lags > 0 is not implemented.")

    X = df[rhs].values
    z = df[lhs].values

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

    results = MVOLSResults(
        nobs,
        params,
        fittedvalues,
        resid,
        cov_e,
        cov_HC0,
        HC0_se,
        HC0_tstat,
        cov_HC1,
        HC1_se,
        HC1_tstat,
        llf,
        aic,
        bic,
        hqc,
    )

    return FullResults(results, ix, Xs, zs)


def hc0(x, e):
    """Compute the HC0 (homoskedastic sandwich) covariance matrix.

    Parameters
    ----------
    x : numpy.ndarray of shape (T, k)
        Regressor matrix.
    e : numpy.ndarray of shape (T, nz)
        Residual matrix.

    Returns
    -------
    cov_HC0 : numpy.ndarray of shape (k*nz, k*nz)
        HC0 covariance matrix computed as ``kron(cov_x_inv, cov_e)``.
    cov_e : numpy.ndarray of shape (nz, nz)
        Residual covariance matrix.
    """
    cov_e, _, cov_x_inv, _, _, _ = init_cov(x, e)
    cov_HC0 = np.kron(cov_x_inv, cov_e)
    return (cov_HC0, cov_e)


def hc1(x, e):
    """Compute the HC1 (heteroskedasticity-robust sandwich) covariance matrix.

    Uses the outer-product of residuals estimator for the middle of the
    sandwich formula.

    Parameters
    ----------
    x : numpy.ndarray of shape (T, k)
        Regressor matrix.
    e : numpy.ndarray of shape (T, nz)
        Residual matrix.

    Returns
    -------
    cov_HC1 : numpy.ndarray of shape (k*nz, k*nz)
        HC1 heteroskedasticity-robust covariance matrix.
    cov_e : numpy.ndarray of shape (nz, nz)
        Residual covariance matrix (also returned for convenience).
    """
    cov_e, cov_x, cov_x_inv, T, nz, k = init_cov(x, e)
    cov_xeex = np.zeros((nz * k, nz * k))
    for tt in range(T):
        x_t = x[tt, :][:, np.newaxis]
        e_t = e[tt, :][:, np.newaxis]
        cov_xeex += np.kron(np.dot(x_t, x_t.T), np.dot(e_t, e_t.T))
    cov_xeex /= T

    cov_X_inv = np.kron(cov_x_inv, np.eye(nz))
    cov_HC1 = np.dot(cov_X_inv, np.dot(cov_xeex, cov_X_inv))
    return (cov_HC1, cov_e)


def init_cov(x, e):
    """Compute basic covariance building-blocks for multivariate OLS.

    Parameters
    ----------
    x : numpy.ndarray of shape (T, k)
        Regressor matrix.
    e : numpy.ndarray of shape (T, nz)
        Residual matrix.

    Returns
    -------
    cov_e : numpy.ndarray of shape (nz, nz)
        Residual cross-product matrix ``e.T @ e / T``.
    cov_x : numpy.ndarray of shape (k, k)
        Regressor cross-product matrix ``x.T @ x / T``.
    cov_x_inv : numpy.ndarray of shape (k, k)
        Inverse of *cov_x*.
    T : int
        Number of time periods / observations.
    nz : int
        Number of dependent variables.
    k : int
        Number of regressors.

    Raises
    ------
    AssertionError
        If the number of rows in *x* and *e* differ.
    """
    T, k = x.shape
    Te, nz = e.shape
    assert T == Te

    cov_e = np.dot(e.T, e) / T
    cov_x = np.dot(x.T, x) / T
    cov_x_inv = np.linalg.inv(cov_x)

    return (cov_e, cov_x, cov_x_inv, T, nz, k)


def standard_errors(V, T):
    """Compute standard errors from a covariance matrix.

    Parameters
    ----------
    V : numpy.ndarray of shape (n, n)
        Covariance matrix.
    T : int
        Sample size used to scale the variance (divides the diagonal by *T*).

    Returns
    -------
    numpy.ndarray of shape (n,)
        Standard errors ``sqrt(diag(V) / T)``.
    """
    se = np.sqrt(np.diagonal(V) / T)
    return se


def least_sq(X, z):
    """Solve the normal equations for OLS.

    Parameters
    ----------
    X : numpy.ndarray of shape (T, k)
        Design matrix.
    z : numpy.ndarray of shape (T,) or (T, nz)
        Outcome vector or matrix.

    Returns
    -------
    numpy.ndarray of shape (k,) or (k, nz)
        OLS coefficient estimate ``(X.T @ X)^{-1} X.T @ z``.
    """
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, z))


def to_pickle(x, path):
    """Serialize an object to a pickle file.

    Parameters
    ----------
    x : any
        Python object to serialize.
    path : str
        Destination file path.

    Returns
    -------
    None
    """
    with open(path, "wb") as f:
        pickle.dump(x, f)


def read_pickle(path):
    """Deserialize an object from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    any
        The deserialized Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def demean_separate(df, var_list, group_list, **kwargs):
    """Demean variables sequentially by each group variable.

    Applies :func:`demean2` once per group in *group_list*, passing the
    result of each call as input to the next. This corresponds to iterative
    within-group demeaning.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (modified by each call to :func:`demean2`).
    var_list : list of str
        Variables to demean.
    group_list : list
        Sequence of group-by variable(s) passed one at a time to
        :func:`demean2`.
    **kwargs
        Additional keyword arguments forwarded to :func:`demean2`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with variables demeaned by each group in succession.
    """

    for group in group_list:
        df = demean2(group, var_list, df, **kwargs)

    return df


def demean2(group_list, var_list, df, prefix=None):
    """Subtract within-group means, optionally writing to new columns.

    Parameters
    ----------
    group_list : str or list of str
        Group-by variable(s) passed to :meth:`pandas.DataFrame.groupby`.
    var_list : list of str
        Variables to demean.
    df : pandas.DataFrame
        Input data (not modified in place; a new DataFrame is returned).
    prefix : str or None, optional
        If ``None`` (default), demeaned values overwrite the original columns.
        If a string, demeaned values are written to ``'<prefix>_<var>'``
        columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with demeaned columns (prefixed or overwritten).
    """

    if prefix is None:
        full_prefix = ""
    else:
        full_prefix = prefix + "_"

    # Precautionarily drop means
    for var in var_list:
        if "MEAN_" + var in df:
            df = df.drop(["MEAN_" + var], axis=1)

    df_mean = df.groupby(group_list)[var_list].mean()
    df_mean = df_mean.rename(columns={var: "MEAN_" + var for var in var_list})
    df = pd.merge(df, df_mean, left_on=group_list, right_index=True)

    for var in var_list:
        df[full_prefix + var] = df[var] - df["MEAN_" + var]

    df = df.drop(["MEAN_" + var for var in var_list], axis=1)

    return df


def weight_regression_params(weights, params=None, cov=None, results=None):
    """Compute a weighted linear combination of regression coefficients and its SE.

    Evaluates ``x = weights @ params`` and the corresponding standard error
    using the delta method: ``se = sqrt(weights @ cov @ weights)``.

    Parameters
    ----------
    weights : array-like of shape (k,)
        Vector of weights defining the linear combination.
    params : array-like of shape (k,) or None, optional
        Coefficient vector. Required if *results* is ``None``.
    cov : array-like of shape (k, k) or None, optional
        Covariance matrix of the coefficients. Required if *results* is
        ``None``.
    results : statsmodels results object or None, optional
        Alternative input: if provided, ``params`` and ``cov_HC0`` are
        extracted from this object.

    Returns
    -------
    x : float
        Weighted sum of coefficients.
    se : float
        Standard error of *x*.

    Raises
    ------
    AssertionError
        If neither (``params`` and ``cov``) nor ``results`` is provided.
    """

    assert ((params is not None) and (cov is not None)) or (results is not None)

    if results is not None:
        params = results.params
        cov = results.cov_HC0

    x = np.dot(params, weights)
    se = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    return x, se


def sum_regression_params(positions, *args, **kwargs):
    """Compute the sum of selected regression coefficients and its standard error.

    Constructs a unit-weight vector with ones at *positions* and delegates to
    :func:`weight_regression_params`.

    Parameters
    ----------
    positions : array-like of int
        Indices of the coefficients to sum.
    *args, **kwargs
        Forwarded to :func:`weight_regression_params`.

    Returns
    -------
    x : float
        Sum of the selected coefficients.
    se : float
        Standard error of the sum.
    """

    if "results" in kwargs and kwargs["results"] is not None:
        n_params = len(kwargs["results"].params)
    elif "params" in kwargs and kwargs["params"] is not None:
        n_params = len(kwargs["params"])
    elif "cov" in kwargs and kwargs["cov"] is not None:
        n_params = np.asarray(kwargs["cov"]).shape[0]
    elif len(args) >= 1 and args[0] is not None:
        n_params = len(args[0])
    elif len(args) >= 2 and args[1] is not None:
        n_params = np.asarray(args[1]).shape[0]
    else:
        raise ValueError(
            "sum_regression_params requires params/cov or results to infer coefficient length."
        )

    e_vec = np.zeros(n_params)
    e_vec[positions] = 1.0

    return weight_regression_params(e_vec, *args, **kwargs)


def collapse(df, method="mean", var_list=None, by=None, wvar=None):
    """Collapse a DataFrame to weighted means or sums within groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    method : {'mean', 'sum'}, optional
        Aggregation method. Defaults to ``'mean'``.
    var_list : list of str or None, optional
        Numeric columns to aggregate. If ``None``, all numeric columns not in
        *by* or *wvar* are used.
    by : list of str or None, optional
        Group-by columns. Must be non-empty.
    wvar : str or None, optional
        Weight variable. Must be provided (use a plain ``groupby`` for
        unweighted aggregation).

    Returns
    -------
    pandas.DataFrame
        Collapsed DataFrame indexed by *by* with aggregated *var_list*
        columns and the weight sum column *wvar*.

    Raises
    ------
    AssertionError
        If *by* is empty, *method* is unsupported, *wvar* is ``None``, or
        *wvar* appears in *var_list*.
    """
    if by is None:
        by = []

    assert by
    assert method in ["mean", "sum"]
    assert wvar is not None  # just do groupby otherwise

    if var_list is None:
        var_list = [
            var
            for var in df.columns
            if (var not in (by + [wvar])) and pd.api.types.is_numeric_dtype(df[var])
        ]

    assert wvar not in var_list

    keep_vars = list(set(var_list + by + [wvar]))
    _df = df[keep_vars].copy()

    for var in var_list:
        if (var not in by) and (var != wvar):
            _df[var] *= _df[wvar]

    collapsed = _df.groupby(by)[var_list + [wvar]].sum()
    if method == "mean":
        for var in var_list:
            collapsed[var] /= collapsed[wvar]

    return collapsed


def safe_sum(x):
    """Sum a Series without skipping NaN values.

    Unlike the default :meth:`pandas.Series.sum`, this returns ``NaN`` if any
    element is ``NaN``, rather than treating NaNs as zero.

    Parameters
    ----------
    x : pandas.Series
        Input series.

    Returns
    -------
    scalar
        Sum of *x*, or ``NaN`` if any element is ``NaN``.
    """
    return x.sum(skipna=False)
