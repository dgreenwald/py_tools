import numpy as np
import pandas as pd
import re
from scipy.linalg import solve_discrete_lyapunov
from tabulate import tabulate

from statsmodels.tsa import stattools
import statsmodels.tsa.arima.model as arima

import py_tools.data as dt


def panel_resampler(df, time_var, freq):
    """Resample a single panel group to a target frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a MultiIndex; time_var must be one of the index levels.
    time_var : str
        Name of the time index level to use for resampling.
    freq : str
        Pandas frequency string (e.g., 'QS', 'MS').

    Returns
    -------
    df_resampled : pd.DataFrame
        Resampled DataFrame indexed by time_var at the given frequency.
    """

    to_drop = [name for name in df.index.names if name != time_var]
    return df.reset_index().drop(columns=to_drop).set_index(time_var).asfreq(freq)


def to_panel(df, panel_vars, time_var, freq):
    """Convert a flat DataFrame to a panel and resample to a target frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with panel and time information.
    panel_vars : str or list of str
        Names of the panel identifier variable(s).
    time_var : str
        Name of the time variable.
    freq : str
        Target frequency for resampling.

    Returns
    -------
    df_panel : pd.DataFrame
        MultiIndex DataFrame with panel variables and time as the index,
        resampled to freq.
    """

    if isinstance(panel_vars, str):
        panel_vars = [panel_vars]

    if df.index.name != [None]:
        df = df.reset_index()

    return (
        df.set_index(panel_vars + [time_var])
        .groupby(panel_vars)
        .apply(panel_resampler, time_var, freq)
    )


def panel_shift(df, panel_vars, periods=1):
    """Shift a panel DataFrame by a given number of periods within each group.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame.
    panel_vars : str or list of str
        Grouping variable(s) defining the panel structure.
    periods : int, optional
        Number of periods to shift (positive = lag, negative = lead).
        Default is 1.

    Returns
    -------
    df_shifted : pd.DataFrame
        Shifted DataFrame with the same index as df.
    """

    return df.groupby(panel_vars).shift(periods=periods)


def merge_date(df1, df2, how="outer", **kwargs):
    """Merge two DataFrames on their date indices.

    Parameters
    ----------
    df1 : pd.DataFrame
        Left DataFrame.
    df2 : pd.DataFrame
        Right DataFrame.
    how : str, optional
        Type of merge ('inner', 'outer', 'left', 'right'). Default is 'outer'.
    **kwargs
        Additional keyword arguments passed to pd.merge.

    Returns
    -------
    df_merged : pd.DataFrame
        Merged DataFrame indexed by the shared date index.
    """

    return pd.merge(df1, df2, left_index=True, right_index=True, how=how, **kwargs)


def merge_date_many(df_list, how="outer", **kwargs):
    """Merge a list of DataFrames on their date indices.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        List of DataFrames to merge sequentially.
    how : str, optional
        Type of merge. Default is 'outer'.
    **kwargs
        Additional keyword arguments passed to pd.merge.

    Returns
    -------
    df_merged : pd.DataFrame
        Merged DataFrame.
    """

    df = df_list[0]
    for ii in range(1, len(df_list)):
        df = pd.merge(
            df, df_list[ii], left_index=True, right_index=True, how=how, **kwargs
        )

    return df


def get_date_index(startdate, periods, freq="QS"):
    """Create a DatetimeIndex of a given length and frequency.

    Parameters
    ----------
    startdate : str or datetime-like
        Start date.
    periods : int
        Number of periods.
    freq : str, optional
        Pandas frequency string. Default is 'QS'.

    Returns
    -------
    idx : pd.DatetimeIndex
        Date range of length periods starting at startdate.
    """
    return pd.date_range(startdate, periods=periods, freq=freq)


def date_index(df, startdate, freq="QS"):
    """Assign a DatetimeIndex to a DataFrame in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to reindex.
    startdate : str or datetime-like
        Start date for the new index.
    freq : str, optional
        Pandas frequency string. Default is 'QS'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with updated DatetimeIndex.
    """
    df.set_index(get_date_index(startdate, len(df), freq=freq), inplace=True)
    return df


def quarter_index(df, yr, q):
    """Assign a quarterly DatetimeIndex starting at a given year and quarter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to reindex.
    yr : int or str
        Starting year.
    q : int or str
        Starting quarter (1-4).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with updated DatetimeIndex.
    """
    mon = 3 * (int(q) - 1) + 1
    return month_index(df, yr, mon)


def month_index(df, yr, mon):
    """Assign a monthly DatetimeIndex starting at a given year and month.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to reindex.
    yr : int or str
        Starting year.
    mon : int or str
        Starting month (1-12).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with updated DatetimeIndex.
    """
    return date_index(df, "{0}/1/{1}".format(mon, yr))


def date_from_year(year):
    """Convert a Series of years to a DatetimeIndex at January 1st.

    Parameters
    ----------
    year : pd.Series of int or str
        Series of year values.

    Returns
    -------
    dates : pd.DatetimeIndex
        Corresponding January 1st dates; NaT on conversion failure.
    """
    date_str = year.astype(str) + "-01-01"
    return pd.to_datetime(date_str, errors="coerce")


def date_from_qtr(yr, q):
    """Convert arrays of year and quarter to a DatetimeIndex.

    Parameters
    ----------
    yr : array-like of int
        Year values.
    q : array-like of int
        Quarter values (1-4).

    Returns
    -------
    dates : pd.DatetimeIndex
        Corresponding first-month-of-quarter dates.
    """
    mon = 3 * (q - 1) + 1
    return date_from_month(yr, mon)


def date_from_month(yr, mon):
    """Convert arrays of year and month to a DatetimeIndex.

    Parameters
    ----------
    yr : array-like of int
        Year values.
    mon : array-like of int
        Month values (1-12).

    Returns
    -------
    dates : pd.DatetimeIndex
        Corresponding first-day-of-month dates.
    """
    date = yr.astype(str) + "-" + mon.astype(str) + "-01"
    return pd.to_datetime(date, errors="coerce")


def date_from_q_string(s):
    """Convert a Series of year-quarter strings to a DatetimeIndex.

    Parameters
    ----------
    s : pd.Series of str
        Quarter strings in 'YYYYQN' format (e.g., '2020Q1').

    Returns
    -------
    dates : pd.DatetimeIndex
        Corresponding first-month-of-quarter dates.
    """

    s = s.astype(str)
    codes, uniq = pd.factorize(s, sort=False)  # uniques only once

    y = uniq.str[:4].astype("int32")
    q = uniq.str[-1].astype("int8")
    m = (q - 1) * 3 + 1
    u_dt = pd.to_datetime(y * 100 + m, format="%Y%m").to_numpy()
    dates = pd.DatetimeIndex(u_dt[codes])  # rebuild full series

    return dates


def resample(df, methods_vars, freq="QS"):
    """Resample a DataFrame using different aggregation methods per variable group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex.
    methods_vars : dict
        Mapping from aggregation method name (e.g., 'mean', 'sum', 'last')
        to a list of column names to aggregate with that method.
    freq : str, optional
        Target frequency. Default is 'QS'.

    Returns
    -------
    df_resamp : pd.DataFrame
        Resampled DataFrame.
    """
    df_resamp = None

    for method, var_list in methods_vars.items():
        if var_list is not None:
            df_new = getattr(df[var_list].resample(freq), method)()

            if df_resamp is None:
                df_resamp = df_new
            else:
                df_resamp = pd.merge(
                    df_resamp, df_new, left_index=True, right_index=True
                )

    return df_resamp


def deflate(
    df,
    var_list,
    index="cpi",
    log=False,
    diff=False,
    per_capita=False,
    reimport=False,
    **kwargs,
):
    """Deflate a set of series by a price or population index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    var_list : list of str
        Variables to deflate.
    index : {'cpi', 'pce', 'pop'}, optional
        Price (or population) index to use. Default is 'cpi'.
    log : bool, optional
        Set to True if the variable is already in log levels; the function
        will subtract the log price index rather than dividing. Default is False.
    diff : bool, optional
        Set to True if the variable is already in differences; the function
        will subtract differenced log prices. Default is False.
    per_capita : bool, optional
        If True, also deflate by population. Default is False.
    reimport : bool, optional
        If True, re-import raw price data instead of using cached files.
        Default is False.
    **kwargs
        Additional keyword arguments passed to the dataset loader.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with deflated series appended.
    new_var_list : list of str
        Names of the newly created deflated variables.
    """

    from py_tools.datasets.loader import load

    new_var_list = []

    if per_capita:
        assert index != "pop"
        df, var_list = deflate(
            df,
            var_list,
            index="pop",
            log=log,
            diff=diff,
            per_capita=False,
            reimport=reimport,
            **kwargs,
        )

    for var in var_list:
        if index == "pop":
            new_var = "PC_" + var
        else:
            new_var = "DEF" + index.upper() + "_" + var

        new_var_list.append(new_var)

        if new_var not in df:
            if index == "cpi":
                dataset = "fred"
                index_name = "cpi_deflator"
            elif index == "pce":
                dataset = "nipa_10109"
                index_name = "pce_deflator"
            elif index == "pop":
                dataset = "nipa_20100"

            if index == "pop":
                index_var = "pop"
            elif index == "pce":
                index_var = index_name
            else:
                index_var = dataset.upper() + "_" + index_name

            if index_var not in df:
                df_new = load([dataset], reimport=reimport, **kwargs)

                if index == "pop":
                    df_new[index_var] = (
                        df_new["NIPA_20100_real_disp_inc"]
                        / df_new["NIPA_20100_real_pc_disp_inc"]
                    )

                df = pd.merge(
                    df, df_new[index_var].to_frame(), left_index=True, right_index=True
                )

            scale = np.log(df[index_var])

            if diff:
                scale = scale.diff()

            if log:
                df[new_var] = df[var] - scale
            else:
                df[new_var] = df[var] / np.exp(scale)

    return (df, new_var_list)


def add_lags(df, var, n_lags, init_lag=1):
    """Create lag columns of a variable and add them to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add lags to (modified in-place).
    var : str
        Name of the variable to lag.
    n_lags : int
        Number of lags to add.
    init_lag : int, optional
        Starting lag order. Default is 1.

    Returns
    -------
    lag_list : list of str
        Names of the newly created lag columns.
    """

    lag_list = []
    for lag in range(init_lag, n_lags + init_lag):
        lag_list += transform(df, [var], lag=lag)

    return lag_list


def transform(df, var_list, lag=0, diff=0, other=None):
    """Apply transformations (lag, difference, or function) to variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to transform (modified in-place).
    var_list : list of str
        Variables to transform.
    lag : int, optional
        Lag order (positive = lag, negative = lead). Default is 0.
    diff : int, optional
        Differencing order. Default is 0 (no differencing).
    other : str, optional
        Name of a NumPy function to apply before lagging/differencing.
        Default is None.

    Returns
    -------
    new_var_list : list of str
        Names of the transformed variables added to df.
    """

    new_var_list = []
    for var in var_list:
        new_var = var

        if other is not None:
            prefix = other.upper() + "_"
            new_var = prefix + new_var

        if diff != 0:
            if diff != 1:
                prefix = "D{}_".format(diff)
            else:
                prefix = "D_"
            new_var = prefix + new_var

        if lag != 0:
            if lag != 1:
                prefix = "L{}_".format(lag)
            else:
                prefix = "L_"
            new_var = prefix + new_var

        new_var = re.sub("-", "_MINUS_", new_var)

        new_var_list.append(new_var)

        if new_var not in df:
            df[new_var] = df[var]

            if other is not None:
                df[new_var] = eval("np.{}(df[new_var])".format(other))

            if diff != 0:
                df[new_var] = df[new_var].diff(diff)

            if lag != 0:
                df[new_var] = df[new_var].shift(lag)

    return new_var_list


def long_horizon_contemp(df, lhs, rhs, horizon, **kwargs):
    """Contemporaneous long-horizon regression (not yet implemented).

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame.
    lhs : str
        Left-hand side variable name.
    rhs : list of str
        Right-hand side variable names.
    horizon : int
        Forecast horizon.
    **kwargs
        Additional keyword arguments.

    Raises
    ------
    Exception
        Always raised; method not yet implemented.
    """

    raise NotImplementedError("long_horizon_contemp is not implemented.")


def long_horizon_predictive(df_in, lhs, rhs, horizon, norm_lhs=False, **kwargs):
    """Run a predictive long-horizon regression.

    The dependent variable is the sum of the next horizon values of lhs.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    lhs : str
        Left-hand side variable name.
    rhs : list of str
        Right-hand side variable names.
    horizon : int
        Forecast horizon (number of periods to sum).
    norm_lhs : bool, optional
        If True, normalize the summed LHS by dividing by horizon. Default is False.
    **kwargs
        Additional keyword arguments passed to dt.regression.

    Returns
    -------
    fr : FullResults
        Regression results object.
    """

    df = df_in[[lhs] + rhs].copy()
    lhs_long = "{0}_{1}_Per_Diff".format(lhs, horizon)
    df[lhs_long] = 0
    for jj in range(1, horizon + 1):
        lhs_diff = transform(df, [lhs], lag=-jj)[0]
        df[lhs_long] += df[lhs_diff]

    if norm_lhs:
        df[lhs_long] /= horizon

    return dt.regression(df, lhs_long, rhs, **kwargs)


def MA(df, lhs_var, rhs_vars, init_lag=1, default_lags=16, lags_by_var=None, **kwargs):
    """Run a distributed lag (moving average) regression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    lhs_var : str
        Left-hand side variable name.
    rhs_vars : list of str
        Right-hand side variable names.
    init_lag : int, optional
        Starting lag order. Default is 1.
    default_lags : int, optional
        Default number of lags for each RHS variable. Default is 16.
    lags_by_var : dict, optional
        Mapping from variable name to number of lags (overrides default_lags).
    **kwargs
        Additional keyword arguments passed to dt.regression.

    Returns
    -------
    fr : FullResults
        Regression results object.
    """

    if lags_by_var is None:
        lags_by_var = {}
    lhs = lhs_var

    rhs = []
    for var in rhs_vars:
        this_lag = lags_by_var.get(var, default_lags)
        for lag in range(init_lag, this_lag + init_lag):
            rhs += transform(df, [var], lag=lag)

    # Get sample indices
    ix, _, _ = dt.match_xy(df[rhs_vars].values, df[lhs_var].values)

    # Run dt.regression
    return dt.regression(df, lhs, rhs, ix=ix, **kwargs)


def ARMA(df, var, p, q, freq="QS", trend="c", display=False):
    """Estimate an ARMA(p, q) model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the series.
    var : str
        Name of the variable to model.
    p : int
        AR order.
    q : int
        MA order.
    freq : str, optional
        Pandas frequency string. Default is 'QS'.
    trend : str, optional
        Trend specification passed to statsmodels ARIMA. Default is 'c'.
    display : bool, optional
        If True, print the model summary. Default is False.

    Returns
    -------
    fr : FullResults
        Regression results object wrapping the fitted ARIMA model.
    """

    ix = pd.notnull(df[var])
    series = df.loc[ix, var]
    mod = arima.ARIMA(series, order=(p, 0, q), freq=freq, trend=trend)
    res = mod.fit()
    if display:
        print(res.summary())

    return dt.FullResults(res, ix=ix, Xs=None, zs=None)


def arma_regression(series, p, q, freq="QS", ix=None):
    """Estimate an ARMA(p, q) model by two-stage OLS.

    Parameters
    ----------
    series : pd.Series
        Time series to model.
    p : int
        AR order.
    q : int
        MA order.
    freq : str, optional
        Pandas frequency string. Default is 'QS'.
    ix : np.ndarray of bool, optional
        Sample selection index. Default is all observations.

    Returns
    -------
    fr : FullResults
        Regression results from the second-stage OLS.
    """

    if ix is None:
        ix = np.ones(len(series), dtype=bool)

    df = series.to_frame(name="x")

    x_lags = add_lags(df, "x", p)
    fr = dt.regression(df.loc[ix, :], "x", x_lags)
    ix_stage1 = ix.copy()
    ix_stage1[ix] = fr.ix
    df.loc[ix_stage1, "u"] = fr.results.resid

    u_lags = add_lags(df, "u", q)
    fr = dt.regression(df.loc[ix, :], "x", x_lags + u_lags)
    ix_stage2 = ix.copy()
    ix_stage2[ix] = fr.ix

    fr.ix = ix_stage2

    return fr


def VAR(df_in, var_list, n_var_lags=1, use_const=True):
    """Estimate a VAR using OLS.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame (copied internally).
    var_list : list of str
        Variable names to include in the VAR.
    n_var_lags : int, optional
        Number of lags. Default is 1.
    use_const : bool, optional
        Include a constant term. Default is True.

    Returns
    -------
    fr : FullResults
        Multivariate OLS results object.
    """

    df = df_in.copy()

    # LHS variables
    lhs = var_list
    rhs = []

    for lag in range(1, n_var_lags + 1):
        for var in var_list:
            rhs += transform(df, [var], lag=lag)

    # RHS variables
    if use_const:
        rhs += ["const"]

    # Regression
    return dt.mv_ols(df, lhs, rhs)


class VECMResults(dt.FullResults):
    """VECM estimation results.

    Extends FullResults with the estimated cointegrating vector.

    Parameters
    ----------
    fr : FullResults
        OLS results from the VECM regression.
    alp : np.ndarray
        Cointegrating vector (normalized so the LHS coefficient is 1).

    Attributes
    ----------
    alp : np.ndarray
        Cointegrating vector.
    """

    def __init__(self, fr, alp):
        """Initialize VECMResults.

        Parameters
        ----------
        fr : FullResults
            OLS results from the VECM regression.
        alp : np.ndarray
            Cointegrating vector.
        """
        dt.FullResults.__init__(self, fr.results, fr.ix, fr.Xs, fr.zs)
        self.alp = alp


def VECM(df, var_list, n_var_lags=1, n_dls_lags=8):
    """Estimate a VECM using dynamic least squares (DLS).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables.
    var_list : list of str
        Ordered variable names; the first variable is the LHS of the
        cointegrating relationship.
    n_var_lags : int, optional
        Number of VAR lags in the VECM. Default is 1.
    n_dls_lags : int, optional
        Number of leads/lags for DLS cointegration estimation. Default is 8.

    Returns
    -------
    vr : VECMResults
        VECM estimation results.
    """

    # LHS variables
    lhs = transform(df, var_list, diff=1)

    # RHS variables
    rhs = ["const"]

    # Estimate cointegrating relationship
    dls_lhs = var_list[0]
    dls_rhs = var_list[1:]
    alp, dlt = run_dls(df, dls_lhs, dls_rhs, n_dls_lags)

    # Get cointegration term
    df["coint_resid"] = df.loc[:, var_list].to_numpy() @ alp
    rhs += transform(df, ["coint_resid"], lag=1)

    for lag in range(1, n_var_lags + 1):
        for var in var_list:
            rhs += transform(df, [var], diff=1, lag=lag)

    # Regression
    return VECMResults(dt.mv_ols(df, lhs, rhs), alp)


def VECM_companion_form(vr, H=None):
    """Convert VECM results to companion form.

    Parameters
    ----------
    vr : VECMResults
        VECM results object.
    H : np.ndarray, optional
        Impact matrix (n_var x n_shock). If None, C is returned as None.

    Returns
    -------
    Phi : np.ndarray
        Companion-form transition matrix.
    C : np.ndarray or None
        Companion-form impact matrix, or None if H is None.
    """

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
    """Compute cumulative impulse responses from a VECM.

    Parameters
    ----------
    vr : VECMResults
        VECM results object.
    H : np.ndarray, shape (n_var, n_shock)
        Impact matrix.
    nt : int
        Number of IRF periods.

    Returns
    -------
    irfs : np.ndarray, shape (n_coint, n_shock, nt)
        Cumulative impulse response functions.
    """

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
        irfs[:, :, tt] = np.dot(Phi, irfs[:, :, tt - 1])

    # Drop lagged terms
    return np.cumsum(irfs[:ny, :, :], axis=2)


class LongHorizonMA:
    """Long-horizon moving average regression (not yet implemented).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    lhs_var : str
        Left-hand side variable name.
    rhs_var : str
        Right-hand side variable name.
    horizon : int
        Forecast horizon.
    n_lags : int, optional
        Number of MA lags. Default is 16.
    """

    def __init__(self, df, lhs_var, rhs_var, horizon, n_lags=16):
        """Initialize LongHorizonMA (not yet implemented).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        lhs_var : str
            Left-hand side variable name.
        rhs_var : str
            Right-hand side variable name.
        horizon : int
            Forecast horizon.
        n_lags : int, optional
            Number of MA lags. Default is 16.

        Raises
        ------
        Exception
            Always raised; not yet implemented.
        """

        raise NotImplementedError("LongHorizonMA is not implemented.")


class LongHorizonVAR:
    """Long-horizon VAR-based regression.

    Computes long-horizon predictive and contemporaneous regression coefficients
    and R-squared values using the analytical VAR covariance structure.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    lhs_var : str
        Left-hand side variable name.
    rhs_vars : list of str
        Right-hand side variable names.
    horizon : int
        Long-horizon window (number of periods).
    n_var_lags : int, optional
        Number of VAR lags. Default is 1.

    Attributes
    ----------
    A : np.ndarray
        Companion-form VAR coefficient matrix.
    Q : np.ndarray
        Companion-form residual covariance matrix.
    C : list of np.ndarray
        Autocovariance matrices C[j] = A^j @ C[0].
    Vk : np.ndarray
        Long-horizon variance-covariance matrix.
    """

    def __init__(self, df, lhs_var, rhs_vars, horizon, n_var_lags=1):
        """Initialize the long-horizon VAR.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        lhs_var : str
            Left-hand side variable name.
        rhs_vars : list of str
            Right-hand side variable names.
        horizon : int
            Long-horizon window.
        n_var_lags : int, optional
            Number of VAR lags. Default is 1.
        """

        self.lhs_var = lhs_var
        self.rhs_vars = rhs_vars
        self.n_var_lags = n_var_lags
        self.horizon = horizon

        var_list = [self.lhs_var] + self.rhs_vars

        self.n_rhs = len(self.rhs_vars)
        self.n_var = len(var_list)

        self.fr = VAR(df, var_list, self.n_var_lags)

        n_A = self.fr.results.params.shape[0] - 1
        self.A = np.zeros((n_A, n_A))
        self.A[: self.n_var, :] = self.fr.results.params[1:, :].T
        if self.n_var_lags > 1:
            self.A[self.n_var :, : -self.n_var] = np.eye(n_A - self.n_var)

        self.Q = np.zeros(self.A.shape)
        self.Q[: self.n_var, : self.n_var] = self.fr.results.cov_HC0

        # C: unconditional covariances
        self.C = []
        self.C.append(solve_discrete_lyapunov(self.A, self.Q))

        for jj in range(1, self.horizon + 1):
            self.C.append(np.dot(self.A, self.C[jj - 1]))

        self.Vk = self.horizon * self.C[0]
        for jj in range(1, self.horizon):
            self.Vk += (self.horizon - jj) * (self.C[jj] + self.C[jj].T)

        # Long-horizon regressions
        self.lhs_ix = 0

    def predictive_reg(self):
        """Compute long-horizon predictive regression coefficients and R-squared.

        Returns
        -------
        bet_lh : np.ndarray, shape (n_rhs,)
            Long-horizon predictive regression coefficients.
        R2 : np.ndarray, shape (n_rhs,)
            R-squared values for each predictor.
        """

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
            R2[ii] = (lh_rh_cov**2) / (lh_var * rh_var)

        return (bet_lh, R2)

    def contemp_reg(self, display=False):
        """Compute long-horizon contemporaneous regression coefficients and R-squared.

        Parameters
        ----------
        display : bool, optional
            If True, print a formatted results table. Default is False.

        Returns
        -------
        bet_lh : np.ndarray, shape (n_rhs,)
            Long-horizon regression coefficients.
        R2 : float
            Overall R-squared.
        """

        # covariance terms
        var_y = self.Vk[0, 0]
        cov_xy = self.Vk[0, 1 : self.n_var]
        cov_xx = self.Vk[1 : self.n_var, 1 : self.n_var]

        # OLS
        bet_lh = np.linalg.solve(cov_xx, cov_xy)
        R2 = np.dot(bet_lh.T, np.dot(cov_xx, bet_lh)) / var_y

        if display:
            # Print first table
            headers = ["LH VAR Regression", ""]
            table = [
                ["Dep. Variable:", self.lhs_var],
                ["Lags:", self.n_var_lags],
                ["Horizon:", self.horizon],
                ["R2:", "{:4.3f}".format(R2)],
            ]

            print("\n\n\n")
            print(tabulate(table, headers, tablefmt="rst", floatfmt="4.3f"))

            headers = ["", "coef"]
            table = []
            for ii in range(self.n_rhs):
                table.append([self.rhs_vars[ii], bet_lh[ii]])

            print("\n")
            print(tabulate(table, headers, tablefmt="rst", floatfmt="4.3f"))
            print("\n\n\n")

        return (bet_lh, R2)

    def orthogonal_contemp_reg(self, display=False):
        """Compute long-horizon regression with orthogonalized regressors.

        Uses a sequential Gram-Schmidt procedure to orthogonalize RHS variables,
        yielding individual R-squared contributions.

        Parameters
        ----------
        display : bool, optional
            If True, print a formatted results table. Default is False.

        Returns
        -------
        bet_lh : np.ndarray, shape (n_rhs,)
            Coefficients on each orthogonalized regressor.
        R2 : np.ndarray, shape (n_rhs,)
            R-squared contribution of each orthogonalized regressor.
        """

        bet_lh = np.zeros(self.n_rhs)
        R2 = np.zeros(self.n_rhs)

        # covariance terms
        var_y = self.Vk[0, 0]
        cov_xy = self.Vk[0, 1 : self.n_var]
        cov_xx = self.Vk[1 : self.n_var, 1 : self.n_var]

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
        R2 = ((bet_lh**2) * var_u) / var_y

        if display:
            # Print first table
            headers = ["LH VAR Orth. Regression", ""]
            table = [
                ["Dep. Variable:", self.lhs_var],
                ["Lags:", self.n_var_lags],
                ["Horizon:", self.horizon],
                ["Total R2", "{:4.3f}".format(np.sum(R2))],
            ]

            print("\n\n\n")
            print(tabulate(table, headers, tablefmt="rst", floatfmt="4.3f"))

            headers = ["", "coef", "R2"]
            table = []
            for ii in range(self.n_rhs):
                table.append([self.rhs_vars[ii], bet_lh[ii], R2[ii]])

            print("\n")
            print(tabulate(table, headers, tablefmt="rst", floatfmt="4.3f"))
            print("\n\n\n")

        return (bet_lh, R2)


def orthogonalize_errors(u):
    """Orthogonalize reduced-form residuals via Cholesky decomposition.

    Parameters
    ----------
    u : np.ndarray, shape (Nt, Ny)
        Reduced-form residuals.

    Returns
    -------
    e : np.ndarray, shape (Nt, Ny)
        Orthogonalized residuals.
    H : np.ndarray, shape (Ny, Ny)
        Lower-triangular Cholesky factor of the residual covariance.
    """
    Sig = np.dot(u.T, u) / u.shape[0]
    H = np.linalg.cholesky(Sig)
    e = (np.linalg.solve(H, u.T)).T
    return (e, H)


# Estimate cointegrating relationship using DLS
def run_dls(df, lhs_var, rhs_vars, n_lags=8, display=False):
    """Estimate a cointegrating relationship using dynamic least squares (DLS).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables.
    lhs_var : str
        Left-hand side (normalizing) variable.
    rhs_vars : list of str
        Right-hand side cointegrating variables.
    n_lags : int, optional
        Number of leads and lags to include. Default is 8.
    display : bool, optional
        If True, print regression summary. Default is False.

    Returns
    -------
    coint_vec : np.ndarray
        Cointegrating vector (normalized so the LHS coefficient is 1).
    const : float
        Estimated constant term.
    """

    n_rhs = len(rhs_vars)
    rhs = ["const"] + rhs_vars

    lhs = [lhs_var]

    for lag in range(-n_lags, n_lags + 1):
        for var in rhs_vars:
            rhs += transform(df, [var], lag=lag, diff=1)

    # Regression
    fr = dt.sm_regression(df, lhs, rhs, display=display)

    coint_vec = np.hstack([np.ones(1), -fr.results.params[1 : n_rhs + 1]])
    const = fr.results.params[0]

    return (coint_vec, const)


def lagged_reg(df_in, lhs, rhs_list, n_lags, use_const=True, copy_df=True):
    """Regress lhs on 1 through n_lags lags of each variable in rhs_list.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    lhs : list of str or str
        Left-hand side variable(s).
    rhs_list : list of str
        Right-hand side variables to lag.
    n_lags : int
        Number of lags to include.
    use_const : bool, optional
        Include a constant term. Default is True.
    copy_df : bool, optional
        Copy the DataFrame before adding lags. Default is True.

    Returns
    -------
    fr : FullResults
        Multivariate OLS regression results.
    """

    if copy_df:
        df = df_in.copy()
    else:
        df = df_in

    rhs = []
    for lag in range(1, n_lags + 1):
        for var in rhs_list:
            rhs += transform(df, [var], lag=lag)

    if use_const:
        rhs += ["const"]

    return dt.mv_ols(df, lhs, rhs)


def detrend_hamilton(df_full, varlist, p=4, h=8):
    """Apply Hamilton's detrending procedure.

    Regresses each variable on its own lags h through h+p-1, using the
    residual as the cyclical component.

    Parameters
    ----------
    df_full : pd.DataFrame
        DataFrame containing the variables to detrend.
    varlist : list of str
        Variable names to detrend.
    p : int, optional
        Number of lags in the projection. Default is 4.
    h : int, optional
        Minimum lag distance. Default is 8.

    Returns
    -------
    df_full : pd.DataFrame
        DataFrame with '_detrend' and '_trend' suffix columns added.
    varlist_detrended : list of str
        Names of the detrended variables.
    fr_list : list of FullResults
        Regression results for each variable.
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

        df_full[var + "_detrend"] = np.nan
        df_full.loc[fr.ix, var + "_detrend"] = fr.results.resid

        df_full[var + "_trend"] = np.nan
        df_full.loc[fr.ix, var + "_trend"] = fr.results.fittedvalues

    varlist_detrended = [var + "_detrend" for var in varlist]

    return df_full, varlist_detrended, fr_list


def detrend_time(df, varlist, time_var=None, suffix="detrend_time"):
    """Remove a linear time trend from each variable in varlist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables.
    varlist : list of str
        Variable names to detrend.
    time_var : str, optional
        Name of the time variable column. If None, a 'TIME' column is created.
    suffix : str, optional
        Suffix appended to detrended variable names. Default is 'detrend_time'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with detrended series appended.
    varlist_detrend : list of str
        Names of the detrended variables.
    """

    if time_var is None:
        assert "TIME" not in df
        df["TIME"] = np.arange(len(df))
        time_var = "TIME"

    for var in varlist:
        fr = dt.regression(df, var, time_var)
        df.loc[fr.ix, var + "_" + suffix] = fr.results.resid

    varlist_detrend = [var + "_" + suffix for var in varlist]

    return df, varlist_detrend


def get_time_trend(df, var, time_var=None):
    """Extract the linear time trend and cycle from a series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variable.
    var : str
        Variable name.
    time_var : str, optional
        Name of the time column. If None, a 'TIME' column is created.

    Returns
    -------
    trend : pd.Series
        Fitted trend values.
    cycle : pd.Series
        Residual (cycle) values.
    """

    if time_var is None:
        assert "TIME" not in df
        df["TIME"] = np.arange(len(df))
        time_var = "TIME"

    fr = dt.regression(df, var, time_var)

    trend = pd.Series(index=df.index)
    trend.loc[fr.ix] = fr.results.fittedvalues

    cycle = pd.Series(index=df.index)
    cycle.loc[fr.ix] = fr.results.resid

    return trend, cycle


def autocorrelations(df_in, var, lags=20):
    """Compute autocorrelations of a variable.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing the variable.
    var : str
        Variable name.
    lags : int, optional
        Maximum lag order. Default is 20.

    Returns
    -------
    table : np.ndarray, shape (lags+1, 2)
        Array with lag values in column 0 and correlations in column 1.
    """

    df = df_in[[var]].copy()
    var_copy = var + "_COPY"
    df[var_copy] = df[var].copy()

    return lead_lag_correlations(df, var, var_copy, max_leads=0, max_lags=lags)


def lead_lag_correlations(
    df_in, var1, var2, lags=None, max_leads=8, max_lags=8, make_plot=False, **kwargs
):
    """Compute correlations between var1 and leads/lags of var2.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing the variables.
    var1 : str
        First variable name.
    var2 : str
        Second variable name.
    lags : range or list, optional
        Explicit lag values to compute. If None, uses max_leads and max_lags.
    max_leads : int, optional
        Maximum lead (negative lag) to include. Default is 8.
    max_lags : int, optional
        Maximum lag to include. Default is 8.
    make_plot : bool, optional
        If True, produce a two-axis plot at the highest-correlation lag.
        Default is False.
    **kwargs
        Additional keyword arguments passed to the plot function.

    Returns
    -------
    table : np.ndarray, shape (n_lags, 2)
        Array with lag values in column 0 and correlations in column 1.
    """

    from py_tools.plot import two_axis

    df = df_in[[var1, var2]].copy()

    if lags is None:
        assert max_lags >= 0
        assert max_leads >= 0
        lags = range(-max_lags, max_leads + 1)

    table = np.zeros((len(lags), 2))

    for ii, lag in enumerate(lags):
        lag_var = "L({0})_{1}".format(lag, var2)
        df[lag_var] = df[var2].shift(lag)

        corrs = df[[var1, lag_var]].corr()
        this_corr = corrs[var1][lag_var]

        table[ii, 0] = lag
        table[ii, 1] = this_corr

    if make_plot:
        best_lag = int(table[np.argmax(np.abs(table[:, 1])), 0])
        best_lag_var2 = "L({0})_{1}".format(best_lag, var2)
        two_axis(df, var1, best_lag_var2, **kwargs)

    return table


def fit_ar1(df_in, var, **kwargs):
    """Fit an AR(1) model to a variable.

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing the variable.
    var : str
        Variable name.
    **kwargs
        Additional keyword arguments passed to dt.regression.

    Returns
    -------
    fr : FullResults
        Regression results.
    """

    df = df_in[[var]].copy()
    df["L_" + var] = df[var].shift()

    return dt.regression(df, var, ["L_" + var], **kwargs)


def rolling_forecast(df_in, lhs, rhs=None, use_const=True, **kwargs):
    """Produce real-time rolling forecasts using expanding-window OLS.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    lhs : list of str
        Left-hand side variable names.
    rhs : list of str, optional
        Right-hand side variable names. Default is no additional regressors.
    use_const : bool, optional
        Include a constant term. Default is True.
    **kwargs
        Additional keyword arguments passed to rolling_forecast_internal.

    Returns
    -------
    forecast : np.ndarray, shape (Nt, Ny)
        Rolling forecasts.
    ix : np.ndarray of bool
        Sample selection index.
    """

    if rhs is None:
        rhs = []

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
    """Compute rolling forecasts using expanding-window OLS.

    Parameters
    ----------
    y : np.ndarray, shape (Nt, Ny)
        Dependent variable matrix.
    X : np.ndarray, shape (Nt, Nx)
        Regressor matrix.
    t_min : int, optional
        Minimum number of observations before forecasting begins.
        Default is Nx + 2.

    Returns
    -------
    forecast : np.ndarray, shape (Nt, Ny)
        One-step-ahead forecasts; NaN for the initial burn-in periods.
    """

    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

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

        # Update forecast
        XX_t += np.dot(x_t, x_t.T)
        Xy_t += np.dot(x_t, y_t.T)

    return forecast


def local_projection(
    df_in, y_var, shock_var, controls=None, periods=16, shock_lags=2, **kwargs
):
    """Estimate impulse responses via Jordà local projections.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    y_var : str
        Response variable name.
    shock_var : str
        Shock (impulse) variable name.
    controls : list of str, optional
        Additional control variable names. Default is None (no controls).
    periods : int, optional
        Number of forward periods to project. Default is 16.
    shock_lags : int, optional
        Number of lags of the shock to include as controls. Default is 2.
    **kwargs
        Additional keyword arguments passed to dt.regression.

    Returns
    -------
    coeffs : np.ndarray, shape (periods+1,)
        Impulse response coefficients at each horizon.
    se : np.ndarray, shape (periods+1,)
        Newey-West standard errors.
    """

    if controls is None:
        controls = []

    df = df_in[[y_var, shock_var] + controls].copy()

    rhs = [shock_var]

    for jj in range(1, shock_lags + 1):
        var_name = "L{0:d}_{1}".format(jj, shock_var)
        df[var_name] = df[shock_var].shift(jj)
        rhs.append(var_name)

    for var in [y_var] + controls:
        var_name = "L_" + var
        df[var_name] = df[var].shift()
        rhs.append(var_name)

    fr_list = []
    for tt in range(periods + 1):
        var_name = "F{0:d}_{1}".format(tt, y_var)
        df[var_name] = df[y_var].shift(-tt)
        fr_list.append(dt.regression(df, var_name, rhs, nw_lags=tt, **kwargs))

    coeffs = np.array([fr.results.params[1] for fr in fr_list])
    se = np.array([fr.results.HC0_se[1] for fr in fr_list])

    return (coeffs, se)


def add_lags_by_group(
    df,
    x_var,
    date_var,
    lag_list,
    group_id=None,
    group_vars=None,
    freq="Q",
    lag_str="lag",
    lead_str="lead",
    leads=False,
    zero_type=None,
):
    """Add lags (and optionally leads) of a variable within panel groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the relevant series.
    x_var : str
        Name of the variable to lag.
    date_var : str
        Name of the date variable column.
    lag_list : list of int
        Lag orders to compute (negative values treated as leads if leads=True).
    group_id : str, optional
        Column name identifying each panel unit. Used directly if provided.
    group_vars : list of str, optional
        Variables to group by if group_id is not provided.
    freq : str, optional
        Pandas frequency string for resampling. Default is 'Q'.
    lag_str : str, optional
        Prefix used for lagged variable names. Default is 'lag'.
    lead_str : str, optional
        Prefix used for lead variable names. Default is 'lead'.
    leads : bool, optional
        If True, negate lag_list values so they are treated as leads.
        Default is False.
    zero_type : {None, 'lag', 'lead'}, optional
        Controls naming when lag == 0. Default is None (uses x_var name).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with lagged/lead columns appended.
    """

    assert (group_id is not None) or (group_vars is not None)

    if group_id is None:
        assert "group_no" not in df
        group_id = "group_no"
        df[group_id] = df.groupby(group_vars).ngroup()

    tmp = (
        df[[x_var, date_var, group_id]]
        .set_index([group_id, date_var])
        .unstack(group_id)
        .resample(freq)
        .asfreq()
    )

    for lag in lag_list:
        if leads:
            lag *= -1

        if lag == 0:
            if zero_type is None:
                lag_var = x_var
            elif zero_type == "lead":
                lag_var = "{0}{1:d}_{2}".format(lead_str, -lag, x_var)
            elif zero_type == "lag":
                lag_var = "{0}{1:d}_{2}".format(lag_str, lag, x_var)

            if lag_var not in df:
                df[lag_var] = df[x_var]

        else:
            if lag < 0:
                lag_var = "{0}{1:d}_{2}".format(lead_str, -lag, x_var)
            else:
                lag_var = "{0}{1:d}_{2}".format(lag_str, lag, x_var)

            if lag_var not in df:
                shifted = tmp.shift(lag).stack(group_id)[x_var].rename(lag_var)
                df = df.join(shifted, on=[date_var, group_id])

    return df


def lowpass_filter(series, freq_ub, nlags=4):
    """Apply a symmetric low-pass filter to a time series.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input time series.
    freq_ub : float
        Upper cutoff frequency (in radians per period).
    nlags : int, optional
        One-sided number of lags in the filter kernel. Default is 4.

    Returns
    -------
    filtered : pd.Series or np.ndarray
        Low-pass filtered series.
    """

    b0 = freq_ub / np.pi
    filtered = b0 * series

    for hh in range(-nlags, nlags + 1):
        if hh != 0:
            bh = np.sin(hh * freq_ub) / (hh * np.pi)
            filtered += bh * series.shift(hh)

    return filtered


def bandpass_filter(
    series, period_lb=None, period_ub=None, freq_lb=None, freq_ub=None, nlags=4
):
    """Apply a symmetric band-pass filter to a time series.

    Accepts either period or frequency bounds (not both for the same bound).

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input time series.
    period_lb : float, optional
        Lower period bound (sets upper frequency cutoff).
    period_ub : float, optional
        Upper period bound (sets lower frequency cutoff).
    freq_lb : float, optional
        Lower frequency cutoff (radians per period).
    freq_ub : float, optional
        Upper frequency cutoff (radians per period).
    nlags : int, optional
        One-sided number of filter lags. Default is 4.

    Returns
    -------
    filtered : pd.Series or np.ndarray
        Band-pass filtered series.
    """

    if period_lb is not None:
        assert freq_ub is None
        freq_ub = 2.0 * np.pi / period_lb

    if period_ub is not None:
        assert freq_lb is None
        freq_lb = 2.0 * np.pi / period_ub

    assert (freq_lb is not None) or (freq_ub is not None)

    if freq_lb is not None:
        filtered_lb = lowpass_filter(series, freq_lb, nlags=nlags)
    else:
        filtered_lb = np.zeros(len(series))

    if freq_ub is not None:
        filtered_ub = lowpass_filter(series, freq_ub, nlags=nlags)
    else:
        filtered_ub = series

    filtered = filtered_ub - filtered_lb
    return filtered


def bandpass_filter_christiano(series, period_lb, period_ub, detrend=False):
    """Apply the Christiano-Fitzgerald band-pass filter.

    Parameters
    ----------
    series : np.ndarray, shape (Nt,)
        Input time series.
    period_lb : float
        Lower period bound.
    period_ub : float or None
        Upper period bound. If None, a low-pass filter is applied.
    detrend : bool, optional
        If True, remove the deterministic linear trend first. Default is False.

    Returns
    -------
    series_new : np.ndarray, shape (Nt,)
        Band-pass filtered series.
    """

    # assert period_lb > 1

    Nt = len(series)

    if detrend:
        dx = (series[-1] - series[0]) / (Nt - 1)
        series = series - np.arange(Nt) * dx

    if period_ub is None:
        a = 0.0
    else:
        a = 2.0 * np.pi / np.float64(period_ub)
    b = 2.0 * np.pi / np.float64(period_lb)

    grid = np.arange(1, Nt)
    B1 = (np.sin(grid * b) - np.sin(grid * a)) / (np.pi * grid)
    B0 = (b - a) / np.pi

    Bc = np.hstack((0.0, np.cumsum(B1)))
    B_til = -0.5 * B0 - Bc

    B_two_sided = np.hstack((B1[::-1], B0, B1))

    BP_mat = np.zeros((Nt, Nt))
    for tt in range(Nt):
        start_ix = Nt - 1 - tt
        BP_mat[tt, :] = B_two_sided[start_ix : start_ix + Nt]

    BP_mat[:, 0] += B_til
    BP_mat[:, -1] += B_til[::-1]

    series_new = BP_mat @ series
    return series_new


def chow_lin_V_default(a, N):
    """Compute the AR(1) error covariance matrix for the Chow-Lin method.

    Parameters
    ----------
    a : float
        Autocorrelation parameter.
    N : int
        Number of observations in the high-frequency series.

    Returns
    -------
    V : np.ndarray, shape (N, N)
        AR(1) covariance matrix with V[i, j] = a^|i-j|.
    """

    V = np.zeros((N, N))
    for tt in range(N):
        V[tt, :] = a ** np.abs(np.arange(N) - tt)

    return V


def chow_lin_inner(Y, Z, B, a, Vfcn=chow_lin_V_default):
    """Perform the inner GLS step of the Chow-Lin interpolation.

    Parameters
    ----------
    Y : np.ndarray, shape (Nt_coarse,)
        Target series at coarse frequency.
    Z : np.ndarray, shape (Nt_fine, k)
        Proxy series at fine frequency.
    B : np.ndarray, shape (Nt_fine, Nt_coarse)
        Aggregation matrix relating fine to coarse observations.
    a : float
        Autocorrelation parameter for the error structure.
    Vfcn : callable, optional
        Function computing the error covariance matrix given (a, N).
        Default is chow_lin_V_default.

    Returns
    -------
    bet_hat : np.ndarray, shape (k,)
        Estimated regression coefficients.
    X_hat : np.ndarray, shape (Nt_fine,)
        Interpolated high-frequency series.
    u_hat : np.ndarray, shape (Nt_fine,)
        Estimated high-frequency residuals.
    """

    N = Z.shape[0]
    V = Vfcn(a, N)

    ZB = Z.T @ B
    BV = B.T @ V
    BVB = BV @ B

    ZB_BVB_inv = np.linalg.solve(BVB.T, ZB.T).T

    bet_hat = np.linalg.solve(ZB_BVB_inv @ ZB.T, ZB_BVB_inv @ Y)

    X_hat = Z @ bet_hat + BV.T @ np.linalg.solve(BVB, Y - B.T @ (Z @ bet_hat))

    u_hat = X_hat - Z @ bet_hat

    return bet_hat, X_hat, u_hat


def chow_lin(Y, Z, B, Vfcn=chow_lin_V_default, a0=0.9, tol=1e-4):
    """Interpolate a coarse series to high frequency using the Chow-Lin method.

    Iterates the GLS inner step until the autocorrelation parameter converges.

    Parameters
    ----------
    Y : np.ndarray, shape (Nt_coarse,)
        Target series at coarse frequency.
    Z : np.ndarray, shape (Nt_fine, k)
        Proxy series at fine frequency.
    B : np.ndarray, shape (Nt_fine, Nt_coarse)
        Aggregation matrix relating fine to coarse observations.
    Vfcn : callable, optional
        Function computing the error covariance matrix. Default is chow_lin_V_default.
    a0 : float, optional
        Initial guess for the autocorrelation parameter. Default is 0.9.
    tol : float, optional
        Convergence tolerance for the autocorrelation parameter. Default is 1e-4.

    Returns
    -------
    X_hat : np.ndarray, shape (Nt_fine,)
        Interpolated high-frequency series.
    """

    a = a0
    done = False

    while not done:
        bet_hat, X_hat, u_hat = chow_lin_inner(Y, Z, B, a, Vfcn=Vfcn)

        a_new = stattools.acf(u_hat, nlags=1, fft=False)[1]

        a_err = np.abs(a_new - a)
        if a_err < tol:
            done = True

        a = a_new

    return X_hat


def interpolate_to_high_frequency(z, freq=4, A=None):
    """Interpolate a low-frequency series to high frequency using spline smoothing.

    Parameters
    ----------
    z : np.ndarray, shape (K,)
        Low-frequency series.
    freq : int, optional
        Number of high-frequency periods per low-frequency period. Default is 4.
    A : np.ndarray, shape (K, N), optional
        Aggregation matrix. If None, assumes averaging over freq periods.

    Returns
    -------
    x_star : np.ndarray, shape (K * freq,)
        Interpolated high-frequency series.
    """

    freq_inv = 1.0 / freq

    K = len(z)
    N = K * freq

    # If A is not provided, assume that z averages over freq periods
    if A is None:
        A = np.zeros((K, N))
        for kk in range(K):
            A[kk, freq * kk : freq * (kk + 1)] = freq_inv

    B = np.zeros((N, N))

    B[0, 0] = 1.0
    B[0, 1] = -1.0

    B[-1, -1] = 1.0
    B[-1, -2] = -1.0

    for jj in range(1, N - 1):
        B[jj, jj - 1] = -1.0
        B[jj, jj] = 2.0
        B[jj, jj + 1] = -1.0

    Phi = np.vstack((np.hstack((B, A.T)), np.hstack((A, np.zeros((K, K))))))

    c = np.hstack((np.zeros(N), z))

    x_lam = np.linalg.solve(Phi, c)
    x_star = x_lam[:N]

    return x_star
