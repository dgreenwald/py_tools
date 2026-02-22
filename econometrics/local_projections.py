from py_tools import data as dt, plot as pl
import numpy as np


def lag_var(df, var, lag):
    """
    Add a lagged or leaded variable to a DataFrame in place.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to modify.
    var : str
        Column name of the variable to lag/lead.
    lag : int
        Number of periods to shift.  Positive values create lags (column named
        ``L{lag}_{var}``); negative values create leads (column named
        ``F{|lag|}_{var}``).  Zero raises an exception.

    Returns
    -------
    df : pandas.DataFrame
        The same DataFrame with the new column added.

    Raises
    ------
    Exception
        If ``lag`` is zero.
    """
    if lag > 0:
        df['L{0}_{1}'.format(lag, var)] = df[var].shift(lag)
    elif lag < 0:
        df['F{0}_{1}'.format(-lag, var)] = df[var].shift(lag)
    else:
        raise Exception

    return df

def add_lags(df, var, max_lags):
    """
    Add multiple lags of a variable to a DataFrame.

    Calls :func:`lag_var` for lags 1 through ``max_lags`` inclusive.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to modify in place.
    var : str
        Column name of the variable to lag.
    max_lags : int
        Maximum lag order to add.

    Returns
    -------
    df : pandas.DataFrame
        The same DataFrame with lag columns ``L1_{var}``, ...,
        ``L{max_lags}_{var}`` added.
    """
    for lag in range(1, max_lags + 1):
        df = lag_var(df, var, lag)

    return df

def add_leads(df, var, max_leads):
    """
    Add multiple leads of a variable to a DataFrame.

    Calls :func:`lag_var` for leads 1 through ``max_leads`` inclusive.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to modify in place.
    var : str
        Column name of the variable to lead.
    max_leads : int
        Maximum lead order to add.

    Returns
    -------
    df : pandas.DataFrame
        The same DataFrame with lead columns ``F1_{var}``, ...,
        ``F{max_leads}_{var}`` added.
    """
    for lead in range(1, max_leads + 1):
        df = lag_var(df, var, -lead)

    return df

def formula_lags(var, max_lags):
    """
    Generate a patsy/statsmodels formula fragment for lags of a variable.

    Parameters
    ----------
    var : str
        Variable name (as it appears in the DataFrame after :func:`add_lags`).
    max_lags : int
        Number of lags to include.

    Returns
    -------
    str
        A formula fragment of the form ``'+ L1_{var} + L2_{var} + ...'``.
        Returns an empty string when ``max_lags`` is zero.
    """
    formula = ''
    for lag in range(1, max_lags + 1):
        formula += '+ L{0}_{1}'.format(lag, var)

    return formula

def get_formula(horizon, y_var, shock_var, control_vars, fe_vars, shock_lags,
                y_lags, control_lags):
    """
    Build a regression formula string for a local projection at a given horizon.

    Parameters
    ----------
    horizon : int
        Forecast horizon.  Positive values use the ``F{horizon}_{y_var}``
        lead column as the dependent variable; zero uses ``y_var`` directly.
    y_var : str
        Name of the outcome variable.
    shock_var : str
        Name of the shock variable of interest (appears contemporaneously on
        the RHS).
    control_vars : list of str
        Names of additional linear control variables (lags only).
    fe_vars : list of str
        Names of fixed-effect variables (included as ``C(var)``).
    shock_lags : int
        Number of lags of ``shock_var`` to include beyond the contemporaneous
        term.
    y_lags : int
        Number of lags of ``y_var`` to include.
    control_lags : dict
        Mapping from control variable name to the number of lags to include.
        Variables absent from this dict default to 1 lag.

    Returns
    -------
    str
        A patsy-compatible formula string suitable for
        ``statsmodels.formula.api``.
    """
    ############################################################################
    # LHS
    ############################################################################

    if horizon > 0:
        formula = 'F{0}_{1} ~'.format(horizon, y_var)
    else:
        formula = '{0} ~'.format(y_var)

    ############################################################################
    # RHS
    ############################################################################

    # shocks
    formula += ' ' + shock_var
    formula += formula_lags(shock_var, shock_lags)

    # y_var
    formula += formula_lags(y_var, y_lags)

    # controls
    for var in control_vars:
        max_lags = control_lags.get(var, 1)
        formula += formula_lags(var, max_lags)

    for var in fe_vars:
        formula += ' C({0})'.format(var)

    return formula

def estimate(df_in, y_var, shock_var, control_vars=None, fe_vars=None,
             shock_lags=2, y_lags=1, periods=20, control_lags=None):
    """
    Estimate impulse responses via local projections.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input data.
    y_var : str
        Name of the outcome (LHS) variable.
    shock_var : str
        Name of the shock variable of interest.
    control_vars : list of str, optional
        Names of additional linear control variables.  Defaults to ``[]``.
    fe_vars : list of str, optional
        Names of fixed-effect variables.  Defaults to ``[]``.
    shock_lags : int, optional
        Number of lags of ``shock_var`` to include beyond the contemporaneous
        term.  Default is 2.
    y_lags : int, optional
        Number of lags of ``y_var`` to include.  Default is 1.
    periods : int, optional
        Maximum forecast horizon (number of local-projection regressions to
        run).  Default is 20.
    control_lags : dict, optional
        Mapping from control variable name to lag order.  Variables absent
        from the dict default to 1 lag.

    Returns
    -------
    fr_list : list
        List of regression result objects (one per horizon).
    x : ndarray of shape (periods,)
        Point estimates of the impulse-response coefficients at each horizon.
    se : ndarray of shape (periods,)
        HC0 standard errors for each horizon's impulse-response coefficient.
    """

    if control_vars is None: control_vars = []
    if fe_vars is None: fe_vars = []
    if control_lags is None: control_lags = {}

    # Copy relevant dataframe columns
    unique_controls = [var for var in control_vars if var not in [y_var, shock_var]]
    
    all_vars = list(set([y_var] + [shock_var] + unique_controls + fe_vars))
    df = df_in[all_vars].copy()

    # Add lags
    df = add_lags(df, shock_var, shock_lags)
    df = add_lags(df, y_var, y_lags)
    for var in unique_controls:
        df = add_lags(df, var, control_lags.get(var, 1))

    # Add leads
    df = add_leads(df, y_var, periods)

    # Run regressions
    x = np.zeros(periods)
    se = np.zeros(periods)
    fr_list = []

    for jj in range(periods):

        formula = get_formula(jj, y_var, shock_var, unique_controls, fe_vars,
                              shock_lags, y_lags, control_lags)

        fr_list.append(dt.formula_regression(df, formula, nw_lags=jj))

    for jj in range(periods):

        x[jj] = fr_list[jj].results.params[1]
        se[jj] = fr_list[jj].results.HC0_se[1]
        
    return fr_list, x, se

class LocalProjection:
    """
    Local projection model for estimating impulse-response functions.

    Wraps :func:`estimate` for one or more outcome variables and provides
    convenience methods for estimation and plotting.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input data used for all estimations.
    labels : dict, optional
        Mapping from variable names to human-readable display labels used in
        plots and titles.

    Attributes
    ----------
    df : pandas.DataFrame or None
    labels : dict
    y_var_list : list of str
        Set after :meth:`estimate` is called.
    shock_var : str
        Set after :meth:`estimate` is called.
    x : ndarray of shape (n_outcomes, periods)
        Stacked impulse-response point estimates; set after :meth:`estimate`.
    se : ndarray of shape (n_outcomes, periods)
        Stacked standard errors; set after :meth:`estimate`.
    """

    def __init__(self, df=None, labels=None):
        """
        Parameters
        ----------
        df : pandas.DataFrame, optional
            Input data.
        labels : dict, optional
            Variable-name-to-display-label mapping.
        """
        self.df = df
        self.labels = labels or {}
        
    def estimate(self, y_var_list, shock_var, **kwargs):
        """
        Estimate local projections for each outcome variable.

        Calls the module-level :func:`estimate` function for every variable in
        ``y_var_list`` and stores results in ``self.x``, ``self.se``, and
        ``self.fr_list_all``.

        Parameters
        ----------
        y_var_list : list of str
            Names of outcome variables to estimate impulse responses for.
        shock_var : str
            Name of the shock variable.
        **kwargs
            Additional keyword arguments forwarded to :func:`estimate`
            (e.g. ``shock_lags``, ``y_lags``, ``periods``).
        """
        self.y_var_list = y_var_list
        self.shock_var = shock_var
        
        self.var_titles = [self.labels.get(y_var, y_var) for y_var in self.y_var_list]
        self.shock_title = self.labels.get(shock_var, shock_var)
        
        self.fr_list_all = {}
        self.x_all = {}
        self.se_all = {}
        
        for y_var in self.y_var_list:
            self.fr_list_all[y_var], self.x_all[y_var], self.se_all[y_var] \
                = estimate(self.df, y_var, self.shock_var, **kwargs)
        
        self.x = np.vstack([self.x_all[y_var] for y_var in self.y_var_list])
        self.se = np.vstack([self.se_all[y_var] for y_var in self.y_var_list])
        
    def plot(self, **kwargs):
        """
        Plot impulse-response functions.

        Calls ``py_tools.plot.projection`` using the estimated coefficients
        and standard errors stored on this object.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to ``py_tools.plot.projection``
            (e.g. ``figsize``, ``nrows``, ``ncols``, ``plotpath``).
        """
        pl.projection(self.x, self.se, self.var_titles, self.shock_title, shock_name=self.shock_var,
                      **kwargs)
