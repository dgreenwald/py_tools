from . import data as dt

def lag_var(df, var, lag):

    if lag > 0:
        df['L{0}_{1}'.format(lag, y_var)] = df[var].shift(lag)
    elif lag < 0:
        df['F{0}_{1}'.format(-lag, y_var)] = df[var].shift(lag)
    else:
        raise Exception

    return df

def add_lags(df, var, max_lags):

    for lag in range(1, max_lags + 1):
        df = lag_var(df, var, lag)

    return df

def add_leads(df, var, max_leads):

    for lead in range(1, max_leads + 1):
        df = lag_var(df, var, -lead)

    return df

def formula_lags(var, max_lags):

    formula = ''
    for lag in range(1, max_lags + 1):
        formula += ' L{0}_{1}'.format(lag, var)

    return formula

def get_formula(horizon, y_var, shock_var, control_vars, fe_vars, shock_lags,
                y_lags, control_lags):

    ############################################################################
    # LHS
    ############################################################################

    if horizon > 0:
        formula = 'F{0}_{1} ~'.format(y_var, horizon)
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

def estimate(df_in, y_var, shock_var, control_vars=[], fe_vars=[],
             shock_lags=2, y_lags=1, periods=20, control_lags={}):
    """
    df_in: pandas dataframe
    y_var: LHS variable
    shock_var: shock of interest
    control_vars: list of linear control variable names
    fe_vars: fixed effect variables
    shock_lags: how many lags of the shock to include (beyond shock_t)
    y_lags: how many lags of the LHS to include (1 means y_{t-1} only)
    periods: maximum horizon
    control_lags: dict containing number of lags for each linear control
    """

    # Copy relevant dataframe columns
    all_vars = list(set([y_var] + [shock_var] + control_vars + fe_vars))
    df = df_in[all_vars].copy()

    # Add lags
    df = add_lags(df, shock_var, shock_lags)
    df = add_lags(df, y_var, y_lags)
    for var in control_vars:
        df = add_lags(df, var, control_lags.get(var, 1))

    # Add leads
    df = add_leads(df, y_var, periods)

    # Run regressions
    x = np.zeros(periods)
    se = np.zeros(periods)
    fr_list = []

    for jj in range(periods):

        formula = get_formula(jj, y_var, shock_var, control_vars, fe_vars,
                              shock_lags, y_lags, control_lags)

        fr_list.append(dt.formula_regression(df, formula, nw_lags=jj))

    for jj in range(periods):

        x[jj] = fr_list[jj].results.params[1]
        se[jj] = fr_list[jj].results.HC0_se[1]
        
    return x, se, fr_list
