from py_tools import data as dt, plot as pl
import numpy as np


def lag_var(df, var, lag):

    if lag > 0:
        df['L{0}_{1}'.format(lag, var)] = df[var].shift(lag)
    elif lag < 0:
        df['F{0}_{1}'.format(-lag, var)] = df[var].shift(lag)
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
        formula += '+ L{0}_{1}'.format(lag, var)

    return formula

def get_formula(horizon, y_var, shock_var, control_vars, fe_vars, shock_lags,
                y_lags, control_lags):

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
    """Local projection object"""
    
    def __init__(self, df=None, labels={}):
        
        self.df = df
        self.labels = labels
        
    def estimate(self, y_var_list, shock_var, **kwargs):
        
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
        
        pl.projection(self.x, self.se, self.var_titles, self.shock_title, shock_name=self.shock_var,
                      **kwargs)
