import numpy as np
import pandas as pd
import os

from . import config, misc
default_dir = config.base_dir() + 'intraday/'

data_dir = default_dir

#def get_month_start(date):
#    
#    ix_start = date.dt.is_month_start
#    month_start = date.copy()
#    month_start[~ix_start] = date[~ix_start] - pd.tseries.offsets.MonthEnd(1)
#    
#    return month_start
DATASET_NAME = "intraday"
DESCRIPTION = "Intraday monetary policy shock dataset loader."
def load(data_dir=default_dir, reimport=False, data_vintage=2019):
    """Load intraday monetary policy shock data resampled to monthly frequency.

    Reads shock data from an Excel file, computes timing-adjusted versions of
    each shock based on how far into the month the FOMC meeting occurred, then
    resamples to monthly frequency by combining the current-month remainder
    with the previous month's carry-over.  Results are cached as a pickle file
    so subsequent calls skip the Excel import unless ``reimport=True``.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the source Excel file and the pickle cache.
        Defaults to the configured intraday data directory.
    reimport : bool, optional
        When ``True``, re-read the Excel file and overwrite the pickle cache
        even if a cached file already exists.  Defaults to ``False``.
    data_vintage : int, optional
        Vintage of the dataset to load.  ``2019`` reads ``tight.xls``; ``2012``
        reads ``tight_June2012.xls``.  Any other value raises an exception.
        Defaults to ``2019``.

    Returns
    -------
    pandas.DataFrame
        Monthly time series of monetary policy shocks with a
        ``pandas.DatetimeIndex`` at month-start frequency.

    Raises
    ------
    Exception
        When ``data_vintage`` is not ``2019`` or ``2012``.
    """
    pkl_file = data_dir + 'intraday_{:d}.pkl'.format(data_vintage)
    
    if reimport or (not os.path.exists(pkl_file)):
    
        if data_vintage == 2019:
            df = pd.read_excel(data_dir + 'tight.xls')
        elif data_vintage == 2012:
            df = pd.read_excel(data_dir + 'tight_June2012.xls')
        else:
            raise Exception
    #    
        df = df.rename(columns={var : var.lower() for var in df.columns})
        shock_list = [var for var in df.columns if var != 'date']
        df['date'] = pd.to_datetime(df['date'])
        
        df['end_date'] = df['date'] + pd.tseries.offsets.MonthEnd(0)
        df['prev_end_date'] = df['date'] - pd.tseries.offsets.MonthEnd(1)
    #    df['start_date'] = get_month_start(df['date'])
        df['days_from_start'] = df['date'] - df['prev_end_date']
        df['length_of_month'] = df['end_date'] - df['prev_end_date']
        df['shock_adj'] = (df['length_of_month'] - df['days_from_start'] + pd.Timedelta('1 day')) / df['length_of_month']
        df['shock_adj_next'] = 1.0 - df['shock_adj']
    
        # Adjust for remaining length of month
        for var in shock_list:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df[var + '_adj'] = df[var] * df['shock_adj']
            df[var + '_adj_next'] = df[var] * df['shock_adj_next']
            df[var + 'abs'] = np.abs(df[var])
            
        shock_adj_list = [var + '_adj' for var in shock_list]
        shock_adj_list += [var + '_adj_next' for var in shock_list]
        df_m = df.set_index('date')[shock_list + shock_adj_list].resample('MS').sum()
        
        # Combine remainder of current month + leftover from last month
        for var in shock_list:
            df_m[var] = df_m[var + '_adj'] + df_m[var + '_adj_next'].shift()
            
        for var in shock_list:
            ix = pd.notnull(df[var])
            if np.any(ix):
                first_date = df.loc[ix, 'prev_end_date'].values[0]
                df_m.loc[:first_date, var] = np.nan
                
        df_m[shock_list].to_pickle(pkl_file)
        
    df_out = pd.read_pickle(pkl_file)
    
    return df_out