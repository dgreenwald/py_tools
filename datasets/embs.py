import numpy as np
import pandas as pd

from . import config
default_dir = config.base_dir() + 'frm/embs/by_coupon/'

# data_dir = '/home/dan/Dropbox/data/frm/'
# embs_dir = data_dir + 'embs/by_coupon/'
DATASET_NAME = "embs"
DESCRIPTION = "eMBS mortgage prepayment dataset loader."
def load(embs_dir=default_dir):
    """Load eMBS mortgage prepayment data by merging per-variable pickle files.

    Reads CPR, RPB, Issuance, WAC, WAM, and WALA pickle files from
    *embs_dir* and merges them on ``coupon``, ``orig_date``, and ``date``.

    Parameters
    ----------
    embs_dir : str, optional
        Directory containing the eMBS ``embs_<var>_clean.pkl`` files.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with columns for each loaded variable indexed by
        coupon, origination date, and observation date.
    """

    var_list = ['CPR', 'RPB', 'Issuance', 'Wac', 'Wam', 'Wala']

    for i_var, var in enumerate(var_list):
        df_new = pd.read_pickle(embs_dir + 'embs_{0}_clean.pkl'.format(var.lower()))
        if i_var == 0:
            df = df_new
        else:
            df = pd.merge(df, df_new, on = ['coupon', 'orig_date', 'date'])

    return df

def aggregate(df_in, weight='rpb'):
    """Compute weighted aggregate CPR, WAC, WAM, and WALA across coupons and vintages.

    For each date, each variable is aggregated as a weighted average using the
    column specified by *weight*.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame containing at least the columns ``cpr``, ``wac``,
        ``wam``, ``wala``, and the weight column.
    weight : str, optional
        Column name to use as the aggregation weight (e.g. ``'rpb'``).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with weighted-average columns
        ``cpr``, ``wac``, ``wam``, and ``wala``.
    """

    var_list = ['cpr', 'wac', 'wam', 'wala']

    df = df_in.loc[pd.notnull(df_in[weight]), :].copy()
    
    for var in var_list:
        df['wtd_' + var] = df[weight] * df[var]

    wtd_list = ['wtd_' + var for var in var_list]
    keep_list = ['date', weight] + wtd_list

    columns = {var : var + '_total' for var in [weight] + wtd_list}

    df_agg = df[keep_list].groupby(['date']).sum().rename(columns=columns) 
    for var in var_list:
        df_agg[var] = df_agg['wtd_' + var + '_total'] / df_agg[weight + '_total']

    return df_agg[var_list] 

def resample_cpr(cpr, freq='QS'):
    """Convert a monthly CPR series to the specified resampling frequency.

    Converts monthly conditional prepayment rates to a lower frequency by
    averaging the implied monthly survival rates and converting back to CPR.

    Parameters
    ----------
    cpr : pandas.Series
        Monthly conditional prepayment rate series (values in percent, e.g.
        ``6.0`` for 6 %).
    freq : str, optional
        Pandas resampling frequency code (e.g. ``'QS'`` for quarter-start).

    Returns
    -------
    pandas.Series
        CPR series resampled to *freq* (values in percent).
    """

    log_survival = np.log((1.0 - 0.01 * cpr) ** (1.0 / 12.0))
    log_survival_resamp = log_survival.resample(freq).mean()
    return 100.0 * (1.0 - np.exp(12.0 * log_survival_resamp))
