import pandas as pd
import warnings
from py_tools.datasets import misc, ramey

from . import config
default_dir = config.base_dir()
DATASET_NAME = "shocks"
DESCRIPTION = "Merged macro shock series loader (TFP, monetary, EBP, and related shocks)."
def load(keep_list=None, master_dirs=None):
    """Load merged macro shock series from multiple sources.

    Combines TFP shocks (Fernald utilization-adjusted, Ramey technology),
    monetary shocks (Romer-Romer, Barakchian-Crowe, Swanson-Zubairy),
    and excess bond premium (EBP) shocks from Gertler-Karadi and
    Eberly-Phillips-Shao into a single quarterly DataFrame.

    Parameters
    ----------
    keep_list : list of str or None, optional
        Column names to retain in the output. If ``None``, defaults to
        ``['ford_tfp', 'e_tfp', 'jpt_tfp', 'mn_tfp_p', 'mn_tfp_s',
        'romer', 'bc_mon', 'sz_mon', 'ep_hat_ebp', 'ep_hat_all_ebp']``.
    master_dirs : dict, optional
        Override directory paths. If ``'base'`` key is absent, the default
        base directory from config is used.

    Returns
    -------
    pandas.DataFrame
        Quarterly time-indexed DataFrame of the requested shock series.
    """
    if master_dirs is not None:
        warnings.warn(
            "master_dirs is deprecated and will be removed in a future version. "
            "Set the PY_TOOLS_DATA_DIR environment variable instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        master_dirs = {}
    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = default_dir
        # home_dir = os.environ['HOME']
        # dirs['base'] = home_dir + '/Dropbox/data/'
        
    if keep_list is None:
        keep_list = ['ford_tfp', 'e_tfp', 'jpt_tfp', 'mn_tfp_p', 'mn_tfp_s',
                     'romer', 'bc_mon', 'sz_mon', 'ep_hat_ebp', 'ep_hat_all_ebp']

    df = misc.load('fernald')[['dtfp_util']]
    
    df = df.join([
            ramey.load('technology'), 
            misc.load('gertler_karadi').resample('QS').mean(),
            ramey.load('monetary').rename(columns={
                'RRSHOCK' : 'romer',
                'BCSHOCK' : 'bc_mon',
                'szshock' : 'sz_mon',
            }).resample('QS').sum(),
            0.01 * 3.0 * pd.read_pickle(dirs['base'] + 'misc/ep_hat_ebp.pkl').resample('QS').mean(),
            ], how='outer')
    
    df['e_tfp'] = df['dtfp_util'] / 400
    
    df = df[keep_list]

    return df
