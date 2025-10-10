import os
import pandas as pd
from py_tools.time_series import date_index
from py_tools.datasets import misc, ramey

from . import defaults
default_dir = defaults.base_dir()

def load(keep_list=None, master_dirs={}):
    """Load data from Ramey shocks file"""

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
