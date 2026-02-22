import pandas as pd

from . import config
default_dir = config.base_dir() + 'dfa/'
DATASET_NAME = "dfa"
DESCRIPTION = "Federal Reserve Distributional Financial Accounts (DFA) dataset loader."
def load(dataset, levels=True, data_dir=default_dir, **kwargs):
    """Load pre-packaged set of DFA variables from CSV.

    Parameters
    ----------
    dataset : str
        Name of the DFA dataset to load (e.g. ``'wealth-shares'``).
    levels : bool, optional
        If ``True``, load the levels file; if ``False``, load the shares file.
    data_dir : str, optional
        Directory containing the DFA CSV files.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the requested DFA variables.
    """
    
    filename = 'dfa-' + dataset + '-'
    
    if levels:
        filename += 'levels'
    else:
        filename += 'shares'
        
    filename += '.csv'
    
    return pd.read_csv(data_dir + filename, **kwargs)