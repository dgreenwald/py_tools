import pandas as pd

from . import config
default_dir = config.base_dir() + 'compustat/'

# def load(dataset='quarterly', data_dir=default_dir):
DATASET_NAME = "compustat"
DESCRIPTION = "Compustat dataset loader (SAS chunk readers)."
def load(data_dir=None, **kwargs):
    """Load Compustat data by chunk.

    Parameters are passed to either ``read_by_chunk`` (default) or
    ``read_single_chunk`` if ``single_chunk=True``.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing Compustat SAS files. Defaults to the
        package-level ``default_dir`` when not provided.
    **kwargs
        Additional keyword arguments forwarded to ``read_by_chunk`` or
        ``read_single_chunk``.  Pass ``single_chunk=True`` to read only
        the first chunk, or ``name`` (str) to specify the SAS file stem,
        ``keep_cols`` (list of str) to select columns, and ``chunksize``
        (int) to control the SAS reader chunk size.

    Returns
    -------
    pandas.DataFrame
        Compustat data loaded from the specified SAS file.
    """
    if data_dir is not None:
        kwargs.setdefault('data_dir', data_dir)

    single_chunk = kwargs.pop('single_chunk', False)
    if single_chunk:
        return read_single_chunk(**kwargs)
    return read_by_chunk(**kwargs)


def read_single_chunk(name, data_dir=default_dir, keep_cols=None, chunksize=10000):
    """Read only the first chunk of a Compustat SAS file.

    Opens the SAS file at ``data_dir + name + '.sas7bdat'`` with the
    given chunk size and returns the first chunk only.  A notice is
    printed to stdout to warn that the full file is not loaded.

    Parameters
    ----------
    name : str
        Stem of the SAS file to read (without the ``.sas7bdat`` extension).
    data_dir : str, optional
        Directory containing the SAS file.  Defaults to the package-level
        ``default_dir``.
    keep_cols : list of str, optional
        Subset of column names to retain.  Columns absent from the file
        are silently dropped.  If ``None``, all columns are kept.
    chunksize : int, optional
        Number of rows per chunk passed to ``pandas.read_sas``.
        Defaults to 10 000.

    Returns
    -------
    pandas.DataFrame
        First chunk of the SAS file, optionally column-filtered.
    """
    filepath = data_dir + name + '.sas7bdat'
    
    print("NOTE: NOT LOADING FULL FILE")
    reader = pd.read_sas(filepath, chunksize=chunksize)
    for df in reader:
        break
    
    if keep_cols is not None:
        df = df[pd.Index.intersection(df.columns, keep_cols)] 
        
    return df

def read_by_chunk(name, data_dir=default_dir, keep_cols=None, chunksize=10000):
    """Read all chunks of a Compustat SAS file and concatenate them.

    Iterates over every chunk produced by ``pandas.read_sas`` and
    returns a single concatenated DataFrame.

    Parameters
    ----------
    name : str
        Stem of the SAS file to read (without the ``.sas7bdat`` extension).
    data_dir : str, optional
        Directory containing the SAS file.  Defaults to the package-level
        ``default_dir``.
    keep_cols : list of str, optional
        Subset of column names to retain in each chunk.  Columns absent
        from the file are silently dropped.  If ``None``, all columns are
        kept.
    chunksize : int, optional
        Number of rows per chunk passed to ``pandas.read_sas``.
        Defaults to 10 000.

    Returns
    -------
    pandas.DataFrame
        Full contents of the SAS file, optionally column-filtered.
    """
    filepath = data_dir + name + '.sas7bdat'

    reader = pd.read_sas(filepath, chunksize=chunksize)
    if keep_cols is None:
        return pd.concat([chunk for chunk in reader])
    return pd.concat([
        chunk[pd.Index.intersection(chunk.columns, keep_cols)]
        for chunk in reader
    ])
