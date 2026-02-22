import os
import pandas as pd
from py_tools import time_series as ts

from . import config
default_dir = config.base_dir() + 'fhfa/'
DATASET_NAME = "fhfa"
DESCRIPTION = "FHFA house price index dataset loader."
def load(dataset, all_transactions=True, reimport=False, data_dir=default_dir):
    """Load FHFA house price index data.

    Parameters
    ----------
    dataset : str
        Geographic level of the index. One of ``'metro'``/``'msa'``,
        ``'state'``, ``'county'``, or ``'zip3'``.
    all_transactions : bool, optional
        If ``True``, load the all-transactions index; if ``False``, load
        the purchase-only index.  Not all dataset/index combinations are
        supported (e.g. ``'metro'`` requires ``all_transactions=True``).
    reimport : bool, optional
        If ``True``, re-read the raw source file and overwrite any cached
        pickle.  If ``False``, use the cached pickle when available.
    data_dir : str, optional
        Path to the directory containing the FHFA source files.

    Returns
    -------
    pandas.DataFrame
        House price index data with a multi-level index (geographic unit,
        date) and ``hpi`` (and possibly additional) columns.

    Raises
    ------
    Exception
        When an unsupported combination of ``dataset`` and
        ``all_transactions`` is requested (e.g. ``dataset='metro'`` with
        ``all_transactions=False``, or ``dataset='county'`` with
        ``all_transactions=False``).
    """
    suffix = dataset
    if all_transactions:
        suffix += '_at'
    else:
        suffix += '_purch'
        
    basepath = data_dir + 'fhfa' + suffix
    pkl_file = basepath + '.pkl'
    
    if reimport or not os.path.exists(pkl_file):

        if dataset in ['metro', 'msa']:
    
            if not all_transactions: raise Exception
    
            # df = pd.read_csv(data_dir + 'HPI_AT_metro.csv')
            df = pd.read_csv(data_dir + 'HPI_AT_metro.csv', header=None, 
                             names=['MSA', 'code', 'year', 'qtr', 'hpi', 'unknown'])
            
            df['date'] = ts.date_from_qtr(df['year'], df['qtr'])
    
            # df['date'] = ts.date_from_qtr(df['year'], df['qtr'])
    
            df['hpi'] = pd.to_numeric(df['hpi'], errors='coerce')
    
            df = df.set_index(['date', 'MSA'])
            df = df.drop(columns=['year', 'qtr', 'unknown'])
            df = df.apply(pd.to_numeric, errors='coerce')
            # df['date'] = df['yr'].astype('str') + '-' + (3*df['qtr'] - 2).astype('str') + '-01'
            # df['date'] = pd.to_datetime(df['date'])
            # df = df.drop(columns=['yr', 'qtr', 'Warning'])
    
        elif dataset == 'state':
    
            if all_transactions:
                df = pd.read_csv(data_dir + 'HPI_AT_state.txt', names=['state', 'year', 'qtr', 'hpi'],
                                 sep='\t')
            else:
                df = pd.read_csv(data_dir + 'HPI_PO_state.txt', sep='\t')
                df = df.drop(columns=['Warning'])
    
            df['date'] = ts.date_from_qtr(df['year'], df['qtr'])
            df = df.set_index(['state', 'date'])
            df = df.apply(pd.to_numeric, errors='coerce')
    
        elif dataset == 'county':
    
            if all_transactions:
                
                df = pd.read_excel(data_dir + 'HPI_AT_BDL_county.xlsx', skiprows=6)
                df = df.rename({var : var.lower() for var in df.columns}, axis=1)
    
                for var in df.columns:
                    if var not in ['state', 'county']:
                        df[var] = pd.to_numeric(df[var], errors='coerce')
                        
                df = df.rename({
                    'county' : 'county_name', 
                    'fips code' : 'fips',
                    'hpi' : 'hpi',
                    'hpi with 1990 base' : 'hpi_1990_base',
                    'hpi with 2000 base' : 'hpi_2000_base',
                    'annual change (%)' : 'annual_change_pct',
                    }, axis=1)
                
                df['date'] = ts.date_from_year(df['year'])
                df = df.set_index(['fips', 'date'])
            else:
                raise Exception
    
        elif dataset == 'zip3':
            
            print("MAKE SURE UP TO DATE WITH INDEXING")
    
            assert all_transactions
    
            filename = 'HPI_AT_3zip'
            excel_file = data_dir + filename + '.xlsx'
            df = pd.read_excel(excel_file, skiprows=4)
            df['date'] = ts.date_from_qtr(df['Year'], df['Quarter'])
            df = df.drop(columns=['Index Type']).rename(columns={'Index (NSA)' : 'hpi'})
    
            df = df.rename({'Three-Digit ZIP Code' : 'zip3'}, axis=1)
            df = df.rename({var : var.lower() for var in df.columns}, axis=1)
            
            df = df.set_index(['zip3', 'date'])
            
        df.to_pickle(pkl_file)
    
    else:
    
        df = pd.read_pickle(pkl_file)
        
    if reimport:
        stata_file = basepath + '.dta'
        df.to_stata(stata_file)

    return df
