import numpy as np
import pandas as pd
# import pickle
# import pdb
import os
from zipfile import ZipFile

from py_tools import in_out

DEFAULT_FANNIE_DIR = os.environ['HOME'] + '/data/fannie/'
DEFAULT_FREDDIE_DIR = os.environ['HOME'] + '/data/freddie/'

def cat(num):
    return list(range(1, num+1))

def to_float(df, var):
    if df[var].dtype == 'object':
        df[var] = pd.to_numeric(df[var], errors='coerce').astype(np.float64)
    return df

def load(year, q, dataset, reimport=False, data_dir=None, columns=None,
         compression='GZIP', **kwargs):

    default_data_dirs = {
        'fannie_acquisition' : DEFAULT_FANNIE_DIR,
        'freddie_acquisition' : DEFAULT_FREDDIE_DIR,
        'freddie_performance' : DEFAULT_FREDDIE_DIR,
    }

    assert dataset in default_data_dirs

    # Set data directory if not specified
    if data_dir is None:
        data_dir = default_data_dirs[dataset]

    # Set up storage directory
    parquet_dir = data_dir + 'storage/'
    in_out.make_dir(parquet_dir)

    # Load parquet file directly
    parquet_file = '{0}{1}_{2:d}Q{3:d}.parquet'.format(parquet_dir, dataset, year, q)
    if os.path.exists(parquet_file) and (not reimport):
        return pd.read_parquet(parquet_file, columns=columns)

    # Otherwise, reimport from txt file
    if dataset == 'fannie_acquisition':
        df = import_fannie_acquisition(year, q, data_dir=data_dir, **kwargs)
    elif dataset == 'freddie_acquisition':
        df = import_freddie_acquisition(year, q, data_dir=data_dir, **kwargs)
    elif dataset == 'freddie_performance':
        df = import_freddie_performance(year, q, data_dir=data_dir, **kwargs)
    else:
        raise Exception

    # Convert variables from string to numeric
    if df is not None:

        numerical_list = ['orig_int_rate', 'orig_upb', 'orig_term', 'orig_ltv',
                          'orig_cltv', 'orig_dti', 'credit_score']
        for col in df.columns:
            if col in numerical_list:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        date_list = ['orig_date', 'first_pay_date']
        for col in df.columns:
            if col in date_list:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Save as parquet
        df.to_parquet(parquet_file, compression=compression)

    return df

def load_data(year, q, dataset, **kwargs):
    if dataset == 'fannie':
        df = load_fannie(year, q, **kwargs)
    elif dataset == 'freddie':
        df = load_freddie(year, q, **kwargs)
    else:
        print("Invalid data set")
        return None

    if df is not None:
        numerical_list = ['orig_int_rate', 'orig_upb', 'orig_term', 'orig_ltv', 'orig_cltv',
                          'orig_dti', 'credit_score']
        for col in df.columns:
            if col in numerical_list:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def import_fannie_acquisition(year, q, data_dir=DEFAULT_FANNIE_DIR, **kwargs):

    col_names = [
        'loan_id', 'channel', 'seller_name', 'orig_int_rate', 'orig_upb', 
        'orig_term', 'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 
        'n_borrowers', 'orig_dti', 'credit_score', 'first_time_flag', 'loan_purpose',
        'prop_type', 'n_units', 'occ_status', 'prop_state', 'zip3', 
        'mi_pct', 'prod_type', 'co_borr_credit_score', 'mi_type', 'reloc'
    ]

    filepath = data_dir + 'Acquisition_{0}Q{1}.txt'.format(year, q)
    compression=None

    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, sep='|', compression=compression,
                           names=col_names, **kwargs)
        return df
    else:
        return None

def import_freddie_acquisition(year, q, data_dir=DEFAULT_FREDDIE_DIR):

    col_names = ['credit_score', 'first_pay_date', 'first_time_flag', 'maturity_date', 'msa', 
            'mi_pct', 'n_units', 'occ_status', 'orig_cltv', 'orig_dti',
            'orig_upb', 'orig_ltv', 'orig_int_rate', 'channel', 'prepay_pen_flag',
            'prod_type', 'prop_state', 'prop_type', 'zip3', 'loan_id',
            'loan_purpose', 'orig_term', 'n_borrowers', 'seller_name', 'servicer_name']

    filename = 'historical_data1_Q{0}{1}'.format(q, year)
    zipname = data_dir + '/' + filename + '.zip'
    txtname = filename + '.txt'

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep='|', names=col_names, index_col=False, **kwargs)
            return df
    else:
        return None

def import_freddie_performance(year, q, data_dir=DEFAULT_FREDDIE_DIR):

    col_names = ['loan_id', 'asof_date', 'current_upb', 'delinq', 'loan_age', 'months_left',
                 'repurchase', 'modification', 'zero_balance', 'zero_balance_date',
                 'current_rate', 'current_deferred_upb', 'last_due_date', 'mi_recoveries',
                 'net_sales_proceeds', 'non_mi_recoveries', 'expenses', 'legal_costs',
                 'maintenance_costs', 'taxes_and_insurance', 'misc_expenses', 'actual_loss',
                 'modification_cost', 'step_modification_flag', 'deferred_pay_mod',
                 'estimated_ltv']

    filename = 'historical_data1_time_Q{0}{1}'.format(q, year)
    zipname = '{0}/historical_data1_Q{1}{2}.zip'.format(data_dir, q, year)
    txtname = filename + '.txt'

    if os.path.isfile(zipname):

        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep='|', names=col_names, index_col=False, **kwargs)

        # Adjust object data types
        df['loan_id'] = df['loan_id'].astype(str)

        for var in ['repurchase', 'modification', 'step_modification_flag']:
            df[var] = df[var].astype('category')

        # Update delinquency to account for REOs
        df['reo'] = df['delinq'] == 'R'
        df.loc[df['reo'], 'delinq'] = -1
        df['delinq'] = pd.to_numeric(df['delinq'], errors='coerce')
        df.loc[pd.isnull(df['delinq']), 'delinq'] = -2
        df['delinq'] = df['delinq'].astype(np.int64)

        # Update net_sales_proceeds codes
        df['net_sales_proceeds_amt'] = pd.to_numeric(df['net_sales_proceeds'], errors='coerce')
        ix = pd.isnull(df['net_sales_proceeds_amt'])
        df.loc[ix, 'net_sales_proceeds_code'] = df.loc[ix, 'net_sales_proceeds']
        df['net_sales_proceeds_code'] = df['net_sales_proceeds_code'].astype('category')
        df = df.drop(columns=['net_sales_proceeds'])

        return df

    else:

        return None

# def load_fannie(year, q, use_pickle=True, reimport=False, **kwargs):
def load_fannie(year, q, reimport=False, **kwargs):

    data_dir = '/nobackup1/dlg/fannie/'
    parquet_dir = data_dir + '/storage/'
    in_out.make_dir(parquet_dir)

    parquet_file = parquet_dir + 'acquisition_{0:d}q{1:d}.parquet'.format(year, q)
    if load_parquet and os.path.exists(parquet_file):
        print("Loading " + parquet_file)
        return pd.read_parquet(parquet_file)

    # if (not use_pickle) or reimport or (not os.path.exists(pkl_file)):
    if True:
        acq_names = [
            'loan_id', 'channel', 'seller_name', 'orig_int_rate', 'orig_upb', 
            'orig_term', 'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 
            'n_borrowers', 'orig_dti', 'credit_score', 'first_time_flag', 'loan_purpose',
            'prop_type', 'n_units', 'occ_status', 'prop_state', 'zip3', 
            'mi_pct', 'prod_type', 'co_borr_credit_score', 'mi_type', 'reloc'
        ]

        filepath = data_dir + 'Acquisition_{0}Q{1}.txt'.format(year, q)
        compression=None

        if os.path.isfile(filepath):
            df = pd.read_csv(filepath, sep='|', compression=compression,
                               names=acq_names, **kwargs)
            # if use_pickle:
                # df.to_pickle(pkl_file)
            return df
        else:
            return None

    # else:
        # return pd.read_pickle(pkl_file)

def load_freddie(year, q, load_parquet=True, save_parquet=True,
                 overwrite_parquet=False, **kwargs):

    data_dir = '/nobackup1/dlg/freddie/data'
    parquet_dir = data_dir + '/storage/'

    parquet_file = parquet_dir + 'acquisition_{0:d}q{1:d}.parquet'.format(year, q)
    if load_parquet and os.path.exists(parquet_file):
        print("Loading " + parquet_file)
        return pd.read_parquet(parquet_file)

    col_names = ['credit_score', 'first_pay_date', 'first_time_flag', 'maturity_date', 'msa', 
            'mi_pct', 'n_units', 'occ_status', 'orig_cltv', 'orig_dti',
            'orig_upb', 'orig_ltv', 'orig_int_rate', 'channel', 'prepay_pen_flag',
            'prod_type', 'prop_state', 'prop_type', 'zip3', 'loan_id',
            'loan_purpose', 'orig_term', 'n_borrowers', 'seller_name', 'servicer_name']

    filename = 'historical_data1_Q{0}{1}'.format(q, year)
    zipname = data_dir + '/' + filename + '.zip'
    txtname = filename + '.txt'

    if os.path.isfile(zipname):
        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep='|', names=col_names, index_col=False, **kwargs)

            if save_parquet:
                if not os.path.exists(parquet_file) or overwrite_parquet:
                    print("Saving " + parquet_file)
                    df.to_parquet(parquet_file)

            return df
    else:
        return None

def load_freddie_performance(year, q, load_parquet=True, save_parquet=True,
                             overwrite_parquet=False, **kwargs):

    data_dir = '/nobackup1/dlg/freddie/data'
    parquet_dir = data_dir + '/storage/'

    parquet_file = parquet_dir + 'perf_{0:d}q{1:d}.parquet'.format(year, q)
    if load_parquet and os.path.exists(parquet_file):
        print("Loading " + parquet_file)
        return pd.read_parquet(parquet_file)

    col_names = ['loan_id', 'asof_date', 'current_upb', 'delinq', 'loan_age', 'months_left',
                 'repurchase', 'modification', 'zero_balance', 'zero_balance_date',
                 'current_rate', 'current_deferred_upb', 'last_due_date', 'mi_recoveries',
                 'net_sales_proceeds', 'non_mi_recoveries', 'expenses', 'legal_costs',
                 'maintenance_costs', 'taxes_and_insurance', 'misc_expenses', 'actual_loss',
                 'modification_cost', 'step_modification_flag', 'deferred_pay_mod',
                 'estimated_ltv']

    filename = 'historical_data1_time_Q{0}{1}'.format(q, year)
    zipname = '{0}/historical_data1_Q{1}{2}.zip'.format(data_dir, q, year)
    txtname = filename + '.txt'

    if os.path.isfile(zipname):

        zf = ZipFile(zipname)
        with zf.open(txtname) as fid:
            df = pd.read_csv(fid, sep='|', names=col_names, index_col=False, **kwargs)

        # Adjust object data types
        df['loan_id'] = df['loan_id'].astype(str)

        for var in ['repurchase', 'modification', 'step_modification_flag']:
            df[var] = df[var].astype('category')

        # Update delinquency to account for REOs
        df['reo'] = df['delinq'] == 'R'
        df.loc[df['reo'], 'delinq'] = -1
        df['delinq'] = pd.to_numeric(df['delinq'], errors='coerce')
        df.loc[pd.isnull(df['delinq']), 'delinq'] = -2
        df['delinq'] = df['delinq'].astype(np.int64)

        # Update net_sales_proceeds codes
        df['net_sales_proceeds_amt'] = pd.to_numeric(df['net_sales_proceeds'], errors='coerce')
        ix = pd.isnull(df['net_sales_proceeds_amt'])
        df.loc[ix, 'net_sales_proceeds_code'] = df.loc[ix, 'net_sales_proceeds']
        df['net_sales_proceeds_code'] = df['net_sales_proceeds_code'].astype('category')
        df = df.drop(columns=['net_sales_proceeds'])

        if save_parquet:
            if not os.path.exists(parquet_file) or overwrite_parquet:
                print("Saving " + parquet_file)
                df.to_parquet(parquet_file)

        return df
    else:
        return None
