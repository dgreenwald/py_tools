import os
import glob
import numpy as np
import pandas as pd

from . import defaults
default_dir = defaults.base_dir() + 'irs/'
# data_dir = '/home/dan/data/irs/'

def compute_fips(df_t):

    df_t['countyfips'] = pd.to_numeric(df_t['countyfips'], errors='coerce')
    df_t['statefips'] = pd.to_numeric(df_t['statefips'], errors='coerce')

    df_t = df_t.dropna(subset=['countyfips', 'statefips'])
    for name in ['countyfips', 'statefips']:
        df_t[name] = df_t[name].astype(np.int64)

    ix = (df_t['countyfips'] > 0) & (df_t['statefips'] > 0)
    df_t = df_t.loc[ix, :]
        
    df_t['fips'] = df_t['countyfips'] + 1000 * df_t['statefips']
    
#    counts = df_t['fips'].value_counts()
#    if np.any(counts > 1):
#        print(counts.loc[counts > 1])

    return df_t

def load_state_county_year(filename, skiprows, target_cols):
    
    df_state = pd.read_excel(filename, skiprows=skiprows, header=None)
    
    # Drop excess columns
    n_cols = len(df_state.columns)
    if n_cols > target_cols:
        df_state = df_state.drop(range(target_cols, n_cols), axis=1)
        
    return df_state

def load_county_year(year, data_dir=default_dir, reimport=False):

    pkl_file = data_dir + 'irs_county_{:d}.pkl'.format(year)
    if reimport or not os.path.exists(pkl_file):
        
        print("Loading year {}".format(year))
        if year <= 2009:
            df_t = import_county_year_to_2009(year, data_dir=default_dir)
        elif year == 2010:
            df_t = import_county_year_2010(year, data_dir=default_dir)
        else:
            df_t = import_county_year_from_2011(year, data_dir=default_dir)

        df_t.to_pickle(pkl_file)
    else:
        df_t = pd.read_pickle(pkl_file) 

    return df_t

def import_county_year_from_2011(year, data_dir=default_dir):

    col_map = {
        'n1' : 'n_returns',
        'mars2' : 'n_joint',
        'prep' : 'n_prepared',
        'n2' : 'n_exemptions',
        'numdep' : 'n_dependents', 
        'a00100' : 'agi',
        'schf' : 'n_farm',
    }

    # double these
    n_map = {
        '00200' : 'wagesal',
        '00300' : 'interest',
        '00600' : 'dividends',
        '00650' : 'qualified_dividends',
        '00900' : 'business',
        '01000' : 'cap_gain',
        '01400' : 'retirement',
        '01700' : 'pension',
        '02300' : 'unemp',
        '02500' : 'social',
        '03300' : 'self_emp_ret',
        '04470' : 'itemized',
        '18425' : 'state_local_inc',
        '18450' : 'state_local_sales',
        '18500' : 'real_estate_tax',
        '18300' : 'taxes_paid',
        '19300' : 'mort_int_paid',
        '19700' : 'contributions',
        '04800' : 'taxable_income',
        '07100' : 'tax_credits',
        '07260' : 'res_energy_credit',
        '07220' : 'child_tax_credit',
        '07180' : 'child_dep_care_credit',
        '11070' : 'add_child_credit',
        '59660' : 'eitc',
        '59720' : 'excess_eitc',
        '09600' : 'alt_minimum_tax',
        '06500' : 'income_tax',
        '10300' : 'tax_liability',
        '11901' : 'tax_due_at_filing',
        '11902' : 'overpayments_refunded',
    }

    # add number and level
    col_map.update({'n' + key : 'n_' + val for key, val in n_map.items()})
    col_map.update({'a' + key : val for key, val in n_map.items()})

    final_map = { key.upper() : val for key, val in col_map.items() } 

    short_year = str(year)[2:]
    df_by_cat = pd.read_csv(data_dir + short_year + 'incyallagi.csv',
                     encoding='latin1').rename(columns=final_map)
    df_by_cat = df_by_cat.rename(columns={col : col.lower() for col in df_by_cat.columns})

    df_by_cat = compute_fips(df_by_cat)

    # for var in ['countyfips', 'statefips']:
        # df_by_cat[var] = pd.to_numeric(df_by_cat[var], errors='coerce').astype(np.int64)

    # df_by_cat = df_by_cat.dropna(subset=['countyfips', 'statefips'])
    # df_by_cat['fips'] = df_by_cat['countyfips'] + 1000 * df_by_cat['statefips']

    df_t = df_by_cat.groupby(['fips']).sum().drop(['agi_stub'], axis=1).reset_index()

    df_t['date'] = '{0:d}-01-01'.format(year)

    return df_t

def import_county_year_2010(year, data_dir=default_dir):

    year_dir = data_dir + str(year) + 'CountyIncome/'

    names = [
        'statefips', 'state', 'countyfips', 'county_name', 'n_returns',
        'n_joint', 'n_prepared', 'n_exemptions', 'n_dependents', 'agi', 
    ]
    
    for item in ['wagesal', 'interest', 'dividends', 'qualified_dividends', 'business']:
        names += ['n_' + item, item]

    names += ['n_farm']

    for item in ['cap_gain', 'retirement', 'pension', 'unemp', 'social', 
            'self_emp_ret', 'itemized', 'state_local_inc', 'state_local_sales', 'real_estate_tax', 
            'taxes_paid', 'mort_int_paid', 'contributions', 'taxable_income', 'tax_credits',
            'res_energy_credit', 'child_tax_credit', 'child_dep_care_credit', 'add_child_credit', 'eitc',
            'excess_eitc', 'alt_minimum_tax', 'income_tax', 'tax_liability', 'tax_due_at_filing',
            'overpayments_refunded']:
        names += ['n_' + item, item]

    skiprows = 6

    df_t = pd.read_excel(year_dir + '10incyall.xls', skiprows=skiprows, header=None)
    df_t = df_t.rename(columns={ii : name for ii, name in enumerate(names)})

    df_t = compute_fips(df_t)
    df_t['date'] = '2010-01-01'

    return df_t

def import_county_year_to_2009(year, data_dir=default_dir):

    year_dir = data_dir + str(year) + 'CountyIncome/'
    
    # if year <= 2009:

    names = ['DROP', 'statefips', 'countyfips', 'county', 'n_returns', 'n_exemptions', 'agi', 'wagesal', 'dividends', 'interest']
    drop_list = ['DROP']

    if year in [1989, 2007]:
        short_year = str(year)[2:]
        year_dir += short_year + 'xls/'
        
    if year <= 2007:
        skiprows = 9
    else:
        skiprows = 8
        names = names[1:]
        drop_list = []
        
    # elif year == 2010:

        # names = [
            # 'countyfips', 'categories', 'n_returns', 'n_joint', 'n_prepared',
            # 'n_exemptions', 'n_dependents', 'agi',
        # ]
        
        # for item in ['wagesal', 'interest', 'dividends', 'qualified_dividends', 'business']:
            # names += ['n_' + item, item]

        # names += ['n_farm']

        # for item in ['cap_gain', 'retirement', 'pension', 'unemp', 'social', 
                # 'self_emp_ret', 'itemized', 'state_local_inc', 'state_local_sales', 'real_estate_tax', 
                # 'taxes_paid', 'mort_int_paid', 'contributions', 'taxable_income', 'tax_credits',
                # 'res_energy_credit', 'child_tax_credit', 'child_dep_care_credit', 'add_child_credit', 'eitc',
                # 'excess_eitc', 'alt_minimum_tax', 'income_tax', 'tax_liability', 'tax_due_at_filing',
                # 'overpayments_refunded']:
            # names += ['n_' + item, item]

        # skiprows = 15
        # drop_list = []

    target_cols = len(names)
    df_t = pd.concat([
#            pd.read_excel(filename, skiprows=9, header=None, names=names)
            load_state_county_year(filename, skiprows, target_cols)
            for filename in glob.glob(year_dir + '*.xls')
            ]).rename(columns={ii : name for ii, name in enumerate(names)})

    df_t = df_t.drop(drop_list, axis=1)
    df_t = compute_fips(df_t)

    # df_t['countyfips'] = pd.to_numeric(df_t['countyfips'], errors='coerce')
    # df_t['statefips'] = pd.to_numeric(df_t['statefips'], errors='coerce')

    # df_t = df_t.drop(drop_list, axis=1).dropna(subset=['countyfips', 'statefips'])
    # for name in ['countyfips', 'statefips']:
        # df_t[name] = df_t[name].astype(np.int64)

    # df_t = df_t.loc[df_t['countyfips'] > 0 & df_t['statefips'] > 0, :]
        
    # df_t['fips'] = df_t['countyfips'] + 1000 * df_t['statefips']

    # Drop binned categories
    # if year >= 2010:

        # drop_cats = ['Under $1', '$1 under $25,000', '$25,000 under $50,000',
                     # '$50,000 under $75,000', '$75,000 under $100,000',
                     # '$100,000 under $200,000', '$200,000 or more']

        # ix = np.zeros(len(df_t), dtype=bool)
        # for cat in drop_cats:
            # ix = np.logical_or(ix, df_t['categories'] == cat)
        # df_t = df_t.drop(df_t.index[ix])
        # df_t = df_t.dropna(subset=['categories']).drop('categories', axis=1)
    
    df_t['date'] = '{0:d}-01-01'.format(year)
    return df_t

def load_county(data_dir=default_dir, reimport=False, reimport_year=False):

    pkl_file = data_dir + 'irs_county.pkl'
    if reimport or not os.path.exists(pkl_file):

        df = pd.concat((load_county_year(year, reimport=reimport_year) for year in range(1989, 2017)), sort=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['fips', 'date']).sort_index()
        df.to_pickle(pkl_file)

    else:

        df = pd.read_pickle(pkl_file)

    return df
