import os
import glob
import numpy as np
import pandas as pd
import re

from . import config, crosswalk
default_dir = config.base_dir() + 'irs/'
DATASET_NAME = "irs"
DESCRIPTION = "IRS tax statistics dataset loader."
def load(data_dir=None, **kwargs):
    """Load IRS dataset variants through a single entrypoint.

    Dispatches to the appropriate loader based on the ``dataset`` keyword
    argument.  Supported variants are ``county`` (default), ``zip``,
    ``zip3_from_county``, ``zip3_from_zip``, ``county_year`` (requires
    ``year``), and ``zip_year`` (requires ``year``).

    Parameters
    ----------
    data_dir : str, optional
        Root directory for IRS data files.  When provided it is forwarded
        to the underlying loader as ``data_dir``.
    **kwargs
        Additional keyword arguments passed to the underlying loader.
        The special key ``dataset`` selects which loader to call.

    Returns
    -------
    pandas.DataFrame
        DataFrame returned by the selected loader.

    Raises
    ------
    ValueError
        If ``dataset`` is not one of the supported variant strings.
    """
    if data_dir is not None:
        kwargs.setdefault('data_dir', data_dir)

    dataset = kwargs.pop('dataset', 'county')
    if dataset == 'county':
        return load_county(**kwargs)
    if dataset == 'zip':
        return load_zip(**kwargs)
    if dataset == 'zip3_from_county':
        return load_zip3_from_county(**kwargs)
    if dataset == 'zip3_from_zip':
        return load_zip3_from_zip(**kwargs)
    if dataset == 'county_year':
        return load_county_year(**kwargs)
    if dataset == 'zip_year':
        return load_zip_year(**kwargs)
    raise ValueError(f"Unsupported dataset variant: {dataset}")


def double_list(these_items):
    """Create a flat list of ``['n_item', 'item']`` pairs for each item.

    Parameters
    ----------
    these_items : list of str
        Base names to expand into count/value pairs.

    Returns
    -------
    list
        Flat list alternating ``'n_<item>'`` and ``'<item>'`` for every
        entry in ``these_items``.
    """
    names = []
    for item in these_items:
        names += ['n_' + item, item]

    return names

def triple_list(these_items):
    """Create a flat list of ``['n_item', 'agi_item', 'item']`` triples for each item.

    Parameters
    ----------
    these_items : list of str
        Base names to expand into count/AGI/value triples.

    Returns
    -------
    list
        Flat list of ``'n_<item>'``, ``'agi_<item>'``, and ``'<item>'``
        for every entry in ``these_items``.
    """
    names = []
    for item in these_items:
        names += ['n_' + item, 'agi_' + item, item]

    return names

def compute_fips(df_t):
    """Compute a 5-digit FIPS code from ``statefips`` and ``countyfips`` columns.

    Coerces both columns to numeric, drops rows where either is NaN or
    non-positive, then sets ``fips = countyfips + 1000 * statefips``.

    Parameters
    ----------
    df_t : pandas.DataFrame
        DataFrame containing ``statefips`` and ``countyfips`` columns.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``df_t`` with a new integer ``fips`` column.
    """
    df_t['countyfips'] = pd.to_numeric(df_t['countyfips'], errors='coerce')
    df_t['statefips'] = pd.to_numeric(df_t['statefips'], errors='coerce')

    df_t.dropna(subset=['countyfips', 'statefips'], inplace=True)
    for name in ['countyfips', 'statefips']:
        df_t[name] = df_t[name].astype(np.int64)

    ix = (df_t['countyfips'] > 0) & (df_t['statefips'] > 0)
    df_t = df_t.loc[ix].copy()
        
    df_t['fips'] = df_t['countyfips'] + 1000 * df_t['statefips']

    return df_t

def load_state_county_year(filename, skiprows, target_cols):
    """Read a single state/county Excel file, trimming to ``target_cols`` columns.

    Parameters
    ----------
    filename : str
        Path to the Excel file to read.
    skiprows : int
        Number of rows to skip at the top of the sheet before the header.
    target_cols : int
        Maximum number of columns to retain; extra columns are dropped.

    Returns
    -------
    pandas.DataFrame
        Raw DataFrame with integer column indices and at most ``target_cols``
        columns.
    """
    # print(filename)
    df_state = pd.read_excel(filename, skiprows=skiprows, header=None)
    
    # Drop excess columns
    n_cols = len(df_state.columns)
    if n_cols > target_cols:
        df_state = df_state.drop(range(target_cols, n_cols), axis=1)
        
    return df_state

def load_county_year(year, data_dir=default_dir, reimport=False):
    """Load or import IRS county-level data for a single year, caching as parquet.

    If the parquet cache exists and ``reimport`` is ``False``, the cached
    file is returned directly.  Otherwise the raw source is imported and
    written to ``<data_dir>/county/irs_county_<year>.parquet``.

    Parameters
    ----------
    year : int
        Tax year to load.
    data_dir : str, optional
        Root directory for IRS data files.
    reimport : bool, optional
        When ``True``, re-import from source even if the parquet cache
        already exists.

    Returns
    -------
    pandas.DataFrame
        County-level IRS data for the requested year.
    """
    parquet_file = data_dir + 'county/irs_county_{:d}.parquet'.format(year)
    if reimport or not os.path.exists(parquet_file):
        
        print("Loading year {}".format(year))
        if year <= 2009:
            df_t = import_county_year_to_2009(year, data_dir=data_dir)
        elif year == 2010:
            df_t = import_county_year_2010(year, data_dir=data_dir)
        else:
            df_t = import_geo_year_from_2011(year, 'county', data_dir=data_dir)
        
        # print(df_t.columns)
        for var in df_t.columns:
            if var == 'county_name':
                df_t[var] = df_t[var].astype(str)
            elif var == 'date':
                df_t[var] = pd.to_datetime(df_t[var])
            elif df_t.dtypes[var] == 'object':
                print("converting to numeric: " + var)
                df_t[var] = pd.to_numeric(df_t[var], errors='coerce')
        
            # if var[:2] == 'n_':
            #     print("converting to int: " + var)
            #     df_t[var] = np.round(df_t[var]).astype('Int64')

        df_t.to_parquet(parquet_file)
    else:
        df_t = pd.read_parquet(parquet_file) 

    return df_t

def import_geo_year_from_2011(year, geo, data_dir=default_dir):
    """Import IRS geographic data for years >= 2011 from CSV source files.

    Reads the raw CSV, renames columns to a standardised schema, filters to
    rows with at least one return, groups by geography, and attaches a
    ``date`` column.

    Parameters
    ----------
    year : int
        Tax year to import (should be >= 2011).
    geo : str
        Geographic level to load; ``'county'`` or ``'zip'``.
    data_dir : str, optional
        Root directory for IRS data files.

    Returns
    -------
    pandas.DataFrame
        Aggregated IRS data for the requested year and geography.

    Raises
    ------
    Exception
        If ``geo`` is not ``'county'`` or ``'zip'``.
    """
    if geo == 'county':
        file_stub = 'incyallagi'
        group_var = 'fips'
    elif geo == 'zip':
        file_stub = 'zpallagi'
        group_var = 'zip'
    else:
        raise Exception

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
    
    if geo == 'zip':
        col_map['zipcode'] = 'zip'

    # add number and level
    col_map.update({'n' + key : 'n_' + val for key, val in n_map.items()})
    col_map.update({'a' + key : val for key, val in n_map.items()})

    final_map = { key.lower() : val for key, val in col_map.items() } 

    short_year = str(year)[2:]
    df_by_cat = pd.read_csv(data_dir + geo + '/' + short_year + file_stub + '.csv',
                     encoding='latin1')
    df_by_cat = df_by_cat.rename(columns={col : col.lower() for col in df_by_cat.columns})
    df_by_cat = df_by_cat.rename(columns=final_map)
    
    ix = df_by_cat['n_returns'] >= 1
    df_by_cat = df_by_cat.loc[ix].copy()

    if geo == 'county':
        df_by_cat = compute_fips(df_by_cat)
    elif geo == 'zip':
        df_by_cat['zip'] = df_by_cat['zip'].astype(np.int64)

    df_t = df_by_cat.groupby([group_var]).sum().drop(['agi_stub'], axis=1).reset_index()
    if 'count' in df_t:
        df_t = df_t.drop(columns=['count'])

    df_t['date'] = '{0:d}-01-01'.format(year)

    return df_t

def import_county_year_2010(year, data_dir=default_dir):
    """Import IRS county-level data for tax year 2010 from an Excel file.

    Parameters
    ----------
    year : int
        Tax year; expected to be ``2010``.
    data_dir : str, optional
        Root directory for IRS data files.

    Returns
    -------
    pandas.DataFrame
        County-level IRS data for 2010 with a ``fips`` column and a
        ``date`` column set to ``'2010-01-01'``.
    """
    year_dir = data_dir + 'county/' + str(year) + 'CountyIncome/'

    names = [
        'statefips', 'state', 'countyfips', 'county_name', 'n_returns',
        'n_joint', 'n_prepared', 'n_exemptions', 'n_dependents', 'agi', 
    ]
    
    names += double_list(['wagesal', 'interest', 'dividends', 'qualified_dividends', 'business'])

    names += ['n_farm']

    names += double_list(['cap_gain', 'retirement', 'pension', 'unemp', 'social', 
            'self_emp_ret', 'itemized', 'state_local_inc', 'state_local_sales', 'real_estate_tax', 
            'taxes_paid', 'mort_int_paid', 'contributions', 'taxable_income', 'tax_credits',
            'res_energy_credit', 'child_tax_credit', 'child_dep_care_credit', 'add_child_credit', 'eitc',
            'excess_eitc', 'alt_minimum_tax', 'income_tax', 'tax_liability', 'tax_due_at_filing',
            'overpayments_refunded'])

    skiprows = 6

    df_t = pd.read_excel(year_dir + '10incyall.xls', skiprows=skiprows, header=None)
    df_t = df_t.rename(columns={ii : name for ii, name in enumerate(names)})

    df_t = compute_fips(df_t)
    df_t['date'] = '2010-01-01'

    return df_t

def import_county_year_to_2009(year, data_dir=default_dir):
    """Import IRS county-level data for tax years <= 2009 from Excel files.

    Concatenates all per-state Excel files found in the year's directory,
    assigns column names, computes the FIPS code, and attaches a ``date``
    column.

    Parameters
    ----------
    year : int
        Tax year to import (should be <= 2009).
    data_dir : str, optional
        Root directory for IRS data files.

    Returns
    -------
    pandas.DataFrame
        County-level IRS data for the requested year with a ``fips`` column
        and a ``date`` column.
    """
    year_dir = data_dir + 'county/' + str(year) + 'CountyIncome/'

    names = ['DROP', 'statefips', 'countyfips', 'county_name', 'n_returns',
             'n_exemptions', 'agi', 'wagesal', 'dividends', 'interest']

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

    target_cols = len(names)
    df_t = pd.concat([
#            pd.read_excel(filename, skiprows=9, header=None, names=names)
            load_state_county_year(filename, skiprows, target_cols)
            for filename in glob.glob(year_dir + '*.xls')
            ]).rename(columns={ii : name for ii, name in enumerate(names)})

    df_t = df_t.drop(drop_list, axis=1)
    df_t = compute_fips(df_t)
    
    df_t['date'] = '{0:d}-01-01'.format(year)
    return df_t

def load_county(data_dir=default_dir, reimport=False, reimport_year=False):
    """Load all years (1989-2016) of IRS county data, concatenating and caching as parquet.

    If the parquet cache exists and ``reimport`` is ``False``, the cached
    file is returned directly.  Otherwise all per-year files are loaded (or
    re-imported when ``reimport_year`` is ``True``) and written to
    ``<data_dir>/county/irs_county.parquet``.

    Parameters
    ----------
    data_dir : str, optional
        Root directory for IRS data files.
    reimport : bool, optional
        When ``True``, rebuild the combined dataset from per-year files even
        if the combined parquet cache already exists.
    reimport_year : bool, optional
        When ``True``, re-import each individual year from its raw source
        even if the per-year parquet cache already exists.

    Returns
    -------
    pandas.DataFrame
        Panel of county-level IRS data indexed by ``(fips, date)``.
    """
    parquet_file = data_dir + 'county/irs_county.parquet'
    if reimport or not os.path.exists(parquet_file):

        df = pd.concat((load_county_year(year, reimport=reimport_year) for year in range(1989, 2017)), sort=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['fips', 'date']).sort_index()
        df.to_parquet(parquet_file)

    else:

        df = pd.read_parquet(parquet_file)

    return df

def load_state_zip_year(filename, year):
    """Read one state ZIP-code Excel file and filter to valid 5-digit ZIP rows.

    Parameters
    ----------
    filename : str
        Path to the state-level ZIP Excel file.
    year : int
        Tax year; controls the number of header rows to skip and the column
        layout used for filtering.

    Returns
    -------
    pandas.DataFrame
        Rows corresponding to valid 5-digit ZIP codes with integer column
        indices.
    """
    print(filename)
    if year in [1998, 2001, 2002]:
       skiprows = 8
    elif year in [2004, 2005, 2006]:
        skiprows = 11
    elif year in [2007]:
        skiprows = 9
    elif year in [2008]:
        skiprows = 10
    elif year in [2009, 2010]:
        skiprows = 6

    df = pd.read_excel(filename, 
                      skiprows=skiprows, 
                       header=None)
    zip_pattern = re.compile(r'\s*\d\d\d\d\d\s*')
    
    if year <= 2005:

        row_names = df[0].astype(str).values
        
        ix = np.zeros(len(df), dtype=bool)
        for ii, name in enumerate(row_names):
            if zip_pattern.match(name):
                ix[ii] = True
            
        df = df.loc[ix, :]

    else:
        
        if year <= 2008:
            ix_bin = 0
            ix_zip = 1
        else:
            ix_bin = 1
            ix_zip = 0
    
        bin_names = df[ix_bin].astype(str).values
        zip_names = df[ix_zip].astype(str).values

        ix = np.zeros(len(df), dtype=bool)
        for ii, (bin_name, zip_name) in enumerate(zip(bin_names, zip_names)):
            if (not bin_name[0] in ['$', 'U']) and zip_pattern.match(zip_name):
                ix[ii] = True

        df = df.loc[ix, :].drop(columns=[ix_bin])
        df.columns = np.arange(df.shape[1])

    return df

def import_zip_year_to_2010(year, data_dir=default_dir):
    """Import IRS ZIP-code data for tax years <= 2010 from Excel files.

    Concatenates all per-state Excel files for the given year, assigns
    column names appropriate for that year's layout, coerces values to
    numeric, and attaches a ``date`` column.

    Parameters
    ----------
    year : int
        Tax year to import (should be <= 2010).
    data_dir : str, optional
        Root directory for IRS data files.

    Returns
    -------
    pandas.DataFrame
        ZIP-level IRS data for the requested year.

    Raises
    ------
    Exception
        If ``year`` is not one of the recognised years with a defined column
        layout.
    """
    year_dir = data_dir + 'zip/' + str(year)
    if year in [2004, 2005, 2006]:
        year_dir += 'ZipCode/'
    else:
        year_dir += 'ZIPCode/'

    if year == 1998:

        names = ['zip', 'n_returns', 'n_exemptions', 'n_dependents', 'agi']
        names += double_list([
            'wagesal', 'interest', 'eitc', 'tax_liability',
        ])
        
        for item in ['schedule_c', 'schedule_f', 'schedule_a']:
            names += ['n_' + item, 'n_sched_' + item]
            
    elif year == 2001:

        names = ['zip', 'n_returns', 'n_exemptions', 'n_dependents', 'agi']
        names += double_list(['wagesal', 'interest', 'tax_liability'])
        names += ['n_' + item for item in ['schedule_c', 'schedule_f', 'schedule_a']]
        
    elif year == 2002:
        
        names = ['zip', 'n_returns', 'n_exemptions', 'n_dependents', 'agi']
        names += double_list(['wagesal', 'interest', 'tax_liability', 'contributions'])
        names += ['n_' + item for item in ['schedule_c', 'schedule_f', 'schedule_a']]

    elif year in [2004, 2005, 2006]:

        names = ['zip', 'n_returns', 'n_exemptions', 'n_dependents', 'agi']
        names += double_list(['wagesal', 'interest', 'dividends', 'cap_gain', 'schedule_c',
                              'schedule_f', 'ira_deduction', 'pension_deduction'])
        names += triple_list(['itemized', 'contributions', 'taxes_paid'])
        
        names += double_list(['alt_minimum_tax', 'income_tax_before_credits', 'tax_liability', 'eitc'])

        names += ['n_prepared']

    elif year in [2007]:

        names = ['zip', 'n_returns', 'n_joint', 'n_prepared', 'n_exemptions', 'n_dependents', 'agi']
        names += double_list(['wagesal', 'interest', 'dividends', 'business'])
        names += ['n_farm']
        names += double_list([
            'cap_gain', 'retirement', 'pension', 'unemp', 'social', 'self_emp_ret',
            'itemized', 'state_local_inc', 'state_local_sales',
            'real_estate_tax', 'taxes_paid', 'mort_int_paid', 'contributions',
            'taxable_income', 'tax_credits', 'res_energy_credit',
            'child_tax_credit', 'child_dep_care_credit', 'eitc', 'excess_eitc',
            'alt_minimum_tax', 'income_tax', 'tax_liability',
            'tax_due_at_filing', 'overpayments_refunded',
        ])

    elif year in [2008]:
        names = [
            'zip', 'n_returns', 'n_joint', 'n_prepared', 'n_exemptions',
            'n_dependents', 'agi', 'wagesal', 'interest', 'dividends',
            # 'qualified_dividends', 
            'business', 'cap_gain', 'retirement',
            'pension', 'unemp', 'social', 'self_emp_ret', 'itemized',
            'state_local_inc', 'state_local_sales', 'real_estate_tax',
            'taxes_paid', 'mort_int_paid', 'contributions', 'taxable_income',
            'tax_credits', 'res_energy_credit', 'child_tax_credit',
            'child_dep_care_credit', 'eitc', 'excess_eitc',
            'alt_minimum_tax', 'income_tax', 'tax_liability',
            'tax_due_at_filing', 'overpayments_refunded',
        ]
    elif year in [2009, 2010]:

        names = ['zip', 'n_returns', 'n_joint', 'n_prepared', 'n_exemptions', 'n_dependents', 'agi']
        names += double_list(['wagesal', 'interest', 'dividends', 'qualified_dividends', 'business'])
        names += ['n_farm']
        names += double_list([
            'cap_gain', 'retirement', 'pension', 'unemp', 'social', 'self_emp_ret',
            'itemized', 'state_local_inc', 'state_local_sales',
            'real_estate_tax', 'taxes_paid', 'mort_int_paid', 'contributions',
            'taxable_income', 'tax_credits', 'res_energy_credit',
            'child_tax_credit', 'child_dep_care_credit',  'add_child_credit', 'eitc', 'excess_eitc',
            'alt_minimum_tax', 'income_tax', 'tax_liability',
            'tax_due_at_filing', 'overpayments_refunded',
        ])

    df_t = pd.concat([
        load_state_zip_year(filename, year)
        for filename in glob.glob(year_dir + '*.xls')
    ])

    df_t = df_t.loc[:, :len(names)-1]
    df_t.columns = names

    df_t = df_t.apply(pd.to_numeric, errors='coerce')

    df_t['date'] = str(year) + '-01-01'

    df_t['zip'] = df_t['zip'].astype(np.int64)
        
    return df_t

def load_zip_year(year, data_dir=default_dir, reimport=False):
    """Load or import IRS ZIP-code data for a single year, caching as parquet.

    If the parquet cache exists and ``reimport`` is ``False``, the cached
    file is returned directly.  Otherwise the raw source is imported,
    filtered to valid ZIP codes (501–99950), deduplicated by grouping, and
    written to ``<data_dir>/zip/irs_zip_<year>.parquet``.

    Parameters
    ----------
    year : int
        Tax year to load (must be <= 2016).
    data_dir : str, optional
        Root directory for IRS data files.
    reimport : bool, optional
        When ``True``, re-import from source even if the parquet cache
        already exists.

    Returns
    -------
    pandas.DataFrame
        ZIP-level IRS data for the requested year.

    Raises
    ------
    Exception
        If ``year`` is greater than 2016.
    """
    parquet_file = data_dir + 'zip/irs_zip_{:d}.parquet'.format(year)
    if reimport or not os.path.exists(parquet_file):
        
        print("Loading year {}".format(year))
        if year <= 2010:
            df_t = import_zip_year_to_2010(year, data_dir=default_dir)
        elif year <= 2016:
            df_t = import_geo_year_from_2011(year, 'zip', data_dir=default_dir)
        else:
            raise Exception
            
        df_t['date'] = pd.to_datetime(df_t['date'])
        
        ix = (df_t['zip'] >= 501) & (df_t['zip'] <= 99950)
        df_t = df_t.loc[ix, :]

        df_t['count'] = 1
        df_count = df_t.groupby('zip')['count'].count()
        if np.amax(df_count) > 1:
    
            print("Duplicates detected")
            ix = df_count > 1
            print(df_count.loc[ix])
            bad_zips = df_count.index[ix]
    
            bad_ix = np.any(df_t['zip'].values[:, np.newaxis] == bad_zips.values[np.newaxis, :], axis=1)
            print(df_t.loc[bad_ix, :])
    
        df_t = df_t.groupby(['zip', 'date']).sum().reset_index()

        df_t.to_parquet(parquet_file)
    else:
        df_t = pd.read_parquet(parquet_file)
        
#    print(df_t.head(10))

    return df_t

def load_zip(data_dir=default_dir, reimport=False, reimport_year=False):
    """Load all available years of IRS ZIP-code data, concatenating and caching as parquet.

    Available years are 1998, 2001, 2002, and 2004-2016.  If the parquet
    cache exists and ``reimport`` is ``False``, the cached file is returned
    directly.  Otherwise all per-year files are loaded (or re-imported when
    ``reimport_year`` is ``True``) and written to
    ``<data_dir>/zip/irs_zip.parquet``.

    Parameters
    ----------
    data_dir : str, optional
        Root directory for IRS data files.
    reimport : bool, optional
        When ``True``, rebuild the combined dataset from per-year files even
        if the combined parquet cache already exists.
    reimport_year : bool, optional
        When ``True``, re-import each individual year from its raw source
        even if the per-year parquet cache already exists.

    Returns
    -------
    pandas.DataFrame
        Panel of ZIP-level IRS data indexed by ``(zip, date)``.
    """
    zip_years = [1998, 2001, 2002] + list(range(2004, 2017))

    parquet_file = data_dir + 'zip/irs_zip.parquet'
    if reimport or not os.path.exists(parquet_file):

        df = pd.concat((load_zip_year(year, reimport=reimport_year) for year in zip_years), sort=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['zip', 'date']).sort_index()
        df.to_parquet(parquet_file)

    else:

        df = pd.read_parquet(parquet_file)

    return df

def load_zip3_from_county(data_dir=default_dir, reimport=False, county_kwargs={},
                          crosswalk_kwargs={}):
    """Aggregate IRS county data to 3-digit ZIP level using a county-to-ZIP crosswalk.

    Merges county-level IRS data with a county-to-ZIP3 crosswalk, scales
    each variable by the crosswalk allocation factor, and sums within each
    ``(zip3, date)`` group.  The result is cached as
    ``<data_dir>/zip/irs_zip3_from_county.parquet``.

    Parameters
    ----------
    data_dir : str, optional
        Root directory for IRS data files.
    reimport : bool, optional
        When ``True``, rebuild the aggregated dataset even if the parquet
        cache already exists.
    county_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`load_county`.
    crosswalk_kwargs : dict, optional
        Extra keyword arguments forwarded to
        ``crosswalk.county_to_zip``.

    Returns
    -------
    pandas.DataFrame
        IRS data aggregated to the 3-digit ZIP level, indexed by
        ``(zip3, date)``.
    """
    
    parquet_file = data_dir + 'zip/irs_zip3_from_county.parquet'
    if reimport or not os.path.exists(parquet_file):
        
        # Load county data and crosswalk then merge
        these_county_kwargs = {'data_dir' : data_dir}
        these_county_kwargs.update(county_kwargs)
        
        df = load_county(**county_kwargs)
        cw = crosswalk.county_to_zip(zip_level=3, **crosswalk_kwargs)
        df = pd.merge(df.reset_index(), cw, on='fips', how='inner')
        
        # Scale variables
        for var in df:
            if var not in ['fips', 'date', 'zip3', 'factor']:
                df[var] = pd.to_numeric(df[var], errors='coerce') * df['factor']
                
        df = df.drop(columns=['fips', 'factor'])
        df = df.groupby(['zip3', 'date']).sum(min_count=1)

        df.to_parquet(parquet_file)
        
    else:
        
        df = pd.read_parquet(parquet_file)
        
    return df

def load_zip3_from_zip(data_dir=default_dir, reimport=False, zip_kwargs={}):
    """Aggregate IRS ZIP-code data to 3-digit ZIP level.

    Loads full ZIP-level data via :func:`load_zip`, derives the 3-digit ZIP
    prefix, and sums within each ``(zip3, date)`` group.  The result is
    cached as ``<data_dir>/zip/irs_zip3_from_zip.parquet``.

    Parameters
    ----------
    data_dir : str, optional
        Root directory for IRS data files.
    reimport : bool, optional
        When ``True``, rebuild the aggregated dataset even if the parquet
        cache already exists.
    zip_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`load_zip`.

    Returns
    -------
    pandas.DataFrame
        IRS data aggregated to the 3-digit ZIP level, indexed by
        ``(zip3, date)``.
    """
    
    parquet_file = data_dir + 'zip/irs_zip3_from_zip.parquet'
    if reimport or not os.path.exists(parquet_file):
        
        # Load ZIP data
        these_zip_kwargs = {'data_dir' : data_dir}
        these_zip_kwargs.update(zip_kwargs)
        
        df = load_zip(**zip_kwargs).reset_index()
        
        # Sum by ZIP-3
        df['zip3'] = df['zip'].astype(str).str[:-2].astype(np.int)
        df = df.drop(columns=['zip'])
        df = df.groupby(['zip3', 'date']).sum(min_count=1)
        
        df.to_parquet(parquet_file)
        
    else:
        
        df = pd.read_parquet(parquet_file)
        
    return df
