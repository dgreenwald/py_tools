import pandas as pd
from . import fred
from py_tools import time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'fof/'
# data_dir = '/home/dan/Dropbox/data/fof/'

def load(dataset, usecols=None, data_dir=default_dir, fof_vintage='2003'):
    """Load pre-packaged set of variables"""
    
    if dataset == 'corporate':
        
        # Format: 'name' : ('table', 'variable')
        var_index = {
            'nonfin_assets' : ('b103', 'LM102010005'),
            'assets' : ('b103', 'FL102000005'),
            'assets_book' : ('b103', 'FL102000115'),
            'liabilities_book' : ('b103', 'FL104190005'),
            'net_worth_book' : ('b103', 'FL102090005'),
            'net_worth_market' : ('b103', 'FL102090005'),
            'equities_outstanding_market' : ('b103', 'LM103164103'),
            # 'net_dividends' : ('u103', 'FU106121075'),
            # 'net_new_equity' : ('u103', 'FU103164103'),
            # 'net_new_paper' : ('u103', 'FU103169100'),
            # 'net_new_bonds' : ('u103', 'FU103163003')
            'gross_investment' : ('f103', 'FA105090005'),
            'net_dividends' : ('f103', 'FA106121075'),
            'net_new_equity' : ('f103', 'FA103164103'),
            'net_new_paper' : ('f103', 'FA103169100'),
            'net_new_bonds' : ('f103', 'FA103163003'),
            'corp_equities_wealth' : ('b101', 'LM153064105'),
            'noncorp_business_wealth' : ('b101', 'LM152090205'),
            'mutual_fund_wealth' : ('b101', 'LM153064205'),
            'equities_outstanding_incl_fin' : ('b1', 'LM883164105'),
            'equities_outstanding_market_fin' : ('l108', 'LM793164105'),
            'net_new_equity_fin' : ('f108', 'FA793164105'),
            'net_dividends_fin' : ('f3', 'FA796121073'),
            'profits_pretax' : ('f103', 'FA106060005'),
            'corp_taxes' : ('f103', 'FA106231005'),
            'iva' : ('f103', 'FA105020601'),
            'foreign_ret_earnings' : ('f103', 'FA106006065'),
            'capex' : ('f103', 'FA105050005'),
            'debt_securities' : ('b103', 'FL104122005'),
            'loans' : ('b103', 'FL104123005'),
            'foreign_deposits' : ('b103', 'FL103091003'),
            'checkable_deposits' : ('b103', 'FL103020000'),
            'time_savings_deposits' : ('b103', 'FL103030003'),
            'mmf_shares' : ('b103', 'FL103034000'),
            'repos' : ('b103', 'FL102051003'),
            'debt_securities_asset' : ('b103', 'LM104022005'),
            'loans_asset' : ('b103', 'FL104023005'),
            'mortgages' : ('b103', 'FL103165005'),
            'nonfinancial_assets_hist_cost' : ('b103', 'FL102010115'),
        }
        
    elif dataset == 'household':
        
        var_index = {
            'assets' : ('b101', 'FL152000005'),
            'nonfinancial_assets' : ('b101', 'LM152010005'),
            'real_estate' : ('b101', 'LM155035015'),
            'real_estate_nonprofit' : ('b101', 'LM165035005'),
            'consumer_durables' : ('b101', 'LM155111005'),
            'residential_mortgages' : ('b101', 'FL153165105'),
            'consumer_credit' : ('b101', 'FL153166000'),
            'disposable_income' : ('f101', 'FA156012005'),
            'gross_income' : ('f101', 'FA156010001'),
            'personal_taxes' : ('f101', 'FA156210005'),
            }
        
    elif dataset == 'financial':
        
        var_index = {
            'assets' : ('l108', 'FL704090005'),
            'liabilities' : ('l108', 'FL794190005'),
            'equity' : ('l108', 'LM793164105'),
            'loans' : ('l108', 'FL794023005'),
            'checkable_deposits' : ('l108', 'FL794023005'),
            'time_savings_deposits' : ('l108', 'FL794023005'),
            'money_market_shares' : ('l108', 'FL794023005'),
            }
        
    elif dataset == 'banks':
        
        var_index = {
            'assets' : ('l110', 'FL704090005'),
            'liabilities' : ('l110', 'FL704190005'),
            'loans' : ('l110', 'FL704023005'),
            'checkable_deposits' : ('l110', 'FL703127005'),
            'time_savings_deposits' : ('l110', 'FL703130005'),
            }
        
    else:
        
        print("Invalid FOF dataset request")
        raise Exception
        
    if usecols is None:
        var_list = sorted(list(var_index.keys()))
    else:
        var_list = [var for var in usecols if var in var_index]
        
    # Get relevant tables and codes
    tables, codes = zip(*[var_index[var] for var in var_list])
    unique_tables = sorted(list(set(tables)))
    codes = [code + '.Q' for code in codes]
    code_index = {code : var for var, code in zip(var_list, codes)}
    
    df_list = []
    vars_loaded = []
    
    for table  in unique_tables:

        these_codes = [this_code for this_table, this_code in
                       zip(tables, codes) if this_table == table]

        these_cols = ['date'] + these_codes

        df_tab = load_table(table, data_dir=data_dir, usecols=these_cols,
                            fof_vintage=fof_vintage)
        df_tab.rename(columns=code_index, inplace=True)
        
        year = df_tab['date'].str[:4].astype(int)
        q = df_tab['date'].str[-1].astype(int)
        
        df_tab['date'] = ts.date_from_qtr(year, q)
        df_tab = df_tab.set_index('date').sort_index()

        new_vars = [var for var in df_tab if var not in vars_loaded]
        df_list.append(df_tab[new_vars].copy())
        vars_loaded += new_vars
        
    df = pd.concat(df_list, axis=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def load_table(table, data_dir=default_dir, fof_vintage='2003', **kwargs):
    """Load single table"""
    
    infile = data_dir + 'all_csv/{0}/csv/{1}.csv'.format(fof_vintage, table)
    df = pd.read_csv(infile)
    
    return df

def load_fred(**kwargs):
    
    print("Deprecated, update to use fof.load()")
    
    var_titles = {
        'HMLBSHNO' : 'debt',
        'HNOREMV' : 'value',
        'HNODPI' : 'income',
        'PI' : 'gross_income',
        'HHMSDODNS' : 'debt_sa',
        'DHUTRC1Q027SBEA' : 'housing_services',
    }

    df = fred.load(code_names=var_titles, **kwargs).resample('QS').mean().loc['1952-01-01':, :]
    df['price_rent'] = df['value'] / df['housing_services']

    return df

def load_prn(data_dir=default_dir):
    
    print("Deprecated, update to use fof.load()")
    
    value_var = 'LM155035015.Q'
    debt_var = 'FL153165105.Q'

    this_dir = data_dir + 'all_prn/'
    df = pd.read_csv(
            this_dir + 'btab101d.prn',
            delimiter=' ',
            usecols=[value_var, debt_var],
            )

    df.rename(columns={value_var : 'value', debt_var : 'debt'}, inplace=True)
    df.set_index(pd.date_range('10/1/1945', periods=len(df), freq='QS'), inplace=True)

    income_var = 'FA156012005.Q' # Disposable Personal Income
    gross_income_var = 'FA156010001.Q' # Disposable Personal Income
    df_a = pd.read_csv(
            this_dir + 'atab101d.prn',
            delimiter=' ',
            usecols=[income_var, gross_income_var],
            )
    df_a.rename(columns={income_var : 'income', gross_income_var : 'gross_income'}, inplace=True)
    df_a.set_index(pd.date_range('10/1/1945', periods=len(df_a), freq='QS'), inplace=True)

    df = pd.merge(df, df_a, left_index=True, right_index=True).loc['1951-10-01':, :]
    for var in ['debt', 'value', 'income', 'gross_income']:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    return df

def load_csv(data_dir=default_dir):
    """Load from CSV files"""
    
    print("Deprecated, update to use fof.load()")

    infile = data_dir + 'csv/fof.csv'
    df = pd.read_csv(infile, skiprows=5)
    df = ts.date_index(df, '1945-10-01', freq='QS')
    df = df.loc['1951-10-01':, :]

# Update names
    raw_labels = {
            '156012005' : 'income',
            '156010001' : 'gross_income',
            '155035005' : 'value',
            '153165105' : 'debt',
            }

    for prefix in ['LM', 'FA', 'FU', 'FL']:
        these_labels = {
                prefix + key + '.Q' : prefix + '_' + val for key, val in raw_labels.items()
                }
        df = df.rename(columns=these_labels)
        
    for col in df.columns:
        if col != 'Time Period':
            df[col] = pd.to_numeric(df[col])

    return df.rename(columns={'FA_income' : 'income', 'FA_gross_income' : 'gross_income', 
                              'FL_debt' : 'debt', 'FL_value' : 'value'})[['income', 'debt', 'value']]