import numpy as np
import os
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir()

idx = pd.IndexSlice

def get_var_index(nipa_table, vintage='1706', add_prefix=False, **kwargs):

    if nipa_table == '10105':
        
        var_index = {
            'gdp' : 'A191RC1',
            'pce' : 'DPCERC1',
            'pce_goods' : 'DGDSRC1',
            'pce_durables' : 'DDURRC1',
            'pce_nondurables' : 'DNDGRC1',
            'pce_services' : 'DSERRC1',
            'invest' : 'A006RC1',
            'invest_fixed' : 'A007RC1',
            'invest_nonres' : 'A008RC1',
            'invest_structures' : 'B009RC1',
            'invest_equip' : 'Y033RC1',
            'invest_ip' : 'Y001RC1',
            'invest_residential' : 'A011RC1',
            'invest_inventory' : 'A014RC1',
            'net_exports' : 'A019RC1',
            'exports' : 'B020RC1',
            'exports_goods' : 'A253RC1',
            'exports_services' : 'A646RC1',
            'imports' : 'B021RC1',
            'imports_goods' : 'A255RC1',
            'imports_services' : 'B656RC1',
            'govt' : 'A822RC1',
            'govt_federal' : 'A823RC1',
            'govt_defense' : 'A824RC1',
            'govt_nondefense' : 'A825RC1',
            'govt_state_local' : 'A829RC1',
        }
    
    elif nipa_table == '10106':
   
        temp_index = {
            'gdp' : 'A191RX1',
            'pce' : 'DPCERX1',
            'pce_goods' : 'DGDSRX1',
            'pce_durables' : 'DDURRX1',
            'pce_nondurables' : 'DNDGRX1',
            'pce_services' : 'DSERRX1',
            'invest' : 'A006RX1',
            'invest_fixed' : 'A007RX1',
            'invest_nonres' : 'A008RX1',
            'invest_nonres_struct' : 'B009RX1',
            'invest_nonres_equip' : 'Y033RX1',
            'invest_nonres_ip' : 'Y001RX1',
            'invest_res' : 'A011RX1',
            'invest_inventory' : 'A014RX1',
            'net_exports' : 'A019RX1',
            'exports' : 'A020RX1',
            'exports_goods' : 'A253RX1',
            'exports_services' : 'A646RX1',
            'imports' : 'A021RX1',
            'imports_goods' : 'A255RX1',
            'imports_services' : 'B656RX1',
            'govt' : 'A822RX1',
            'govt_federal' : 'A823RX1',
            'govt_federal_defense' : 'A824RX1',
            'govt_federal_nondefense' : 'A825RX1',
            'govt_nonfed' : 'A829RX1',
            'residual' : 'A960RX1',
        }

        var_index = {
            'real_' + key : val for key, val in temp_index.items()
        }

    elif nipa_table == '10109':

        var_index = {
            'pce_deflator' : 'DPCERD3',
        }

    elif nipa_table == '11000':

        var_index = {
            'gdi' : 'A261RC1',
            'comp' : 'A4002C1',
            'comp_wage_sal' : 'A4102C1',
            'comp_wage_sal_domestic' : 'W270RC1',
            'comp_wage_sal_row' : 'B4189C1',
            'comp_supplements' : 'A038RC1',
            'prod_taxes' : 'W056RC1',
            'prod_subsidies' : 'A107RC1',
            'net_op_surplus' : 'W271RC1',
            'net_op_surplus_private' : 'W260RC1',
            'net_interest' : 'W272RC1',
            'business_transfer' : 'B029RC1',
            'proprietor_income' : 'A041RC1',
            'rental_income' : 'A048RC1',
            'corp_profits' : 'A445RC1',
            'corp_taxes' : 'A054RC1',
            'corp_after_tax_profits' : 'W273RC1',
            'net_dividends' : 'A449RC1',
            'corp_after_tax_undistributed' : 'W274RC1',
            'net_op_surplus_govt' : 'A108RC1',
            'consumption_fixed' : 'A262RC1',
            'consumption_fixed_private' : 'A024RC1',
            'consumption_fixed_govt' : 'A264RC1',
        }

    elif nipa_table == '11200':

        var_index = {
            'income' : 'A032RC1',
            'comp' : 'A033RC1',
            'comp_wage_sal' : 'A034RC1',
            'comp_wage_sal_gov' : 'B202RC1',
            'comp_wage_sal_other' : 'A132RC1',
            'comp_supp' : 'A038RC1',
            'proprietors' : 'A041RC1',
            'rental' : 'A048RC1',
            'corp_profits' : 'A051RC1',
            'corp_taxes' : 'A054RC1',
            'corp_after_tax_profits' : 'A551RC1',
            'corp_net_dividends' : 'B056RC1',
            'corp_undistributed_profits' : 'A127RC1',
            'net_interest_misc' : 'W255RC1',
            'prod_taxes' : 'W056RC1',
            'prod_subsidies' : 'A107RC1',
            'business_transfer' : 'B029RC1',
            'gov_surplus' : 'A108RC1',
        }

    elif nipa_table == '11400':

        # Corporate nonfinancial
        cnf_index = {
            # Nonfinancial
            'gross_value_added' : 'A455RC1',
            'cons_fixed_cap' : 'B456RC1',
            'net_value_added' : 'A457RC1',
            'compensation' : 'A460RC1',
            'wage_sal' : 'B461RC1',
            'wage_sal_supp' : 'B462RC1',
            'prod_taxes' : 'W325RC1',
            'net_op_surplus' : 'W326RC1',
            'net_interest' : 'B471RC1',
            'transfer_payments' : 'W327RC1',
            'profits' : 'A463RC1',
            'corp_taxes' : 'B465RC1',
            'after_tax_profits' : 'W328RC1',
            'net_dividends' : 'B467RC1',
            'undistributed_profits' : 'W332RC1',
            'gross_value_added_chained' : 'B455RX1',
            'net_value_added_chained' : 'A457RX1',
        }

        # Total corporate
        corp_index = {
            'gross_value_added' : 'A451RC1',
            'cons_fixed_cap' : 'A438RC1',
            'net_value_added' : 'A439RC1',
            'compensation' : 'A442RC1',
            'wage_sal' : 'A443RC1',
            'wage_sal_supp' : 'A444RC1',
            'prod_taxes' : 'W321RC1',
            'net_op_surplus' : 'W322RC1',
            'net_interest' : 'A453RC1',
            'transfer_payments' : 'W323RC1',
            'profits' : 'A445RC1',
            'corp_taxes' : 'A054RC1',
            'after_tax_profits' : 'W273RC1',
            'net_dividends' : 'A449RC1',
            'undistributed_profits' : 'W274RC1',
        }

        # Put these together
        var_index = {}
        var_index.update({
            key + '_corp_nonfin' : val for key, val in cnf_index.items()
        })
        var_index.update({
            key + '_corp' : val for key, val in corp_index.items()
        })

    elif nipa_table == '20100':

        var_index = {
            # 'wage_sal' : 'A576RC1',
            'compensation' : 'A033RC1',
            'personal_income' : 'A065RC1',
            'transfer_payments' : 'A577RC1',
            'employer_pension_ins' : 'B040RC1',
            'personal_social' : 'A061RC1',
            'employer_social' : 'B039RC1',
            'proprietors_income' : 'A041RC1',
            'rental_income' : 'A048RC1',
            'dividends' : 'B703RC1',
            'interest' : 'A064RC1',
            'personal_current_taxes' : 'W055RC1',
            'disp_inc' : 'A067RC1',
            'real_disp_inc' : 'A067RX1',
            'real_pc_disp_inc' : 'A229RX0',
        }

        if vintage in ['1604', '1706']:
            var_index.update({
                'wage_sal' : 'A034RC1',
            })
        elif vintage == '1302':
            var_index.update({
                'wage_sal' : 'A576RC1',
            })

    elif nipa_table == '70405':

        var_index = {
            'housing_output' : 'A2007C1',
            'gross_housing_va' : 'A2009C1',
            'gross_owner_va' : 'B1300C1',
            'gross_tenant_va' : 'B1301C1',
            'net_housing_va' : 'B952RC1',
            'taxes' : 'B1031C1',
            'net_op_surplus' : 'W165RC1',
            'net_interest' : 'B1037C1',
            'rental_income' : 'B1035C1',
        }
        
    elif nipa_table == '71200':
        
        var_index = {
            'owner_services' : 'A2013C',
            }
        
    else:
        
        var_index = {}

    if add_prefix:
        temp = var_index.copy()
        var_index = {'NIPA_{0:d}_{1}'.format(nipa_table, name) : code for name, code in temp.items()}

    return var_index

def load(nipa_table=None, nipa_source='xls', **kwargs):

    assert(nipa_table is not None) # Need to pick a table

    if nipa_source == 'xls':
        return load_xls(nipa_table, **kwargs)
    elif nipa_source == 'flat':
        return load_flat(nipa_table, **kwargs)
    else:
        raise Exception

def load_flat(nipa_table=None, data_dir=default_dir+'nipa/', var_list=None,
              reimport=False, billions=True, var_index=None, vintage='2003',
              named_only=True, nipa_vintage=None, freq='Q', **kwargs):
    
    if nipa_vintage is not None:
        print("Need to switch to using vintage argument")
        vintage = nipa_vintage

    vintage_dir = data_dir + vintage + '/'
    pkl_file = vintage_dir + 'nipadata{}.pkl'.format(freq)

    if not os.path.exists(pkl_file) or reimport:
        infile = 'nipadata{}.txt'.format(freq)
        df = pd.read_csv(vintage_dir + infile).rename(columns={'%SeriesCode' : 'Series'})
        
        # Convert numbers
        df['Value'] = df['Value'].str.replace(',', '')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # Convert dates
        if freq == 'Q':
            df[['y', 'q']] = df['Period'].str.split(pat='Q', expand=True)
            df['q'] = df['q'].astype(np.int64)
            df['m'] = df['q'] * 3 - 2
            df['Date'] = df['y'].astype(str) + '-' + df['m'].astype(str) + '-01'
            df = df.drop(columns=['y', 'q', 'm', 'Period'])
        elif freq == 'A':
            df['Date'] = df['Period'].astype(str) + '-01-01'
            df = df.drop(columns=['Period'])
        else:
            raise Exception
            
        df['Date'] = pd.to_datetime(df['Date'])

        df['Series'] = df['Series'].astype(str)
        df = df.set_index(['Series', 'Date'])
        df.to_pickle(pkl_file)
    else:
        df = pd.read_pickle(pkl_file)

    if var_index is None:
        assert nipa_table is not None
        var_index = get_var_index(nipa_table, vintage=vintage, 
                                  **kwargs)

    # Get rid of final digits
    numbers = [str(ii) for ii in range(10)]
    for key, val in var_index.items():
        if val[-1] in numbers:
            val = val[:-1]
            var_index[key] = val
            
    # var_index = {key : val[:-1] for key, val in var_index.items()}

    if named_only:
        # Drop to only named variables
        
        if var_list is None:
            var_list = list(var_index.keys())
    
        code_list = [var_index[var] for var in var_list]
    
        df = df.loc[idx[code_list, :], :]
    
    if billions: 
        df['Value'] /= 1000
    df = df.reset_index().pivot(index='Date', columns='Series', values='Value')

    reverse_index = {val : key for key, val in var_index.items()}
    df = df.rename(columns=reverse_index)

    return df

def load_xls(nipa_table, vintage='1706', nipa_quarterly=True,
         master_dirs={}, **kwargs):
    """Load NIPA table, specify table (e.g., 20100) vintage (when downloaded) and whether quarterly data"""

    dirs = master_dirs.copy()
    if 'base' not in dirs:
        dirs['base'] = default_dir
        # home_dir = os.environ['HOME']
        # dirs['base'] = home_dir + '/Dropbox/data/'

    data_dir = dirs['base'] + 'nipa/' + vintage + '/'

    if nipa_quarterly:
        freq_str = ' Qtr'
    else:
        freq_str = ' Ann'

    sheet_name = nipa_table + freq_str

    ################################################################################
    # LOAD FILES
    ################################################################################

    # File names
    table_group = nipa_table[0]
    curr_file_path = data_dir + 'Section{}All_xls.xls'.format(table_group)
    hist_file_path = data_dir + 'Section{}All_Hist.xls'.format(table_group)

    # Load current file
    df_t = pd.read_excel(
        curr_file_path,
        sheet_name=sheet_name,
        skiprows=7,
        # header=[0, 1],
        index_col=2,
    )
    df_curr = clean_nipa(df_t, nipa_quarterly=nipa_quarterly)
    df_curr = df_curr.apply(pd.to_numeric, errors='coerce')
    # df_curr = df_curr.convert_objects(convert_dates=False, convert_numeric=True)

    # Load historical file
    df_t = pd.read_excel(
        hist_file_path,
        sheet_name=sheet_name,
        skiprows=7,
        # header=[0, 1],
        index_col=2,
    )
    df_hist = clean_nipa(df_t, nipa_quarterly=nipa_quarterly)
    df_hist = df_hist.apply(pd.to_numeric, errors='coerce')

    # Combine datasets
    start_date = df_curr.index[0]
    df_hist_sample = df_hist.loc[:start_date, :]
    df = df_hist_sample.iloc[:-1, :].append(df_curr)

    ################################################################################
    # RENAME SERIES
    ################################################################################

    var_index = get_var_index(nipa_table)

    full_list = sorted(list(var_index.keys()))
    codes = [var_index[var] for var in full_list]
    # df = df.loc[:, codes]
    df = df.loc[:,~df.columns.duplicated()]
    df.rename(columns = {code : var for var, code in zip(full_list, codes)}, inplace=True)

    if nipa_table == '11400':
        df['earnings_corp'] = df['after_tax_profits_corp'] + df['net_interest_corp']
        df['earnings_corp_nonfin'] = df['after_tax_profits_corp_nonfin'] + df['net_interest_corp_nonfin']
    elif nipa_table == '20100':
        df['employee_net_social'] = df['personal_social'] - df['employer_social']
        df['total_other'] = df['proprietors_income'] + df['rental_income'] + df['dividends'] + df['interest']
        df['tax_share'] = df['wage_sal'] / (df['wage_sal'] + df['total_other'])
        df['tax'] = df['tax_share'] * df['personal_current_taxes']
        df['nyd'] = (df['wage_sal'] + df['transfer_payments'] + df['employer_pension_ins']
                     - df['employee_net_social'] - df['tax'])
        df['comp_no_transfers'] = df['wage_sal'] + df['employer_pension_ins'] + df['employer_social']
        df['total_comp'] = df['comp_no_transfers'] + df['transfer_payments']

    return df

def clean_nipa(df_t, nipa_quarterly=True):

    df_t = df_t.loc[df_t.index != ' ', :]
    df_t = df_t.loc[pd.notnull(df_t.index), :]
    del df_t['Line']
    del df_t['Unnamed: 1']

    df = df_t.transpose()
    start_date = df.index[0]

    if nipa_quarterly:
        yr = int(np.floor(start_date))
        q = int(10 * (start_date - yr) + 1)
        mon = int(3 * (q - 1) + 1)

        ts.date_index(df, '{0}/1/{1}'.format(mon, yr))
    else:
        ts.date_index(df, '1/1/{0}'.format(int(start_date)), freq='AS')

    return df
