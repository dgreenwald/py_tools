# TODO: data about type of series (log, real, units, etc)
# TODO: default index for deflation

import datetime
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web

import py_tools.utilities as ut

drop_dir = '/home/dan/Dropbox/'
base_dir = drop_dir + 'data/'
pkl_dir = base_dir + 'pkl/'
gll_dir = drop_dir + 'gll/Dan/data/'
gll_pred_dir = drop_dir + 'gll/Dan/Predictability/data/'

def date_index(df, startdate, freq='QS'):
    df.set_index(pd.date_range(startdate, periods=len(df), freq=freq), inplace=True)
    return df

def resample(df, methods_vars, freq='QS'):
    df_resamp = None

    for method, var_list in methods_vars.items():

        if var_list is not None:

            df_new = getattr(df[var_list].resample(freq), method)()

            # if len(var_list) == 1:
                # df_new = df_new.to_frame()
            if df_resamp is None:
                df_resamp = df_new
            else:
                df_resamp = pd.merge(df_resamp, df_new, left_index=True, right_index=True)

    return df_resamp

def get_suffix(dataset, **kwargs):

    if dataset[:4] == 'nipa':

        nipa_vintage = kwargs.get('nipa_vintage', '1604')
        suffix = nipa_vintage

    else:
        
        suffix = ''

    if suffix != '':
        suffix = '_' + suffix

    return suffix

def load(dataset_list, reimport=False, no_prefix=True, **kwargs):

    df = None
    for dataset in dataset_list:

        suffix = get_suffix(dataset, **kwargs)
        pkl_file = pkl_dir + dataset + suffix + '.pkl'
        
        if not os.path.exists(pkl_file) or reimport:

            df_new = load_dataset(dataset, **kwargs)
            if len(df_new.columns) > 1 or not no_prefix:
                columns = {col : dataset.upper() + '_' + col for col in df_new.columns}
                df_new.rename(columns=columns, inplace=True)

            # Convert to numeric
            # if to_numeric:
                # df_new.apply(lambda x: pd.to_numeric(x, errors='coerce'))

            df_new.to_pickle(pkl_file)

        else:

            df_new = pd.read_pickle(pkl_file)

        if df is None:
            df = df_new
        else:
            df = pd.merge(df, df_new, left_index=True, right_index=True, how='outer')

    return df

def load_dataset(dataset, **kwargs):

    if dataset == 'fof':

        data_dir = base_dir + 'fof/'

        var_index = {
            'liabilities_book' : ('b103', 'FL104190005'),
            'net_worth_book' : ('b103', 'FL102090005'),
            # 'net_dividends' : ('u103', 'FU106121075'),
            # 'net_new_equity' : ('u103', 'FU103164103'),
            # 'net_new_paper' : ('u103', 'FU103169100'),
            # 'net_new_bonds' : ('u103', 'FU103163003')
            'net_dividends' : ('a103', 'FA106121075'),
            'net_new_equity' : ('a103', 'FA103164103'),
            'net_new_paper' : ('a103', 'FA103169100'),
            'net_new_bonds' : ('a103', 'FA103163003')
        }

        full_list = sorted(list(var_index.keys()))

        tables, codes = zip(*[var_index[var] for var in full_list])
        codes = [code + '.Q' for code in codes]

        df = None

        unique_tables = sorted(list(set(tables)))
        for table in unique_tables:
            prefix, suffix = ut.split_str(table, 1)
            infile = prefix + 'tab' + suffix + 'd.prn'

            these_codes = [this_code for this_table, this_code in zip(tables, codes) if this_table == table]
            usecols = ['DATES'] + these_codes

            df_new = pd.read_table(
                data_dir + infile,
                delimiter=' ',
                usecols=usecols,
            )
            df_new.rename(columns = {code : var for var, code in zip(full_list, codes)}, inplace=True)

            yr, q = (int(string) for string in ut.split_str(df_new.ix[0, 'DATES'], 4))
            mon = 3 * (q - 1) + 1
            date_index(df_new, '{0}/1/{1}'.format(mon, yr))
            del df_new['DATES']

            df_new = df_new.convert_objects(convert_dates=False, convert_numeric=True)
            # df_new.apply(lambda x: pd.to_numeric(x, errors='coerce'))

            if df is not None:
                df = pd.merge(df, df_new, left_index=True, right_index=True)
            else:
                df = df_new

        # Drop missing observations
        df = df.ix['1951-10-01':, :]

        # Convert to billions
        df /= 1000.0

    elif dataset == 'stockw':
    
        data_dir = gll_dir
        infile = 'stockw.csv'
        df = pd.read_table(data_dir + infile, sep=',', 
                               names=['dates', 'stockw'], usecols=['stockw'])
        # df['stockw'] = np.log(df['stockw_level'])

        # del df['dates']
        # del df['stockw_level']

        df = date_index(df, '1/1/1952')

    elif dataset == 'crsp':

        data_dir = gll_dir
        infile = 'crsp.csv' 

        df_m = pd.read_table(data_dir + infile, sep=',',
                               names=['date', 'vwretd', 'vwretx'],
                               usecols=['vwretd', 'vwretx'])
        df_m = date_index(df_m, '12/1/1925', freq='MS')
        df_m['P'] = (df_m['vwretx'] + 1.0).cumprod()
        df_m['D'] = np.hstack((np.nan, df_m['P'][:-1])) * (df_m['vwretd'] - df_m['vwretx'])

        # df = df_m['P'].resample('QS').last().to_frame()
        # df = pd.merge(df, df_m['D'].resample('QS').sum().to_frame(),
                        # left_index=True, right_index=True)

        methods_vars = {
            'last' : ['P'],
            'sum' : ['D'],
        }
        df = resample(df_m, methods_vars)

        df['D4'] = df['D']
        for jj in range(1, 4):
            df['D4'] += df['D'].shift(jj)
        df['D4'] *= 0.25

        df['p'] = np.log(df['P'])
        df['d4'] = np.log(df['D4'])
        # df['dd'] = df['d'].diff()
        df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
        df['re'] = np.log(df['Re'])
        df['pd'] = df['p'] - df['d4']

    elif dataset == 'crsp_q':

        data_dir = gll_pred_dir
        infile = 'crsp.csv'

        df = pd.read_table(
            data_dir + infile,
            sep=',',
            # names=['date', 'vwretd', 'vwretx'],
            usecols=['vwretd', 'vwretx'],
        )

        df = date_index(df, '10/1/1925')

        df['P'] = (df['vwretx'] + 1.0).cumprod()
        df['D'] = np.hstack((np.nan, df['P'][:-1])) * (df['vwretd'] - df['vwretx'])

        df['D4'] = df['D']
        for jj in range(1, 4):
            df['D4'] += df['D'].shift(jj)
        df['D4'] *= 0.25

        df['p'] = np.log(df['P'])
        df['d4'] = np.log(df['D4'])
        # df['dd'] = df['d'].diff()
        df['Re'] = np.hstack((np.nan, (df['P'][1:] + df['D'][1:]).values / df['P'][:-1].values))
        df['re'] = np.log(df['Re'])
        df['pd'] = df['p'] - df['d4']

    elif dataset == 'cay':

        data_dir = gll_dir
        infile = 'caydata.txt'

        df = pd.read_table(data_dir + infile, delim_whitespace=True, 
                           names=['dates', 'c', 'a', 'y', 'cay'])
        df = date_index(df, '1/1/1952')

    elif dataset == 'cay_source':

        cay_source_vintage = kwargs.get('cay_source_vintage', '1302')
        data_dir = drop_dir + 'Dan Greenwald Files/CreateCAY/data_{}/'.format(cay_source_vintage)

        df = pd.read_excel(
            data_dir + 'source_{0}_{1}_rats.xlsx'.format(cay_source_vintage[:2], cay_source_vintage[2:]),
            sheetname='Sheet1',
        )

        df = date_index(df, '1/1/1947')

    elif dataset == 'bls_ls':

        data_dir = gll_dir
        infile = 'bls_labor_share.csv'

        df = pd.read_table(data_dir + infile, sep=',',
                               names=['dates', 'bls_ls'],
                               usecols=['bls_ls'])
        df = date_index(df, '1/1/1947')
        # start = datetime.datetime(1900, 1, 1)
        # end = datetime.date.today()
        # df_fred = web.DataReader(['GDPC1'], "fred", start, end)
        # df = pd.merge(df, df_fred, left_index=True, right_index=True) 
        # df['bls_cap_inc'] = np.log((1.0 - df['bls_ls']) * df['GDPC1'])
        # df['log_bls_ls'] = np.log(df['bls_ls'])

    elif dataset == 'fred':

        var_index = {
            'cpi_deflator' : 'CPIAUCSL',
            'pce_deflator' : 'PCEPI',
            'real_gdp' : 'GDPC1',
        }

        var_list = sorted(list(var_index.keys()))
        codes = [var_index[var] for var in var_list]

        start = datetime.datetime(1900, 1, 1)
        end = datetime.datetime.today()
        df = web.DataReader(codes, "fred", start, end)
        df.rename(columns = {code : var for var, code in zip(var_list, codes)}, inplace=True)

    elif dataset == 'payouts':

        df = load(['nipa_11400', 'fof'], reimport=True)
        df['net_payouts'] = (df['FOF_net_dividends'] + df['NIPA_11400_net_interest_corp_nonfin']
                             - df['FOF_net_new_equity'] - df['FOF_net_new_paper'] - df['FOF_net_new_bonds'])
        neg_ix = df['net_payouts'].values < 0.0
        if np.any(neg_ix):
            max_negative_ix = int(np.amax(np.arange(len(df))[neg_ix]))
            df = df.ix[max_negative_ix + 1:, :]

        df = df['net_payouts'].to_frame() 

    elif dataset == 'fernald':

        df = pd.read_excel(
            gll_dir + 'quarterly_tfp.xls',
            sheetname='quarterly',
            skiprows=1,
        )

        df['tfp_util'] = np.cumsum(df['dtfp_util'] / 400.0)
        df['tfp'] = np.cumsum(df['dtfp'] / 400.0)

        df = date_index(df, '01/01/1947')

    elif dataset[:4] == 'nipa':

        nipa_vintage = kwargs.get('nipa_vintage', '1604')
        data_dir = base_dir + 'nipa/' + nipa_vintage + '/'

        # TODO: allow annual?
        table = dataset[5:]
        sheetname = table + ' Qtr'

        i_file = table[0]

        # File names
        curr_file_path = data_dir + 'Section{}All_xls.xls'.format(i_file)
        hist_file_path = data_dir + 'Section{}All_Hist.xls'.format(i_file)

        # # Table name
        # table_name = '{0:d}{1:0.2d}{2:0.2d}'.format(*table)
        # if freq == 'Q':
            # table_name += ' Qtr'
        # elif freq == 'A':
            # table_name += ' Ann'
        # else:
            # print("Bad frequency")
            # raise Exception

        # Load current file
        df_t = pd.read_excel(
            curr_file_path,
            sheetname=sheetname,
            skiprows=7,
            # header=[0, 1],
            index_col=2,
        )
        df_curr = clean_nipa(df_t)
        # df_curr = df_curr.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        df_curr = df_curr.convert_objects(convert_dates=False, convert_numeric=True)

        # Load historical file
        df_t = pd.read_excel(
            hist_file_path,
            sheetname=sheetname,
            skiprows=7,
            # header=[0, 1],
            index_col=2,
        )
        df_hist = clean_nipa(df_t)
        # df_hist = df_hist.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        df_hist = df_hist.convert_objects(convert_dates=False, convert_numeric=True)

        if table == '10106':
       
            var_index = {
                'real_gdp' : 'A191RX1',
                'real_pce' : 'DPCERX1',
                'real_private_inv' : 'A006RX1',
                'real_net_exports' : 'A019RX1',
                'real_govt_expend' : 'A822RX1',
            }

        elif table == '10109':

            var_index = {
                'pce_deflator' : 'DPCERD3',
            }

        elif table == '11400':

            # Corporate nonfinancial
            cnf_index = {
                # Nonfinancial
                # 'gross_value_added' : 'A455RC1',
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
                # 'gross_value_added_chained' : 'B455RX1',
                'net_value_added_chained' : 'A457RX1',
            }

            # Total corporate
            corp_index = {
                'cons_fixed_cap' : 'A438RC1',
                'net_value_added' : 'A439RC1',
                'compensation' : 'A442RC1',
                'wage_sal' : 'A443RC1',
                'wage_sal_supp' : 'A444RC1',
                'taxes' : 'W321RC1',
                'net_op_surplus' : 'W322RC1',
                'net_interest' : 'A453RC1',
                'transfer_payments' : 'W323RC1',
                'profits' : 'A445RC1',
                'taxes' : 'A054RC1',
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

        elif table == '20100':

            # var_index = {
                # 'wage_sal' : 'A576RC1',
                # 'transfer_payments' : 'A577RC1',
                # 'proprieters_income' : 'A041RC1',
                # 'personal_contribs'


                # 'disposable_income' : 'A067RC1',
                # 'personal_income' : 'A065RC1',
                # 'transfers' : 'A577RC1',
                # 'employee_contributions' : 'A061RC1',
                # 'personal_taxes' : 'W055RC1',
                # 'real_disp_inc' : 'A067RX1',
                # 'real_pc_disp_inc' : 'A229RX0',
            # }

            # if nipa_vintage == '1604':
            var_index = {
                # 'wage_sal' : 'A576RC1',
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
                'real_disp_inc' : 'A067RX1',
                'real_pc_disp_inc' : 'A229RX0',
            }
            # elif nipa_vintage == '1302':
                # var_index = {
                    # 'wage_sal' : 'A576RC1',
                    # 'transfer_payments' : 'A577RC1',
                    # 'employer_pension_ins' : 'B040RC1',
                    # 'personal_social' : 'A061RC1',
                    # 'employer_social' : 'B039RC1',
                    # 'proprietors_income' : 'A041RC1',
                    # 'rental_income' : 'A048RC1',
                    # 'personal_dividends' : 'B703RC1',
                    # 'personal_interest_income' : 'A064RC1',
                    # 'personal_current_taxes' : 'W055RC1',
                # } 
            # else:
                # raise Exception

            if nipa_vintage == '1604':
                var_index.update({
                    'wage_sal' : 'A034RC1',
                })
            elif nipa_vintage == '1302':
                var_index.update({
                    'wage_sal' : 'A576RC1',
                })

        # prefix = table_name.replace(' ', '_')
        # var_index = {table_str + '_' + key : val for key, val in var_index.items()}

        full_list = sorted(list(var_index.keys()))

        start_date = df_curr.index[0]
        df = df_hist.ix[:start_date, :].append(df_curr)

        codes = [var_index[var] for var in full_list]
        df = df.ix[:, codes]
        df.rename(columns = {code : var for var, code in zip(full_list, codes)}, inplace=True)

        if table == '11400':
            df['earnings_corp'] = df['after_tax_profits_corp'] + df['net_interest_corp']
            df['earnings_corp_nonfin'] = df['after_tax_profits_corp_nonfin'] + df['net_interest_corp_nonfin']
        elif table == '20100':
            df['employee_net_social'] = df['personal_social'] - df['employer_social']
            df['total_other'] = df['proprietors_income'] + df['rental_income'] + df['dividends'] + df['interest']
            df['tax_share'] = df['wage_sal'] / (df['wage_sal'] + df['total_other'])
            df['tax'] = df['tax_share'] * df['personal_current_taxes']
            df['nyd'] = (df['wage_sal'] + df['transfer_payments'] + df['employer_pension_ins']
                         - df['employee_net_social'] - df['tax'])
            df['comp_no_transfers'] = df['wage_sal'] + df['employer_pension_ins'] + df['employer_social']
            df['total_comp'] = df['comp_no_transfers'] + df['transfer_payments']

    elif dataset == 'shiller':

        # colnames = [
            # 'Date', 'P', 'D', 'E', 'CPI', 'Date Frac', 'GS10', 
            # 'Real P', 'Real D', 'Real E', 'CAPE',
        # ]

        colnames = [
            'Date', 'P', 'D', 'E', 'CPI', 'date_frac', 'GS10', 
            'real_P', 'real_D', 'real_E', 'CAPE',
        ]

        df_m = pd.read_excel(
            base_dir + 'shiller/ie_data.xls',
            sheetname='Data',
            skiprows=7,
            # colnames=colnames,
            # usecols=usecols,
        )

        df_m = df_m.rename(columns={
            'Rate GS10' : 'GS10',
            'Fraction' : 'date_frac',
            'Price' : 'real_P',
            'Dividend' : 'real_D',
            'Earnings' : 'real_E',
        })

        df_m = df_m[colnames]
        df_m = date_index(df_m, '01/01/1871', freq='MS')

        methods_vars = {
            # 'sum' : ['Dividend', 'Earnings'],
            # 'last' : ['Price', 'CAPE'],
            'sum' : ['real_D', 'real_E'],
            'last' : ['real_P', 'CAPE'],
        }

        df = resample(df_m, methods_vars)

    elif dataset == 'tb3ms':

        df_m = pd.read_csv(
            gll_dir + 'TB3MS.csv',
            names=['date', 'tb3ms'],
            skiprows=1,
            usecols=['tb3ms'],
        ) 

        df_m['rf'] = 0.25 * np.log(1.0 + 0.01 * df_m['tb3ms'])
        df_m = df_m['rf'].to_frame()

        df_m = date_index(df_m, '01/01/1934', freq='MS')

        df = resample(df_m, {'first' : 'rf'}).to_frame()

    elif dataset == 'uc':

        # usecols = ['u{:02d}'.format(ii + 1) for ii in range(12)]

        # df_m = pd.read_excel(
            # gll_pred_dir + 'macro_uncertainty.xlsx',
            # sheetname='data',
            # usecols=usecols,
        # )

        # df_m = date_index(df_m, '7/01/1960', freq='MS')

        usecols = ['{:d}'.format(ii + 1) for ii in range(12)]

        df_m = pd.read_excel(
            gll_pred_dir + 'ut_cons.xls',
            sheetname='All',
            # skiprows=1,
            names=(['Dates'] + usecols),
            # usecols=usecols,
        )

        df_m = df_m[usecols]
        df_m = date_index(df_m, '07/01/1960', freq='MS')

        methods_vars = {
            'first' : usecols,
        }

        df = resample(df_m, methods_vars)

    return df

def clean_nipa(df_t):

    df_t = df_t.ix[df_t.index != ' ', :]
    df_t = df_t.ix[pd.notnull(df_t.index), :]
    del df_t['Line']
    del df_t['Unnamed: 1']

    df = df_t.transpose()
    start_date = df.index[0]
    yr = int(np.floor(start_date))
    q = int(10 * (start_date - yr) + 1)
    mon = int(3 * (q - 1) + 1)

    date_index(df, '{0}/1/{1}'.format(mon, yr))

    return df
