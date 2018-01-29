import os
import numpy as np
import pandas as pd
import py_tools.time_series as ts

from . import defaults
default_dir = defaults.base_dir() + 'saez/'

def update_names(df, usecols):

    mapping = {
        ii : name for ii, name in enumerate(usecols)
    }

    if len(usecols) < len(df.columns):
        mapping.update({
            ii : 'TEMP_VARIABLE_{}'.format(ii) 
            for ii in range(len(usecols), len(df.columns))
        })

    return df.rename(columns=mapping)[usecols]

# data_dir = '/home/dan/Dropbox/data/saez/'

def load(table='shares', reimport=False, data_dir=default_dir):

    pkl_file = table + '.pkl'

    if reimport or not os.path.exists(pkl_file):

        if table == 'shares':

            xls_file = data_dir + 'TabFig2015prel' + '.xls'

            base_names = [
                'year',
                'top_10', 'top_5', 'top_1', 'top_0_5', 'top_0_1', 'top_0_01',
                'blank',
                'top_10_to_5', 'top_5_to_1', 'top_1_to_0_5', 'top_0_5_to_0_1', 'top_0_1_to_0_01',
            ]

            tables = ['A2', 'A3']
            labels = ['excl', 'incl']

            df_list = []
            for ii, (table, label) in enumerate(zip(tables, labels)):
                full_table = 'Table ' + table
                prefix = 'share_{}_cg_'.format(label)
                df_list.append(pd.read_excel(
                    xls_file, sheet_name=full_table, skiprows=5,
                    skip_footer=4, header=None,
                    names=[prefix + name for name in base_names],
                ))
                df_list[ii].drop([prefix + 'year', prefix + 'blank'], axis=1, inplace=True)
                df_list[ii] = ts.date_index(df_list[ii], '1913-01-01', freq='AS')

            df = ts.merge_date_many(df_list)

        elif table == 'sources':

            xls_file = data_dir + 'TabFig2015prel' + '.xls'

            names = []
            drop_list = []
            pct_list = ['10', '5', '1', '0_5']
            for pct in pct_list:

                var_list = ['year', 'wage', 'entrep', 'divid', 'interest', 'rents']
                drop_list += ['year_p' + pct]
                if pct != '0_5':
                    var_list += ['blank']
                    drop_list += ['blank_p' + pct]

                names += [
                    '{0}_p{1}'.format(var, pct) for var in var_list
                ] 

            df = pd.read_excel(
                xls_file, sheet_name='Table A7', skiprows=6,
                skip_footer=6, header=None,
                names=names,
            )

            df.drop(drop_list, axis=1, inplace=True)
            df = ts.date_index(df, '1916-01-01', freq='AS')

        elif table == 'wealth_by_asset':

            xls_file = data_dir + 'AppendixTablesAggregates.xlsx'

            names = []
            usecols = ['year', 
                     #1-5
                     'net_hh_wealth', 'net_housing', 'owner_gross_housing',
                     'mortgages_owner', 'tenant_gross_housing',
                     #6-10
                     'mortages_tenant', 'equities', 'equities_non_s',
                     'equities_s_corps', 'fixed_income',
                     #11-15
                     'taxable_bonds', 'munis', 'non_interest',
                     'non_mortg_debt', 'prop_and_part',
                     #16-19
                     'pensions_and_ins', 'pensions', 'life_ins', 'iras',
                     #20-23 (excluded)
                     'unfunded_pens', 'funded_social', 'consumer_durables', 'npish_net_wealth'
                     ]

            # names = ['year'] + usecols + ['blank_{}'.format(ii) for ii in range(6)]

            df = pd.read_excel(
                xls_file, sheet_name='TableA1', skiprows=9, header=None,
                skipfooter=9, 
                # names=names, 
            )

            df = update_names(df, usecols)
            df.drop(['year'], axis=1, inplace=True)            

            df = ts.date_index(df, '1913-01-01', freq='AS')

        elif table == 'distribution_by_asset':

            xls_file = data_dir + 'AppendixTablesDistributions.xlsx'

            assets = ['equity', 'net_housing', 'business', 'fixed_income',
                      'non_mortg_debt', 'total_debt']

            pcts = ['bottom_90', 'top_10', 'top_5', 'top_1', 'top_0_5', 'top_0_1', 'top_0_01',
                    'top_10_to_1', 'top_10_to_5', 'top_5_to_1', 'top_1_to_0_1', 'top_1_to_0_5',
                    'top_0_5_to_0_1', 'top_0_1_to_0_01', 'share_of_tot']

            df_list = len(assets) * [None]
            for ii, asset in enumerate(assets):

                table = 'TableB{}'.format(ii + 7) 

                usecols = ['year'] + ['{0}_{1}'.format(asset, pct) for pct in pcts]

                df_list[ii] = pd.read_excel(
                    xls_file, sheet_name=table, skiprows=8, header=None,
                     skipfooter=4, 
                )

                df_list[ii] = update_names(df_list[ii], usecols)

                start_year = int(df_list[ii]['year'].iloc[0])
                print('ii = {}'.format(ii))
                print('asset = ' + asset)
                print('{}-01-01'.format(start_year))
                df_list[ii].drop(['year'], axis=1, inplace=True)
                df_list[ii] = ts.date_index(df_list[ii], '{}-01-01'.format(start_year), freq='AS')

                print(df_list[ii].head(10))

            df = ts.merge_date_many(df_list)

        # elif table == ...

        # Save pickle file
        df.to_pickle(pkl_file)

    else:

        # Load pickle file
        df = pd.read_pickle(pkl_file)

    return df
