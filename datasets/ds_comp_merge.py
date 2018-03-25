#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 20:26:03 2018

@author: MaryGong
"""

##Code for merging compustat with fincov stuff


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from time import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
from py_tools import local_projections as lp
from py_tools import data as dt
from py_tools.datasets import misc





##### CODE FOR QUARTERLY COMPUSTAT. THE COVENANTS "FIXED CHARGE COVERAGE" and
#Max. Senior Debt-to- EBITDA" and "Max. Senior Leverage" and "CAPEX" are omitted because lack of
#quarterly variables that correspond directly to annual ones. 

################################################################################
# PRELIMINARIES
################################################################################


# Name of data directory
#data_dir = '/Users/MaryGong/Dropbox (MIT)/firm_credit/'
#data_dir = '/home/dan/Dropbox/firm_credit/'
#data_dir = '/nobackup1/dlg/firm_credit/'




################################################################################
# DEFINE FUNCTIONS
################################################################################

def read_single_chunk(file, keep_cols=None, chunksize=10000):
    
    print("NOTE: NOT LOADING FULL FILE")
    reader = pd.read_sas(file, chunksize=chunksize)
    for df in reader:
        break
    
    if keep_cols is not None:
        df = df[pd.Index.intersection(df.columns, keep_cols)] 
        
    return df

def read_by_chunk(file, keep_cols, chunksize=10000):
    
    reader = pd.read_sas(file, chunksize=chunksize)
    return pd.concat([chunk[pd.Index.intersection(chunk.columns, keep_cols)] 
                            for chunk in reader])
    
def keep_ds_companies_only(df, companies, keep_cols):

    df = clean(df)
    cols = pd.Index.intersection(df.columns, keep_cols)
    ix = df['gvkey'].isin(companies)
    return df.loc[ix, cols]
    
def read_ds_companies_only(file, companies, keep_cols, filetype,chunksize=10000):
    
    if filetype=='csv':
        reader = pd.read_csv(file, chunksize=chunksize)
        return pd.concat([keep_ds_companies_only(chunk, companies, keep_cols)
                      for chunk in reader])
    elif filetype=='sas':
        reader = pd.read_sas(file, chunksize=chunksize)
        return pd.concat([keep_ds_companies_only(chunk, companies, keep_cols)
                      for chunk in reader])
    else:
        print('error')
        return
    
def clean(df):
    
    for var in ['PackageID', 'BorrowerCompanyID', 'gvkey']:
        if var in df:
            df[var] = df[var].astype(np.int64)
            
    for var in ['comment']:
        if var in df:
            df.drop(var, axis=1, inplace=True)
            
    return df


def ratio(data, numerator, denominator, name):
    #Find where denomintor is zero
    ix_denom= denominator==0
    data.loc[ix_denom, name] = 666
    data.loc[~ix_denom, name] = numerator/denominator
    data[name]=pd.to_numeric(data[name])
    return 



def toc(start):
    end = time()
    print("Time elapsed: {:3.2g}s".format(end - start))
    return None



def annualize(varlist, file, group):
    for var in varlist:
        file[var+'_ann']=4*group[var].shift(0)
    return 


def merge_compustat(keep_cols, annualize_vars, file, data_dir, companies, ds_linked, covenant_list, datetime):
    chunksize = 10000
    print("Loading compustat data, {} observations at a time...".format(chunksize))
    start = time()
    #Load the relevant datasets
    if file=='quarterly':
        compustat = read_ds_companies_only(data_dir+'compustat/co_ifndq.csv',companies, keep_cols, 'csv', chunksize)
    elif file=='annual':
        compustat1 = read_ds_companies_only(data_dir+'compustat/co_afnd1.sas7bdat',companies, keep_cols,'sas', chunksize)
        compustat2 = read_ds_companies_only(data_dir+'compustat/co_afnd2.sas7bdat',companies, keep_cols,'sas', chunksize)
        compustat =pd.merge(compustat1, compustat2, how='outer', on=['gvkey', 'datadate', 'indfmt', 'datafmt', 'consol'])
    else:
        print('error')
        return
        #convert to date-time
    

    if datetime==1:
        compustat['datadate'] = pd.to_datetime(compustat['datadate'].astype(np.int64), unit='D', origin=pd.Timestamp('1960-01-01'))
    

    group=compustat.groupby(['gvkey','indfmt', 'datafmt', 'consol' ])
    
    #LAG ALL VARIABLES JUST IN CASE I NEED THEM LATER

    

    compustat = clean(compustat)
    #Create lagged and annualized variables, and relevant ratios
    if file=='quarterly':
        qvars=['oibdpq', 'xintq', 'dlcq', 'dlttq', 'atq','intanq', 'ltq', 
               'actq','lctq', 'rectq', 'cheq', 'saleq', 'revtq', 'xoprq', 
               'ivstq', 'ivltq' ]
        
        for var in qvars:
            if var in keep_cols:
                compustat[var+'1']=group[var].shift(1)

        if 'oibdpq' in keep_cols:
            compustat['oibdpq']=compustat['oibdpq']*1000000
        annualize(annualize_vars, compustat, group)
                  
    if file=='annual':
        if 'dlc' in keep_cols:
            compustat['dlc1'] = group['dlc'].shift(1)
        if 'oibdp' in keep_cols:
            compustat['oibdp']=compustat['oibdp']*1000000
        if 'capx' in keep_cols:
            compustat['capx']=compustat['capx']*1000000
        return (compustat)
   

    toc(start) # Report time
    print("Merging datasets...")
    start = time()

    #drop all companies from compustat dataset who are not in dealscan
    #    df = df[df['gvkey'].isin(companies)]

    # Create linked dataset
    ds_date = pd.merge(ds_linked, compustat[['gvkey', 'datadate']], how='inner', on='gvkey')
    #ds_date = pd.merge(ds_date, compustat[['gvkey', 'datadate']], how='inner', on='gvkey')
    # Go through and see if covenant is active at that date
    ix_inactive = np.logical_or(
        ds_date['datadate'] < ds_date['MinStartDate'],
        ds_date['datadate'] > ds_date['MaxEndDate'],
        )

    # Drop all inactive values
    for cov in covenant_list:
        ds_date.loc[ix_inactive, cov] = np.nan
        ds_date.loc[ix_inactive, cov + ' Count'] = np.nan

    # Take most binding active covenant to get down to company level
    gb = ds_date.groupby(['gvkey', 'datadate'])
    ds_by_company_date_data = {}
    for cov in covenant_list:
    
        # Count all covenants of that type in package
        ds_by_company_date_data[cov + ' Count'] = gb[cov + ' Count'].sum()
        # Get binding ratios
        if cov.startswith("Min"):
            ds_by_company_date_data[cov] = gb[cov].min()
        elif cov.startswith("Max"):
            ds_by_company_date_data[cov] = gb[cov].max()
        else:
            #        print("Bad covenant name: {}".format(cov))
            ds_by_company_date_data[cov] = gb[cov].mean()

    # Company-Date level covenant info        
    ds_by_company_date = pd.DataFrame(data=ds_by_company_date_data)
    
    # Merge into full compustat
    comp_ds = pd.merge(compustat, ds_by_company_date, left_on=['gvkey', 'datadate'], right_index=True, how='outer')
    toc(start) # Report time
    
    return (comp_ds)

def distance_to_covenant(covname, data):
    if covname.startswith("Min"):
        data[covname+' distance']=data[covname+'_compustat']-data[covname]
    elif covname.startswith("Max"):
        data[covname+' distance']=data[covname]-data[covname+'_compustat']
    return

def percent_violations(group, cov):
    nv=group[cov+' distance']>=0
    v=group[cov+' distance']<0
    if (sum(v)+sum(nv))>0:
        pv=sum(v)/(sum(v)+sum(nv))
        return pv
    else: 
        return np.nan

#create new dataset which drops all companies that violate a covenant in the first year of that covenant
def filter_companies(g, covenants):
    max_violations=0
    for cov in covenants:
        if g['init distance '+cov].min()<0:
            max_violations=max_violations+1
    return (max_violations==0)

def clean_companies(covenants, data):
    #compute distance to covenant measure where distance<1 is a violation. 
    for cov in covenants:
        distance_to_covenant(cov, data)
        
    for cov in covenants:
        #throw out everything except for first occurrence of a unique value under 
        #the covenant limit (which I assume is the date the covenant starts)
        ix=data.duplicated(subset=['gvkey', 'indfmt', 'datafmt', 'consol', cov], keep='first')
        ix=np.logical_or(ix, np.isnan(data[cov]))
        ix=~ix
        data.loc[ix, cov+' first']=data[cov]
    initial_year={}
    for cov in covenants:
        if cov.startswith("Min."):
            ratio(data, data[cov+'_compustat'],  data[cov],'init distance '+cov)
        else:
            ratio(data,  data[cov], data[cov+'_compustat'],'init distance '+cov)
        data.drop(labels=cov+' first', axis=1)
        initial_year['init distance '+cov]=data['init distance '+cov].describe()
    #create descriptive table for initial year observations
    initial_year=pd.DataFrame(initial_year)
            
    
    gb_comp_ds=data.groupby(by=['gvkey','indfmt', 'datafmt', 'consol'])
    finalcompanies_comp_ds=gb_comp_ds.filter(lambda g: filter_companies(g, covenants))

    return(finalcompanies_comp_ds)


def final_merge(data_dir, datetime, avars, qvars, annualizevars, interp_list, changevars):
    
    # Read in file
    print("Loading dealscan data...")
    start = time()
    fincov = clean(pd.read_sas(data_dir + 'dealscan/financialcovenant.sas7bdat'))

    #decodes bytes into strings (gets rid of the b in front of covenant type names)
    fincov['CovenantType'] = fincov['CovenantType'].str.decode('UTF-8')

    
    # Create dummy variables
    covenant_list = fincov['CovenantType'].unique()
    for cov in covenant_list:
        ix = fincov['CovenantType'] == cov
        if cov in ['Max. Capex', 'Min. EBITDA']:
            fincov.loc[ix, cov] = fincov['InitialAmt']
        else:
            fincov.loc[ix, cov] = fincov['InitialRatio']
    
        fincov[cov + ' Count'] = 0
        fincov.loc[ix, cov + ' Count'] = 1

    # Get total number of covenants
    gb = fincov.groupby('PackageID')
    covs_by_package_data = {}

    # Get binding ratios
    for cov in covenant_list:
    
        # Count all covenants of that type in package
        covs_by_package_data[cov + ' Count'] = gb[cov + ' Count'].sum()
    
        # Get binding ratios
        if cov.startswith("Min"):
            covs_by_package_data[cov] = gb[cov].min()
        elif cov.startswith("Max"):
            covs_by_package_data[cov] = gb[cov].max()
        else:
            print("Bad covenant name: {}".format(cov))
            covs_by_package_data[cov] = gb[cov].mean()
        
        covs_by_package = pd.DataFrame(data=covs_by_package_data)
        covs_by_package['PackageID'] = covs_by_package.index

    package = clean(pd.read_sas(data_dir + 'dealscan/package.sas7bdat'))
    package = package.drop(['Currency','ExchangeRate','Ticker'], axis=1)


    # Get just first and last dates    
    cov_dates_file = data_dir + 'ds_dates.feather'
    if not os.path.exists(cov_dates_file):
    
        facility = clean(pd.read_sas(data_dir+'dealscan/facility.sas7bdat'))
        min_date = facility.groupby(['PackageID'])['FacilityStartDate'].min()
        max_date = facility.groupby(['PackageID'])['FacilityEndDate'].max()
    
        fac_dates = pd.DataFrame(data={'MinStartDate' : min_date, 'MaxEndDate' : max_date})
        fac_dates['PackageID'] = fac_dates.index
        fac_dates.index = range(len(fac_dates))
    
        fac_dates.to_feather(cov_dates_file)
    
    else:
    
        fac_dates = pd.read_feather(cov_dates_file)
    

    # Dealscan merge on PackageID
    dealscan = pd.merge(covs_by_package, package, how='outer', on='PackageID')
    dealscan = pd.merge(dealscan, fac_dates, how='outer', on='PackageID')

    toc(start) # Report time

    
    ################################################################################
    # LOAD LINKING TABLE
    ################################################################################

    print("Loading linking table...")
    start = time()
    # Dealscan and Compustat link (save to feather format if not already stored)
    link_name = 'DealScanCOMPUSTAT_Link'
    link_file = data_dir + link_name + '.feather'
    if (not os.path.exists(link_file)):
    
        xlsx_file = data_dir + link_name + '.xlsx'
        link_table = pd.read_excel(xlsx_file, sheet_name='Data').rename(
            columns={'bcoid' : 'BorrowerCompanyID'} # shorter rename command
            )
    
        link_table.to_feather(link_file)
    
    else:    
        link_table = pd.read_feather(link_file)

    link_table.drop(['facid', 'facstartdate', 'link_status'],
                       axis=1, inplace=True)

    link_table=link_table.drop_duplicates(
        subset=['BorrowerCompanyID', 'company','ticker',
                'ds_vintage','gvkey', 'coname_h','smbl_h', 'fic']
        )

    toc(start) # Report time
    
    
    
    ################################################################################
    # MERGE DEALSCAN TO LINK TABLE
    ################################################################################

    print("Merging datasets...")
    start = time()

    # Merges relevant dealscan data to the dealscan-compustat link on borrower company ID. 
    ds_linked = pd.merge(dealscan, link_table, how='inner', on='BorrowerCompanyID') 

    # Drop observations without valid start and end dates
    ds_linked = ds_linked.dropna(subset=['MinStartDate', 'MaxEndDate'])

    #Get list of companies that have dealscan information 
    companies = ds_linked.gvkey.unique()
    np.savetxt("companies_with_covenants.csv", companies, delimiter=",")



    ################################################################################
    # LOAD COMPUSTAT DATA
    ################################################################################
    keep_cols_q=qvars
    annualize_variables=annualizevars
    comp_ds_q=merge_compustat(keep_cols_q,annualize_variables, 'quarterly' , data_dir, companies, ds_linked, covenant_list, datetime)
    print("finished loading quarterly")
    comp_ds_q = comp_ds_q.groupby(['gvkey','indfmt', 'datafmt', 'consol']).apply(lambda x: x.set_index('datadate').resample('Q',convention='end').mean())

    comp_ds_q['gvkey']=comp_ds_q.index.get_level_values(0)
    comp_ds_q['indfmt']=comp_ds_q.index.get_level_values(1)
    comp_ds_q['datafmt']=comp_ds_q.index.get_level_values(2)
    comp_ds_q['consol']=comp_ds_q.index.get_level_values(3)
    comp_ds_q['datadate'] = comp_ds_q.index.get_level_values(4)
    comp_ds_q=comp_ds_q.set_index(np.arange(len(comp_ds_q)))
    
    comp_ds_q=comp_ds_q.sort_values(by=['gvkey', 'datadate'])    
    
    keep_cols_a=avars
    
    comp_ds_a=merge_compustat(keep_cols_a,annualize_variables, 'annual' , data_dir, companies, ds_linked, covenant_list, datetime)
    print("finished loading annual")
    comp_ds_a = comp_ds_a.groupby(['gvkey','indfmt', 'datafmt', 'consol']).apply(lambda x: x.set_index('datadate').resample('Q',convention='end').mean())

    comp_ds_a['gvkey']=comp_ds_a.index.get_level_values(0)
    comp_ds_a['indfmt']=comp_ds_a.index.get_level_values(1)
    comp_ds_a['datafmt']=comp_ds_a.index.get_level_values(2)
    comp_ds_a['consol']=comp_ds_a.index.get_level_values(3)
    comp_ds_a['datadate'] = comp_ds_a.index.get_level_values(4)
    comp_ds_a=comp_ds_a.set_index(np.arange(len(comp_ds_a)))
    
    comp_ds_a=comp_ds_a.sort_values(by=['gvkey', 'datadate'])

    group=comp_ds_a.groupby(['gvkey','indfmt', 'datafmt', 'consol'])
    

    #interpolate missing values
    foo={}
    for var in interp_list:
        foo[var+'_ann']=group.apply(lambda x: x[var].interpolate(method='linear'))
    foo['datadate']=group.apply(lambda x: x['datadate'])
    #comp_ds_a[var+'q']=comp_ds_a[var+'_ann']/(4*comp_ds_a['ones'])
    foo=pd.DataFrame(data=foo)
    foo['gvkey'] = foo.index.get_level_values(0)
    foo['indfmt'] = foo.index.get_level_values(1)
    foo['datafmt'] = foo.index.get_level_values(2)
    foo['consol'] = foo.index.get_level_values(3)

    foo=foo.set_index(np.arange(len(foo)))
    foo['ones']=1

    for var in interp_list:
        foo[var+'q']=foo[var+'_ann']/(4*foo['ones'])
    
    comp_ds_a=pd.merge(comp_ds_a, foo, how='outer', on=['gvkey', 'datadate', 
                                                       'indfmt', 'datafmt', 'consol'])

    interp_list_a=[s + '_ann' for s in interp_list]
    interp_list_q=[s + 'q' for s in interp_list]
    
    interp_lists=interp_list_a+interp_list_q
    
    comp_ds_a=comp_ds_a[avars+interp_lists]


    #Merge quarterly and annual datasets
    comp_ds_q=pd.merge(comp_ds_q, comp_ds_a,how='outer', on=['gvkey', 'datadate', 'indfmt', 'datafmt', 'consol'])
    comp_ds_q['ones']=1
    
    #Create relevant ratios for financial covenants
    ratio(comp_ds_q, comp_ds_q['dlttq']+comp_ds_q['dlcq'], comp_ds_q['oibdpq'], 'Max. Debt to EBITDA_compustat')
    ratio(comp_ds_q, comp_ds_q['oibdpq'], comp_ds_q['xintq'], 'Min. Interest Coverage_compustat')
    ratio(comp_ds_q, comp_ds_q['dlttq']+comp_ds_q['dlcq'], comp_ds_q['atq'], 'Max. Leverage ratio_compustat')
    ratio(comp_ds_q, comp_ds_q['actq'], comp_ds_q['lctq'], 'Min. Current Ratio_compustat')
    ratio(comp_ds_q, comp_ds_q['dlttq']+comp_ds_q['dlcq'], comp_ds_q['atq']-comp_ds_q['intanq']-comp_ds_q['ltq'], 'Max. Debt to Tangible Net Worth_compustat')# Debt to tangible net worth
    ratio(comp_ds_q, comp_ds_q['oibdpq'], comp_ds_q['xintq']+comp_ds_q['dlcq1'], 'Min. Debt Service Coverage_compustat')# Debt to service coverage
    ratio(comp_ds_q, comp_ds_q['oibdpq'], comp_ds_q['ones'], 'Min. EBITDA_compustat')
    ratio(comp_ds_q, comp_ds_q['rectq']+comp_ds_q['cheq'],  comp_ds_q['lctq'], 'Min. Quick Ratio_compustat')
    ratio(comp_ds_q, comp_ds_q['dlttq']+comp_ds_q['dlcq'],  comp_ds_q['atq']-comp_ds_q['ltq'], 'Max. Debt to Equity_compustat')
    ratio(comp_ds_q, comp_ds_q['atq']-comp_ds_q['ltq'], comp_ds_q['atq'], 'Min. Equity to Asset Ratio_compustat')
    ratio(comp_ds_q, comp_ds_q['oibdpq'],  comp_ds_q['intpn_ann'], 'Min. Cash Interest Coverage_compustat')
    ratio(comp_ds_q, comp_ds_q['oibdpq'], comp_ds_q['xintq']+comp_ds_q['xrent_ann']+comp_ds_q['dlcq1'], 'Min. Fixed Charge Coverage_compustat')
    ratio(comp_ds_q, comp_ds_q['capx_ann'],comp_ds_q['ones'], 'Max. Capex_compustat')
    ratio(comp_ds_q, comp_ds_q['dlttq']+comp_ds_q['dlcq'], comp_ds_q['oibdpq'], 'Max. Senior Debt to EBITDA_compustat')
    ratio(comp_ds_q, comp_ds_q['dlttq']+comp_ds_q['dlcq']-comp_ds_q['ds_ann'], comp_ds_q['atq'], 'Max. Senior Leverage_compustat')

    comp_ds_q['totdebtq']=comp_ds_q['dlttq']+comp_ds_q['dlcq']
    #### make dataset of percentage of violations for each covenant
    covenants_q=['Min. Current Ratio', 'Min. Interest Coverage', 'Max. Debt to EBITDA', 
           'Max. Senior Debt to EBITDA', 'Min. Fixed Charge Coverage',
           'Max. Leverage ratio', 'Min. Quick Ratio', 'Min. Debt Service Coverage', 
           'Max. Debt to Equity','Min. Cash Interest Coverage', 'Max. Capex', 
           'Min. EBITDA', 'Max. Senior Leverage', 'Min. Equity to Asset Ratio',
           'Max. Debt to Tangible Net Worth']
    

    
    #Define other interesting LHS variables
    group=comp_ds_q.groupby(['gvkey','indfmt', 'datafmt', 'consol'])
    

    for var in changevars:
        comp_ds_q['change'+var]=group[var].shift(0)-group[var].shift(1)
    

    
    
    
    
    
    
    final_comp_ds_q=clean_companies(covenants_q, comp_ds_q)
 

    return (final_comp_ds_q, comp_ds_q)