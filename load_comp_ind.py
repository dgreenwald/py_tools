#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:55:54 2018

@author: MaryGong
"""


##Code for merging compustat with fincov stuff

import sys
sys.path.insert(0, "/Users/MaryGong/Desktop/RA_work")
import os
import matplotlib

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


#set directory
#data_dir = os.environ.get('FIRM_CREDIT_DIR', '/home/dan/Dropbox/firm_credit/')
data_dir ='/Users/MaryGong/Dropbox (MIT)/firm_credit/'
#data_dir = '/home/dan/Dropbox/firm_credit/'





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

def keep_industries_only(df, industries, keep_cols):
    df = clean(df)
    cols = pd.Index.intersection(df.columns, keep_cols)
    a=[str(x)[:2] for x in df['naicsh']]
    a=pd.DataFrame(a, columns=['twodigit'])
    df=pd.concat([df,a], axis=1, join='inner' )
    ix = df['twodigit'].isin(industries)
    df=df.loc[ix, cols]
    return df

def read_industries_only(file, industries, keep_cols, filetype):
    reader = pd.read_sas(file)
    return keep_industries_only(reader,industries, keep_cols)
                     
    
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

def get_industry_companies(industrycodes, data_dir):
    industry_file=data_dir+'industry_file.feather'  
    if os.path.exists(industry_file):
        ind=pd.read_feather(industry_file)
    else:
        ind_cols=['gvkey', 'consol', 'popsrc', 'fyr', 'naicsh', 'sich', 'datadate']
        ind = read_industries_only(data_dir+'compustat/co_industry.sas7bdat', industrycodes, ind_cols, 'sas')
        ind['datadate']=pd.to_datetime(ind['datadate'].astype(np.int64), unit='D', origin=pd.Timestamp('1960-01-01'))
        ind['gvkey']=ind['gvkey'].astype(np.int64)
        ind = ind.groupby(['gvkey','consol', 'popsrc', 'fyr']).apply(lambda x: x.set_index('datadate').resample('Q',convention='end').max())
        
        ind['gvkey'] = ind.index.get_level_values(0)
        ind['consol'] = ind.index.get_level_values(1)
        ind['popsrc'] = ind.index.get_level_values(2)
        ind['fyr'] = ind.index.get_level_values(3)
        ind['datadate'] = ind.index.get_level_values(4)
        ind=ind.set_index(np.arange(len(ind)))
        
        ind['year']=pd.DatetimeIndex(ind['datadate']).year
    
        #Need to interpolate industries for quarters
        indpre=ind[['gvkey', 'consol', 'popsrc', 'fyr', 'year','datadate']].copy()

        foo={}
        interplist=['naicsh','sich']
        group=ind.groupby(['gvkey', 'consol', 'popsrc', 'fyr', 'year'])
        for var in interplist:
            foo[var]=group[var].max()

        foo=pd.DataFrame(data=foo)
        foo['gvkey'] = foo.index.get_level_values(0)
        foo['consol'] = foo.index.get_level_values(1)
        foo['popsrc'] = foo.index.get_level_values(2)
        foo['fyr'] = foo.index.get_level_values(3)
        foo['year']=foo.index.get_level_values(4)
        foo=foo.set_index(np.arange(len(foo)))

        ind=pd.merge(indpre, foo, how='outer', on=['gvkey', 'consol', 'popsrc', 'fyr', 'year'])
        ind.to_feather(industry_file)
    return (ind, ind['gvkey'].unique())



def merge_compustat_industries(industrycodes, keep_cols, data_dir, datetime):
    chunksize = 10000
    print("Loading compustat data, {} observations at a time...".format(chunksize))

    ind,ind_companies=get_industry_companies(industrycodes, data_dir)
    compustat = read_ds_companies_only(data_dir+'compustat/co_ifndq.csv',ind_companies, keep_cols, 'csv', chunksize)
  
        #convert to date-time
    if datetime==1:
        compustat['datadate'] = pd.to_datetime(compustat['datadate'].astype(np.int64), unit='D', origin=pd.Timestamp('1960-01-01'))
     
    #LAG ALL VARIABLES JUST IN CASE I NEED THEM LATER   
    compustat = clean(compustat)
    #Create lagged and annualized variables, and relevant ratios


    if 'oibdpq' in keep_cols:
        compustat['oibdpq']=compustat['oibdpq']*1000000                

    return (ind, compustat)



def final_merge(data_dir, datetime, industrycodes, qvars):
    


    ################################################################################
    # LOAD COMPUSTAT DATA
    ################################################################################
    keep_cols_q=qvars

    ind, comp_ds_q=merge_compustat_industries(industrycodes, keep_cols_q, data_dir, datetime)
    print("finished loading quarterly")
    comp_ds_q = comp_ds_q.groupby(['gvkey','indfmt', 'datafmt', 'consol', 'popsrc', 'fyr']).apply(lambda x: x.set_index('datadate').resample('Q',convention='end').mean())

    comp_ds_q['gvkey']=comp_ds_q.index.get_level_values(0)
    comp_ds_q['indfmt']=comp_ds_q.index.get_level_values(1)
    comp_ds_q['datafmt']=comp_ds_q.index.get_level_values(2)
    comp_ds_q['consol']=comp_ds_q.index.get_level_values(3)
    comp_ds_q['popsrc']=comp_ds_q.index.get_level_values(4)
    comp_ds_q['fyr']=comp_ds_q.index.get_level_values(5)
    comp_ds_q['datadate'] = comp_ds_q.index.get_level_values(6)
    comp_ds_q=comp_ds_q.set_index(np.arange(len(comp_ds_q)))
    comp_ds_q=comp_ds_q.sort_values(by=['gvkey', 'datadate'])    
    
    ind['fyr']=ind['fyr'].astype(np.int64)
    #Merge quarterly and annual datasets
    comp_ds_q=pd.merge(comp_ds_q, ind, how='outer',  on=['gvkey','consol','popsrc','datadate', 'fyr'])
    comp_ds_q=comp_ds_q.sort_values(by=['gvkey', 'datadate'])
    
    comp_ds_q=comp_ds_q.set_index(np.arange(len(comp_ds_q)))
    
    #interpolate the missing industry values for companies that are not listed in multiple industries
    
    ix= np.isnan(comp_ds_q['naicsh'])
    missingdf=comp_ds_q.loc[ix].copy()
    missinglist=missingdf['gvkey'].unique()
    
    iy=comp_ds_q['gvkey'].isin(missinglist)
    missingcomp=comp_ds_q.loc[iy].copy()
    groupmissing=missingcomp.groupby('gvkey')
    foo={}
    foo['max']=groupmissing['naicsh'].max()
    foo['min']=groupmissing['naicsh'].min()
    foo=pd.DataFrame(foo)
    foo['gvkey']=foo.index.get_level_values(0)
    foo=foo.set_index(np.arange(len(foo)))
    foo['naicsh']=np.nan
    ix=foo['max']==foo['min']
    foo.loc[ix, 'naicsh']=foo['max']
    iy=~np.isnan(foo['naicsh'])
    foo=foo.loc[iy]
    foo=foo[['gvkey', 'naicsh']]
    
    foo['naicsh2']=foo['naicsh']
    foo=foo[['gvkey', 'naicsh2']]
    
    
    comp_ds_q=pd.merge(comp_ds_q, foo, how='outer', on='gvkey')
    
    comp_ds_q['naicsh']=comp_ds_q[['naicsh','naicsh2']].max(axis=1)
    
    
    
    
    

    
    comp_ind_file= data_dir + 'comp_ind.feather'
    comp_ds_q.to_feather(comp_ind_file)
    
    return (comp_ds_q)