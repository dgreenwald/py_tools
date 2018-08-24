#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:29:55 2018

@author: MaryGong
"""



import sys
sys.path.insert(0, "/Users/MaryGong/Desktop/RA_work")
import pandas as pd
import numpy as np
import os
from time import time
from operator import add
from py_tools import data as dt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pylab as pl
from matplotlib import collections  as mc
#set directory
#data_dir = os.environ.get('FIRM_CREDIT_DIR', '/home/dan/Dropbox/firm_credit/')
#data_dir ='/Users/MaryGong/Dropbox (MIT)/firm_credit/'
data_dir = '/Users/MaryGong/Dropbox (MIT)/firm_credit/'



#controls = ['lag1_ebitda_rate', 'lag1_m2b', 'lag1_cash_rate',  'lag1_asset_rate', 'lag1_debt_rate']




directory=data_dir+'compustat/ds_comp_q_93_07_resampled.feather'

#directory=data_dir+'compustat/ds_comp_93_07_resampled.feather'
df_orig = pd.read_feather(directory)
df = df_orig.copy()




group=df.groupby('gvkey')

df['lead1_violation']=group['violation'].shift(-1)
df['lag1_violation']=group['violation'].shift(1)
df['lag2_violation']=group['violation'].shift(2)
df['lag3_violation']=group['violation'].shift(3)
df['lag4_violation']=group['violation'].shift(4)


#VIOLATION 0: No violation in previous period
#NEW VIOLATION: No violation in previous 2 years

df['violation0']=df['violation'].copy()
ix=df['lag1_violation']==1
df.loc[ix, 'violation0']=0

df['new_violation']=df['violation'].copy()




if directory==data_dir+'compustat/ds_comp_q_93_07_resampled.feather':
    lhs_vars = ['net_debt_issue_at_rate', 'D_net_debt_issue_at_rate',
                'capx_at_rate', 'D_capx_at_rate',] 
    
    controls=['lag1_lnat', 'lag1_tan_at_rate', 'lag1_debt_at_rate', 
          'lag1_netw_at_rate', 'lag1_ch_at_rate', 'lag1_oibdp_at_rate', 
          'oibdp_at_rate', 'lag1_xint_at_rate', 'xint_at_rate']

#Define relevant variables    
    df['lag1_at']=group['atq'].shift(1)
    df['lag1_intan']=group['intanq'].shift(1)
    df['lag1_lt']=group['ltq'].shift(1)    
    df['lag1_oibdp']=group['oibdpq'].shift(1)  
    df['lag1_xint']=group['xintq'].shift(1)  
    df['lag1_cheq']=group['cheq'].shift(1)  
    df['lag1_D_interest']=group['D_interest'].shift(1)
    df['lag1_D_interest_eop']=group['D_interest_eop'].shift(1)
    df['lag1_D_interest_eoq']=group['D_interest_eoq'].shift(1)


    df['net_debt_issue_at_rate']=df['net_debt_issue']/df['lag1_at']
    df['capx_at_rate']=df['capxq']/df['lag1_at']

  
    df['lag1_lnat']=np.log(df['lag1_at'])
    df['lag1_tan_at_rate']=df['lag1_ppentq']/df['lag1_at']
    df['lag1_debt_at_rate']=df['lag1_debt']/df['lag1_at']
    df['lag1_ch_at_rate']=df['lag1_cheq']/df['lag1_at']
    df['lag1_oibdp_at_rate']=df['lag1_oibdp']/df['lag1_at']
    df['oibdp_at_rate']=df['oibdpq']/df['lag1_at']
    df['lag1_xint_at_rate']=df['lag1_xint']/df['lag1_at']
    df['xint_at_rate']=df['xintq']/df['lag1_at']

    df['lag1_netw_at_rate']=(df['lag1_at']-df['lag1_lt'])/df['lag1_at']

    ix=np.logical_or(ix, df['lag2_violation']==1)
    ix=np.logical_or(ix, df['lag3_violation']==1)
    ix=np.logical_or(ix, df['lag4_violation']==1)
    df.loc[ix, 'new_violation']=0

elif directory==data_dir+'compustat/ds_comp_93_07_resampled.feather':
 
    controls=['lag1_lnat', 'lag1_m2b', 'lag1_tan_at_rate', 'lag1_debt_at_rate', 
          'lag1_netw_at_rate', 'lag1_ch_at_rate', 'lag1_oibdp_at_rate', 
          'oibdp_at_rate', 'lag1_xint_at_rate', 'xint_at_rate']

    lhs_vars = ['net_debt_issue_at_rate', 'D_net_debt_issue_at_rate','gross_debt_issue_at_rate',
                 'D_gross_debt_issue_at_rate','capx_at_rate', 
                 'D_capx_at_rate',] 


#Define relevant variables    
    df['lag1_xint']=group['xint'].shift(1)      
    df['lag1_oibdp']=group['oibdp'].shift(1)        
    df['lag1_lt']=group['lt'].shift(1)       
    df['lag1_at']=group['at'].shift(1)
    df['lag1_intan']=group['intan'].shift(1)
    df['lag1_D_interest']=group['D_interest'].shift(1)
    df['lag1_D_interest_eop']=group['D_interest_eop'].shift(1)
    df['lag1_D_interest_eoy']=group['D_interest_eoy'].shift(1)    
    
    df['net_debt_issue_at_rate']=df['net_debt_issue']/df['lag1_at']
    df['gross_debt_issue_at_rate']=df['dltis']/df['lag1_at']
    df['capx_at_rate']=df['capx']/df['lag1_at']     

    
    df['lag1_lnat']=np.log(df['lag1_at'])
    df['lag1_tan_at_rate']=(df['lag1_at']-df['lag1_intan'])/df['lag1_at']
    df['lag1_debt_at_rate']=df['lag1_debt']/df['lag1_at']
    df['lag1_ch_at_rate']=df['lag1_cash']/df['lag1_at']
    df['lag1_oibdp_at_rate']=df['lag1_oibdp']/df['lag1_at']
    df['oibdp_at_rate']=df['oibdp']/df['lag1_at']
    df['lag1_xint_at_rate']=df['lag1_xint']/df['lag1_at']
    df['xint_at_rate']=df['xint']/df['lag1_at']

    df['lag1_netw_at_rate']=(df['lag1_at']-df['lag1_lt'])/df['lag1_at']

    ix=np.logical_or(ix, df['lag2_violation']==1)
    df.loc[ix, 'new_violation']=0
   
    
    
#    df['lead2_violation']=group['violation'].shift(-2)
#    ix=np.logical_and(pd.isnull(df['lead2_violation']),df['violation']==1)

"""
df_dropouts=df.loc[ix].copy()

dropouts=df_dropouts['gvkey'].unique()
dropouts=pd.DataFrame(data=dropouts, columns=['gvkey'])
dropouts['dropped']=1


df_nodrops=df.copy()
df_nodrops=pd.merge(dropouts, df_nodrops, how='outer', on='gvkey')
ix=df_nodrops['dropped']==1
df_nodrops=df_nodrops.loc[~ix].copy()

df_nodrops=df_nodrops.set_index(np.arange(len(df_nodrops)))

"""





# Drop infs
df = df.replace([np.inf, -np.inf], np.nan)

# New violation: no violation in previous 2 years
# Violation 0: first year of violation (no violation in the previous year)




df=df.sort_values(['gvkey', 'datadate'])
df=df.set_index(np.arange(len(df)))
group=df.groupby('gvkey')



df['lag1_net_debt_issue_at_rate']=group['net_debt_issue_at_rate'].shift(1)
df['lag1_capx_at_rate']=group['capx_at_rate'].shift(1)

df['D_net_debt_issue_at_rate']=df['net_debt_issue_at_rate']-df['lag1_net_debt_issue_at_rate']
df['D_capx_at_rate']=df['capx_at_rate']-df['lag1_capx_at_rate']


leadlags=[1, 2, 3, 4, 5]

for ll in leadlags:
    df['lead'+str(ll)+'_new_violation']=group['new_violation'].shift(-ll)
    df['lag'+str(ll)+'_new_violation']=group['new_violation'].shift(ll)
    df['lead'+str(ll)+'_violation0']=group['violation0'].shift(-ll)
    df['lag'+str(ll)+'_violation0']=group['violation0'].shift(ll)



if directory==data_dir+'compustat/ds_comp_q_93_07_resampled.feather':  
    leads=list(range(1, 16))
    lags=list(range(-16, 0))
    newvioll4=[]
    vio0ll4=[]
    for l in lags:
        newvioll4.append('lag'+str(-l)+'_new_violation')
        vio0ll4.append('lag'+str(-l)+'_violation0')
    
    newvioll4.append('new_violation')
    vio0ll4.append('violation0')
    
    for l in leads:
        newvioll4.append('lead'+str(l)+'_new_violation')
        vio0ll4.append('lead'+str(l)+'_violation0')


    leads=list(range(1, 8))
    lags=list(range(-8, 0))
    newvioll2=[]
    vio0ll2=[]
    for l in lags:
        newvioll2.append('lag'+str(-l)+'_new_violation')
        vio0ll2.append('lag'+str(-l)+'_violation0')
    
    newvioll2.append('new_violation')
    vio0ll2.append('violation0')
    
    for l in leads:
        newvioll2.append('lead'+str(l)+'_new_violation')
        vio0ll2.append('lead'+str(l)+'_violation0')


elif directory==data_dir+'compustat/ds_comp_93_07_resampled.feather': 
    df['lag1_gross_debt_issue_at_rate']=group['gross_debt_issue_at_rate'].shift(1)

    df['D_gross_debt_issue_at_rate']=df['gross_debt_issue_at_rate']-df['lag1_gross_debt_issue_at_rate']

    
    
    
    newvioll4=['lag4_new_violation','lag3_new_violation','lag2_new_violation', 'lag1_new_violation' ,
               'new_violation', 'lead1_new_violation', 'lead2_new_violation', 
               'lead3_new_violation', 'lead4_new_violation']
    vio0ll4=['lag4_violation0','lag3_violation0','lag2_violation0', 'lag1_violation0' , 'violation0', 
               'lead1_violation0', 'lead2_violation0', 'lead3_violation0','lead4_violation0']

    newvioll2=['lag2_new_violation', 'lag1_new_violation' ,
               'new_violation', 'lead1_new_violation', 'lead2_new_violation']
    vio0ll2=['lag2_violation0', 'lag1_violation0' , 'violation0', 
               'lead1_violation0', 'lead2_violation0']




        
if True:
    rhs_sets=[['D_interest'], ['D_interest_eop'], ['D_interest_eoq'],
              ['lag1_D_interest'], ['lag1_D_interest_eop'],['lag1_D_interest_eoq']]
    
    for rhs_set in rhs_sets:
    
        df_use=df.copy()
        
        lhs_vars=['violation','violation0', 'new_violation']
        
        for lhs_var in lhs_vars:  
            

            rhs_vars=controls+rhs_set
            cols=lhs_vars+rhs_vars
    
    
            df_use = df.dropna(subset=cols).copy()
            df_use = dt.demean2( ['gvkey'],cols, df_use)
            df_use = dt.demean2(['datadate'], cols, df_use)
        
            _ = dt.regression(df_use, lhs_var, rhs_vars, display='True', 
#                          cluster_groups=df_use['gvkey'], 
                          )
        input("Press Enter to continue...")



        


if True:


    rhs_sets=[['violation', 'lag1_violation' ],
              ['new_violation', 'lag1_new_violation'], 
              ['violation0', 'lag1_violation0'], 
              ['new_violation', 'lag1_new_violation', 'violation0', 'lag1_violation0']]
    
    for rhs_set in rhs_sets:
    
        df_use=df.copy()
        
        for lhs_var in lhs_vars:  


            rhs_vars=controls+rhs_set
            cols=lhs_vars+rhs_vars
    
    
            df_use = df.dropna(subset=cols).copy()
            df_use = dt.demean2(['gvkey'], cols, df_use)
            df_use = dt.demean2(['datadate'], cols, df_use)
        
            _ = dt.regression(df_use, lhs_var, rhs_vars, display='True', 
                          cluster_groups=df_use[['gvkey']], 
                          )
        input("Press Enter to continue...")
        
       
if True:
    df_use=df.copy()

    rhs_sets=[newvioll2, vio0ll2]
    
    for rhs_set in rhs_sets:
    

        for lhs_var in lhs_vars:  


            rhs_vars=controls+rhs_set
            cols=lhs_vars+rhs_vars
    
            df_use = df.dropna(subset=cols).copy()
            df_use = dt.demean2(['gvkey'], cols, df_use)
            df_use = dt.demean2(['datadate'], cols, df_use)
        
            _ = dt.regression(df_use, lhs_var, rhs_vars, display='True', 
                          cluster_groups=df_use[['gvkey']], 
                          )
        input("Press Enter to continue...")