#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 17:22:14 2018

@author: MaryGong
"""



import sys
sys.path.insert(0, "/Users/MaryGong/Desktop/RA_work")
import pandas as pd
import numpy as np
import os
from time import time
from py_tools import data as dt
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pylab as pl
from matplotlib import collections  as mc
#set directory
#data_dir = os.environ.get('FIRM_CREDIT_DIR', '/home/dan/Dropbox/firm_credit/')
#data_dir ='/Users/MaryGong/Dropbox (MIT)/firm_credit/'
data_dir = '/Users/MaryGong/Dropbox (MIT)/firm_credit/'


comp_ds=pd.read_feather(data_dir+'comp_ds.feather')




df_mkt=pd.read_sas(data_dir+'compustat/co_amkt.sas7bdat')

coded=['gvkey', 'cfflag', 'curcd']
for variable in coded:
    df_mkt[variable]=df_mkt[variable].str.decode('UTF-8')
    

    
ix= df_mkt['curcd']=='USD'    
df_mkt=df_mkt.loc[ix].copy()
ix= df_mkt['cfflag']=='F'    
df_mkt=df_mkt.loc[ix].copy()


df_mkt['gvkey']=df_mkt['gvkey'].astype(np.int64)


df_mkt=df_mkt[['gvkey','mkvalt', 'prcc', 'prch', 'prcl','clsm','cfflag','datadate']]
df_mkt=df_mkt.set_index(np.arange(len(df_mkt)))

df_mkt.columns=['gvkey','mkvalt', 'prcc', 'prch', 'prcl','month','cfflag','datadate']
df_mkt['year']=df_mkt['datadate'].dt.year
df_mkt=df_mkt[['gvkey','mkvalt', 'prcc', 'prch', 'prcl', 'month', 'year']]

df_annual=pd.merge(comp_ds, df_mkt, how='outer', on=['gvkey', 'year', 'month'])






df_state=pd.read_sas(data_dir+'compustat/company.sas7bdat')

df_state=df_state[['gvkey', 'city', 'state', 'county', 'ipodate']]
df_state['ipoyear']=df_state['ipodate'].dt.year

tostr=['gvkey', 'city', 'state', 'county']

for variable in tostr:
    df_state[variable]=df_state[variable].str.decode('UTF-8')
df_state['gvkey']=df_state['gvkey'].astype(np.int64)




ix=df_state['state']=='QC'
ix=np.logical_or(ix, df_state['state']=='ON')
ix=np.logical_or(ix, df_state['state']=='BC')
ix=np.logical_or(ix, df_state['state']=='AB')
ix=np.logical_or(ix, df_state['state']=='PR')
ix=np.logical_or(ix, df_state['state']=='MB')
ix=np.logical_or(ix, df_state['state']=='NB')
ix=np.logical_or(ix, df_state['state']=='SK')
ix=np.logical_or(ix, df_state['state']=='PE')
ix=np.logical_or(ix, df_state['state']=='NS')
ix=np.logical_or(ix, df_state['state']=='NF')
ix=np.logical_or(ix, df_state['state']=='VI')
ix=np.logical_or(ix, df_state['state']=='GU')
ix=np.logical_or(ix, pd.isnull(df_state['state']))


df_state=df_state.loc[~ix].copy()
df_annual=pd.merge(df_state, df_annual, how='outer', on=['gvkey'])


ix=pd.isnull(df_annual['state'])
df_annual=df_annual.loc[~ix].copy()




df_annual=df_annual.sort_values(by=['gvkey', 'datadate'])
df_annual=df_annual.set_index(np.arange(len(df_annual)))


##############
### Throw out firms with acquisitions/mergers
#############

ix=df_annual['aqc']>.1
iy=df_annual['aqc']<-.1
iz=np.logical_or(ix, iy)

df_annual=df_annual.loc[~iz].copy()


#####################
### CLEAN BY INDUSTRY
df_ind=pd.read_sas(data_dir+'compustat/co_industry.sas7bdat')
df_ind['gvkey']=df_ind['gvkey'].str.decode('UTF-8')
df_ind['consol']=df_ind['consol'].str.decode('UTF-8')
df_ind['popsrc']=df_ind['popsrc'].str.decode('UTF-8')


df_ind['gvkey']=df_ind['gvkey'].astype(np.int64)


ix=~np.isnan(df_ind['sich'])
df_ind=df_ind.loc[ix].copy()
df_ind['sich']=df_ind['sich'].astype(np.int64)
df_ind['sich']=df_ind['sich'].astype(str)

df_ind['sic2']=df_ind['sich'].apply(lambda x: x[:2])
df_ind=df_ind.drop_duplicates(['gvkey','consol', 'year'])
df_ind=df_ind[['gvkey', 'consol', 'year', 'sic2']]


df_annual_ind=pd.merge(df_annual, df_ind, how='outer', on=['gvkey', 'consol', 'year'])
df_annual_ind['ones']=1

bad=['10', '12', '13', '14', '15', '17', '49','60', '61', '62', '63', '64', '65',  '67']

ix=np.isnan(df_annual_ind['ones'])
for industry in bad:
    ix=np.logical_or(ix, df_annual_ind['sic2']==industry)

df_annual_ind=df_annual_ind.loc[~ix].copy()


ix=~pd.isnull(df_annual_ind['datadate'])
df_ind=df_annual_ind.loc[ix].copy()
df_ind=df_ind.sort_values(['gvkey', 'datadate'])
df_ind=df_ind.set_index(np.arange(len(df_ind)))



##############################
## Keep only those with 3 consecutive years
##############################




df_ind=df_ind.sort_values(['gvkey', 'month','datafmt','indfmt', 'consol','datadate'])
df_ind=df_ind.set_index(np.arange(len(df_ind)))

group=df_ind.groupby(by=['gvkey','month', 'datafmt','indfmt', 'consol'])
df_ind['lag1']=group['year'].shift(0)-group['year'].shift(1)
df_ind['lag2']=group['year'].shift(1)-group['year'].shift(2)

ix=np.logical_and(df_ind['lag1']==1, df_ind['lag2']==1)
df_cons3=df_ind.loc[ix].copy()
df_cons3=df_cons3[['gvkey', 'month', 'datafmt','indfmt', 'consol']]
df_cons3=df_cons3.drop_duplicates(['gvkey', 'month', 'datafmt','indfmt', 'consol'])
df_cons3['cons3']=1

df=pd.merge(df_ind, df_cons3, how='outer', on=['gvkey', 'month', 'datafmt','indfmt', 'consol'])

ix=df['cons3']==1
df=df.loc[ix].copy()



###
## Throw out companies once they skip a reporting year
#####


for yr in list(range(int(df['year'].min()), int(df['year'].max()))):
    ix=df['lag1']>1
    iy=df['year']>=yr
    iz=np.logical_and(ix, iy)
    df=df.loc[~iz].copy()
    

#########
## Define company age
    
group=df.groupby('gvkey')
birth={}
birth['birth']=group['year'].min()
birth=pd.DataFrame(birth)
birth['gvkey']=birth.index.get_level_values(0)
birth=birth.set_index(np.arange(len(birth)))



df=pd.merge(df, birth, how='outer', on='gvkey')

dfa_final=df.sort_values(['gvkey', 'month','datafmt','indfmt', 'consol','datadate'])
dfa_final=dfa_final.set_index(np.arange(len(dfa_final)))


dfa_final['debt']=dfa_final['dltt']+dfa_final['dlc']
dfa_final['m2b']=(dfa_final['csho']*dfa_final['prcc']+dfa_final['at']-dfa_final['ceq']-dfa_final['txdb'])/dfa_final['at']



group=dfa_final.groupby(by=['gvkey','month', 'datafmt','indfmt', 'consol']) # Why by month?
dfa_final['lag1_m2b']=group['m2b'].shift(1)
dfa_final['lag1_ppent']=group['ppent'].shift(1)
dfa_final['capx_rate']=dfa_final['capx']/dfa_final['lag1_ppent']
dfa_final['ebitda_rate']=dfa_final['oibdp']/dfa_final['lag1_ppent']
dfa_final['lag1_at']=group['at'].shift(1)
dfa_final['lag1_debt']=group['debt'].shift(1)
dfa_final['lead1_debt']=group['debt'].shift(-1)

dfa_final['debt_growth']=dfa_final['debt']-dfa_final['lag1_debt']
dfa_final['debt_growth_rate']=dfa_final['debt_growth']/dfa_final['lag1_ppent']


dfa_final['lead1_debt_growth']=dfa_final['lead1_debt']-dfa_final['debt']
dfa_final['lead1_debt_growth_rate']=dfa_final['lead1_debt_growth']/dfa_final['lag1_ppent']



dfa_final['age']=dfa_final['year']-dfa_final['birth']
dfa_final['young']=0
ix=dfa_final['age']<=5
dfa_final.loc[ix, 'young']=1


horizons=list(range(0, 4))
for horizon in horizons:
    dfa_final['lead'+str(horizon)+'_capx']=group['capx'].shift(-horizon)
    dfa_final['lead'+str(horizon)+'_capx_rate']=dfa_final['lead'+str(horizon)+'_capx']/dfa_final['lag1_ppent']




dfa_final=dfa_final.sort_values(['gvkey', 'month','datafmt','indfmt', 'consol','datadate'])
dfa_final=dfa_final.set_index(np.arange(len(dfa_final)))
group=dfa_final.groupby(by=['gvkey','month', 'datafmt','indfmt', 'consol'])

dfa_final['m2b']=(dfa_final['csho']*dfa_final['prcc']+dfa_final['at']-dfa_final['ceq']-dfa_final['txdb'])/dfa_final['at']

dfa_final['lag1_ebitda_rate']=group['ebitda_rate'].shift(1)





#####
##READ IN T-BILL DATA



tbill=pd.read_excel(data_dir+'TB3MS.xlsx')
tbill.columns=['datadate', 'TB3MS']
tbill=tbill.sort_values('datadate')
tbill=tbill.set_index(np.arange(len(tbill)))

tbill['datadate']=pd.to_datetime(tbill['datadate'])
tbill['year']=tbill['datadate'].dt.year
tbill['month']=tbill['datadate'].dt.month


lags=list(range(1, 12))
for lag in lags:
    tbill['lag'+str(lag)+'_TB3MS']=tbill['TB3MS'].shift(lag)

tbill['oneyearmean']=tbill['TB3MS']
for lag in lags:
    tbill['oneyearmean']=tbill['oneyearmean']+tbill['lag'+str(lag)+'_TB3MS']

tbill['oneyearmean']=tbill['oneyearmean']/12


ix=np.isnan(tbill['TB3MS'])
for lag in lags:
    ix=np.logical_or(ix, np.isnan(tbill['lag'+str(lag)+'_TB3MS']))

tbill.loc[ix, 'oneyearmean']=np.nan


ix12=tbill['month']==12
tbill_endofyear=tbill.loc[ix12].copy()
tbill_endofyear=tbill_endofyear[['year', 'TB3MS']]
tbill_endofyear.columns=['year', 'TB3MS_eoy']

tbill_annual=tbill[['year', 'month', 'oneyearmean']]
tbill_annual.columns=['year', 'month', 'TB3MS']


dfa=pd.merge(dfa_final, tbill_annual, how='outer', on=['year', 'month'])
dfa=pd.merge(dfa, tbill_endofyear, how='outer', on='year')



dfa['interest']=dfa['TB3MS']+6
dfa['ebitda_rate_interest']=dfa['ebitda_rate']/dfa['interest']
dfa['ebitda_interest']=dfa['ebitda']/dfa['interest']

med=tbill['TB3MS'].quantile(q=.5)+6
dfa['interest_med']=0
ix=dfa['interest']>med
dfa.loc[ix, 'interest_med']=1


dfa['lag1_debt_rate'] = dfa['lag1_debt'] / dfa['lag1_ppent']


dfa=dfa.replace([np.inf, -np.inf], np.nan)

# Remove unusual formats
dfa = dfa.set_index(['gvkey', 'indfmt', 'datafmt', 'consol', 'datadate']).sort_index()

idx = pd.IndexSlice
dfa_sub = dfa.loc[idx[:, 'INDL', 'STD', 'C', :], :]
dfa_sub = dfa_sub.reset_index().drop(['indfmt', 'datafmt', 'consol'], axis=1)
dfa_sub = dfa_sub.set_index(['gvkey', 'datadate']).sort_index()
dfa_sub = dfa_sub.groupby(level=0).resample('A', level=1).last()





# Add cash
dfa_sub['lag1_cash'] = dfa_sub.groupby(level=0)['ch'].shift()
dfa_sub['lag1_cash_rate'] = dfa_sub['lag1_cash'] / dfa_sub['lag1_ppent']
dfa_sub['lag1_asset_rate'] = dfa_sub['lag1_at'] / dfa_sub['lag1_ppent']

# Debt issuance
dfa_sub['gross_debt_issue_rate'] = dfa_sub['dltis'] / dfa_sub['lag1_ppent']
dfa_sub['net_debt_issue']=dfa_sub['dltis']-dfa_sub['dltr']
dfa_sub['net_debt_issue_rate']=dfa_sub['net_debt_issue']/dfa_sub['lag1_ppent']

dfa_sub['lag1_interest'] = dfa_sub.groupby(level=0)['interest'].shift()

dfa_sub['gvkey']=dfa_sub.index.get_level_values(0)
dfa_sub['datadate']=dfa_sub.index.get_level_values(1)
dfa_sub=dfa_sub.set_index(np.arange(len(dfa_sub)))
dfa=dfa_sub.sort_values(['gvkey', 'datadate'])















dfa['debt_d2e']=dfa['max_debt_to_ebitda']*dfa['ebitda']
dfa['ddebt_d2e']=dfa['debt_d2e']-dfa['debt']


dfa['debt_d2at']=dfa['max_net_debt_to_assets']*dfa['at']
dfa['ddebt_d2at']=dfa['debt_d2at']-dfa['debt']

dfa['xint_rate']=dfa['xint']/dfa['debt']
dfa['debt_ic']=dfa['oibdp']/(dfa['xint_rate']*dfa['min_interest_coverage'])
dfa['ddebt_ic']=dfa['debt_ic']-dfa['debt']


dfa['intpn_rate']=dfa['intpn']/dfa['debt']
dfa['debt_cic']=dfa['oibdp']/(dfa['intpn_rate']*dfa['min_cash_interest_coverage'])
dfa['ddebt_cic']=dfa['debt_cic']-dfa['debt']


dfa['debt_d2eq']=dfa['max_debt_to_equity']*(dfa['at']-dfa['lt'])
dfa['ddebt_d2eq']=dfa['debt_d2eq']-dfa['debt']

dfa['distance']=dfa[['ddebt_d2e','ddebt_d2at','ddebt_ic','ddebt_cic','ddebt_d2eq']].min(axis=1, skipna=True)
dfa['distance_rate_lag']=dfa['distance']/dfa['lag1_ppent']
dfa['distance_rate']=dfa['distance']/dfa['ppent']



ix=pd.isnull(dfa['distance'])
dfa['distance0']=dfa['distance'].copy()
dfa.loc[ix, 'distance0']=0
dfa['distance_rate_lag0']=dfa['distance_rate_lag'].copy()
dfa.loc[ix, 'distance_rate_lag0']=0
dfa['distance_rate0']=dfa['distance_rate'].copy()
dfa.loc[ix, 'distance_rate0']=0


dfa['dbar']=dfa[['debt_d2e','debt_d2at','debt_ic','debt_cic','debt_d2eq']].min(axis=1, skipna=True)
dfa['dbar_rate']=dfa['dbar']/dfa['lag1_ppent']

dfa=dfa.sort_values(['gvkey', 'datadate'])
dfa=dfa.set_index(np.arange(len(dfa)))
group=dfa.groupby(['gvkey'])
dfa['lag1_dbar']=group['dbar'].shift(1)
dfa['lag1_distance']=group['distance'].shift(1)
dfa['lag1_dbar_rate']=group['dbar_rate'].shift(1)
dfa['lag1_distance_rate']=group['distance_rate'].shift(1)
dfa['lead1_net_debt_issue']=group['net_debt_issue'].shift(1)
dfa['lead1_net_debt_issue_rate']=group['net_debt_issue_rate'].shift(1)




dfa['D_distance_rate']=dfa['distance_rate']-dfa['lag1_distance_rate']
dfa['D_dbar_rate']=dfa['dbar_rate']-dfa['lag1_dbar_rate']
dfa['D_dbar_int_dist_rate']=dfa['D_dbar_rate']*dfa['distance']


dfa['flag_d2eq']= 0
dfa.loc[dfa['max_debt_to_equity_count']>0, 'flag_d2eq']=1


dfa['flag_cic']= 0
dfa.loc[dfa['min_cash_interest_coverage_count']>0, 'flag_cic']=1

dfa['flag_d2at']= 0
dfa.loc[dfa['max_net_debt_to_assets_count']>0, 'flag_d2at']=1

dfa['flag_ic']= 0
dfa.loc[dfa['min_interest_coverage_count']>0, 'flag_ic']=1

dfa['flag_d2e']= 0
dfa.loc[dfa['max_debt_to_ebitda_count']>0, 'flag_d2e']=1


cov_flag_list=['flag_d2eq','flag_cic','flag_d2at','flag_ic','flag_d2e']
rhs_var_list=['D_dbar_rate', 'D_distance_rate', 'distance_rate', 'dbar_rate', 'D_dbar_int_dist_rate']
for cov in cov_flag_list:
    for rhsvar in rhs_var_list:
        dfa[cov+'_int_'+rhsvar]=dfa[cov]*dfa[rhsvar]

dfa=dfa.replace([np.inf, -np.inf], np.nan)


ix = dfa['ebitda_rate'] > 0.0
dfa = dfa.loc[ix].copy()
dfa=dfa.set_index(np.arange(len(dfa)))





dfa=dfa.replace([np.inf, -np.inf], np.nan)

dfa['violation']=0
ix=dfa['distance']<0
dfa.loc[ix, 'violation']=1

dfa=dfa.sort_values(['gvkey', 'datadate'])
dfa=dfa.set_index(np.arange(len(dfa)))

group=dfa.groupby('gvkey')
dfa['lead1_violation']=group['violation'].shift(-1)
dfa['lead2_violation']=group['violation'].shift(-2)
dfa['lead3_violation']=group['violation'].shift(-3)
dfa['lead4_violation']=group['violation'].shift(-4)
dfa['lag1_violation']=group['violation'].shift(1)
dfa['lag2_violation']=group['violation'].shift(2)
dfa['lag3_violation']=group['violation'].shift(3)
dfa['lag4_violation']=group['violation'].shift(4)
dfa['lag1_TB3MS']=group['TB3MS'].shift(1)
dfa['lag1_TB3MS_eoy']=group['TB3MS_eoy'].shift(1)
dfa['D_interest']=dfa['TB3MS']-dfa['lag1_TB3MS']
dfa['D_interest_EOY']=dfa['TB3MS_eoy']-dfa['lag1_TB3MS_eoy']

dfa['lag1_distance_rate']=group['distance_rate'].shift(1)
dfa['lag1_distance_rate0']=group['distance_rate0'].shift(1)




dfa.to_feather(data_dir+'compustat/ds_comp_resampled.feather')




ix=dfa['year']>1992
iy=dfa['year']<2008
iz=np.logical_and(ix, iy)
dfa=dfa.loc[iz]
dfa=dfa.sort_values(by=['gvkey', 'datadate'])
dfa=dfa.set_index(np.arange(len(dfa)))
dfa.to_feather(data_dir+'compustat/ds_comp_93_07_resampled.feather')
