#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:44:06 2018

@author: MaryGong
"""

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


comp_ds=pd.read_feather(data_dir+'compustat/comp_ds_q.feather')




df_mkt=pd.read_sas(data_dir+'compustat/co_amkt.sas7bdat')

coded=['gvkey', 'cfflag', 'curcd']
for variable in coded:
    df_mkt[variable]=df_mkt[variable].str.decode('UTF-8')
    
df_mkt['gvkey']=df_mkt['gvkey'].astype(np.int64)
df_mkt['year']=df_mkt['datadate'].dt.year

df_mkt['month']=df_mkt['clsm'].copy()

ix=np.isnan(df_mkt['clsm'])
df_mkt.loc[ix, 'month']=df_mkt['datadate'].dt.month


q2=np.logical_or(df_mkt['month']==4,df_mkt['month']==5)
q2=np.logical_or(q2,df_mkt['month']==6)

q3=np.logical_or(df_mkt['month']==7,df_mkt['month']==8)
q3=np.logical_or(q3,df_mkt['month']==9)

q4=np.logical_or(df_mkt['month']==10,df_mkt['month']==11)
q4=np.logical_or(q4,df_mkt['month']==12)


df_mkt['quarter']=1

df_mkt.loc[q2, 'quarter']=2
df_mkt.loc[q3, 'quarter']=3
df_mkt.loc[q4, 'quarter']=4



# KEEP ONLY THOSE DENOTED IN USD
ix= df_mkt['curcd']=='USD'    
df_mkt=df_mkt.loc[ix].copy()


# DROP ALL OBSERVATIONS WHERE PRCC IS MISSING 
ix=np.isnan(df_mkt['prcc'])
df_mkt=df_mkt.loc[~ix].copy()




#Drop all duplicates on gvkey, date, and prcc
ix=df_mkt.duplicated(['gvkey', 'year', 'month', 'prcc'])
df_mkt=df_mkt.loc[~ix].copy()




# drop duplicates on gvkey and date where closing month is missing
ix=df_mkt.duplicated(['gvkey', 'month', 'year'], keep=False)
ix=np.logical_and(ix,np.isnan(df_mkt['clsm']))
df_mkt=df_mkt.loc[~ix].copy()






# Drop duplicates on company, date, keeping the version that is 'fiscal'
ix=df_mkt.duplicated(['gvkey', 'year', 'month'], keep=False)
ix=np.logical_and(ix, df_mkt['cfflag']=='C')
df_mkt=df_mkt.loc[~ix].copy()

ix=df_mkt.duplicated(['gvkey', 'month', 'year'], keep=False)
iy=np.logical_and(ix, df_mkt['datadate']=='1987-02-28' )


df_mkt=df_mkt.loc[~iy].copy()


df_mkt['day']=1






df_mkt=df_mkt[['gvkey','mkvalt', 'prcc', 'prch', 'prcl','month', 'year', 'quarter', 'day']]

df_mkt['month']=df_mkt['month'].astype(np.int64)
df_mkt['datadate']=pd.to_datetime(df_mkt[['year', 'month', 'day']])


df_mkt = df_mkt.set_index(['gvkey', 'datadate']).sort_index()
df_mkt = df_mkt.groupby(level=0).resample('Q', level=1).last()

df_mkt['gvkey']=df_mkt.index.get_level_values(0)
df_mkt['datadate']=df_mkt.index.get_level_values(1)

df_mkt=df_mkt.set_index(np.arange(len(df_mkt)))
df_mkt=df_mkt.sort_values(['gvkey', 'datadate'])
df_mkt=df_mkt.set_index(np.arange(len(df_mkt)))


df_mkt=df_mkt[['gvkey','mkvalt', 'prcc', 'prch', 'prcl', 'quarter', 'year']]


df_annual=pd.merge(comp_ds, df_mkt, how='outer', on=['gvkey', 'year', 'quarter'])

#PRCC, TXDB




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

ix=df_annual['aqpq']>.1
iy=df_annual['aqpq']<-.1
iz=np.logical_or(ix, iy)

df_annual=df_annual.loc[~iz].copy()


#####################
### CLEAN BY INDUSTRY
df_ind=pd.read_sas(data_dir+'compustat/co_industry.sas7bdat')
df_ind['gvkey']=df_ind['gvkey'].str.decode('UTF-8')
df_ind['consol']=df_ind['consol'].str.decode('UTF-8')
df_ind['popsrc']=df_ind['popsrc'].str.decode('UTF-8')


df_ind['gvkey']=df_ind['gvkey'].astype(np.float64)


ix=~np.isnan(df_ind['sich'])
df_ind=df_ind.loc[ix].copy()
df_ind['sich']=df_ind['sich'].astype(np.int64)
df_ind['sich']=df_ind['sich'].astype(str)

df_ind['sic2']=df_ind['sich'].apply(lambda x: x[:2])
df_ind=df_ind.drop_duplicates(['gvkey','consol', 'year' ,'sic2'])
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



df_ind=df_ind.sort_values(['gvkey', 'month','datafmt','indfmt', 'consol','datadate'])
df_ind=df_ind.set_index(np.arange(len(df_ind)))




#####
##READ IN T-BILL DATA



tbill=pd.read_excel(data_dir+'TB3MS.xlsx')
tbill.columns=['datadate', 'TB3MS']
tbill=tbill.sort_values('datadate')
tbill=tbill.set_index(np.arange(len(tbill)))

tbill['datadate']=pd.to_datetime(tbill['datadate'])
tbill['year']=tbill['datadate'].dt.year
tbill['month']=tbill['datadate'].dt.month
tbill['quarter']=tbill['datadate'].dt.quarter


lags=list(range(1, 3))
for lag in lags:
    tbill['lag'+str(lag)+'_TB3MS']=tbill['TB3MS'].shift(lag)

tbill['oneqmean']=tbill['TB3MS']
for lag in lags:
    tbill['oneqmean']=tbill['oneqmean']+tbill['lag'+str(lag)+'_TB3MS']

tbill['oneqmean']=tbill['oneqmean']/3


ix=np.isnan(tbill['TB3MS'])
for lag in lags:
    ix=np.logical_or(ix, np.isnan(tbill['lag'+str(lag)+'_TB3MS']))

tbill.loc[ix, 'oneyqmean']=np.nan


ix3=tbill['month']==3 
ix6=tbill['month']==6
ix9=tbill['month']==9
ix12=tbill['month']==12

ix=np.logical_or(ix3, ix6)
ix=np.logical_or(ix, ix9)
ix=np.logical_or(ix, ix12)

tbill_eoq=tbill.loc[ix].copy()
tbill_eoq=tbill_eoq[['year', 'TB3MS']]
tbill_eoq.columns=['year', 'TB3MS_eoq']

tbill_annual=tbill[['year', 'month', 'TB3MS', 'oneqmean']]
tbill_annual.columns=['year', 'month', 'TB3MS_eop','TB3MS']


dfa=pd.merge(df_ind, tbill_annual, how='outer', on=['year', 'month'])
dfa=pd.merge(dfa, tbill_eoq, how='outer', on='year')

print(1)

dfa['interest']=dfa['TB3MS']





###########################
### REMOVE UNUSUAL FORMATS AND RESAMPLE
###########################
dfa = dfa.set_index(['gvkey', 'indfmt', 'datafmt', 'consol', 'datadate']).sort_index()

idx = pd.IndexSlice
dfa_sub = dfa.loc[idx[:, 'INDL', 'STD', 'C', :], :]
dfa_sub = dfa_sub.reset_index().drop(['indfmt', 'datafmt', 'consol'], axis=1)
dfa_sub = dfa_sub.set_index(['gvkey', 'datadate']).sort_index()
dfa_sub = dfa_sub.groupby(level=0).resample('Q', level=1).last()

dfa_sub['gvkey']=dfa_sub.index.get_level_values(0)
dfa_sub['datadate']=dfa_sub.index.get_level_values(1)




dfa_sub=dfa_sub.set_index(np.arange(len(dfa_sub)))
dfa=dfa_sub.sort_values(['gvkey', 'datadate'])

df=dfa.copy()

print(2)

#########
## Define company age
    
group=df.groupby('gvkey')
birth={}
birth['birth']=group['year'].min()
birth=pd.DataFrame(birth)
birth['gvkey']=birth.index.get_level_values(0)
birth=birth.set_index(np.arange(len(birth)))



df=pd.merge(df, birth, how='outer', on='gvkey')

dfa_final=df.sort_values(['gvkey', 'datadate'])
dfa_final=dfa_final.set_index(np.arange(len(dfa_final)))


dfa_final['debt']=dfa_final['dlttq']+dfa_final['dlcq']
dfa_final['m2b']=(dfa_final['cshoq']*dfa_final['prcc']+dfa_final['atq']-dfa_final['ceqq']-dfa_final['txdbq'])/dfa_final['atq']



group=dfa_final.groupby(by=['gvkey']) # Why by month?
dfa_final['lag1_icaptq']=group['icaptq'].shift(1)
dfa_final['lag1_dlttq']=group['dlttq'].shift(1)
dfa_final['lag1_m2b']=group['m2b'].shift(1)
dfa_final['lag1_ppentq']=group['ppentq'].shift(1)
dfa_final['capxq']=dfa_final['ppentq']-dfa_final['lag1_ppentq']

ix=dfa_final['capxq']<0
dfa_final.loc[ix, 'capxq']=0

dfa_final['capxq_rate']=dfa_final['capxq']/dfa_final['lag1_ppentq']
dfa_final['ebitda_rate']=dfa_final['oibdpq']/dfa_final['lag1_ppentq']
dfa_final['lag1_atq']=group['atq'].shift(1)
dfa_final['lag1_debt']=group['debt'].shift(1)
dfa_final['lead1_debt']=group['debt'].shift(-1)


dfa_final['debt_growth']=dfa_final['debt']-dfa_final['lag1_debt']
dfa_final['debt_growth_rate']=dfa_final['debt_growth']/dfa_final['lag1_ppentq']


dfa_final['lead1_debt_growth']=dfa_final['lead1_debt']-dfa_final['debt']
dfa_final['lead1_debt_growth_rate']=dfa_final['lead1_debt_growth']/dfa_final['lag1_ppentq']



dfa_final['age']=dfa_final['year']-dfa_final['birth']
dfa_final['young']=0
ix=dfa_final['age']<=5
dfa_final.loc[ix, 'young']=1

print(3)

horizons=list(range(0, 4))
for horizon in horizons:
    dfa_final['lead'+str(horizon)+'_capxq']=group['capxq'].shift(-horizon)
    dfa_final['lead'+str(horizon)+'_capxq_rate']=dfa_final['lead'+str(horizon)+'_capxq']/dfa_final['lag1_ppentq']




dfa_final=dfa_final.sort_values(['gvkey', 'datadate'])
dfa_final=dfa_final.set_index(np.arange(len(dfa_final)))
group=dfa_final.groupby(by=['gvkey'])

dfa_final['m2b']=(dfa_final['cshoq']*dfa_final['prcc']+dfa_final['atq']-dfa_final['ceqq']-dfa_final['txdbq'])/dfa_final['atq']

dfa_final['lag1_ebitda_rate']=group['ebitda_rate'].shift(1)


dfa=dfa_final.copy()


## DEFINE OTHER VARIABLES
dfa['lag1_debt_rate'] = dfa['lag1_debt'] / dfa['lag1_ppentq']
dfa['debt_rate'] = dfa['debt'] / dfa['ppentq']


dfa=dfa.replace([np.inf, -np.inf], np.nan)



dfa_sub=dfa.copy()
group=dfa_sub.groupby('gvkey')

# Add cash
dfa_sub['lag1_cash'] = group['chq'].shift(1)
dfa_sub['lag1_cash_rate'] = dfa_sub['lag1_cash'] / dfa_sub['lag1_ppentq']
dfa_sub['cash_rate'] = dfa_sub['chq'] / dfa_sub['ppentq']
dfa_sub['lag1_asset_rate'] = dfa_sub['lag1_atq'] / dfa_sub['lag1_ppentq']
dfa_sub['asset_rate'] = dfa_sub['atq'] / dfa_sub['ppentq']
















# Debt issuance
dfa_sub['net_debt_issue']=dfa_sub['dlttq']-dfa_sub['lag1_dlttq']

dfa_sub['net_debt_issue_rate']=dfa_sub['net_debt_issue']/dfa_sub['lag1_ppentq']

dfa_sub['lag1_interest'] = group['interest'].shift(1)



dfa=dfa_sub.copy()


dfa['debt_d2e']=dfa['max_debt_to_ebitda']*dfa['oibdpq']
dfa['ddebt_d2e']=dfa['debt_d2e']-dfa['debt']


dfa['debt_d2at']=dfa['max_net_debt_to_assets']*dfa['atq']
dfa['ddebt_d2at']=dfa['debt_d2at']-dfa['debt']

dfa['xint_rate']=dfa['xintq']/dfa['debt']
dfa['debt_ic']=dfa['oibdpq']/(dfa['xint_rate']*dfa['min_interest_coverage'])
dfa['ddebt_ic']=dfa['debt_ic']-dfa['debt']


dfa['debt_cic']=dfa['oibdpq']/(dfa['xint_rate']*dfa['min_cash_interest_coverage'])
dfa['ddebt_cic']=dfa['debt_cic']-dfa['debt']


dfa['debt_d2eq']=dfa['max_debt_to_equity']*(dfa['atq']-dfa['ltq'])
dfa['ddebt_d2eq']=dfa['debt_d2eq']-dfa['debt']

dfa['distance']=dfa[['ddebt_d2e','ddebt_d2at','ddebt_ic','ddebt_cic','ddebt_d2eq']].min(axis=1, skipna=True)
dfa['distance_rate_lag']=dfa['distance']/dfa['lag1_ppentq']
dfa['distance_rate']=dfa['distance']/dfa['ppentq']



ix=pd.isnull(dfa['distance'])
dfa['distance0']=dfa['distance'].copy()
dfa.loc[ix, 'distance0']=0
dfa['distance_rate_lag0']=dfa['distance_rate_lag'].copy()
dfa.loc[ix, 'distance_rate_lag0']=0
dfa['distance_rate0']=dfa['distance_rate'].copy()
dfa.loc[ix, 'distance_rate0']=0


dfa['dbar']=dfa[['debt_d2e','debt_d2at','debt_ic','debt_cic','debt_d2eq']].min(axis=1, skipna=True)
dfa['dbar_rate']=dfa['dbar']/dfa['lag1_ppentq']

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
dfa.loc[ix, 'ebitda_rate']=np.nan
dfa=dfa.set_index(np.arange(len(dfa)))





dfa=dfa.replace([np.inf, -np.inf], np.nan)

dfa['violation']=np.nan
ix=dfa['distance']<0
dfa.loc[ix, 'violation']=1

ix=dfa['distance']>0
dfa.loc[ix, 'violation']=0




dfa=dfa.sort_values(['gvkey', 'datadate'])
dfa=dfa.set_index(np.arange(len(dfa)))


dfa['lag1_TB3MS']=group['TB3MS'].shift(1)
dfa['lag1_TB3MS_eoq']=group['TB3MS_eoq'].shift(1)
dfa['lag1_TB3MS_eop']=group['TB3MS_eop'].shift(1)

dfa['D_interest']=dfa['TB3MS']-dfa['lag1_TB3MS']
dfa['D_interest_eoq']=dfa['TB3MS_eoq']-dfa['lag1_TB3MS_eoq']
dfa['D_interest_eop']=dfa['TB3MS_eop']-dfa['lag1_TB3MS_eop']


print(4)



dfa['lag1_distance_rate']=group['distance_rate'].shift(1)
dfa['lag1_distance_rate0']=group['distance_rate0'].shift(1)




dfa['lag1_capxq_rate']=group['capxq_rate'].shift(1)
dfa['lag1_net_debt_issue_rate']=group['net_debt_issue_rate'].shift(1)






columns=['capxq_rate', 'net_debt_issue_rate']
for col in columns:
    dfa['D_'+col]=dfa[col]-dfa['lag1_'+col]
    dfa['lead1_D_'+col]=dfa['lead1_'+col]-dfa[col]



##############################
## Keep only those with 3 consecutive years
##############################



years={}
group=df_ind.groupby(by=['gvkey'])
years['quarters']=group['ones'].sum()
years=pd.DataFrame(years)

years['gvkey']=years.index.get_level_values(0)
years=years.set_index(np.arange(len(years)))

dfa=pd.merge(dfa, years, how='outer', on='gvkey')

ix=dfa['quarters']>=8
dfa=dfa.loc[ix].copy()

dfa=dfa.sort_values(['gvkey', 'datadate'])
dfa=dfa.set_index(np.arange(len(dfa)))

"""




###
## Throw out companies once they skip a reporting year
#####


for yr in list(range(int(df['year'].min()), int(df['year'].max()))):
    ix=df['lag1']>1
    iy=df['year']>=yr
    iz=np.logical_and(ix, iy)
    df=df.loc[~iz].copy()


"""








dfa.to_feather(data_dir+'compustat/ds_comp_q_resampled.feather')




ix=dfa['violation']==1


leadlag=list(range(1, 21))

group=dfa.groupby('gvkey')

dropcols=[]

for ll in leadlag:
    dfa['lead'+str(ll)+'_violation']=group['violation'].shift(-ll)
    dfa['lag'+str(ll)+'_violation']=group['violation'].shift(ll)
    dropcols.append('lead'+str(ll)+'_violation')
    dropcols.append('lag'+str(ll)+'_violation')
    ix=np.logical_or(ix, dfa['lead'+str(ll)+'_violation']==1)
    ix=np.logical_or(ix, dfa['lag'+str(ll)+'_violation']==1)


print(5)

dfa=dfa.drop(dropcols, axis=1)


df_eventstudy=dfa.loc[ix].copy()
df_eventstudy=df_eventstudy.sort_values(['gvkey', 'datadate'])
df_eventstudy=df_eventstudy.set_index(np.arange(len(df_eventstudy)))

df_eventstudy.to_feather(data_dir+'compustat/ds_comp_q_eventstudy_resampled.feather')



ix=dfa['year']>1992
iy=dfa['year']<2008
iz=np.logical_and(ix, iy)
dfa_sub=dfa.loc[iz].copy()
dfa_sub=dfa_sub.sort_values(by=['gvkey', 'datadate'])
dfa_sub=dfa_sub.set_index(np.arange(len(dfa_sub)))
dfa_sub.to_feather(data_dir+'compustat/ds_comp_q_93_07_resampled.feather')
