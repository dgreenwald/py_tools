#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:35:26 2018

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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pylab as pl
from matplotlib import collections  as mc
#set directory
#data_dir = os.environ.get('FIRM_CREDIT_DIR', '/home/dan/Dropbox/firm_credit/')
#data_dir ='/Users/MaryGong/Dropbox (MIT)/firm_credit/'
data_dir = '/Users/MaryGong/Dropbox (MIT)/firm_credit/'


lhs_vars = ['net_debt_issue_rate', 'gross_debt_issue_rate','lead1_debt_growth_rate', 
            'lead0_capx_rate',] 
controls = ['lag1_ebitda_rate', 'lag1_m2b', 'lag1_cash_rate', 'lag1_asset_rate', 'lag1_debt_rate']


rhs=['ebitda_cov_ebitda', 'ic_cov_D_interest', 'ic_cov_D_interest_ebitda', 
     'ic_tight_D_interest', 'ic_tight_D_interest_ebitda', 
     'lag1_distance_D_interest', 'lag1_distance_D_interest_ebitda', 
     ]

rhs_abs=['ebitda_cov_ebitda', 'ic_cov_D_interest', 'ic_tight_D_interest', 
     'lag1_distance_D_interest',  'abs_ic_cov_D_interest_ebitda', 'abs_ic_tight_D_interest_ebitda', 
     'abs_lag1_distance_D_interest_ebitda', 'abs_neg_ic_cov_D_interest_ebitda', 'abs_neg_ic_tight_D_interest_ebitda', 
     'abs_neg_lag1_distance_D_interest_ebitda',]


rhs_eoy=['ebitda_cov_ebitda','ic_cov_D_interest_eoy', 'ic_cov_D_interest_eoy_ebitda', 
     'ic_tight_D_interest_eoy', 'ic_tight_D_interest_eoy_ebitda', 
     'lag1_distance_D_interest_eoy', 'lag1_distance_D_interest_eoy_ebitda',
]

rhs_eoy_abs=['ebitda_cov_ebitda','ic_cov_D_interest_eoy', 'ic_tight_D_interest_eoy', 
     'lag1_distance_D_interest_eoy',  'abs_ic_cov_D_interest_eoy_ebitda', 'abs_ic_tight_D_eoy_interest_ebitda', 
     'abs_lag1_distance_D_interest_eoy_ebitda',
     'abs_neg_ic_cov_D_interest_eoy_ebitda', 'abs_neg_ic_tight_D_eoy_interest_ebitda', 
     'abs_neg_lag1_distance_D_interest_eoy_ebitda',]



all_cols = list(set(lhs_vars + controls + rhs + rhs_eoy))


df_orig = pd.read_feather(data_dir+'compustat/ds_comp_93_07_resampled.feather')


df = df_orig.copy()

# Drop infs
df = df.replace([np.inf, -np.inf], np.nan)



# Define additional variables
df['young'] = df['age'] <= 7
df['pays_dividend'] = (df['dvc'] > 0) | (df['cdvc'] > 0) | (df['dv'] > 0)
df['no_dividend'] = ~df['pays_dividend']
df['young_no_div'] = df['young'] & ~df['pays_dividend']

ebitda_cols=['min_interest_coverage_count','min_cash_interest_coverage_count',
              'min_fixed_charge_coverage_count','min_debt_service_coverage_count',
              'max_debt_to_ebitda_count','max_senior_debt_to_ebitda_count','min_ebitda_count']

df['ebitda_cov'] = df[ebitda_cols].apply(max,axis=1)

#Find all observations where there is active covenant depending on ebitda
ix=df['ebitda_cov']>0

df['ebitda_cov']=0
df.loc[ix, 'ebitda_cov']=1

# Ebitda cov * ebitda
df['ebitda_cov_ebitda']=df['ebitda_cov']*df['ebitda_rate']



#Active interest coverage covenant 
df['ic_cov']=0
ix=df['min_interest_coverage_count']>0
df.loc[ix, 'ic_cov']=1


#Interest coverage covenant is tightest
df['ic_tight'] = 0
ix = df['ddebt_ic'] == df['distance']
df.loc[ix, 'ic_tight'] = 1


df['D_interest_ebitda']=df['D_interest']*df['ebitda_rate']
df['D_interest_eoy_ebitda']=df['D_interest_EOY']*df['ebitda_rate']

ix = df['D_interest_ebitda']>0
ix_eoy = df['D_interest_eoy_ebitda']>0



df['D_interest_ebitda0']=-1
df.loc[ix, 'D_interest_ebitda0']=1

df['D_interest_eoy_ebitda0']=-1
df.loc[ix_eoy, 'D_interest_eoy_ebitda0']=1

def_cols=['ic_cov','ic_tight','lag1_distance','lag1_distance_rate0']

for col in def_cols:

    df[col+'_D_interest']=df[col]*df['D_interest']
    df[col+'_D_interest_eoy']=df[col]*df['D_interest_EOY']    
    df['abs_'+col+'_D_interest_ebitda']=0
    df['abs_'+col+'_D_interest_eoy_ebitda']=0
    df['abs_neg_'+col+'_D_interest_ebitda']=0
    df['abs_neg_'+col+'_D_interest_eoy_ebitda']=0

    if np.logical_or(col=='lag1_distance', col=='lag1_distance_rate0'):

    
        df[col+'_D_interest_ebitda']=df[col]*df['D_interest_ebitda']*df['lag1_ppent']
        df[col+'_D_interest_eoy_ebitda']=df[col]*df['D_interest_eoy_ebitda']*df['lag1_ppent']
        
        df.loc[ix, 'abs_'+col+'_D_interest_ebitda']=df[col]*df['D_interest_ebitda0']*df['D_interest_ebitda']*df['lag1_ppent']
        df.loc[ix_eoy,'abs_'+col+'_D_interest_eoy_ebitda']=df[col]*df['D_interest_eoy_ebitda0']*df['D_interest_eoy_ebitda']*df['lag1_ppent']

        df.loc[~ix,'abs_neg_'+col+'_D_interest_ebitda']=df[col]*df['D_interest_ebitda0']*df['D_interest_ebitda']*df['lag1_ppent']
        df.loc[~ix_eoy,'abs_neg_'+col+'_D_interest_eoy_ebitda']=df[col]*df['D_interest_eoy_ebitda0']*df['D_interest_eoy_ebitda']*df['lag1_ppent']

    else:
        df[col+'_D_interest_ebitda']=df[col]*df['D_interest_ebitda']
        df[col+'_D_interest_eoy_ebitda']=df[col]*df['D_interest_eoy_ebitda']
        

        df.loc[~ix,'abs_neg_'+col+'_D_interest_ebitda']=df[col]*df['D_interest_ebitda0']*df['D_interest_ebitda']
        df.loc[~ix_eoy,'abs_neg_'+col+'_D_interest_eoy_ebitda']=df[col]*df['D_interest_eoy_ebitda0']*df['D_interest_eoy_ebitda']


        df.loc[ix,'abs_'+col+'_D_interest_ebitda']=df[col]*df['D_interest_ebitda0']*df['D_interest_ebitda']
        df.loc[ix_eoy,'abs_'+col+'_D_interest_eoy_ebitda']=df[col]*df['D_interest_eoy_ebitda0']*df['D_interest_eoy_ebitda']


# Winsorize regression variables


df[all_cols] = dt.winsorize(df, all_cols)

def set_bins(df, bin_var, n_bins, quantiles):
    
    df[bin_var + '_bin'] = np.nan
    for ii in range(n_bins):
        print(ii)
        if ii < n_bins - 1:
            q = df.groupby('datadate')[bin_var].quantile(quantiles[ii])
#            print("\n\nBin var = {0}, upper bound of bin = {1:d}:\n".format(bin_var, ii))
#            print(q)
            df = pd.merge(df, q.to_frame(name=bin_var+'_q{:d}'.format(ii)),
                          left_on='datadate', right_index=True)
            if ii > 0:
                ix = (df[bin_var] <= df[bin_var + '_q{:d}'.format(ii)]) & (df[bin_var] > df[bin_var + '_q{:d}'.format(ii-1)])
            else:
                ix = (df[bin_var] <= df[bin_var + '_q{:d}'.format(ii)])
        else:
            ix = (df[bin_var] > df[bin_var + '_q{:d}'.format(ii-1)])
        
        df.loc[ix, bin_var + '_bin'] = ii
        
    df = df.drop(columns=[bin_var + '_q{:d}'.format(ii) for ii in range(n_bins - 1)])
        
    return df



"""""
This function makes plots the estimated coefs for quartile, quintile, and decile regressions.
Takes arguments 
1: dataset to use
2: the name that the file should be saved as (MAY WANT TO CHANGE THE DIRECTORY HERE)
3: Which variable should be used for calculating quantiles
4: A list of LHS variables
5: list of non-control RHS variables
6: Indicator for quartile
7: Indicator for quintile
8: Indicator for decile
9: Whether to also plot the 95% confidence interval


def debt_regression_plots(df_use, filename, binned_on, rhs_var=[],quantiles=[ .025, .25, .5, .75,.975], plot_se=False):    
    
    
        
    all_cols= lhs_vars+controls+rhs_var
    df=df_use.copy()
        


    df=df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=all_cols)
    df[all_cols]=dt.winsorize(df,all_cols)
    
    
    for lhs_var in lhs_vars:
        midpts=[quantiles[1]/2]
        for pt in list(range(1, len(quantiles)-1)):
            if pt<len(quantiles)-2:
                midpts.append((quantiles[pt]+quantiles[pt+1])/2)
            else:
                midpts.append((1+quantiles[pt])/2)
        xpoints=list(df[binned_on].quantile(midpts))
        qcuts=list(df[binned_on].quantile(quantiles))
  
        fr=[]
        coefs = np.zeros((len(quantiles)-1, len(rhs_var)))
        se = np.zeros((len(quantiles)-1, len(rhs_var)))
        lb95=np.zeros((len(quantiles)-1, len(rhs_var)))
        ub95=np.zeros((len(quantiles)-1, len(rhs_var)))
        zeros=[0]*(len(quantiles)-1)
        q=0
        
        blah=[]
        blah_se=[]
        while q<=len(quantiles)-2:
            
            lbq=qcuts[q]
            ubq=qcuts[q+1]
            
            ix=np.logical_and(lbq<=df[binned_on], df[binned_on]<ubq) 
            dfq=df.loc[ix].copy()
            dfq=dt.demean2(['gvkey'], all_cols, dfq)
            dfq=dt.demean2(['datadate'], all_cols, dfq)
            fr.append(dt.regression(dfq, lhs_var, controls, intercept=False))
            coefs[q]=fr[q].results.params[len(controls):len(controls)+len(rhs_var)]
            se[q]=fr[q].results.HC0_se[len(controls):len(controls)+len(rhs_var)]
            conf95=fr[q].results.conf_int(alpha=.05, cols=None)
            lb95[q]=conf95[len(controls):len(controls)+len(rhs_var), 0]
            ub95[q]=conf95[len(controls):len(controls)+len(rhs_var), 1]
             
            blah_se.append([(qcuts[q], lb95[q][0]), (qcuts[q+1], lb95[q][0])])
            blah_se.append([(qcuts[q], ub95[q][0]), (qcuts[q+1], ub95[q][0])])
            blah.append([(qcuts[q], coefs[q][0]), (qcuts[q+1], coefs[q][0])])
            
            q=q+1
            ''' This part plots coefs as line segments. I think this technically makes more sense than plotting
            midpoints but the graphs are much more confusing to read. 
       
        zeroline=[[(qcuts[0], 0), (qcuts[q], 0)]]
        coef_lines = mc.LineCollection(blah, linewidths=1)
        se_lines=mc.LineCollection(blah_se, colors='g', linewidths=1, linestyle='--')
        z_line=mc.LineCollection(zeroline, colors='k', linewidths=1, linestyle=':')
        fig, ax = pl.subplots()
        ax.add_collection(coef_lines)
        ax.add_collection(se_lines)
        ax.add_collection(z_line)
        ax.autoscale()
        ax.margins(0.1)
        '''                      
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)       
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()    
        plt.xticks(fontsize=14)  
        plt.yticks( fontsize=14) 
        if plot_se:
            plt.plot(xpoints, coefs, xpoints, lb95, 'g--',xpoints, ub95, 'g--',xpoints, zeros,'k:' )
        
        else:
            plt.plot(xpoints, coefs, xpoints, zeros,'k:' )
        plt.xlabel(binned_on)
        plt.title(lhs_var)
        plt.legend(rhs_var)
        plt.savefig('/Users/MaryGong/Desktop/RA_work/'+lhs_var+'_'+filename+'.png', bbox_inches="tight");  
    
    
    return 


"""""

input("Press Enter to continue...")

# Baseline RHS variable sets. First does it including everything, then does it by pairs
if True:
    df_use=df.copy()

    rhs_sets=[rhs, rhs_abs, ['ebitda_cov_ebitda'], ['ic_cov_D_interest','ic_cov_D_interest_ebitda'],
              ['ic_cov_D_interest','abs_ic_cov_D_interest_ebitda','abs_neg_ic_cov_D_interest_ebitda']
              ['ic_tight_D_interest','ic_tight_D_interest_ebitda'],
              ['ic_tight_D_interest','abs_ic_tight_D_interest_ebitda','abs_neg_ic_tight_D_interest_ebitda'],
              ['lag1_distance_D_interest','lag1_distance_D_interest_ebitda'],
              ['lag1_distance_D_interest','abs_lag1_distance_D_interest_ebitda','abs_neg_lag1_distance_D_interest_ebitda'],
              ['lag1_distance0_D_interest', 'lag1_distance_rate0_D_interest_ebitda'],
              ['lag1_distance0_D_interest', 'abs_lag1_distance_rate0_D_interest_ebitda','abs_neg_lag1_distance_rate0_D_interest_ebitda'],]

    for rhs_set in rhs_sets:

        rhs_vars=controls+rhs_set
        cols=lhs_vars+rhs_vars
    
        df_use = df.dropna(subset=cols).copy()
        df_use = dt.demean2(['gvkey'], cols, df_use)
        df_use = dt.demean2(['datadate'], cols, df_use)


    
        for lhs_var in lhs_vars:
            _ = dt.regression(df_use, lhs_var, rhs_vars, display='True', 
                          cluster_groups=df_use[['gvkey']], 
                          )
        input("Press Enter to continue...")


# Using end-of-year interest rates instead of the past-year average
if True:
    df_use=df.copy()


    rhs_sets=[rhs_eoy, ['ic_cov_D_interest_eoy','ic_cov_D_interest_eoy_ebitda'],
              ['ic_cov_D_interest_eoy','abs_ic_cov_D_interest_eoy_ebitda', 'abs_neg_ic_cov_D_interest_eoy_ebitda'],
              ['ic_tight_D_interest_eoy','ic_tight_D_interest_eoy_ebitda'],
              ['ic_tight_D_interest_eoy','abs_ic_tight_D_interest_eoy_ebitda','abs_neg_ic_tight_D_interest_eoy_ebitda' ],
              ['lag1_distance_D_interest_eoy','lag1_distance_D_interest_eoy_ebitda'],
              ['lag1_distance_D_interest_eoy','abs_lag1_distance_D_interest_eoy_ebitda', 'abs_neg_lag1_distance_D_interest_eoy_ebitda'],
              ['lag1_distance_rate0_D_interest_eoy', 'lag1_distance_rate0_D_interest_eoy_ebitda'],
              ['lag1_distance_rate0_D_interest_eoy', 'abs_lag1_distance_rate0_D_interest_eoy_ebitda', 'abs_neg_lag1_distance_rate0_D_interest_eoy_ebitda']]

#    rhs_sets=[rhs_eoy, ['ic_cov_D_interest_eoy','ic_cov_D_interest_eoy_ebitda'],
#              ['ic_cov_D_interest','ic_cov_D_interest_ebitda'],
#              ['ic_tight_D_interest_eoy','ic_tight_D_interest_eoy_ebitda'],
#              ['ic_tight_D_interest','ic_tight_D_interest_ebitda'],
#              ['lag1_distance_D_interest_eoy','lag1_distance_D_interest_eoy_ebitda'],
#              ['lag1_distance_D_interest','lag1_distance_D_interest_ebitda'],
#              ['lag1_distance_rate0_D_interest_eoy', 'lag1_distance_rate0_D_interest_eoy_ebitda'],
#              ['lag1_distance_rate0_D_interest', 'lag1_distance_rate0_D_interest_ebitda']]

    for rhs_set in rhs_sets:

        rhs_vars=controls+rhs_set
        cols=lhs_vars+rhs_vars
    
        df_use = df.dropna(subset=cols).copy()
        df_use = dt.demean2(['gvkey'], cols, df_use)
        df_use = dt.demean2(['datadate'], cols, df_use)

    
        for lhs_var in lhs_vars:
            _ = dt.regression(df_use, lhs_var, rhs_vars, display='True', 
                          cluster_groups=df_use[['gvkey']], intercept=False
                          )
        input("Press Enter to continue...")


    
# By type, 
if True:
    rhs_sets=[rhs, ['ebitda_cov_ebitda'], ['ic_cov_D_interest','ic_cov_D_interest_ebitda'],
              ['ic_tight_D_interest','ic_tight_D_interest_ebitda'],
              ['lag1_distance_D_interest','lag1_distance_D_interest_ebitda'], 
              ['lag1_distance_rate0_D_interest', 'lag1_distance_rate0_D_interest_ebitda']]
    
    ixyoung=df['young']==True
    df_young=df.loc[ixyoung].copy()    
    df_no_young=df.loc[~ixyoung].copy()
    
    ixyoungnodiv=df['young_no_div']==True
    df_youngnodiv=df.loc[ixyoungnodiv].copy()
    df_no_youngnodiv=df.loc[~ixyoungnodiv].copy()
    
    for rhs_set in rhs_sets:
        rhs_vars=controls+rhs_set
        cols=lhs_vars+rhs_vars
        
        df_young_use = df_young.dropna(subset=cols).copy()
        df_young_use=dt.demean2(['gvkey'], cols, df_young_use)
        df_young_use = dt.demean2(['datadate'], cols, df_young_use)
        
        df_no_young_use = df_no_young.dropna(subset=cols).copy()
        df_no_young_use=dt.demean2(['gvkey'], cols, df_no_young_use)
        df_no_young_use = dt.demean2(['datadate'], cols, df_no_young_use)
        
        
        df_youngnodiv_use = df_youngnodiv.dropna(subset=cols).copy()
        df_youngnodiv_use=dt.demean2(['gvkey'], cols, df_youngnodiv_use)
        df_youngnodiv_use = dt.demean2(['datadate'], cols, df_youngnodiv_use)
        
        df_no_youngnodiv_use = df_no_youngnodiv.dropna(subset=cols).copy()
        df_no_youngnodiv_use=dt.demean2(['gvkey'], cols, df_no_youngnodiv_use)
        df_no_youngnodiv_use = dt.demean2(['datadate'], cols, df_no_youngnodiv_use)
        
        
        df_use_young =pd.concat([df_young_use, df_no_young_use], ignore_index=True)
        df_use_youngnodiv =pd.concat([df_youngnodiv_use, df_no_youngnodiv_use], ignore_index=True)

    

        for lhs_var in lhs_vars:
            formula = '{0} ~ C(young) * ({1})'.format(lhs_var, ' + '.join(rhs_vars))
            _ = dt.formula_regression(df_use_young, formula, display='True', intercept=False
    #                      cluster_groups=df[['gvkey']]
                          )
    #    
            formula = '{0} ~ C(young_no_div) * ({1})'.format(lhs_var, ' + '.join(rhs_vars))
            _ = dt.formula_regression(df_use_youngnodiv, formula, display='True', intercept=False
    #                      cluster_groups=df[['gvkey']]
                          )
        input("Press Enter to continue...")

#Binned regressions
if True:
    n_bins = 4
    quants= np.linspace(0.0, 1.0, n_bins+1)
    quantiles = np.linspace(0.0, 1.0, n_bins+1)[1:-1]
#    bin_vars = ['distance']
    midpts=[quants[1]/2]
    for pt in list(range(1, len(quants)-1)):
        if pt<len(quants)-2:
            midpts.append((quants[pt]+quants[pt+1])/2)
        else:
            midpts.append((1+quants[pt])/2)
        
    bin_vars = ['at', 'ebitda', 'sale', 'D_dbar_rate']        
    rhs_sets=[rhs, ['ebitda_cov_ebitda'], ['ic_cov_D_interest','ic_cov_D_interest_ebitda'],
              ['ic_tight_D_interest','ic_tight_D_interest_ebitda'],
              ['lag1_distance_D_interest','lag1_distance_D_interest_ebitda'], 
              ['lag1_distance0_D_interest', 'lag1_distance0_D_interest_ebitda']]

    for rhs_set in rhs_sets:
        rhs_vars=controls+rhs_set
        cols=lhs_vars+rhs_vars
        
        df_binned = df.dropna(subset=cols).copy()
    
        for bin_var in bin_vars:
            bin_name = bin_var + '_bin'
        
            df_binned = set_bins(df_binned, bin_var, n_bins, quantiles)

            xpoints=list(df[bin_var].quantile(midpts))            
            for binn in list(range(0, n_bins)):
                ix=df_binned[bin_name]==binn
            
                if binn==0:
                    df_use=df_binned.loc[ix].copy()
                    df_use = dt.demean2(['gvkey'], all_cols, df_use)
                    df_use = dt.demean2(['datadate'], all_cols, df_use)
                else:
                    df_use1=df_binned.loc[ix].copy()
                    df_use1 = dt.demean2(['gvkey'], all_cols, df_use1)
                    df_use1 = dt.demean2(['datadate'], all_cols, df_use1)
                
                    df_use=pd.concat([df_use, df_use1], ignore_index=True)
                
            for lhs_var in lhs_vars:          
                formula = '{0} ~ C({1}) * ({2})'.format(lhs_var, bin_name, ' + '.join(rhs_vars))
  
                fr = dt.formula_regression(df_use, formula, display='True',intercept=False
        #                      cluster_groups=df[['gvkey']] 
                    )                
                control_coefs=np.zeros((len(quants)-1, len(rhs_vars)))
                coefs = np.zeros((len(quants)-1, len(rhs_set)))
                #se = np.zeros((len(quants)-1, len(rhs_set)))
                #lb95=np.zeros((len(quants)-1, len(rhs_set)))
                #ub95=np.zeros((len(quants)-1, len(rhs_set)))
                zeros=[0]*(len(quants)-1)        
                
                q=0
                while q<=len(quants)-2:
                    
                    control_coefs[q]=fr.results.params[3+q::n_bins]
                    coefs[q]=fr.results.params[(len(controls)+1)*n_bins::n_bins]
                    #se[q]=fr.results.HC0_se[(len(controls)+1)*n_bins::n_bins]
                    #conf95=fr.results.conf_int(alpha=.05, cols=None)
                    #lb95[q]=conf95[(len(controls)+1)*n_bins::n_bins, 0]
                    #ub95[q]=conf95[(len(controls)+1)*n_bins::n_bins, 1]
                    
                    if q>0:
                        coefs[q]=list( map(add, coefs[q], coefs[q-1]) )
                        control_coefs[q]=list(map(add, control_coefs[q], control_coefs[q-1]))
                    
                    q=q+1
                
                plt.figure(figsize=(10, 8))
                ax = plt.subplot(111)  
                ax.spines["top"].set_visible(False)  
                ax.spines["right"].set_visible(False)       
                ax.get_xaxis().tick_bottom()  
                ax.get_yaxis().tick_left()    
                plt.xticks(fontsize=14)  
                plt.yticks( fontsize=14) 

                plt.plot(xpoints, control_coefs, xpoints, zeros,'k:' )
                plt.xlabel(bin_var)
                plt.title(lhs_var)
                plt.legend(rhs_vars)
                
                filename=''
                for var in rhs_set:
                    filename=filename+var+'_'
                plt.savefig('/Users/MaryGong/Desktop/RA_work/'+lhs_var+'_'+filename+str(n_bins)+'.png', bbox_inches="tight");  
    
            input("Press Enter to continue...")

