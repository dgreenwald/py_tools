import numpy as np
import pandas as pd

from . import defaults, misc
default_dir = defaults.base_dir() + 'zillow/'
# data_dir = '/home/dan/data/irs/'

def load_county(data_dir=default_dir, dataset='Zhvi_AllHomes'):

    # df_wide = pd.read_csv(data_dir+'County/County_' + dataset + '.csv')
    df_wide = load(data_dir=data_dir, dataset=dataset, geo='County')
#    cw = pd.read_csv(data_dir+'CountyCrossWalk_Zillow2.csv')
    cw = load_crosswalk(data_dir=data_dir)

    df_long = pd.melt(df_wide, id_vars=df_wide.columns[:7], value_vars=df_wide.columns[7:]).rename(columns={
            'variable' : 'date',
            'value' : dataset,
            })

    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long_cw = pd.merge(df_long, cw, left_on='RegionID', right_on='CountyRegionID_Zillow')

    return df_long_cw.set_index(['FIPS', 'date'])

def load_state(data_dir=default_dir, dataset='Zhvi_AllHomes'):
    
    df_wide = load(data_dir=data_dir, dataset=dataset, geo='State')
    df_long = pd.melt(df_wide, id_vars=['RegionName'], value_vars=df_wide.columns[3:])
    
    state_codes = misc.load('state_codes')
    state_codes['state_name'] = state_codes['state_name'].str.title()
    
    df_long = pd.merge(df_long, state_codes, left_on='RegionName', right_on='state_name')
    df_long = df_long.drop(columns=['RegionName'])
    df_long = df_long.rename(columns={'variable' : 'date', 'value' : dataset})
    df_long['date'] = pd.to_datetime(df_long['date'])
    
    return df_long

def load_crosswalk(data_dir=default_dir):
    
    return pd.read_csv(data_dir+'CountyCrossWalk_Zillow2.csv')

def load(data_dir=default_dir, dataset='Zhvi_AllHomes', geo='State'):

    df = pd.read_csv(data_dir+geo+'/'+geo+'_'+dataset+'.csv')

    return df
