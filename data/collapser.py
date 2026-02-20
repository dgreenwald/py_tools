#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:08:09 2020

@author: dan
"""

import numpy as np
import pandas as pd

from py_tools import in_out, stats as st

def get_weighted_quantile_inner(df, var_list, weight_var, q, **kwargs):
    
    weights = df[weight_var].values
    
    data = {
        var : st.weighted_quantile(df[var].values, weights, [q], **kwargs)
        for var in var_list
        }
    
    return pd.DataFrame(data=data)

def collapse_quantile(df, by_list, weight_var=None, var_list=None, q=0.5, **kwargs):
    
    if var_list is None:
        var_list = [var for var in df.columns if var not in (by_list + [weight_var])]
    
    if weight_var is None:
        return df.groupby(by_list)[var_list].median()
    
    df_out = df.groupby(by_list).apply(get_weighted_quantile_inner, var_list, weight_var, q, **kwargs)

    #df_out = df_out.reset_index().drop(columns=['level_1']).set_index(by_list)
    # PD: does not necessarily called level_1
    # TODO: probably a more robust way to do this
    col_name = 'level_{}'.format(len(by_list))
    df_out = df_out.reset_index().drop(columns=col_name).set_index(by_list)

    return df_out


def concat(collapser_list, check=True):
    
    col0 = collapser_list[0]
    var_list = col0.var_list.copy()
    weight_var = col0.weight_var
    by_list = col0.by_list.copy()
    
    if check:
        for col in collapser_list[1:]:
            assert col.var_list == var_list
            assert col.weight_var == weight_var
            assert col.by_list == by_list
    
    dfc = pd.concat([col.dfc for col in collapser_list], axis=0)
    
    col = Collapser(dfc=dfc, var_list=var_list, weight_var=weight_var, 
                     by_list=by_list)
    col.collapse(by_list, inplace=True)
    return col

def load_collapser(filename, add_suffix=True, by_list=None, weight_var=None):

    if by_list is None: by_list = []
        
    col = Collapser(by_list=by_list, weight_var=weight_var)
    col.load(filename, add_suffix=add_suffix)
    return col

def create_suffix(by_list, weight_var):
    
    return '_'.join([''] + by_list + ['wtd_by', weight_var])

def collapse(df, by_list, var_list=None, weight_var=None, weight_suffix=False):

    if var_list is None: var_list = []
    
    coll = Collapser(df, var_list=var_list, by_list=by_list, 
                     weight_var=weight_var)
    
    return coll.get_data(weight_suffix=weight_suffix)

def collapse_multiweight(df, weight_dict, by_list=None):

    if by_list is None: by_list = []
    
    return pd.concat(
        [collapse(df, by_list, var_list=[var], weight_var=weight_var)
         for var, weight_var in weight_dict.items()],
        axis=1)

def collapse_multiquantile(df, by_list, q_list, weight_var=None, var_list=None, **kwargs):
    
    dfout_list = []
    for q in q_list:
        dfout = collapse_quantile(df, by_list, 
                                  weight_var= weight_var,
                                  var_list=var_list, q=q, **kwargs)
        
        dfout.columns = [col + '_p{}'.format(q*100) for col in dfout.columns]
        dfout_list.append(dfout)
    return pd.concat(dfout_list, axis=1)
        
def collapse_multiweight_multiquantile(df, weight_dict, q_list, by_list= []):
    return pd.concat(
        [collapse_multiquantile(df, by_list, q_list = q_list, weight_var = weight_var, var_list=[var])
         for var, weight_var in weight_dict.items()],
        axis=1)
    


class Collapser:
    
    """Class for flexibly collapsing data sets"""
    
    def __init__(self, df=None, var_list=None, weight_var=None, by_list=None, 
                 dfc=None, **kwargs):

        if by_list is None: by_list = []
        if var_list is None: var_list = []
    
        assert (df is None) or (dfc is None)
        
        if df is not None:
            self.set_data(df, var_list, weight_var, by_list=by_list, inplace=True, 
                     **kwargs)
        else:
            self.dfc = dfc
            self.var_list = var_list
            self.weight_var = weight_var
            self.by_list = by_list
    
    def set_data(self, df, var_list, weight_var, by_list, collapse=True,
            inplace=True, scale=False):
        
        copy_list = [var for var in by_list if var not in df.index.names]
        if copy_list:
            self.dfc = df[copy_list].copy()
        else:
            self.dfc = pd.DataFrame(index=df.index.copy())
        
        self.var_list = var_list
        self.weight_var = weight_var
        self.by_list = by_list
        
        if not self.var_list:
            self.var_list = [var for var in df.columns if var not in self.by_list]
            
        if self.weight_var is None:
            weight = np.ones((len(df), 1))
        else:
            weight = df[self.weight_var].values[:, np.newaxis]
            
        if scale:
            weight = weight / np.mean(weight)
            
        # Old way, caused fragmentation errors
        
        # for var in self.var_list:
            
        #     self.dfc[var + '_num'] = df[var] * weight
        #     self.dfc[var + '_denom'] = pd.notnull(df[var]).astype(np.int) * weight
            
        df_num = (df[self.var_list] * weight)
        df_denom = pd.notnull(df[self.var_list]) * weight
        
        df_num = df_num.rename({var : var + '_num' for var in df_num.columns}, axis=1)
        df_denom = df_denom.rename({var : var + '_denom' for var in df_denom.columns}, axis=1)
        
        self.dfc = pd.concat([self.dfc, df_num, df_denom], axis=1)
            
        if collapse:
            return self.collapse(self.by_list, inplace=inplace)
            
    def get_data(self, weight_suffix=False, include_denom=False):
        
        df_num, df_denom = self.get_numerators_and_denominators()
        
        df_out = df_num / df_denom
        if weight_suffix:
            df_out = df_out.rename({var : var + '_' + self.weight_var for var in self.var_list}, axis=1)
            
        if include_denom:
            df_out = pd.concat([df_out, self.dfc[[var + '_denom' for var in self.var_list]]], axis=1)
            
        return df_out
    
    def get_numerators_and_denominators(self):
        
        df_num = self.dfc[[var + '_num' for var in self.var_list]].rename({var + '_num' : var for var in self.var_list}, axis=1)
        df_denom = self.dfc[[var + '_denom' for var in self.var_list]].rename({var + '_denom' : var for var in self.var_list}, axis=1)
        
        return df_num, df_denom
            
    def collapse(self, by_list=None, inplace=False, method='mean'):

        if by_list is None: by_list = []
        
        singleton = (not by_list)
        if singleton:
            dfc_old = self.dfc.copy()
            dfc_old['TEMP'] = 0
            by_list = ['TEMP']
        else:
            dfc_old = self.dfc
        
        if method == 'mean':
            dfc_new = dfc_old.groupby(by_list).sum()
        elif method == 'median':
            raise Exception
            # dfc_new = dfc_old.groupby(by_list).agg(st.weighted_quantile, )
            
        if singleton:
            dfc_new = dfc_new.reset_index().drop(columns=['TEMP'])
            
        if inplace:
            self.dfc = dfc_new
            self.by_list = by_list
            return None
        else:
            return Collapser(dfc=dfc_new, var_list=self.var_list.copy(), 
                             weight_var=self.weight_var, by_list=by_list)
        
    def resample(self, by_list, time_var, freq, inplace=False):
        
        dfc_new = self.dfc.groupby(by_list).resample(freq, level=time_var).sum()
        by_list_new = list(dfc_new.index.names)
        
        if inplace:
            self.dfc = dfc_new
            self.by_list = by_list_new
            return None
        else:
            return Collapser(dfc=dfc_new, var_list=self.var_list.copy(), 
                             weight_var=self.weight_var, by_list=by_list_new)
        
    def loc(self, sliced, copy=False):
        
        dfc_sliced = self.dfc.loc[sliced, :]
        if copy:
            dfc_new = dfc_sliced.copy()
        else:
            dfc_new = dfc_sliced
            
        return Collapser(dfc=dfc_new, var_list=self.var_list.copy(),
                         weight_var=self.weight_var, by_list=self.by_list.copy())
    
    def get_weight(self, var):
        
        return self.dfc[var + '_denom'].values
        
    def save(self, filename, add_suffix=True, fmt='parquet'):
        
        if add_suffix:
            suffix = create_suffix(self.by_list, self.weight_var)
        else:
            suffix = ''
            
        fullname = filename + suffix
        
        if fmt == 'parquet':
            self.dfc.to_parquet(fullname + '_data.parquet')
        elif fmt == 'pickle':
            self.dfc.to_pickle(fullname + '_data.pkl')
        else:
            raise Exception
        
        for item in ['var_list', 'weight_var', 'by_list']:
            in_out.save_pickle(getattr(self, item), fullname + '_' + item + '.pkl')

        return None
    
    def load(self, filename, add_suffix=True, fmt='parquet'):
        
        if add_suffix:
            suffix = create_suffix(self.by_list, self.weight_var)
        else:
            suffix = ''
            
        fullname = filename + suffix
        
        if fmt == 'parquet':
            self.dfc = pd.read_parquet(fullname + '_data.parquet')
        elif fmt == 'pickle':
            self.dfc = pd.read_pickle(fullname + '_data.pkl')
        else:
            raise Exception
        
        for item in ['var_list', 'weight_var', 'by_list']:
            setattr(self, item, in_out.load_pickle(fullname + '_' + item + '.pkl'))

        return None
    
    def rename(self, name_map):
        
        self.dfc.index.names = [name_map.get(var, var) for var in self.dfc.index.names]
        self.by_list = [name_map.get(var, var) for var in self.by_list]
        self.var_list = [name_map.get(var, var) for var in self.var_list]
        self.weight_var = name_map.get(self.weight_var, self.weight_var)
        
        name_map_dfc = {}
        for suffix in ['num', 'denom']:
            name_map_dfc.update({
                key + '_' + suffix : val + '_' + suffix for key, val in name_map.items()
                })
        
        self.dfc = self.dfc.rename(columns=name_map_dfc)
        
        return None
