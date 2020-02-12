#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:08:09 2020

@author: dan
"""

import numpy as np
import pandas as pd
#import cPickle as pickle

from py_tools import in_out

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

def load_collapser(filename, add_suffix=True, by_list=[], weight_var=None):
        
    col = Collapser(by_list=by_list, weight_var=weight_var)
    col.load(filename, add_suffix=add_suffix)
    return col

def create_suffix(by_list, weight_var):
    
    return '_'.join([''] + by_list + ['wtd_by', weight_var])

def collapse(df, by_list, var_list=[], weight_var=None, weight_suffix=False):
    
    coll = Collapser(df, var_list=var_list, by_list=by_list, 
                     weight_var=weight_var)
    
    return coll.get_data(weight_suffix=weight_suffix)

class Collapser:
    
    """Class for flexibly collapsing data sets"""
    
    def __init__(self, df=None, var_list=[], weight_var=None, by_list=[], 
                 dfc=None, **kwargs):
    
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
            weight = np.ones(len(df))
        else:
            weight = df[self.weight_var].copy()
            
        if scale:
            weight = weight / np.mean(weight)
            
        for var in self.var_list:
            
            self.dfc[var + '_num'] = df[var] * weight
            self.dfc[var + '_denom'] = pd.notnull(df[var]).astype(np.int) * weight
            
        if collapse:
            return self.collapse(self.by_list, inplace=inplace)
            
    def get_data(self, weight_suffix=False):
        
        if weight_suffix:
            suffix = '_' + self.weight_var
        else:
            suffix = ''
        
        df_out = pd.DataFrame(index=self.dfc.index.copy())
        for var in self.var_list:
            df_out[var + suffix] = self.dfc[var + '_num'] / self.dfc[var + '_denom']
            
        return df_out
            
    def collapse(self, by_list, inplace=False):
        
        dfc_new = self.dfc.groupby(by_list).sum()
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
        
        self.dfc = self.dfc.rename(columns=name_map)
        self.dfc.index.names = [name_map.get(var, var) for var in self.dfc.index.names]
        self.by_list = [name_map.get(var, var) for var in self.by_list]
        
        return None