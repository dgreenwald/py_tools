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
    var_list = col0.var_list
    weight_var = col0.weight_var
    by_list = col0.by_list
    
    if check:
        for col in collapser_list[1:]:
            assert col.var_list == var_list
            assert col.weight_var == weight_var
            assert col.by_list == by_list
    
    df = pd.concat([col.df for col in collapser_list], axis=0)
    
    return Collapser(df, var_list=var_list, weight_var=weight_var, by_list=by_list)

class Collapser:
    
    """Class for flexibly collapsing data sets"""
    
    def __init__(self, df=None, var_list=None, weight_var=None, by_list=None):
    
        
        self.df = None
        self.var_list = None
        self.weight_var = None
        self.by_list = None
        
        if df is not None:
            self.set(df, var_list, weight_var, by_list=by_list)
    
    def set(self, df, var_list, weight_var, by_list=None):
        
        self.df = pd.DataFrame(index=df.index.copy())
        self.var_list = var_list
        self.weight_var = weight_var
        self.by_list = by_list
        
        if df is not None:
            self.df = pd.DataFrame(index=df.index.copy())
            
        for var in self.var_list:
            self.df[var + '_num'] = df[var] * df[self.weight_var]
            self.df[var + '_denom'] = pd.notnull(df[var]).astype(np.int) * df[self.weight_var]
            
        if self.by_list is not None:
            self.collapse(self.by_list)
            
    def get(self):
        
        df_out = pd.DataFrame(index=self.df.index.copy())
        for var in self.var_list:
            df_out[var] = self.df[var + '_num'] / self.df[var + '_denom']
            
        return df_out
            
    def collapse(self, by_list, weight_var=None):
        
        if self.by_list is not None:
            self.df = self.df.reset_index()

        self.by_list = by_list
        self.df = self.df.groupby(by_list).sum()
        
        return None
        
    def save(self, filename, fmt='parquet'):
        
        if fmt == 'parquet':
            self.df.to_parquet(filename + '_data.parquet')
        elif fmt == 'pickle':
            self.df.to_pickle(filename + '_data.pkl')
        else:
            raise Exception
        
        for item in ['var_list', 'weight_var', 'by_list']:
            in_out.save_pickle(filename + '_' + item + '.pkl', getattr(self, item))

        return None
    
    def load(self, filename, fmt='parquet'):
        
        if fmt == 'parquet':
            self.df = pd.read_parquet(filename + '_data.parquet')
        elif fmt == 'pickle':
            self.df = pd.read_pickle(filename + '_data.pkl')
        else:
            raise Exception
        
        for item in ['var_list', 'weight_var', 'by_list']:
            setattr(self, item, in_out.load_pickle(filename + '_' + item + '.pkl'))

        return None