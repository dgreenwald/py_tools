#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:01:55 2023

@author: dan
"""

class ModSpec:
    
    def __init__(self, mod=None, add_list=[], remove_list=[], label=None):
        
        self.features = add_list.copy()
        if mod is not None:
            self.features += [feature for feature in mod.features if feature not in remove_list]
            
        self.features = sorted(list(set(self.features)))
        
        if label is None:
            self.label = '_'.join(self.features)

    def __contains__(self, key):

        return (key in self.features)
    
    def __str__(self):
        
        return self.label