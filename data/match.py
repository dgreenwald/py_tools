#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def pairwise_match(treated_score, untreated_score, caliper=0.0, 
                   replacement=True):
    """One-to-one matching based on propensity scores"""
    
    # Get differences for all combinations
    dist_mat = np.abs(treated_score[:, np.newaxis] - untreated_score[np.newaxis, :])
    
    # Apply caliper if specified
    if caliper > 0.0:
        ix = dist_mat > caliper
        dist_mat[ix] = np.inf
        
    # Keep only rows that have an acceptable solution
    good_rows = np.any(np.isfinite(dist_mat), axis=1)
    dist_mat = dist_mat[good_rows, :]
    
    if replacement:
        
        min_ix = np.argmin(dist_mat, axis=1)
        
    else:
        
        # For now, only works with replacement
        raise Exception
        
    return good_rows, min_ix