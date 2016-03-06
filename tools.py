# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:46:14 2016

@author: dan
"""

import numpy as np
from walker import WalkerRandomSampling

def get_unit_vecs(P, tol=1e-8, normalize=False):
    
    vals, vecs = np.linalg.eig(P)
    unit = np.abs(vals - 1.0) < tol
    unit_vecs = vecs[:, unit]
    
    if normalize:
        unit_vecs /= np.sum(unit_vecs, axis=0)
        
    return unit_vecs

def check_ergodic(invariant, tol=1e-8):
    
    invariant_test = invariant - invariant[0, :]
    is_ergodic = np.amax(np.abs(invariant_test)) < tol
    return is_ergodic
    
def update_value(V):
    
    indices = np.argmax(V, axis=1)
    v = V[np.arange(V.shape[0]), indices]
    return indices, v
    
def get_transition(indices):
    
    n = len(indices)
    transition = np.zeros((n, n))
    transition[np.arange(n), indices] = 1
    return transition
    
def sim_discrete(P, N, i0=0):
    
    samplers = [WalkerRandomSampling(P[ii, :]) for ii in range(P.shape[0])]
    
    ix = np.zeros(N).astype(int)
    ix[0] = i0
    
    for ii in range(1, N):
        ix[ii] = samplers[ix[ii-1]].random(1)
    
    return ix
    
def sim_policy(index_list, z_ix_sim, i0=0):
    
    Nsim = len(z_ix_sim)
    ix = np.zeros(Nsim).astype(int)
    ix[0] = i0
    
    for ii in range(0, Nsim):
        if ii > 0:
            ix_old = ix[ii-1]
        else:
            ix_old = i0

        ix[ii] = index_list[z_ix_sim[ii]][ix_old]
        
    return ix

def sim_life_cycle(index_lists, z_ix_sim, i0=0):

    Nt = len(z_ix_sim)
    ix = np.zeros(Nt).astype(int)

    for tt in range(0, Nt):
        if tt > 0:
            ix_old = ix[tt-1]
        else:
            ix_old = i0

        ix[tt] = index_lists[tt][z_ix_sim[tt]][ix_old]

    return ix
