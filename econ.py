# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:46:14 2016

@author: dan
"""

import numpy as np
import scipy.sparse as sp
from . import walker
# from walker import WalkerRandomSampling

# def lse(x, axis=None):
    # """Safely computes log(sum(exp(x)))"""

    # x_star = np.amax(x)
    # x_til = x - x_star
    # return x_star + np.log(np.sum(np.exp(x_til), axis=axis))

def get_unit_vecs(P, tol=1e-8, normalize=False, **kwargs):
    
    vals, vecs = np.linalg.eig(P)
    unit = np.abs(vals - 1.0) < tol
    unit_vecs = np.real(vecs[:, unit])
    
    if normalize:
        unit_vecs /= np.sum(unit_vecs, axis=0)
        
    return unit_vecs

def stationary_doubling(P, tol=1e-12, pi_seed=None, maxit=500, normalize=True, **kwargs):
    """Find invariant distribution of a Markov chain by iteration."""
    
    if pi_seed is None:
        pi = np.ones(P.shape[0]) / P.shape[0]
    else:
        pi = pi_seed
        
    Pj = P.copy()

    for it in range(maxit):
        pi_new = pi @ Pj
        Pj = Pj @ Pj
        dist = np.max(np.abs(pi_new - pi))
        # print(dist)
        if dist < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new
    
    if normalize:
        pi = pi / np.sum(pi)

    return pi

def ergodic_dist(P, doubling=False, **kwargs):
    
    if doubling:
        return stationary_doubling(P, **kwargs)
    else:
        return get_unit_vecs(P.T, normalize=True, **kwargs) 

def check_ergodic(invariant, tol=1e-8):
    
    invariant_test = invariant - invariant[0, :]
    is_ergodic = np.amax(np.abs(invariant_test)) < tol
    return is_ergodic

def markov_std(P, vals):

    Ev = np.dot(P, vals)
    Ev2 = np.dot(P, vals ** 2)
    V = Ev2 - (Ev ** 2)
    sig = np.sqrt(V)

    return sig
    
def update_value(V):
    
    indices = np.argmax(V, axis=1)
    v = V[np.arange(V.shape[0]), indices]
    return indices, v
    
def get_transition(indices, sparse=False):
    
    n = len(indices)
    
    if sparse:
        transition = sp.lil_matrix((n, n))
        transition = transition.tocsr()
    else:
        transition = np.zeros((n, n))
        
    transition[np.arange(n), indices] = 1
    return transition
    
def sim_discrete(P, N, i0=0):
    
    samplers = [walker.WalkerRandomSampling(P[ii, :]) for ii in range(P.shape[0])]
    
    ix = np.zeros(N).astype(int)
    ix[0] = i0
    
    for ii in range(1, N):
        ix[ii] = samplers[ix[ii-1]].random(1)
    
    return ix

def sim_iid(p, N):

    sampler = walker.WalkerRandomSampling(p)
    return sampler.random(N)

def sim_discrete_from_ergodic(P, N, pi_star=None):

    if pi_star is None:
        pi_star = ergodic_dist(P)
        
    i0 = np.random.choice(len(pi_star), p=pi_star.ravel())

    return sim_discrete(P, N, i0)

def multi_choice(p):
    """Draw from many probability vectors in parallel.  Each row of the p
    matrix should be one vector."""

    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)

    return choices
    
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

def sim_ar1(rho, sig, mu=0.0, Nsim=100, e=None, x0=None):

    x = np.zeros(Nsim)
    if e is None:
        e = np.random.randn(Nsim)

    if x0 is None:
        sig0 = sig / np.sqrt(1.0 - rho ** 2)
        x[0] = sig0 * e[0] 
    else:
        x[0] = x0

    for jj in range(1, Nsim):
        x[jj] = rho * x[jj-1] + sig * e[jj]

    x += mu
    return x

def sim_ar1_multi(rho, sig, Nper, Nsim=1, mu=0.0, e=None, x0=None):
    
    x = np.zeros((Nsim, Nper))
    if e is None:
        e = np.random.randn(Nsim, Nper)
    else:
        assert e.shape == (Nsim, Nper)
        
    if x0 is None:
        sig0 = sig / np.sqrt(1.0 - rho ** 2)
        x[:, 0] = sig0 * e[:, 0]
    else:
        assert len(x0) == Nsim
        
    for jj in range(1, Nper):
        x[:, jj] = rho * x[:, jj-1] + sig * e[:, jj]
        
    x += mu
    return x

def sim_cir(rho, sig, mu=0.0, Nsim=100, e=None, x0=None, bound=False):
    """Cox-Ingersoll-Ross Process"""

    x = np.zeros(Nsim)
    if e is None:
        e = np.random.randn(Nsim)

    if x0 is None:
        raise Exception
        sig0 = sig / np.sqrt(1.0 - rho ** 2)
        x[0] = mu + mu * sig0 * e[0] 
    else:
        x[0] = x0

    if bound:
        x[0] = np.abs(x[0])

    for jj in range(1, Nsim):
        x[jj] = (1.0 - rho) * mu + rho * x[jj-1] + np.sqrt(np.abs(x[jj-1])) * sig * e[jj]
        if bound:
            x[jj] = np.abs(x[jj])

    return x

# Rouwenhorst approximation
def discrete_approx(rho, sig_e, N, cons=0.0):

    q = 0.5 * (1.0 + rho)
    sig_z = np.sqrt((sig_e ** 2)/(1 - rho ** 2))
    psi = np.sqrt(N - 1.0) * sig_z

    y = np.linspace(-psi, psi, N) + cons

    P = np.array([[q, 1.0 - q], [1.0 - q, q]])

    for ii in range(2, N):
        P_old = P
        P = np.zeros((ii+1, ii+1))

        P[:-1, :-1] += q * P_old
        P[:-1, 1:] += (1.0 - q) * P_old
        P[1:, :-1] += (1.0 - q) * P_old
        P[1:, 1:] += q * P_old

        P[1:-1, :] *= 0.5

    P /= np.sum(P, axis=1)

    return (y, P)
