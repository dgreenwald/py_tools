# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:09:18 2016

@author: dan
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import py_tools.econ as ec
from py_tools.utilities import tic, toc

def to_2d(x):
    
    if len(x.shape) == 1:
        return x[:, np.newaxis]
    elif len(x.shape) == 2:
        return x
    else:
        return None

def combine_grids(x, y):
    
    x_2d = to_2d(x)
    y_2d = to_2d(y)
    
    assert (x_2d is not None) and (y_2d is not None)
    
    x_stack = np.repeat(x_2d, len(y), axis=0)
    y_stack = np.tile(y_2d, (len(x), 1))
    
    return np.hstack((x_stack, y_stack))

def combine_grids_from_list(grids_list):
    
    stack = to_2d(grids_list[0])
    assert stack is not None
    
    for ii in range(1, len(grids_list)):
        stack = combine_grids(stack, grids_list[ii])
        
    return stack

def combine_markov_chains(grids_list, P_list):
    
    stack = grids_list[0]
    P_stack = P_list[0]
    
    for ii in range(1, len(grids_list)):
        stack = combine_grids(stack, grids_list[ii])
        P_stack = np.kron(P_stack, P_list[ii])
        
    return stack, P_stack

def drop_low_probs(P, tol=1e-6):
    
    P_new = P.copy()
    ix = P_new < tol
    P_new[ix] = 0
    P_sum = np.sum(P_new, axis=1)
    P_new = P_new / P_sum[:, np.newaxis]
    
    return P_new

class DiscreteModel:
    """Object for discrete modeling. 
    
    INPUTS:
    
    bet: the discount factor

    flow_list: list of flow values for various x, z combinations

    x_grid: the set of endogenous state values 
        size: (Nx x kx) = (# of gridpoints x # of variables)

    z_grid: the set of exogenous state values
        size: (Nz x kz) = (# of gridpoints x # of variables)

    Pz: the transition matrix for the exogenous states

    """

    def __init__(self, bet, flow_list, x_grid, z_grid, Pz, index_list=None,
                 P_tol=1e-6):
        """Constructor"""
        
        self.bet = np.array(bet)
        self.flow_list = flow_list
        self.x_grid = x_grid
        self.z_grid = z_grid
        self.Pz = drop_low_probs(Pz, tol=P_tol)
        # self.sparse = sparse

        # Sizes
        self.Nx, self.kx = self.x_grid.shape
        self.Nz, self.kz = self.z_grid.shape
        
        # self.z_states = np.repeat(np.arange(self.Nz), self.Nx)
        # self.x_states = np.tile(np.arange(self.Nx), self.Nz)
        
        # Sparse identity  matrix
        self.Ixz = sp.eye(self.Nx * self.Nz)

        # Discounted transition probs for exogenous states
        self.bP = self.bet[:, np.newaxis] * self.Pz
        
        # Create sparse matrices
        self.Pzs = sp.csr_matrix(self.Pz)
        self.bPs = sp.csr_matrix(self.bP)
        
        # Discounted transition probs for combined states
        self.bP1 = np.kron(self.bP, np.ones((self.Nx, 1)))

        # Initializations
        # self.v_list = self.Nz * [np.zeros((self.Nx, 1))]
        
        if index_list is None:
            self.index_list = self.Nz * [np.arange(self.Nx).astype(int)]
        else:
            self.index_list = index_list
            
        # Check for nans
        opt_flow = self.get_opt_flow()
        assert np.all(np.isfinite(opt_flow))
        
        # self.flow_mat = np.concatenate([flow_i[np.newaxis, :, :] for flow_i in self.flow_list], axis=0)
        # self.index_mat = np.concatenate([index[np.newaxis, :] for index in self.index_list], axis=0)
        
    def get_opt_flow(self):
        
        opt_flow_list = [self.flow_list[ii][np.arange(self.Nx), 
                                            np.squeeze(self.index_list[ii])][:, np.newaxis] 
                         for ii in range(self.Nz)]
        
        opt_flow = np.vstack(opt_flow_list)
        return opt_flow

    def solve(self):
        """Solve model"""

        it = 0
        done = False

        R = np.vstack(self.flow_list)
        indices = np.vstack(self.index_list)
        
        # Temp code
        self.Pzs = sp.csr_matrix(self.Pz)
        self.bPs = sp.csr_matrix(self.bP)
        self.bP1s = sp.kron(self.bPs, np.ones((self.Nx, 1)))
        
        # end

        while not done:
            
            it += 1
            
            indices_old = indices
            
            # Howard improvement step
            opt_flow = self.get_opt_flow()
            
            # print("Dense:")
            # start = tic()
            # bP_trans = self.get_P_trans(discount=True, sparse=self.sparse)
            # v = np.linalg.solve((np.eye(self.Nx * self.Nz) - bP_trans), opt_flow)
            # toc(start)
            
            # Sparse version
            print("Sparse:")
            start = tic()
            bP_trans_sparse = self.get_P_trans(discount=True, sparse=True)
            v = sp.linalg.spsolve(self.Ixz - bP_trans_sparse, opt_flow)
            toc(start)
            
            # print("Sparse2:")
            # start = tic()
            # bP_trans_sparse = self.get_P_trans(discount=True, sparse=True)
            # vs = sp.linalg.spsolve(self.Ixz - bP_trans_sparse, opt_flow)
            # toc(start)
            
            # Update step
            V = v.reshape((self.Nz, self.Nx))
            W = np.dot(self.bP1, V)
            Q = R + W
            indices, v = ec.update_value(Q)
            self.index_list = np.split(indices, self.Nz)
            
            # toc(start)
            
            done = np.all(indices_old == indices)
            
        print('Converged in {} iterations'.format(it))

        self.V = v.reshape((self.Nz, self.Nx))
        self.I = indices.reshape((self.Nz, self.Nx))
        
        return None
    
    def get_P_trans(self, discount=False, sparse=True):
        
        transition_list = [ec.get_transition(index, sparse=False) for index in self.index_list]
        
        if sparse:
            
            if discount:
                this_P = self.bPs
            else:
                this_P = self.Pzs
            
            P_trans_list = [sp.kron(this_P[ii, :], transition_list[ii]) for ii in range(self.Nz)]
            P_trans = sp.vstack(P_trans_list)
            
        else:
            
            if discount:
                this_P = self.bP
            else:
                this_P = self.Pz
            
            P_trans_list = [np.kron(this_P[ii, :], transition_list[ii]) for ii in range(self.Nz)]
            P_trans = np.vstack(P_trans_list)
            
        return P_trans
    
    def compute_stationary_dist(self, tol=1e-6):
        
        # transition_list = [ec.get_transition(index) for index in self.index_list]
        # P_trans = np.vstack([np.kron(self.Pz[ii, :], transition_list[ii]) for ii in range(self.Nz)])
        
        P_trans = self.get_P_trans(discount=False, sparse=True)
        vals, vecs = sp.linalg.eigs(P_trans.T)
        
        assert np.abs(np.real(vals[0]) - 1.0) < tol
        # ix = np.argmax(np.abs(vals))
        
        self.pi_star = np.real(vecs[:, 0])
        self.pi_star = self.pi_star / np.sum(self.pi_star)
        # self.pi_star = vecs[:, 0] / np.sum(vecs[:, 0])
        
        # check = P_trans.T @ self.pi_star
        return None

    def sim(self, Nsim, ix0=0, iz0=0):
        
        """Simulate from solution"""
        z_ix_sim = ec.sim_discrete(self.Pz, Nsim, i0=iz0)
        z_sim = self.z_grid[z_ix_sim, :]

        x_ix_sim = ec.sim_policy(self.index_list, z_ix_sim, i0=ix0)
        x_sim = self.x_grid[x_ix_sim, :]
        return (x_sim, z_sim)

class LifeCycleModel:
    """Life cycle model"""

    def __init__(self, bet, flow_lists, terminal_list, x_grid, z_grid, Pz):
        """Constructor"""
        self.bet = bet
        self.flow_lists = flow_lists
        self.terminal_list = terminal_list
        self.x_grid = x_grid
        self.z_grid = z_grid
        self.Pz = Pz

        # Initial calculations
        self.Nx, self.kx = self.x_grid.shape
        self.Nz, self.kz = self.z_grid.shape
        self.Nt = len(self.flow_lists)

        self.bP = self.bet * self.Pz
        self.bP1 = np.kron(self.bP, np.ones((self.Nx, 1)))

        # Initializations
        self.v_lists = self.Nt * [self.Nz * [np.zeros((self.Nx, 1))]]
        self.index_lists = self.Nt * [self.Nz * [np.zeros((self.Nx, 1)).astype(int)]]

    def solve(self):
        """Solve model"""

        v_next = np.vstack([vi.T for vi in self.terminal_list])

        for tt in range(self.Nt-1, -1, -1):

            R = np.vstack(self.flow_lists[tt])
            W = np.dot(self.bP1, v_next)
            V = R + W
            indices, v = ec.update_value(V)

            self.v_lists[tt] = np.split(v, self.Nz)
            self.index_lists[tt] = np.split(indices, self.Nz)
            
            v_next = np.vstack([vi.T for vi in self.v_lists[tt]])
        
        return None

    def sim(self, Nsim):
        """Simulate from solution"""
        z_sim = np.zeros((self.kz, self.Nt, Nsim))
        x_sim = np.zeros((self.kx, self.Nt, Nsim))
        
        for ii in range(Nsim):
            z_ix_sim = ec.sim_discrete(self.Pz, self.Nt)
            z_sim[:, :, ii] = self.z_grid[z_ix_sim, :].T  
            
            x_ix_sim = ec.sim_life_cycle(self.index_lists, z_ix_sim)
            x_sim[:, :, ii] = self.x_grid[x_ix_sim, :].T
            
        return (x_sim, z_sim)
