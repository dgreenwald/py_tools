# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:09:18 2016

@author: dan
"""

import matplotlib.pyplot as plt
import numpy as np
import tools

class DiscreteModel:
    """Object for discrete modeling"""

    def __init__(self, bet, flow_list, x_grid, z_grid, Pz):
        """Constructor"""
        self.bet = bet
        self.flow_list = flow_list
        self.x_grid = x_grid
        self.z_grid = z_grid
        self.Pz = Pz

        # Initial calculations
        self.Nx = len(self.x_grid)
        self.Nz = len(self.z_grid)

        self.bP = self.bet * self.Pz
        self.bP1 = np.kron(self.bP, np.ones((self.Nx, 1)))

        # Initializations
        self.v_list = self.Nz * [np.zeros((self.Nx, 1))]
        self.index_list = self.Nz * [np.zeros((self.Nx, 1)).astype(int)]

    def solve(self):
        """Solve model"""

        it = 0
        done = False

        R = np.vstack(self.flow_list)
        indices = np.vstack(self.index_list)

        while not done:
            
            it += 1
            indices_old = indices
            
            # Howard improvement step
            transition_list = [tools.get_transition(index) for index in self.index_list]
            bP_trans_list = [np.kron(self.bP[ii, :], transition_list[ii]) for ii in range(self.Nz)]
            bP_trans = np.vstack(bP_trans_list)
            
            opt_flow_list = [self.flow_list[ii][np.arange(self.Nx), np.squeeze(self.index_list[ii])][:, np.newaxis] 
                             for ii in range(self.Nz)]
            opt_flow = np.vstack(opt_flow_list)
            v = np.linalg.solve((np.eye(self.Nx * self.Nz) - bP_trans), opt_flow)
            self.v_list = np.split(v, self.Nz)
            
            # Update step
            v_next = np.vstack([vi.T for vi in self.v_list])
            W = np.dot(self.bP1, v_next)
            V = R + W
            indices, v = tools.update_value(V)
            self.index_list = np.split(indices, self.Nz)
            
            done = np.all(indices_old == indices)
            
        print('Converged in {} iterations'.format(it))

    def sim(self, Nsim):
        z_ix_sim = tools.sim_discrete(self.Pz, Nsim)
        z_sim = self.z_grid[z_ix_sim]

        state_ix_sim = tools.sim_policy(self.index_list, z_ix_sim)
        x_sim = self.x_grid[state_ix_sim]
        return (x_sim, z_sim)
