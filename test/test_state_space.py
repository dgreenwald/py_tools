#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:09:55 2018

@author: dan
"""

import matplotlib.pyplot as plt
import numpy as np
from py_tools import state_space
from scipy.stats import multivariate_normal as mvn

A = np.array(([0.8, 0.1], [0.05, 0.9]))
Q = np.diag([0.2, 0.5])
R = np.eye(2)
Z = np.eye(2)
H = np.zeros((2, 2))

# Simulate data
P0 = np.diag([0.8, 0.8])
x0 = np.zeros(2)

# Set normal
dist = mvn(mean=np.zeros(2), cov=Q)
dist_init = mvn(mean=x0, cov=P0)

# Simulate data and compute true likelihood
Nt = 20
x_sim = np.zeros((Nt, 2))
e_sim = dist.rvs(Nt-1)
L_all = np.zeros(Nt)

x_sim[0, :] = dist_init.rvs(1)
L_all[0] = dist_init.logpdf(x_sim[0, :])

for tt in range(1, 20):
    x_sim[tt, :] = np.dot(A, x_sim[tt-1, :]) + np.dot(R, e_sim[tt-1, :])
    L_all[tt] = dist.logpdf(e_sim[tt-1, :])

L_true = np.sum(L_all)
print(L_all)

y_sim = x_sim.copy()

ssm = state_space.StateSpaceModel(A, R, Q, Z, H)
sse = state_space.StateSpaceEstimates(ssm, y_sim)
sse.kalman_filter(x_init=x0, P_init=P0)