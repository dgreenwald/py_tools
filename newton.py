#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:15:16 2023

@author: dan
"""

import numpy as np
from . import numerical as nm

def root(fun, x0, args=None, kwargs=None, grad=None, tol=1e-8,
         gradient_kwargs=None, max_iterations=50, max_backstep_iterations=10,
         verbose=True):

    if args is None: args = ()
    if kwargs is None: kwargs = {}
    if gradient_kwargs is None: gradient_kwargs = {}
    
    # Initialization
    x = np.array(x0).copy()
    f_val = fun(x, *args, **kwargs)
    dist = np.linalg.norm(f_val)
    res = {}
    
    iteration = 0
    
    if verbose:
        print("Iteration {0:d}: |f| = {1:g}".format(iteration, dist))
    
    while (dist > tol) and (iteration <= max_iterations):
        
        iteration += 1
        
        # Get Jacobian
        if grad is None:
            grad_val = nm.gradient(fun, x, args=args, kwargs=kwargs, f_val=f_val, **gradient_kwargs)
        else:
            grad_val = grad(x, *args, **kwargs)
            
        # Use Jacobian to compute step size
        step = -np.linalg.solve(grad_val.T, f_val)
        
        # Move in step direction
        backstep_iteration = 0
        dist_new = dist + 1.0
        while (dist_new > dist) and (backstep_iteration <= max_backstep_iterations):
            backstep_iteration += 1
            x_new = x + step
            f_val_new = fun(x_new, *args, **kwargs)
            dist_new = np.linalg.norm(f_val_new)
            step *= 0.5
            
        if dist_new < dist:
            x = x_new
            f_val = f_val_new
            dist = dist_new
        else:
            res['success'] = False
            res['failure_cause'] = 'max_backstep_iterations'
            return res
        
        if verbose:
            print("Iteration {0:d}: |f| = {1:g}".format(iteration, dist))
        
    if dist < tol:
        res['x'] = x
        res['f_val'] = f_val
        res['dist'] = dist
        res['success'] = True
    else:
        res['success'] = False
        res['failure_cause'] = 'max_iterations'
        
    return res
