#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:15:16 2023

@author: dan
"""

import numpy as np
from . import numerical as nm

def secant(fcn, args, x0, x1, tol=1e-6, max_it_inner=20, max_it_outer=50):
    
    it_outer = 0
    
    f0 = fcn(x0, *args)
    f1 = fcn(x1, *args)
    
    while True:
        
        dist = np.abs(f1)[0]
        print(f"Iteration {it_outer:d}: |f| = {dist:g}")
        
        slope = (f1 - f0) / (x1 - x0)
        step = -f1 / slope
        
        done = False
        it_inner = 0
        while not done:
            
            x2 = x1 + step
            f2 = fcn(x2, *args)
            if np.abs(f2) < np.abs(f1):
                done = True
            else:
                step *= 0.5
                it_inner += 1
                if it_inner > max_it_inner:
                    return None
                
        if np.abs(f2) < tol:
        
            return x2
        
        else:
        
            it_outer += 1
            if it_outer > max_it_outer:
                return None
            
            x0 = x1
            f0 = f1
            
            x1 = x2
            f1 = f2

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
