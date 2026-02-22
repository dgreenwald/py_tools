#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:15:16 2023

@author: dan
"""

import numpy as np
from . import core as nm

def secant(fcn, args, x0, x1, tol=1e-6, max_it_inner=20, max_it_outer=50, verbose=True):
    """Find a root of a scalar function using the secant method.

    Applies a secant step at each outer iteration, then uses a
    line-search (halving the step) to ensure progress.

    Parameters
    ----------
    fcn : callable
        Scalar-valued function whose root is sought.  Called as
        ``fcn(x, *args)``.
    args : tuple
        Positional arguments passed to *fcn* after *x*.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.  Must differ from *x0*.
    tol : float, optional
        Convergence tolerance on ``|f(x)|``, by default ``1e-6``.
    max_it_inner : int, optional
        Maximum number of line-search halvings per outer iteration,
        by default ``20``.
    max_it_outer : int, optional
        Maximum number of secant iterations, by default ``50``.
    verbose : bool, optional
        If ``True`` (default), print the residual at each iteration.

    Returns
    -------
    float or None
        The root *x* such that ``|f(x)| < tol``, or ``None`` if the
        method failed to converge.
    """
    it_outer = 0

    f0 = fcn(x0, *args)
    f1 = fcn(x1, *args)

    while True:

        dist = float(np.abs(f1))
        if verbose:
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
    """Find a root of a vector-valued function using Newton's method.

    At each iteration the Jacobian is estimated (or supplied) and a
    Newton step is taken.  A back-tracking line search halves the step
    while the residual fails to decrease.

    Parameters
    ----------
    fun : callable
        Function whose root is sought.  Called as
        ``fun(x, *args, **kwargs)`` and must return a 1-D array with
        the same length as *x0*.
    x0 : array-like
        Initial guess.
    args : tuple, optional
        Positional arguments forwarded to *fun*, by default ``()``.
    kwargs : dict, optional
        Keyword arguments forwarded to *fun*, by default ``{}``.
    grad : callable or None, optional
        Function returning the Jacobian matrix at a given *x*, called
        as ``grad(x, *args, **kwargs)``.  If ``None`` (default), the
        Jacobian is estimated by finite differences using
        :func:`~py_tools.numerical.core.gradient`.
    tol : float, optional
        Convergence tolerance on the Euclidean norm of the residual,
        by default ``1e-8``.
    gradient_kwargs : dict, optional
        Extra keyword arguments forwarded to the finite-difference
        gradient estimator, by default ``{}``.
    max_iterations : int, optional
        Maximum number of Newton iterations, by default ``50``.
    max_backstep_iterations : int, optional
        Maximum number of step-halving attempts per iteration, by
        default ``10``.
    verbose : bool, optional
        If ``True`` (default), print the residual norm at each
        iteration.

    Returns
    -------
    dict
        Result dictionary with keys:

        ``'success'`` : bool
            Whether convergence was achieved.
        ``'x'`` : numpy.ndarray
            Solution vector (present only on success).
        ``'f_val'`` : numpy.ndarray
            Residual at the solution (present only on success).
        ``'dist'`` : float
            Final residual norm (present only on success).
        ``'failure_cause'`` : str
            Reason for failure (present only on failure).
            Either ``'max_backstep_iterations'`` or
            ``'max_iterations'``.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if gradient_kwargs is None:
        gradient_kwargs = {}
    
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
        # print(grad_val)
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
