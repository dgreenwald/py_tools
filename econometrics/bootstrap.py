#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:06:33 2024

@author: dan
"""

import numpy as np
from py_tools import data as dt
from scipy.optimize import root_scalar


def ar1_bootstrap_inner(rho, mu, eps_boot, x_init):
    """
    Inner loop for AR(1) bootstrap simulation.

    Simulates ``Nboot`` AR(1) paths using the supplied bootstrap residuals and
    returns the mean OLS rho estimate across those paths.

    Parameters
    ----------
    rho : float
        AR(1) persistence coefficient used to simulate paths.
    mu : float
        Unconditional mean of the AR(1) process.
    eps_boot : ndarray of shape (Nboot, Nt_eps)
        Bootstrap residual draws (rows = bootstrap replications,
        columns = time periods).
    x_init : float
        Initial value for each simulated path.

    Returns
    -------
    rho_sim_avg : float
        Mean OLS rho estimate computed across all ``Nboot`` simulated paths.
    """
    Nboot, Nt_eps = eps_boot.shape

    X = np.zeros((Nboot, Nt_eps + 1))
    X[:, 0] = x_init
    for tt in range(1, Nt_eps + 1):
        X[:, tt] = (1.0 - rho) * mu + rho * X[:, tt - 1] + eps_boot[:, tt - 1]

    X_next = X[:, 1:]
    X_pre = X[:, :-1]

    X_next_til = X_next - np.mean(X_next, axis=1)[:, np.newaxis]
    X_pre_til = X_pre - np.mean(X_pre, axis=1)[:, np.newaxis]

    cov = np.sum(X_next_til * X_pre_til, axis=1) / (Nt_eps - 1)
    var = np.sum(X_pre_til**2, axis=1) / (Nt_eps - 1)

    rho_sim = cov / var
    rho_sim_avg = np.mean(rho_sim)
    return rho_sim_avg


def objfcn_ar1_bootstrap(rho, rho_ols, mu, eps_boot, x_init):
    """
    Objective function for AR(1) bootstrap bias correction.

    Returns the difference between the OLS estimate and the bootstrap-simulated
    average rho.  Pass to ``scipy.optimize.root_scalar`` to find the true rho
    whose simulated OLS average equals ``rho_ols``.

    Parameters
    ----------
    rho : float
        Candidate AR(1) persistence coefficient.
    rho_ols : float
        OLS estimate of rho obtained from the actual data.
    mu : float
        Unconditional mean of the AR(1) process.
    eps_boot : ndarray of shape (Nboot, Nt_eps)
        Bootstrap residual draws.
    x_init : float
        Initial value for each simulated path.

    Returns
    -------
    float
        ``rho_ols - rho_sim_avg``.  Zero when ``rho`` reproduces ``rho_ols``
        on average.
    """
    rho_sim_avg = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
    return rho_ols - rho_sim_avg


def ar1_bootstrap_bias(df, obs_var, Nboot=100000, iterate=False):
    """
    Estimate a bias-corrected AR(1) persistence coefficient via bootstrap.

    Runs OLS on the observed series to obtain ``rho_ols``, resamples residuals
    ``Nboot`` times, and either applies a one-step additive bias correction or
    iteratively solves for the rho whose simulated OLS average matches
    ``rho_ols``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the observed series.
    obs_var : str
        Column name of the AR(1) variable.
    Nboot : int, optional
        Number of bootstrap replications. Default is 100000.
    iterate : bool, optional
        If ``True``, use ``scipy.optimize.root_scalar`` for iterative bias
        correction.  If ``False`` (default), apply a single additive
        correction: ``rho_corrected = 2 * rho_ols - rho_sim_avg``.

    Returns
    -------
    rho : float
        Bias-corrected AR(1) persistence estimate.

    Notes
    -----
    Progress information (``rho_ols``, bias, and ``rho_new``) is printed to
    stdout during execution.
    """
    _df = df[[obs_var]].dropna()
    _df["L_" + obs_var] = _df[obs_var].shift()

    fr = dt.regression(_df, obs_var, ["L_" + obs_var])
    mu = fr.results.params[0]
    rho_ols = fr.results.params[1]
    print("rho_ols: " + repr(rho_ols))

    rho = rho_ols
    resid = fr.results.resid

    Nt_eps = len(resid)
    eps_boot = np.random.choice(resid, Nboot * Nt_eps, replace=True).reshape(
        (Nboot, Nt_eps)
    )
    x_init = _df.iloc[0][obs_var]

    if iterate:
        rho_guess = 0.5 * (rho_ols + 1.0)
        args = (rho_ols, mu, eps_boot, x_init)
        res = root_scalar(objfcn_ar1_bootstrap, args=args, x0=rho_ols, x1=rho_guess)
        rho = res.root

    else:
        rho_sim_avg = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        bias = rho_ols - rho_sim_avg
        rho = rho_ols + bias
        print("original bias: " + repr(bias))

    print("rho_new: " + repr(rho))

    return rho
