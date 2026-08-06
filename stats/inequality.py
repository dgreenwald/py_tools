#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:27:08 2021

@author: dan
"""

import numpy as np
from scipy import integrate
from . import core as stats


def compute_gini(df, var, wvar=None):
    """Compute the Gini coefficient and Lorenz curve for a variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing the variable of interest.
    var : str
        Name of the column in ``df`` whose inequality is to be measured.
    wvar : str, optional
        Name of the column in ``df`` containing observation weights.  If
        ``None`` (default), equal weights are assumed (unit counts).

    Returns
    -------
    gini_coeff : float
        Gini coefficient in ``[0, 1]``.  A value of 0 indicates perfect
        equality; 1 indicates maximal inequality.
    c_weight : numpy.ndarray
        Cumulative population shares (x-axis of the Lorenz curve).
    c_shares : numpy.ndarray
        Cumulative income/wealth shares (y-axis of the Lorenz curve).
    """
    if wvar is None:
        df_by_var = df.groupby(var)[var].count()
    else:
        df_by_var = df.groupby(var)[wvar].sum()

    df_by_var = df_by_var.to_frame(name="weight").sort_index().reset_index()
    ix = df_by_var["weight"] > 1e-12
    df_by_var = df_by_var.loc[ix]
    df_by_var["w_share"] = df_by_var[var] * df_by_var["weight"]

    df_by_var["w_share"] /= np.sum(df_by_var["w_share"])
    df_by_var["weight"] /= np.sum(df_by_var["weight"])

    c_shares = np.cumsum(df_by_var["w_share"].values)
    c_weight = np.cumsum(df_by_var["weight"].values)

    area_under = integrate.simpson(c_shares, c_weight)
    gini_coeff = 1.0 - 2.0 * area_under

    return gini_coeff, c_weight, c_shares


def get_top_shares(df, var, shares=None, wvar=None):
    """Compute top-share statistics for one or more percentile thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    var : str
        Name of the column in ``df`` for which top shares are computed.
    shares : array_like, optional
        Percentile thresholds (as percentages) defining the top groups.
        For example, ``[10, 1]`` returns the top-10 % and top-1 % shares.
        Defaults to ``[10]``.
    wvar : str, optional
        Name of the column in ``df`` containing observation weights.  If
        ``None`` (default), equal weights are assumed.

    Returns
    -------
    list of float
        Top-share fractions (in ``[0, 1]``) corresponding to each threshold
        in ``shares``, in the same order.
    """
    shares = np.atleast_1d([10] if shares is None else shares)
    top_shares = []

    if wvar is None:
        weight = np.ones(len(df))
    else:
        weight = df[wvar].values

    for share in shares:
        q = 1.0 - (0.01 * share)
        cutoff = stats.weighted_quantile(df[var].values, weight, q)
        ix = df[var] >= cutoff
        top_share = np.sum(df.loc[ix, var] * weight[ix]) / np.sum(df[var] * weight)
        top_shares.append(top_share)

    return top_shares
