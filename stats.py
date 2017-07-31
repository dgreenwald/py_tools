import numpy as np
from scipy.stats import norm

def weighted_quantile(values, weights, quantiles, sort=True):

    if sort:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
    
    cumulative_weights = np.cumsum(weights) - weights[0]
    cumulative_weights /= cumulative_weights[-1]

    return np.interp(quantiles, cumulative_weights, values)

def std_norm_z_star(p_val, two_sided=True):

    if two_sided:
        p = 1.0 - 0.5 * (1.0 - p_val)
    else:
        p = p_val

    return (norm.ppf(p))

def std_norm_bands(p_val):

    half_tail_prob = 0.5 * (1.0 - p_val)
    p_lb = half_tail_prob
    p_ub = 1.0 - half_tail_prob

    z_lb = norm.ppf(p_lb)
    z_ub = norm.ppf(p_ub)

    return (z_lb, z_ub)
