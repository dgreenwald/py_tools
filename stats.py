import numpy as np
from scipy.stats import norm

def weighted_quantile(values_in, weights_in, quantiles, sort=True):

    if sort:
        sorter = np.argsort(values_in)
        values = values_in[sorter].astype(np.float64)
        weights = weights_in[sorter].astype(np.float64)
    
    cumulative_weights = np.cumsum(weights) - weights[0]
    cumulative_weights /= cumulative_weights[-1]

    return np.interp(quantiles, cumulative_weights, values)

def wq_by_col(values, weights, quantiles, **kwargs):

    Ncol = values.shape[1]
    output = np.zeros((len(quantiles), Ncol))
    for icol in range(Ncol):
        output[:, icol] = weighted_quantile(
            values[:, icol], weights, quantiles, **kwargs
        )

    return output

def weighted_mean(values_in, weights_in):

    num = np.dot(values_in, weights_in)
    denom = np.sum(weights_in)

    return (num / denom)

def weighted_std(values_in, weights_in):

    residuals = values_in - weighted_mean(values_in, weights_in)
    num = np.dot(residuals**2, weights_in)
    denom = np.sum(weights_in)

    return np.sqrt(num / denom)

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
