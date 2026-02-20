import numpy as np
from scipy.stats import norm, lognorm

def weighted_quantile(values_in, weights_in, quantiles, sort=True, C=0.5):
    
    # assert (np.all(np.isfinite(values_in)) and np.all(np.isfinite(weights_in)))
    
    ix_keep = np.isfinite(values_in) & np.isfinite(weights_in) & (weights_in > 0.0)
    if np.sum(ix_keep) == 1:
        return np.full(len(quantiles), values_in[ix_keep][0])
    elif np.sum(ix_keep) == 0:
        return np.full(len(quantiles), np.nan)

    if sort:
        sorter = np.argsort(values_in[ix_keep])
        values = values_in[ix_keep][sorter].astype(np.float64)
        weights = weights_in[ix_keep][sorter].astype(np.float64)
    
    S = np.cumsum(weights)
    q_grid = (S - C * weights) / (S[-1] + (1.0 - 2.0 * C) * weights)

    return np.interp(quantiles, q_grid, values)

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

def weighted_var(values_in, weights_in):
    
    residuals = values_in - weighted_mean(values_in, weights_in)
    num = np.dot(residuals**2, weights_in)
    denom = np.sum(weights_in)
    
    return num / denom

def weighted_std(values_in, weights_in):

#    residuals = values_in - weighted_mean(values_in, weights_in)
#    num = np.dot(residuals**2, weights_in)
#    denom = np.sum(weights_in)
#
#    return np.sqrt(num / denom)
    
    return np.sqrt(weighted_var(values_in, weights_in))

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

def draw_norm(Sig):

    C = np.linalg.cholesky(Sig)
    return np.dot(C, np.random.randn(Sig.shape[0]))

def draw_norm_multi(Sig, n):

    C = np.linalg.cholesky(Sig)
    return np.dot(np.random.randn(n, Sig.shape[0]), C.T)

def my_lognorm(mu, sig):
    
    return lognorm(s=sig, scale=np.exp(mu))
