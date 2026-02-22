import numpy as np
from scipy.stats import norm, lognorm

def weighted_quantile(values_in, weights_in, quantiles, sort=True, C=0.5):
    """Compute weighted quantiles of a 1-D array.

    Parameters
    ----------
    values_in : array_like
        Input data values.
    weights_in : array_like
        Non-negative weights corresponding to each value.  Entries that are
        non-finite or non-positive are excluded from the calculation.
    quantiles : array_like
        Quantile levels to evaluate, each in ``[0, 1]``.
    sort : bool, optional
        If ``True`` (default) values are sorted before computing quantiles.
        Set to ``False`` only when ``values_in`` is already sorted.
    C : float, optional
        Interpolation constant that controls how quantiles are mapped to
        cumulative weights.  The default ``0.5`` produces a result equivalent
        to the "hazen" plotting position.

    Returns
    -------
    numpy.ndarray
        Quantile values with the same length as ``quantiles``.  Returns an
        array filled with ``nan`` when no valid observations are present, or
        a constant array when only one valid observation exists.
    """
    ix_keep = np.isfinite(values_in) & np.isfinite(weights_in) & (weights_in > 0.0)
    if np.sum(ix_keep) == 1:
        return np.full(len(quantiles), values_in[ix_keep][0])
    elif np.sum(ix_keep) == 0:
        return np.full(len(quantiles), np.nan)

    if sort:
        sorter = np.argsort(values_in[ix_keep])
    else:
        sorter = np.arange(np.sum(ix_keep))

    values = values_in[ix_keep][sorter].astype(np.float64)
    weights = weights_in[ix_keep][sorter].astype(np.float64)
    
    S = np.cumsum(weights)
    q_grid = (S - C * weights) / (S[-1] + (1.0 - 2.0 * C) * weights)

    return np.interp(quantiles, q_grid, values)

def wq_by_col(values, weights, quantiles, **kwargs):
    """Compute weighted quantiles for each column of a 2-D array.

    Parameters
    ----------
    values : numpy.ndarray
        2-D array of shape ``(N, K)`` where quantiles are computed along axis 0
        for each of the ``K`` columns.
    weights : array_like
        1-D array of length ``N`` with non-negative observation weights shared
        across all columns.
    quantiles : array_like
        Quantile levels to evaluate, each in ``[0, 1]``.
    **kwargs
        Additional keyword arguments forwarded to :func:`weighted_quantile`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(quantiles), K)`` containing the weighted
        quantiles for each column.
    """
    Ncol = values.shape[1]
    output = np.zeros((len(quantiles), Ncol))
    for icol in range(Ncol):
        output[:, icol] = weighted_quantile(
            values[:, icol], weights, quantiles, **kwargs
        )

    return output

def weighted_mean(values_in, weights_in):
    """Compute the weighted mean of an array.

    Parameters
    ----------
    values_in : array_like
        Input data values.
    weights_in : array_like
        Non-negative weights corresponding to each value.

    Returns
    -------
    float
        Weighted arithmetic mean of ``values_in``.
    """
    num = np.dot(values_in, weights_in)
    denom = np.sum(weights_in)

    return (num / denom)

def weighted_var(values_in, weights_in):
    """Compute the weighted variance of an array.

    Parameters
    ----------
    values_in : array_like
        Input data values.
    weights_in : array_like
        Non-negative weights corresponding to each value.

    Returns
    -------
    float
        Weighted variance of ``values_in`` (biased estimator using total
        weight as the denominator).
    """
    residuals = values_in - weighted_mean(values_in, weights_in)
    num = np.dot(residuals**2, weights_in)
    denom = np.sum(weights_in)
    
    return num / denom

def weighted_std(values_in, weights_in):
    """Compute the weighted standard deviation of an array.

    Parameters
    ----------
    values_in : array_like
        Input data values.
    weights_in : array_like
        Non-negative weights corresponding to each value.

    Returns
    -------
    float
        Square root of the weighted variance; see :func:`weighted_var`.
    """
    return np.sqrt(weighted_var(values_in, weights_in))

def std_norm_z_star(p_val, two_sided=True):
    """Return the standard-normal critical value for a given confidence level.

    Parameters
    ----------
    p_val : float
        Desired confidence level (e.g. ``0.95`` for a 95 % interval).
    two_sided : bool, optional
        If ``True`` (default) the critical value corresponds to a two-sided
        interval.  If ``False``, a one-sided upper-tail critical value is
        returned.

    Returns
    -------
    float
        Critical value ``z*`` such that ``P(Z <= z*) = p`` for the
        appropriate tail probability.
    """
    if two_sided:
        p = 1.0 - 0.5 * (1.0 - p_val)
    else:
        p = p_val

    return (norm.ppf(p))

def std_norm_bands(p_val):
    """Return the lower and upper standard-normal critical values for a confidence band.

    Parameters
    ----------
    p_val : float
        Desired confidence level (e.g. ``0.95`` for a 95 % band).

    Returns
    -------
    z_lb : float
        Lower critical value (negative for symmetric bands).
    z_ub : float
        Upper critical value.
    """
    half_tail_prob = 0.5 * (1.0 - p_val)
    p_lb = half_tail_prob
    p_ub = 1.0 - half_tail_prob

    z_lb = norm.ppf(p_lb)
    z_ub = norm.ppf(p_ub)

    return (z_lb, z_ub)

def draw_norm(Sig):
    """Draw a single multivariate normal sample with covariance ``Sig``.

    Parameters
    ----------
    Sig : numpy.ndarray
        Symmetric positive-definite covariance matrix of shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Random draw of shape ``(n,)`` from ``N(0, Sig)``.
    """
    C = np.linalg.cholesky(Sig)
    return np.dot(C, np.random.randn(Sig.shape[0]))

def draw_norm_multi(Sig, n):
    """Draw multiple multivariate normal samples with covariance ``Sig``.

    Parameters
    ----------
    Sig : numpy.ndarray
        Symmetric positive-definite covariance matrix of shape ``(k, k)``.
    n : int
        Number of independent draws to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, k)`` where each row is an independent draw from
        ``N(0, Sig)``.
    """
    C = np.linalg.cholesky(Sig)
    return np.dot(np.random.randn(n, Sig.shape[0]), C.T)

def my_lognorm(mu, sig):
    """Create a log-normal distribution object parameterised by its log-scale mean and std.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution (i.e. ``E[log X]``).
    sig : float
        Standard deviation of the underlying normal distribution (i.e.
        ``Std[log X]``).

    Returns
    -------
    scipy.stats.rv_frozen
        A frozen ``scipy.stats.lognorm`` distribution object.
    """
    return lognorm(s=sig, scale=np.exp(mu))
