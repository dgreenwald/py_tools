"""Long-run covariance and GMM inference helpers.

Moment arrays have observations in rows. Covariance matrices returned by
``long_run_cov`` and the overlap helpers are for sqrt(T) times the sample
moment mean; ``sandwich`` applies the final 1/T factor.
"""

import numpy as np


def _matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty, finite 2-D array")
    return values


def long_run_cov(u, lags, kernel="hh"):
    """Estimate the long-run covariance of a T-by-m moment series.

    ``hh`` and ``truncated`` include lags 1 through ``lags`` with unit
    weight. ``nw`` uses CP's bandwidth k=``lags`` and weights 1-j/k,
    thus its last nonzero lag is k-1. ``bartlett`` uses maxlags=``lags``
    and weights 1-j/(lags+1), matching statsmodels HAC without corrections.
    All lag products are divided by T, including at nonzero lags.
    """
    u = _matrix(u, "u")
    if not isinstance(lags, (int, np.integer)) or lags < 0 or lags >= len(u):
        raise ValueError("lags must be an integer between 0 and T-1")
    if kernel not in {"hh", "truncated", "nw", "bartlett"}:
        raise ValueError("unknown kernel")

    T = len(u)
    S = u.T @ u / T
    for j in range(1, lags + 1):
        if kernel == "nw":
            weight = 1.0 - j / lags
        elif kernel == "bartlett":
            weight = 1.0 - j / (lags + 1)
        else:
            weight = 1.0
        cross = u[j:].T @ u[:-j] / T
        S += weight * (cross + cross.T)
    return (S + S.T) / 2


def _regression_moments(x, e):
    x = _matrix(x, "x")
    e = np.asarray(e, dtype=float)
    if e.ndim == 1:
        e = e[:, None]
    e = _matrix(e, "e")
    if len(x) != len(e):
        raise ValueError("x and e must have the same number of rows")
    return x, e, (x[:, :, None] * e[:, None, :]).reshape(len(x), -1)


def simplified_hh(x, e, k):
    """CP simplified HH moment covariance for a scalar regression residual.

    Uses mean(e**2) times the Bartlett weighted autocovariance of x with
    bandwidth k. The x columns may include a constant.
    """
    x, e, _ = _regression_moments(x, e)
    if e.shape[1] != 1:
        raise ValueError("simplified_hh requires one residual column")
    return e[:, 0].dot(e[:, 0]) / len(e) * long_run_cov(x, k, "nw")


def no_overlap_cov(x, e, step=12):
    """Moment covariance from nonoverlapping subsamples, averaged by offset.

    Each offset contributes ``step`` times its zero-lag moment covariance,
    so the result is scaled for the full-sample sandwich calculation.
    """
    _, _, moments = _regression_moments(x, e)
    if not isinstance(step, (int, np.integer)) or step < 1 or step > len(moments):
        raise ValueError("step must be an integer between 1 and T")
    pieces = [moments[offset::step] for offset in range(step)]
    return step * sum(part.T @ part / len(part) for part in pieces) / step


def sandwich(d, S, T):
    """Exactly identified GMM parameter covariance, including 1/T."""
    d = _matrix(d, "d")
    S = _matrix(S, "S")
    if d.shape[0] != d.shape[1] or S.shape != (len(d), len(d)):
        raise ValueError("d must be square and S must match its rows")
    if T <= 0:
        raise ValueError("T must be positive")
    inv_d = np.linalg.inv(d)
    return inv_d @ S @ inv_d.T / T


def _symmetric_pinv(V, rank=None):
    V = _matrix(V, "V")
    if V.shape[0] != V.shape[1]:
        raise ValueError("V must be square")
    eigenvalues, eigenvectors = np.linalg.eigh((V + V.T) / 2)
    tolerance = max(V.shape) * np.finfo(float).eps * max(1.0, np.max(np.abs(eigenvalues)))
    positive = np.flatnonzero(eigenvalues > tolerance)
    if rank is not None:
        if not isinstance(rank, (int, np.integer)) or rank < 1 or len(positive) < rank:
            raise ValueError("requested rank exceeds the positive numerical rank")
        positive = positive[-rank:]
    return (eigenvectors[:, positive] / eigenvalues[positive]) @ eigenvectors[:, positive].T


def wald(theta, V, R=None, r=None, pinv_rank=None):
    """Return the Wald quadratic form for R @ theta = r."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    V = _matrix(V, "V")
    if V.shape != (len(theta), len(theta)):
        raise ValueError("V does not match theta")
    R = np.eye(len(theta)) if R is None else np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[1] != len(theta):
        raise ValueError("R does not match theta")
    r = np.zeros(R.shape[0]) if r is None else np.asarray(r, dtype=float).reshape(-1)
    if len(r) != R.shape[0]:
        raise ValueError("r does not match R")
    diff = R @ theta - r
    return float(diff @ _symmetric_pinv(R @ V @ R.T, pinv_rank) @ diff)


def gmm_jt_inefficient(g, S, d, a, T, df):
    """Hansen's inefficient-estimator JT statistic from sample mean moments.

    ``a`` maps moments to estimating equations, and ``d`` is their
    Jacobian with respect to the restricted parameters. ``df`` selects
    the largest positive eigenvalues of the projected covariance.
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    S = _matrix(S, "S")
    d = _matrix(d, "d")
    a = _matrix(a, "a")
    m, p = d.shape
    if S.shape != (m, m) or len(g) != m or a.shape != (p, m):
        raise ValueError("incompatible moment, covariance, d, or a shapes")
    if T <= 0:
        raise ValueError("T must be positive")
    projection = np.eye(m) - d @ np.linalg.solve(a @ d, a)
    C = projection @ S @ projection.T
    return float(T * g @ _symmetric_pinv(C, df) @ g)
