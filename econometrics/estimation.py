import numpy as np
from scipy.optimize import minimize

from py_tools.numerical import gradient


def objfcn(b, err_fcn, *args):
    """
    NLS objective function (RMSE).

    Parameters
    ----------
    b : array_like
        Parameter vector.
    err_fcn : callable
        Function ``err_fcn(b, *args)`` returning a 1-D array of residuals.
    *args
        Additional arguments forwarded to ``err_fcn``.

    Returns
    -------
    float
        Root-mean-square of the residuals returned by ``err_fcn``.
    """
    return np.sqrt(np.mean(err_fcn(b, *args) ** 2))


def se_nls(err_fcn, b, args=()):
    """
    Compute heteroskedasticity-robust standard errors for NLS estimates.

    Uses the sandwich (HC0) formula: ``V = (G'G)^{-1} (4/T * sum e_t^2 g_t
    g_t') (G'G)^{-1}``, where ``G`` is the Jacobian of the residuals and
    ``g_t`` is the gradient at observation ``t``.

    Parameters
    ----------
    err_fcn : callable
        Function ``err_fcn(b, *args)`` returning a 1-D array of residuals of
        length ``T``.
    b : array_like of shape (Nb,)
        Parameter vector at which to evaluate the standard errors.
    args : tuple, optional
        Additional arguments forwarded to ``err_fcn``.

    Returns
    -------
    se : ndarray of shape (Nb,)
        Heteroskedasticity-robust standard errors.
    V : ndarray of shape (Nb, Nb)
        Robust covariance matrix scaled by ``1/T``.
    e : ndarray of shape (T,)
        Residuals evaluated at ``b``.
    """
    e = err_fcn(b, *args)
    grad = gradient(err_fcn, b, args=args)

    Nb = len(b)
    outer_vals = np.zeros((Nb, Nb))
    inner_vals = np.zeros((Nb, Nb))

    Nt = len(e)
    for tt in range(Nt):
        grad_t = grad[:, tt][:, np.newaxis]
        grad_t2 = np.dot(grad_t, grad_t.T)
        outer_vals += grad_t2
        inner_vals += (e[tt] ** 2) * grad_t2

    inner_vals *= 4.0 / Nt
    outer_vals /= Nt

    outer_vals_inv = np.linalg.inv(outer_vals)
    V = np.dot(outer_vals_inv, np.dot(inner_vals, outer_vals_inv))
    se = np.sqrt(np.diagonal(V) / Nt)

    return se, V, e


def nls(err_fcn, b0, args=(), **kwargs):
    """
    Estimate a nonlinear least squares (NLS) model.

    Minimises the RMSE of ``err_fcn`` via ``scipy.optimize.minimize`` and
    returns estimates together with heteroskedasticity-robust standard errors.

    Parameters
    ----------
    err_fcn : callable
        Function ``err_fcn(b, *args)`` returning a 1-D array of residuals.
    b0 : array_like
        Initial parameter guess.
    args : tuple, optional
        Extra arguments passed to ``err_fcn``.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

    Returns
    -------
    output : dict
        Dictionary with the following keys:

        ``b_hat`` : ndarray
            Estimated parameter vector.
        ``e_hat`` : ndarray
            Residuals evaluated at ``b_hat``.
        ``V`` : ndarray
            Robust covariance matrix of ``b_hat``.
        ``se`` : ndarray
            Robust standard errors of ``b_hat``.
        ``res`` : OptimizeResult
            Full result object from ``scipy.optimize.minimize``.
    """
    all_args = (err_fcn,) + args
    res = minimize(objfcn, b0, args=all_args, **kwargs)

    # Compute standard errors
    b_hat = res.x
    se, V, e_hat = se_nls(err_fcn, b_hat, args=args)

    output = {
        "b_hat": b_hat,
        "e_hat": e_hat,
        "V": V,
        "se": se,
        "res": res,
    }

    return output
