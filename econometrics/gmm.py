import numpy as np
import scipy


def compute_g(params, h, data):
    """
    Compute the sample average of moment conditions.

    Parameters
    ----------
    params : array_like
        Parameter vector passed to ``h``.
    h : callable
        Moment function ``h(data_t, params)`` evaluated at a single time-period
        slice ``data[:, t]``.  Returns a 1-D array of length ``Nmom``.
    data : ndarray of shape (Nvar, T)
        Data matrix; columns index time periods.

    Returns
    -------
    g : ndarray of shape (Nmom,)
        Sample average ``(1/T) * sum_{t=1}^{T} h(data[:, t], params)``.
    """
    Nt = data.shape[1]

    g = h(data[:, 0], params)
    for tt in range(1, Nt):
        g += h(data[:, tt], params)
    g /= Nt

    return g


def compute_dg(params, dh, data):
    """
    Compute the sample average of the Jacobian of moment conditions.

    Parameters
    ----------
    params : array_like
        Parameter vector passed to ``dh``.
    dh : callable
        Jacobian function ``dh(data_t, params)`` returning a 2-D array of
        shape ``(Nmom, Nparams)`` for a single time-period slice.
    data : ndarray of shape (Nvar, T)
        Data matrix; columns index time periods.

    Returns
    -------
    dg : ndarray of shape (Nmom, Nparams)
        Sample average of the Jacobian:
        ``(1/T) * sum_{t=1}^{T} dh(data[:, t], params)``.
    """
    Nt = data.shape[1]

    dg = dh(data[:, 0], params)
    for tt in range(1, Nt):
        dg += dh(data[:, tt], params)
    dg /= Nt

    return dg


def compute_d2g(params, d2h, data):
    """
    Compute the sample average of the second derivatives of moment conditions.

    Parameters
    ----------
    params : array_like
        Parameter vector passed to each element of ``d2h``.
    d2h : list of callable
        List of length ``Nmom``.  Each element ``d2h[i](data_t, params)``
        returns a 2-D array of shape ``(Nparams, Nparams)`` — the Hessian of
        the ``i``-th moment condition with respect to ``params``.
    data : ndarray of shape (Nvar, T)
        Data matrix; columns index time periods.

    Returns
    -------
    d2g : list of ndarray
        List of length ``Nmom`` where each element is an ``(Nparams, Nparams)``
        array containing the sample-average Hessian of the corresponding
        moment condition.
    """
    Nt = data.shape[1]

    d2g = []
    for d2h_i in d2h:
        d2g_i = d2h_i(data[:, 0], params)
        for tt in range(1, Nt):
            d2g_i += d2h_i(data[:, tt], params)
        d2g_i /= Nt
        d2g.append(d2g_i)

    return d2g


def obj_gmm(params, h, data, W, dh=None):
    """
    GMM objective function.

    Parameters
    ----------
    params : array_like
        Parameter vector.
    h : callable
        Moment function; see :func:`compute_g`.
    data : ndarray of shape (Nvar, T)
        Data matrix.
    W : ndarray of shape (Nmom, Nmom)
        Weighting matrix.
    dh : callable, optional
        Jacobian of ``h``; accepted for interface compatibility but not used
        inside this function.

    Returns
    -------
    float
        ``0.5 * g(params)' @ W @ g(params)``.
    """
    g = compute_g(params, h, data)
    return 0.5 * np.dot(g.T, np.dot(W, g))


def jac_gmm(params, h, data, W, dh=None):
    """
    Gradient of the GMM objective function with respect to params.

    Parameters
    ----------
    params : array_like
        Parameter vector.
    h : callable
        Moment function; see :func:`compute_g`.
    data : ndarray of shape (Nvar, T)
        Data matrix.
    W : ndarray of shape (Nmom, Nmom)
        Weighting matrix.
    dh : callable, optional
        Jacobian of ``h``; see :func:`compute_dg`.  Required for correct
        output; if ``None``, behaviour is undefined.

    Returns
    -------
    ndarray of shape (Nparams,)
        Gradient ``dg' @ W @ g`` evaluated at ``params``.
    """
    g = compute_g(params, h, data)
    dg = compute_dg(params, dh, data)

    return np.dot(dg.T, np.dot(W, g))


def hess_gmm(params, h, dh, d2h, data, W):
    """
    Hessian of the GMM objective function with respect to params.

    Parameters
    ----------
    params : array_like
        Parameter vector.
    h : callable
        Moment function; see :func:`compute_g`.
    dh : callable
        Jacobian of ``h``; see :func:`compute_dg`.
    d2h : list of callable
        Second-derivative functions; see :func:`compute_d2g`.
    data : ndarray of shape (Nvar, T)
        Data matrix.
    W : ndarray of shape (Nmom, Nmom)
        Weighting matrix.

    Returns
    -------
    hess : ndarray of shape (Nparams, Nparams)
        Hessian ``dg' @ W @ dg + sum_i (W @ g)_i * d2g_i`` evaluated at
        ``params``.
    """
    g = compute_g(params, h, data)
    dg = compute_dg(params, dh, data)
    d2g = compute_d2g(params, d2h, data)

    Wg = np.dot(W, g)
    hess = np.dot(dg.T, np.dot(W, dg))
    for ii, d2g_i in enumerate(d2g):
        hess += Wg[ii] * d2g_i

    return hess


def solve_gmm(params_guess, h, data, W=None, dh=None, d2h=None, algorithm=None):
    """
    Estimate a GMM model by minimising the GMM objective function.

    Parameters
    ----------
    params_guess : array_like
        Initial parameter guess.
    h : callable
        Moment function ``h(data_t, params)`` returning an ``(Nmom,)`` array.
    data : ndarray of shape (Nvar, T)
        Data matrix; columns index time periods.
    W : ndarray of shape (Nmom, Nmom), optional
        Weighting matrix.  Defaults to the identity matrix.
    dh : callable, optional
        Jacobian of ``h``; see :func:`compute_dg`.  If provided, the analytic
        gradient is used during optimisation.
    d2h : list of callable, optional
        Second-derivative functions; see :func:`compute_d2g`.  If both ``dh``
        and ``d2h`` are provided, the analytic Hessian is used and the default
        algorithm becomes ``'dogleg'``; otherwise ``'BFGS'`` is used.
    algorithm : str, optional
        Optimisation algorithm forwarded to ``scipy.optimize.minimize``.
        Overrides the automatic selection described above.

    Returns
    -------
    res : OptimizeResult
        Result object from ``scipy.optimize.minimize``.
    """
    # Get number of moments
    Nmom = len(h(data[:, 1], params_guess))

    if W is None:
        W = np.eye(Nmom)

    if dh is not None:
        jac = jac_gmm
    else:
        jac = False

    if d2h is not None:
        hess = hess_gmm
    else:
        hess = None

    if algorithm is None:
        if (dh is not None) and (d2h is not None):
            algorithm = "dogleg"
        else:
            algorithm = "BFGS"

    return scipy.optimize.minimize(
        obj_gmm,
        params_guess,
        args=(h, data, W, dh),
        method=algorithm,
        jac=jac,
        hess=hess,
    )
