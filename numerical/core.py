import numpy as np
import itertools

def quad_form(A, X):
    """Compute the quadratic form ``A' X A``.

    Parameters
    ----------
    A : numpy.ndarray
        Array of shape ``(n, m)``.
    X : numpy.ndarray
        Square array of shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(m, m)`` equal to ``A.T @ X @ A``.
    """
    return np.dot(A.T, np.dot(X, A))

def rsolve(b, A):
    """Solve the right-hand linear system ``b A^{-1}``.

    Equivalent to solving ``A' x' = b'`` and transposing the result,
    i.e. computing ``b @ inv(A)``.

    Parameters
    ----------
    b : numpy.ndarray
        Right-hand side array of shape ``(m, n)``.
    A : numpy.ndarray
        Square coefficient matrix of shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Solution array of shape ``(m, n)``.
    """
    return np.linalg.solve(A.T, b.T).T

def gradient(f, x, args=None, kwargs=None, step=1e-5, two_sided=True, f_val=None):
    """Estimate the Jacobian of *f* at *x* by finite differences.

    Parameters
    ----------
    f : callable
        Function to differentiate.  Must accept ``f(x, *args, **kwargs)``
        and return a scalar or 1-D array.
    x : numpy.ndarray
        Point at which the Jacobian is evaluated.  1-D array of length
        ``n``.
    args : tuple, optional
        Positional arguments passed to *f*, by default ``()``.
    kwargs : dict, optional
        Keyword arguments passed to *f*, by default ``{}``.
    step : float, optional
        Finite-difference step size, by default ``1e-5``.
    two_sided : bool, optional
        If ``True`` (default), use a central-difference approximation;
        otherwise use a forward-difference approximation.
    f_val : numpy.ndarray or None, optional
        Pre-computed value ``f(x, *args, **kwargs)``.  Only used when
        *two_sided* is ``False``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, m)`` where ``grad[i, j]`` is the partial
        derivative of the *j*-th output of *f* with respect to
        ``x[i]``.  This is the *transpose* of the conventional
        Jacobian convention (which has shape ``(m, n)``).
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    x = x.copy()

    if (not two_sided) and (f_val is None):
        f_val = f(x, *args, **kwargs)

    grad = None
    for ii in range(len(x)):

        x[ii] += step
        f_hi = f(x, *args, **kwargs)

        if two_sided:

            x[ii] -= (2.0 * step)
            f_lo = f(x, *args, **kwargs)
            x[ii] += step

            df_i = np.array(f_hi - f_lo) / (2.0 * step)

        else:

            x[ii] -= step

            df_i = np.array(f_hi - f_val) / step

        if grad is None:

            if df_i.shape == ():
                ncols = 1
            else:
                ncols = len(df_i)

            grad = np.zeros((len(x), ncols))

        grad[ii, :] = df_i

    return grad

def hessian(f, x_in, eps=1e-4):
    """Estimate the Hessian of *f* at *x_in* by finite differences.

    Uses a second-order central-difference formula applied element-wise
    to each pair of coordinates.

    Parameters
    ----------
    f : callable
        Scalar-valued function accepting a 1-D array.
    x_in : numpy.ndarray
        Point at which the Hessian is evaluated.  1-D array of length
        ``n``.
    eps : float, optional
        Step size for finite differences, by default ``1e-4``.

    Returns
    -------
    numpy.ndarray
        Symmetric Hessian matrix of shape ``(n, n)``.
    """
    x = x_in.copy()
    n = len(x)
    H = np.zeros((n, n))
    for ii, jj in itertools.product(range(n), repeat=2):

        if ii <= jj:

            x[ii] += eps
            x[jj] += eps
            H[ii, jj] += f(x)

            x[jj] -= 2.0 * eps
            if ii != jj:
                H[ii, jj] -= f(x)
            else:
                H[ii, jj] -= 2.0 * f(x)

            x[ii] -= 2.0 * eps
            H[ii, jj] += f(x)

            x[jj] += 2.0 * eps
            if ii != jj:
                H[ii, jj] -= f(x)

            x[ii] += eps
            x[jj] -= eps

        else:

            H[ii, jj] = H[jj, ii]

    return H / (4.0 * (eps ** 2))

def svd_inv(A, sv_tol=1e-8, **kwargs):
    """Compute a pseudo-inverse of *A* using singular value decomposition.

    Singular values smaller than *sv_tol* in absolute value are treated
    as zero and excluded from the inversion.

    Parameters
    ----------
    A : numpy.ndarray
        Square (or rectangular) matrix to invert.
    sv_tol : float, optional
        Threshold below which singular values are considered zero, by
        default ``1e-8``.
    **kwargs
        Currently unused; reserved for future extensions.

    Returns
    -------
    numpy.ndarray
        Pseudo-inverse of *A* with the same shape as ``A.T``.
    """
    u, s, vh = np.linalg.svd(A)
    s_inv = np.zeros(s.shape)

    ix = np.abs(s) > sv_tol
    s_inv[ix] = 1.0 / s[ix]

    S_inv = np.diag(s_inv)
    A_inv = np.dot(vh.T, np.dot(S_inv, u.T))
    return A_inv

def ghquad_norm(degree, mu=0.0, sig=1.0):
    """Return nodes and weights for Gauss-Hermite quadrature over a normal distribution.

    Scales the standard Gauss-Hermite rule so that the weights sum to one
    and the nodes match the ``N(mu, sig^2)`` distribution.

    Parameters
    ----------
    degree : int
        Number of quadrature nodes (polynomial degree).
    mu : float, optional
        Mean of the normal distribution, by default ``0.0``.
    sig : float, optional
        Standard deviation of the normal distribution, by default ``1.0``.

    Returns
    -------
    x : numpy.ndarray
        Quadrature nodes of shape ``(degree,)``.
    w : numpy.ndarray
        Quadrature weights of shape ``(degree,)`` summing to one.
    """
    x, w = np.polynomial.hermite.hermgauss(degree)

    # Re-scale weights (currently add to pi^{1/2})
    w /= np.sum(w)

    # Re-scale nodes to adjust for mu and sigma
    x *= np.sqrt(2.0) * sig
    x += mu

    return x, w

def gauss_legendre_norm(degree, a=-1.0, b=1.0):
    """Return nodes and weights for Gauss-Legendre quadrature on ``[a, b]``.

    Rescales the standard Gauss-Legendre rule from ``[-1, 1]`` to the
    interval ``[a, b]``.

    Parameters
    ----------
    degree : int
        Number of quadrature nodes.
    a : float, optional
        Left endpoint of the integration interval, by default ``-1.0``.
    b : float, optional
        Right endpoint of the integration interval, by default ``1.0``.

    Returns
    -------
    x : numpy.ndarray
        Quadrature nodes of shape ``(degree,)`` on ``[a, b]``.
    w : numpy.ndarray
        Quadrature weights of shape ``(degree,)``.
    """
    x, w = np.polynomial.legendre.leggauss(degree)

    # Re-scale weights
    w *= 0.5 * (b - a)

    # Re-scale nodes
    x = 0.5 * ((x + 1.0) * (b - a)) + a

    return x, w

def logit(x, lb=0.0, ub=1.0):
    """Apply the logit (log-odds) transform to map ``(lb, ub)`` to the real line.

    Parameters
    ----------
    x : array-like
        Values to transform.  Must satisfy ``lb < x < ub``.
    lb : float, optional
        Lower bound of the bounded interval, by default ``0.0``.
    ub : float, optional
        Upper bound of the bounded interval, by default ``1.0``.

    Returns
    -------
    numpy.ndarray
        Transformed values in ``(-inf, +inf)``.
    """
    return np.log(x - lb) - np.log(ub - x)

def logistic(x, lb=0.0, ub=1.0):
    """Apply the logistic transform to map the real line to ``(lb, ub)``.

    This is the inverse of :func:`logit`.

    Parameters
    ----------
    x : array-like
        Values to transform.
    lb : float, optional
        Lower bound of the output interval, by default ``0.0``.
    ub : float, optional
        Upper bound of the output interval, by default ``1.0``.

    Returns
    -------
    numpy.ndarray
        Transformed values in ``(lb, ub)``.
    """
    return lb + (ub - lb) / (1.0 + np.exp(-x))

def bound_transform(vals, lb, ub, to_bdd=True):
    """Transform values between unbounded and bounded representations.

    Applies element-wise transformations based on which bounds are
    finite:

    * Both bounds finite: logistic / logit transform.
    * Lower bound only: exponential / log shift from *lb*.
    * Upper bound only: negative exponential / log shift from *ub*.

    Parameters
    ----------
    vals : numpy.ndarray
        Values to transform.
    lb : numpy.ndarray
        Element-wise lower bounds.  Use ``-numpy.inf`` for no lower
        bound.
    ub : numpy.ndarray
        Element-wise upper bounds.  Use ``numpy.inf`` for no upper
        bound.
    to_bdd : bool, optional
        If ``True`` (default), map unbounded values to the bounded
        domain.  If ``False``, map bounded values to the unbounded
        domain.

    Returns
    -------
    numpy.ndarray
        Transformed values with the same shape as *vals*.
    """
    trans_vals = vals.copy()

    # Indices
    ix_lb = lb > -np.inf
    ix_ub = ub < np.inf

    ix_both = ix_lb & ix_ub
    ix_lb_only = ix_lb & (~ix_ub)
    ix_ub_only = (~ix_lb) & ix_ub

    if to_bdd:
        trans_vals[ix_both] = logistic(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = lb[ix_lb_only] + np.exp(vals[ix_lb_only])
        trans_vals[ix_ub_only] = ub[ix_ub_only] - np.exp(vals[ix_ub_only])
    else:
        trans_vals[ix_both] = logit(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = np.log(vals[ix_lb_only] - lb[ix_lb_only])
        trans_vals[ix_ub_only] = np.log(ub[ix_ub_only] - vals[ix_ub_only])

    return trans_vals

def robust_cholesky(A, min_eig=1e-12):
    """Compute a Cholesky-like factor of *A*, clipping small eigenvalues.

    Decomposes *A* via eigen-decomposition and clips eigenvalues to
    ``min_eig`` before constructing the square-root factor, so the
    result is well-defined even when *A* is only positive semi-definite.

    Parameters
    ----------
    A : numpy.ndarray
        Square symmetric matrix of shape ``(n, n)``.
    min_eig : float, optional
        Minimum eigenvalue threshold, by default ``1e-12``.

    Returns
    -------
    numpy.ndarray
        Matrix ``L`` of shape ``(n, n)`` such that ``L @ L.T`` is a
        positive semi-definite approximation to *A*.
    """
    vals, vecs = np.linalg.eig(A)
    vals = np.maximum(vals, min_eig)
    Dhalf = np.diag(np.sqrt(vals))
    return vecs @ Dhalf

def my_chol(A_in):
    """Compute the lower-triangular Cholesky factor of *A_in*.

    A pure-Python implementation of the Cholesky–Banachiewicz algorithm.
    The input matrix must be symmetric and positive definite.

    Parameters
    ----------
    A_in : numpy.ndarray
        Symmetric positive-definite matrix of shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Lower-triangular matrix ``L`` of shape ``(n, n)`` such that
        ``L @ L.T == A_in``.
    """
    A = A_in.copy()

    N = A.shape[0]
    L = np.zeros((N, N))

    for jj in range(N):
        Dj = np.sqrt(A[jj, jj])
        aj = A[jj+1:, jj]
        Aj = aj[:, np.newaxis]
        L[jj, jj] = Dj
        L[jj+1:, jj] = aj / Dj
        if jj < N - 1:
            A[jj+1:, jj+1:] -= (Aj @ Aj.T) / (Dj**2)

    return L
