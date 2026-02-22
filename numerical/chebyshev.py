import numpy as np
from py_tools.utilities import cartesian


def grid(n_pts):
    """Return Chebyshev-Gauss-Lobatto nodes on the interval ``[-1, 1]``.

    Generates ``n_pts`` nodes at the extrema of the ``(n_pts - 1)``-th
    Chebyshev polynomial (also known as Chebyshev nodes of the second
    kind or Chebyshev-Gauss-Lobatto nodes) via a sine mapping.

    Parameters
    ----------
    n_pts : int
        Number of grid points.

    Returns
    -------
    numpy.ndarray
        1-D array of ``n_pts`` nodes in ``[-1, 1]``.
    """
    return np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_pts))


def poly(x, n):
    """Evaluate the first *n* Chebyshev polynomials at each point in *x*.

    Each column of the result corresponds to one polynomial evaluated
    across all input points.

    Parameters
    ----------
    x : array-like
        Evaluation points in ``[-1, 1]``.  1-D array of length ``N``.
    n : int
        Number of polynomials to compute (orders 0 through ``n - 1``).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, n)`` where column ``j`` contains the values
        of the *j*-th Chebyshev polynomial at every point in *x*.
    """

    T = np.ones((n, len(x)))
    if n > 1:
        T[1, :] = x

    for jj in range(2, n):
        T[jj, :] = 2.0 * x * T[jj - 1, :] - T[jj - 2, :]

    return T.T


def basis_and_gradient(x, n):
    """Compute Chebyshev basis values and their derivatives at *x*.

    Parameters
    ----------
    x : array-like
        Evaluation points in ``[-1, 1]``.  1-D array of length ``N``.
    n : int
        Number of Chebyshev polynomials (orders 0 through ``n - 1``).

    Returns
    -------
    T : numpy.ndarray
        Basis matrix of shape ``(N, n)`` (Chebyshev polynomial values).
    dT : numpy.ndarray
        Derivative matrix of shape ``(N, n)`` (derivatives of each
        polynomial with respect to *x*).
    """
    T = poly(x, n).T
    dT = np.zeros((n, len(x)))

    if n > 1:
        dT[1, :] = 1.0

    for jj in range(2, n):
        dT[jj, :] = 2.0 * T[jj - 1, :] + 2.0 * x * dT[jj - 1, :] - dT[jj - 2, :]

    return T.T, dT.T


def gradient(x, n):
    """Compute the derivatives of the first *n* Chebyshev polynomials at *x*.

    Parameters
    ----------
    x : array-like
        Evaluation points in ``[-1, 1]``.  1-D array of length ``N``.
    n : int
        Number of Chebyshev polynomials (orders 0 through ``n - 1``).

    Returns
    -------
    numpy.ndarray
        Derivative matrix of shape ``(N, n)``.
    """
    _, dT = basis_and_gradient(x, n)
    return dT


def tensor(X, n_vec):
    """Build a tensor-product Chebyshev basis matrix.

    Each row of *X* is one observation with *k* variables.  The
    resulting basis contains all tensor products of the univariate
    Chebyshev polynomials specified by *n_vec*.

    Parameters
    ----------
    X : numpy.ndarray
        Observation matrix of shape ``(N, k)``, or 1-D of length ``k``
        for a single observation.  Each column corresponds to one
        variable and should lie in ``[-1, 1]``.
    n_vec : sequence of int
        Number of Chebyshev polynomials to use for each variable.
        Length must equal ``k``.

    Returns
    -------
    numpy.ndarray
        Tensor-product basis matrix of shape
        ``(N, prod(n_vec))``.
    """

    if len(X.shape) == 1:
        X = X[np.newaxis, :]

    k = X.shape[1]
    TX = poly(X[:, -1], n_vec[-1])
    return recurse_tensor(X, TX, n_vec, k - 2)


def recurse_tensor(X, TX, n_vec, level):
    """Recursively construct a tensor-product Chebyshev basis.

    Called internally by :func:`tensor`.  At each recursion level the
    existing partial basis *TX* is expanded by the Chebyshev polynomials
    for the variable at *level*.

    Parameters
    ----------
    X : numpy.ndarray
        Observation matrix of shape ``(N, k)``.
    TX : numpy.ndarray
        Partial tensor-product basis accumulated so far.
    n_vec : sequence of int
        Number of Chebyshev polynomials for each variable.
    level : int
        Current variable index being processed.  Recursion stops when
        *level* reaches ``-1``.

    Returns
    -------
    numpy.ndarray
        Completed (or partially completed) tensor-product basis matrix.
    """

    if level >= 0:
        this_basis = poly(X[:, level], n_vec[level])
        TX_new = np.hstack(
            [this_basis[:, jj][:, np.newaxis] * TX for jj in range(n_vec[level])]
        )
        return recurse_tensor(X, TX_new, n_vec, level - 1)
    else:
        return TX


class ChebFcn:
    """Chebyshev polynomial approximation for a univariate function.

    Fits a degree-``n`` Chebyshev approximation to a function or a set
    of values on the interval ``[lb, ub]`` and provides methods for
    evaluating the approximation and its derivative.

    Parameters
    ----------
    n : int
        Number of Chebyshev basis functions (polynomial order + 1).
    lb : float, optional
        Left endpoint of the approximation domain, by default ``-1.0``.
    ub : float, optional
        Right endpoint of the approximation domain, by default ``1.0``.

    Attributes
    ----------
    n : int
        Number of basis functions.
    grid : numpy.ndarray
        Chebyshev nodes on ``[-1, 1]``.
    basis : numpy.ndarray
        Chebyshev basis matrix evaluated at ``grid``.
    lb : float
        Left domain endpoint.
    ub : float
        Right domain endpoint.
    scaled_grid : numpy.ndarray
        Chebyshev nodes scaled to ``[lb, ub]``.
    coeffs : numpy.ndarray
        Chebyshev coefficients, set after calling :meth:`fit_vals` or
        :meth:`fit_fcn`.
    """

    def __init__(self, n, lb=-1.0, ub=1.0):

        self.n = n
        self.grid = grid(n)
        self.basis = poly(self.grid, self.n)

        self.lb = lb
        self.ub = ub

        self.a_to_grid = -(self.ub + self.lb) / (self.ub - self.lb)
        self.b_to_grid = 2.0 / (self.ub - self.lb)

        self.a_from_grid = 0.5 * (self.ub + self.lb)
        self.b_from_grid = (self.ub - self.lb) / 2.0

        self.scaled_grid = self.scale_from_grid(self.grid)

    def make_grid(self):
        """Return a copy of the Chebyshev grid nodes on ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            1-D array of ``n`` Chebyshev nodes.
        """
        return self.grid.copy()

    def fit_vals(self, vals):
        """Fit the Chebyshev approximation to function values at the grid.

        Solves for the coefficient vector ``coeffs`` such that the
        Chebyshev expansion matches *vals* at the grid nodes.

        Parameters
        ----------
        vals : numpy.ndarray
            Function values at the Chebyshev grid nodes.  Length must
            equal ``n``.
        """
        self.coeffs = np.linalg.solve(self.basis, vals)

    def fit_fcn(self, fcn):
        """Fit the Chebyshev approximation by evaluating a function on the grid.

        Parameters
        ----------
        fcn : callable
            Function to approximate.  Must accept a 1-D array of points
            in ``[lb, ub]`` and return a 1-D array of the same length.
        """
        vals = fcn(self.scaled_grid)
        self.fit_vals(vals)

    def get_basis(self, x_in):
        """Return the Chebyshev basis matrix evaluated at *x_in*.

        Input values are clipped to ``[lb, ub]`` before scaling.

        Parameters
        ----------
        x_in : numpy.ndarray
            Evaluation points.  1-D array of length ``N``.

        Returns
        -------
        numpy.ndarray
            Basis matrix of shape ``(N, n)``.
        """
        x_scaled = self.scale_x(x_in)
        return poly(x_scaled, self.n)

    def get_basis_and_gradient(self, x_in):
        """Return the Chebyshev basis and its derivative at *x_in*.

        Input values are clipped to ``[lb, ub]`` before scaling.

        Parameters
        ----------
        x_in : numpy.ndarray
            Evaluation points.  1-D array of length ``N``.

        Returns
        -------
        T : numpy.ndarray
            Basis matrix of shape ``(N, n)``.
        dT : numpy.ndarray
            Derivative matrix of shape ``(N, n)``.
        """
        x_scaled = self.scale_x(x_in)
        return basis_and_gradient(x_scaled, self.n)

    def get_basis_gradient_only(self, x_in):
        """Return only the derivative of the Chebyshev basis at *x_in*.

        Input values are clipped to ``[lb, ub]`` before scaling.

        Parameters
        ----------
        x_in : numpy.ndarray
            Evaluation points.  1-D array of length ``N``.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape ``(N, n)``.
        """
        x_scaled = self.scale_x(x_in)
        return gradient(x_scaled, self.n)

    def evaluate(self, x_in):
        """Evaluate the fitted Chebyshev approximation at *x_in*.

        Parameters
        ----------
        x_in : numpy.ndarray
            Evaluation points.  1-D array of length ``N``.

        Returns
        -------
        numpy.ndarray
            Approximated function values of shape ``(N,)``.
        """
        Tx = self.get_basis(x_in)
        return Tx @ self.coeffs

    def gradient(self, x_in):
        """Evaluate the derivative of the fitted approximation at *x_in*.

        Parameters
        ----------
        x_in : numpy.ndarray
            Evaluation points.  1-D array of length ``N``.

        Returns
        -------
        numpy.ndarray
            Derivative values of shape ``(N,)``.
        """
        _, dTx = self.get_basis_and_gradient(x_in)
        return np.dot(dTx, self.coeffs)

    def evaluate_with_gradient(self, x_in):
        """Evaluate the approximation and its derivative simultaneously.

        Parameters
        ----------
        x_in : numpy.ndarray
            Evaluation points.  1-D array of length ``N``.

        Returns
        -------
        fx : numpy.ndarray
            Approximated function values of shape ``(N,)``.
        dfx : numpy.ndarray
            Derivative values of shape ``(N,)``.
        """
        Tx, dTx = self.get_basis_and_gradient(x_in)
        fx = Tx @ self.coeffs
        dfx = dTx @ self.coeffs
        return fx, dfx

    def scale_x(self, x_in):
        """Clip *x_in* to ``[lb, ub]`` and scale to the grid interval ``[-1, 1]``.

        Parameters
        ----------
        x_in : numpy.ndarray
            Input values.

        Returns
        -------
        numpy.ndarray
            Scaled values in ``[-1, 1]``.
        """
        x = x_in.copy()
        x = np.minimum(x, self.ub)
        x = np.maximum(x, self.lb)
        return self.scale_to_grid(x)

    def scale_to_grid(self, vals):
        """Scale values from ``[lb, ub]`` to the grid interval ``[-1, 1]``.

        Parameters
        ----------
        vals : array-like
            Values in ``[lb, ub]``.

        Returns
        -------
        numpy.ndarray
            Scaled values in ``[-1, 1]``.
        """
        return self.a_to_grid + (self.b_to_grid * vals)

    def scale_from_grid(self, vals):
        """Scale values from the grid interval ``[-1, 1]`` to ``[lb, ub]``.

        Parameters
        ----------
        vals : array-like
            Values in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            Scaled values in ``[lb, ub]``.
        """
        return self.a_from_grid + (self.b_from_grid * vals)


class TensorChebFcn:
    """Tensor-product Chebyshev approximation for a multivariate function.

    Builds a tensor-product Chebyshev basis over a *k*-dimensional
    hypercube ``[lb, ub]^k`` and fits a linear combination of basis
    functions to supplied values or a callable.

    Parameters
    ----------
    n_vec : sequence of int
        Number of Chebyshev basis functions for each dimension.
        Length determines the number of dimensions ``k``.
    lb : float, optional
        Common lower bound for all dimensions, by default ``-1.0``.
    ub : float, optional
        Common upper bound for all dimensions, by default ``1.0``.

    Attributes
    ----------
    k : int
        Number of dimensions.
    n_vec : sequence of int
        Polynomial counts per dimension.
    grid : numpy.ndarray
        Tensor-product Chebyshev grid of shape
        ``(prod(n_vec), k)`` on ``[-1, 1]^k``.
    basis : numpy.ndarray
        Tensor-product basis matrix evaluated at ``grid``.
    lb : float
        Lower domain bound.
    ub : float
        Upper domain bound.
    scaled_grid : numpy.ndarray
        Chebyshev grid scaled to ``[lb, ub]^k``.
    coeffs : numpy.ndarray
        Fitted coefficients, set after calling :meth:`fit_vals` or
        :meth:`fit_fcn`.
    """

    def __init__(self, n_vec, lb=-1.0, ub=1.0):

        self.k = len(n_vec)
        self.n_vec = n_vec
        self.grid = cartesian((grid(n_vec[ii]) for ii in range(self.k)))
        self.basis = tensor(self.grid, self.n_vec)

        self.lb = lb
        self.ub = ub

        self.a_to_grid = -(self.ub + self.lb) / (self.ub - self.lb)
        self.b_to_grid = 2.0 / (self.ub - self.lb)

        self.a_from_grid = 0.5 * (self.ub + self.lb)
        self.b_from_grid = (self.ub - self.lb) / 2.0

        self.scaled_grid = self.scale_from_grid(self.grid)

    def fit_vals(self, vals):
        """Fit the tensor Chebyshev approximation to values at the grid.

        Parameters
        ----------
        vals : numpy.ndarray
            Function values at the tensor-product grid nodes.  Length
            must equal ``prod(n_vec)``.
        """
        self.coeffs = np.linalg.solve(self.basis, vals)

    def fit_fcn(self, fcn):
        """Fit the tensor Chebyshev approximation by evaluating a function.

        Parameters
        ----------
        fcn : callable
            Function to approximate.  Must accept a 2-D array of shape
            ``(prod(n_vec), k)`` and return a 1-D array of length
            ``prod(n_vec)``.
        """
        vals = fcn(self.scaled_grid)
        self.fit_vals(vals)

    def evaluate(self, x, chunksize=100000):
        """Evaluate the fitted tensor approximation at *x*.

        Large inputs are processed in chunks of size *chunksize* to
        limit peak memory usage.

        Parameters
        ----------
        x : numpy.ndarray
            Evaluation points of shape ``(N, k)``.  Values are clipped
            to ``[lb, ub]`` before evaluation.
        chunksize : int, optional
            Maximum number of points evaluated in one pass, by default
            ``100000``.

        Returns
        -------
        numpy.ndarray
            Approximated function values of shape ``(N,)``.
        """
        x = np.minimum(x, self.ub)
        x = np.maximum(x, self.lb)
        x_grid = self.scale_to_grid(x)

        Nx = x_grid.shape[0]
        if Nx > chunksize:
            vals = np.zeros(Nx)
            count = 0
            while count < Nx:
                vals[count : count + chunksize] = eval_tensor(
                    x_grid[count : count + chunksize, :], self.n_vec, self.coeffs
                )
                count += chunksize
        else:
            # Tx = tensor(x_grid, self.n_vec)
            # vals= np.dot(Tx, self.coeffs)
            vals = eval_tensor(x_grid, self.n_vec, self.coeffs)

        return vals

    def scale_to_grid(self, vals):
        """Scale values from ``[lb, ub]`` to the grid interval ``[-1, 1]``.

        Parameters
        ----------
        vals : array-like
            Values in ``[lb, ub]``.

        Returns
        -------
        numpy.ndarray
            Scaled values in ``[-1, 1]``.
        """
        return self.a_to_grid + (self.b_to_grid * vals)

    def scale_from_grid(self, vals):
        """Scale values from the grid interval ``[-1, 1]`` to ``[lb, ub]``.

        Parameters
        ----------
        vals : array-like
            Values in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            Scaled values in ``[lb, ub]``.
        """
        return self.a_from_grid + (self.b_from_grid * vals)


def eval_tensor(x, n_vec, coeffs):
    """Evaluate a fitted tensor-product Chebyshev approximation at *x*.

    Parameters
    ----------
    x : numpy.ndarray
        Evaluation points of shape ``(N, k)`` in ``[-1, 1]^k``.
    n_vec : sequence of int
        Number of Chebyshev basis functions per dimension.
    coeffs : numpy.ndarray
        Coefficient vector of length ``prod(n_vec)``.

    Returns
    -------
    numpy.ndarray
        Approximated function values of shape ``(N,)``.
    """
    Tx = tensor(x, n_vec)
    return np.dot(Tx, coeffs)
