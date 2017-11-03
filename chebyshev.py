import numpy as np

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def grid(n_pts):

    return np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_pts))

def poly(x, n):
    """Note: each column is one polynomial"""

    T = np.ones((n, len(x)))
    if n > 1:
        T[1, :] = x

    for jj in range(2, n):
        T[jj, :] = 2.0 * x * T[jj-1, :] - T[jj-2, :]

    return T.T

def tensor(X, n_vec):
    """Each row of X should be one observation
    n_vec should contain the levels for each variable"""
    
    k = X.shape[1]
    TX = poly(X[:, -1], n_vec[-1])
    return recurse_tensor(X, TX, n_vec, k - 2)
    
def recurse_tensor(X, TX, n_vec, level):
    """Recursively construct tensor"""
    
    if level >= 0:
        this_basis = poly(X[:, level], n_vec[level])
        TX_new = np.hstack((
                this_basis[:, jj][:, np.newaxis] * TX for jj in range(n_vec[level])
                ))
        return recurse_tensor(X, TX_new, n_vec, level - 1)
    else:
        return TX

class ChebFcn:
    """Chebyshev polynomial approximation"""
    
    def __init__(self, n):

        self.n = n
        self.grid = grid(n)
        self.basis = poly(self.grid, self.n)
        
    def grid(self):
        
        return self.grid.copy()
    
    def fit_vals(self, vals):
        
        self.coeffs = np.linalg.solve(self.basis, vals)
        
    def fit_fcn(self, fcn):
        
        vals = fcn(self.grid)
        self.fit_vals(vals)
        
    def evaluate(self, x):
        
        Tx = poly(x, self.n)
        return np.dot(Tx, self.coeffs)
    
class TensorChebFcn:
    
    def __init__(self, n_vec):
        
        self.k = len(n_vec)
        self.n_vec = n_vec
        self.grid = cartesian((grid(n_vec[ii]) for ii in range(self.k)))
        self.basis = tensor(self.grid, self.n_vec)
        
    def grid(self):
        
        return self.grid.copy()
    
    def fit_vals(self, vals):
        
        self.coeffs = np.linalg.solve(self.basis, vals)
        
    def fit_fcn(self, fcn):
        
        vals = fcn(self.grid)
        self.fit_vals(vals)
        
    def evaluate(self, x):
        
        Tx = tensor(x, self.n_vec)
        return np.dot(Tx, self.coeffs)