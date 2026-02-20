import numpy as np
from py_tools.utilities import cartesian

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

def basis_and_gradient(x, n):
    
    T = poly(x, n).T
    dT = np.zeros((n, len(x)))
    
    if n > 1:
        dT[1, :] = 1.0
        
    for jj in range(2, n):
        dT[jj, :] = 2.0 * T[jj-1, :] + 2.0 * x * dT[jj-1, :] - dT[jj-2, :]
        
    return T.T, dT.T

def gradient(x, n):
    
    _, dT = basis_and_gradient(x, n)
    return dT

def tensor(X, n_vec):
    """Each row of X should be one observation
    n_vec should contain the levels for each variable"""
    
    if len(X.shape) == 1:
        X = X[np.newaxis, :]

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
        
    def grid(self):
        
        return self.grid.copy()
    
    def fit_vals(self, vals):
        
        self.coeffs = np.linalg.solve(self.basis, vals)
        
    def fit_fcn(self, fcn):
        
        vals = fcn(self.scaled_grid)
        self.fit_vals(vals)
        
    def get_basis(self, x_in):
        
        x_scaled = self.scale_x(x_in)
        return poly(x_scaled, self.n)
        
    def get_basis_and_gradient(self, x_in):
        
        x_scaled = self.scale_x(x_in)
        return basis_and_gradient(x_scaled, self.n)
        
    def get_basis_gradient_only(self, x_in):
        
        x_scaled = self.scale_x(x_in)
        return gradient(x_scaled, self.n)
        
    def evaluate(self, x_in):
        
        Tx = self.get_basis(x_in)
        return Tx @ self.coeffs
    
    def gradient(self, x_in):
        
        _, dTx = self.get_basis_and_gradient(x_in)
        return np.dot(dTx, self.coeffs)
    
    def evaluate_with_gradient(self, x_in):
        
        Tx, dTx = self.get_basis_and_gradient(x_in)
        fx = Tx @ self.coeffs
        dfx = dTx @ self.coeffs
        return fx, dfx
        
    def scale_x(self, x_in):
        
        x = x_in.copy()
        x = np.minimum(x, self.ub)
        x = np.maximum(x, self.lb)
        return self.scale_to_grid(x)
    
    def scale_to_grid(self, vals):

        return self.a_to_grid + (self.b_to_grid * vals)

    def scale_from_grid(self, vals):

        return self.a_from_grid + (self.b_from_grid * vals)
    
class TensorChebFcn:
    
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
        
        self.coeffs = np.linalg.solve(self.basis, vals)
        
    def fit_fcn(self, fcn):
        
        vals = fcn(self.scaled_grid)
        self.fit_vals(vals)
        
    def evaluate(self, x, chunksize=100000):
        
        x = np.minimum(x, self.ub)
        x = np.maximum(x, self.lb)
        x_grid = self.scale_to_grid(x)

        Nx = x_grid.shape[0]
        if Nx > chunksize:
            vals = np.zeros(Nx)
            count = 0
            while count < Nx:
                vals[count : count + chunksize] = eval_tensor(
                    x_grid[count : count + chunksize, :],
                    self.n_vec, self.coeffs
                )
                count += chunksize
        else:
            # Tx = tensor(x_grid, self.n_vec)
            # vals= np.dot(Tx, self.coeffs) 
            vals = eval_tensor(x_grid, self.n_vec, self.coeffs)

        return vals

    def scale_to_grid(self, vals):

        return self.a_to_grid + (self.b_to_grid * vals)

    def scale_from_grid(self, vals):

        return self.a_from_grid + (self.b_from_grid * vals)

def eval_tensor(x, n_vec, coeffs):

    Tx = tensor(x, n_vec)
    return np.dot(Tx, coeffs)
