import numpy as np
import itertools

def quad_form(A, X):
    return np.dot(A.T, np.dot(X, A))

def numerical_hessian(f, x, eps=1e-4):
    
    n = len(x)
    H = np.zeros((n, n))
    for ii, jj in itertools.product(range(n), repeat=n):
        
        x[ii] += eps
        x[jj] += eps
        H[ii, jj] += f(x)
        
        x[jj] -= 2.0 * eps
        H[ii, jj] -= f(x)
        
        x[ii] -= 2.0 * eps
        H[ii, jj] += f(x)
        
        x[jj] += 2.0 * eps
        H[ii, jj] -= f(x)
        
        x[ii] += eps
        x[jj] -= eps
        
    return H / (4.0 * (eps ** 2))

def test_func(x):
    
    return (x[0] ** 2) * (x[1] ** 3)