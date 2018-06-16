import numpy as np
import itertools

def gradient(f, x, args=(), step=1e-5):

    grad = None
    for ii in range(len(x)):

        x[ii] += step
        f_hi = f(x, *args)
        x[ii] -= (2.0 * step)
        f_lo = f(x, *args)
        x[ii] += step

        df_i = np.array(f_hi - f_lo) / (2.0 * step)

        if grad is None:

            if df_i.shape == ():
                ncols = 1 
            else:
                ncols = len(df_i)

            grad = np.zeros((len(x), ncols))

        grad[ii, :] = df_i

    return grad

def quad_form(A, X):
    return np.dot(A.T, np.dot(X, A))

def hessian(f, x_in, eps=1e-4):
    
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
    
    u, s, vh = np.linalg.svd(A)
    s_inv = np.zeros(s.shape)
    
    ix = np.abs(s) > sv_tol
    s_inv[ix] = 1.0 / s[ix]
    
    S_inv = np.diag(s_inv)
    A_inv = np.dot(vh.T, np.dot(S_inv, u.T))
    return A_inv

