import numpy as np
from scipy.optimize import minimize

from py_tools.numerical import gradient

def objfcn(b, err_fcn, *args):

    return np.sqrt(np.mean(err_fcn(b, *args) ** 2))

def se_nls(err_fcn, b, args=()):

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

    inner_vals *= (4.0 / Nt)
    outer_vals /= Nt

    outer_vals_inv = np.linalg.inv(outer_vals)
    V = np.dot(outer_vals_inv, np.dot(inner_vals, outer_vals_inv))
    se = np.sqrt(np.diagonal(V) / Nt)

    return se, V, e

def nls(err_fcn, b0, args=(), **kwargs):

    all_args = (err_fcn,) + args
    res = minimize(objfcn, b0, args=all_args, **kwargs)

    # Compute standard errors 
    b_hat = res.x
    se, V, e_hat = se_nls(err_fcn, b_hat, args=args)

    output = {
        'b_hat' : b_hat,
        'e_hat' : e_hat,
        'V' : V,
        'se' : se,
        'res' : res,
    }

    return output
