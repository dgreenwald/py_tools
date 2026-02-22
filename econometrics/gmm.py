import numpy as np
import scipy

def compute_g(params, h, data):

    Nt = data.shape[1]

    g = h(data[:, 0], params)
    for tt in range(1, Nt):
        g += h(data[:, tt], params)
    g /= Nt

    return g

def compute_dg(params, dh, data):

    Nt = data.shape[1]

    dg = dh(data[:, 0], params)
    for tt in range(1, Nt):
        dg += dh(data[:, tt], params)
    dg /= Nt

    return dg

def compute_d2g(params, d2h, data):

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

    g = compute_g(params, h, data)
    return 0.5 * np.dot(g.T, np.dot(W, g))

def jac_gmm(params, h, data, W, dh=None):

    g = compute_g(params, h, data)
    dg = compute_dg(params, dh, data)

    return np.dot(dg.T, np.dot(W, g))

def hess_gmm(params, h, dh, d2h, data, W):

    g = compute_g(params, h, data)
    dg = compute_dg(params, dh, data)
    d2g = compute_d2g(params, d2h, data)

    Wg = np.dot(W, g)
    hess = np.dot(dg.T, np.dot(W, dg))
    for ii, d2g_i in enumerate(d2g):
        hess += Wg[ii] * d2g_i

    return hess

def solve_gmm(params_guess, h, data, W=None, dh=None, d2h=None, algorithm=None):

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
            algorithm = 'dogleg'
        else:
            algorithm = 'BFGS'

    return scipy.optimize.minimize(obj_gmm, params_guess, args=(h, data, W, dh),
                                   method=algorithm, jac=jac, hess=hess)
