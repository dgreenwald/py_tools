import numpy as np
from scipy.optimize import minimize
from . import math

def logit(x, lb=0.0, ub=1.0):
    return np.log(x - lb) - np.log(ub - x)

def logistic(x, lb=0.0, ub=1.0):
    return lb + (ub - lb) / (1.0 + np.exp(-x))

def check_bounds(x, bounds):

    for ii in range(len(x)):
        lb, ub = bounds[ii]
        if lb is not None and x[ii] < lb:
            return False
        if ub is not None and x[ii] > lb:
            return False

    return True

def svd_inv(A, sv_tol=1e-8):
    
    u, s, vh = np.linalg.svd(A)
    s_inv = np.zeros(s.shape)
    
    ix = np.abs(s) > sv_tol
    s_inv[ix] = 1.0 / s[ix]
    
    S_inv = np.diag(s_inv)
    A_inv = np.dot(vh.T, np.dot(S_inv, u.T))
    return A_inv

class MCMC:
    """Class for Markov Chain Monte Carlo sampler"""

    def __init__(self, log_like, args=(), bounds=None, names=None,
                 bounds_dict={}):
        """Constructor -- need to finish"""

        self.log_like = log_like
        self.args = args
        self.bounds = bounds 
#        self.tol = tol
#        self.jump_scale = jump_scale

        if self.bounds is None and names is not None:
            self.bounds = []
            for name in names:
                self.bounds.append(bounds_dict.get(name, (None, None)))

        self.Npar = len(self.bounds)

    def log_like_args(self, params):

        return self.log_like(params, *self.args)

    def objfcn(self, unbdd_params):

        params = self.transform(unbdd_params, to_bdd=True)
        return -self.log_like(params, *self.args)

    def find_mode(self, x0, method='bfgs', tol=1e-8):
        
        self.tol = tol

        x0_u = self.transform(x0, to_bdd=False)
        res = minimize(self.objfcn, x0_u, tol=self.tol)
        self.params_hat = self.transform(res.x, to_bdd=True)
        self.log_like_hat = -res.fun
        return res

    def transform(self, vals, to_bdd=True):

        trans_vals = vals.copy()
        for ii, (lb, ub) in enumerate(self.bounds):
            if to_bdd:
                if lb is not None:
                    if ub is not None:
                        trans_vals[ii] = logistic(vals[ii], lb=lb, ub=ub)
                    else:
                        trans_vals[ii] = lb + np.exp(vals[ii])
                elif ub is not None:
                    trans_vals[ii] = ub - np.exp(vals[ii])
            else:
                if lb is not None:
                    if ub is not None:
                        trans_vals[ii] = logit(vals[ii], lb=lb, ub=ub)
                    else:
                        trans_vals[ii] = np.log(vals[ii] - lb)
                elif ub is not None:
                    trans_vals[ii] = -np.log(ub - vals[ii])

        return trans_vals

    def compute_hessian(self, **kwargs):

        self.H = -math.numerical_hessian(self.log_like_args, self.params_hat)
        self.H_inv = svd_inv(self.H, **kwargs)
        self.CH_inv = np.linalg.cholesky(self.H_inv)

        return None

    def metropolis_hastings(self, x, L, x_try, log_u=None):

        # Keep old if trial is out of bounds
        if not check_bounds(x_try, self.bounds):
            return (x, L, False)

        L_try = self.log_like_args(x_try)
        
        if log_u is None:
            log_u = np.log(np.random.rand())

        if log_u < L_try - L:
            return (x_try, L_try, True)
        else:
            return (x, L, False)

    def sample(self, Nsim, jump_scale=1.0):

        self.Nsim = Nsim
        self.jump_scale = jump_scale

        self.draws = np.zeros((self.Nsim, self.Npar))
        self.acc = np.zeros(self.Nsim, dtype=bool)
        
        e = np.random.randn(self.Nsim, self.Npar)
        log_u = np.log(np.random.rand(self.Nsim))

        x = self.params_hat
        L = self.log_like_hat

        for ii in range(Nsim):
            x_try = self.params_hat + self.jump_scale * np.dot(self.CH_inv, e[ii, :])
            x, L, self.acc[ii] = self.metropolis_hastings(x, L, x_try, log_u=log_u[ii])
            self.draws[ii, :] = x

        return None
