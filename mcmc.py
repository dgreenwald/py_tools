import numpy as np
from scipy.optimize import minimize
from . import math

def logit(x):
    return np.log(x) - np.log(1.0 - x)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

class MCMC:
    """Class for Markov Chain Monte Carlo sampler"""

    def __init__(self, log_like, args=(), t_types=None, names=None,
                 trans_dict={}, tol=1e-8):
        """Constructor -- need to finish"""

        self.log_like = log_like
        self.args = args
        self.t_types = t_types
        self.tol=tol

        if self.t_types is None and names is not None:
            self.t_types = []
            for name in names:
                self.t_types.append(trans_dict.get(name, None))

    def objfcn(self, unbdd_params):

        params = self.transform(unbdd_params, to_bdd=True)
        return -self.log_like(params, *self.args)

    def find_mode(self, x0_u, method='bfgs'):

        res = minimize(self.objfcn, x0_u, args=self.args, tol=self.tol)
        self.params_hat = self.transform(res.x, to_bdd=True)
        return res

    def transform(self, vals, to_bdd=True):

        trans_vals = vals.copy()
        for ii, t_type in enumerate(self.t_types):
            if to_bdd:
                if t_type == 'logistic':
                    trans_vals[ii] = logistic(vals[ii])
                elif t_type == 'exp':
                    trans_vals[ii] = np.exp(vals[ii])
            else:
                if t_type == 'logistic':
                    trans_vals[ii] = logit(vals[ii])
                elif t_type == 'exp':
                    trans_vals[ii] = np.log(vals[ii])

        return trans_vals

    def compute_hessian(self, sv_tol=1e-8):

        self.H = math.numerical_hessian(self.log_like, self.params_hat)

        # Invert
        u, s, vh = np.linalg.svd(self.H)
        s_inv = np.zeros(s.shape)
        
        ix = np.abs(s) > sv_tol
        s_inv[ix] = 1.0 / s[ix]
        
        S_inv = np.diag(s_inv)
        self.H_inv = np.dot(vh.T, np.dot(S_inv, u.T))

        return None

    # def sample(Nsim):


