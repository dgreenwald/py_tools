import numpy as np
from scipy.optimize import minimize
import py_tools.numerical as nm

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

def svd_inv(A, sv_tol=1e-8, **kwargs):
    
    u, s, vh = np.linalg.svd(A)
    s_inv = np.zeros(s.shape)
    
    ix = np.abs(s) > sv_tol
    s_inv[ix] = 1.0 / s[ix]
    
    S_inv = np.diag(s_inv)
    A_inv = np.dot(vh.T, np.dot(S_inv, u.T))
    return A_inv

def save_file(x, out_dir, name, suffix=None):

    if out_dir[-1] != '/':
        out_dir += '/'

    outfile = out_dir + name
    if suffix is not None:
        outfile += '_' + suffix

    outfile += '.npy'

    np.save(outfile, x)

def load_file(out_dir, name, suffix=None):

    if out_dir[-1] != '/':
        out_dir += '/'

    outfile = out_dir + name
    if suffix is not None:
        outfile += '_' + suffix

    outfile += '.npy'

    return np.load(outfile)

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

        self.params_hat = None
        self.L_hat = None
        self.CH_inv = None
        self.draws = None
        self.acc_rate = None

    def log_like_args(self, params):

        return self.log_like(params, *self.args)

    def objfcn(self, unbdd_params):

        params = self.transform(unbdd_params, to_bdd=True)
        return -self.log_like(params, *self.args)

    def find_mode(self, x0, method='bfgs', tol=1e-8, **kwargs):
        
        self.tol = tol

        x0_u = self.transform(x0, to_bdd=False)
        res = minimize(self.objfcn, x0_u, tol=self.tol)
        self.params_hat = self.transform(res.x, to_bdd=True)
        self.L_hat = -res.fun
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

    def compute_hessian(self, x0=None, **kwargs):

        if x0 is None:
            x0 = self.params_hat.copy()
        self.H = -nm.hessian(self.log_like_args, x0)
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

    def sample(self, Nsim, jump_scale=1.0, stride=1, x0=None, **kwargs):

        self.Nsim = Nsim
        self.jump_scale = jump_scale

        Ntot = Nsim * stride
        self.draws = np.zeros((self.Nsim, self.Npar))
        self.acc_rate = 0.0
        
        e = np.random.randn(Ntot, self.Npar)
        log_u = np.log(np.random.rand(Ntot))

        if x0 is not None:
            x = x0.copy()
            L = self.log_like_args(x)
        else:
            x = self.params_hat.copy()
            L = self.L_hat

        for ii in range(Ntot):
            x_try = x + self.jump_scale * np.dot(self.CH_inv, e[ii, :])
            x, L, acc = self.metropolis_hastings(x, L, x_try, log_u=log_u[ii])
            self.acc_rate += acc
            if ii % stride == 0:
                self.draws[ii // stride, :] = x

        self.acc_rate /= Ntot

        return None

    def save_all(self, out_dir, suffix=None, **kwargs):

        for name in ['params_hat', 'L_hat', 'CH_inv', 'draws', 'acc_rate']:
            self.save_item(name, out_dir, suffix=suffix)

        return None

    def save_item(self, name, out_dir, suffix=None):

        obj = getattr(self, name)
        if obj is not None:
            save_file(obj, out_dir, name, suffix)

        return None

    def load_item(self, name, out_dir, suffix=None):

        setattr(self, name, load_file(out_dir, name, suffix))

        return None

    def run_all(self, x0, Nsim, out_dir=None, **kwargs):
        """Find mode, run MCMC chain, and save"""

        self.find_mode(x0, **kwargs)
        self.compute_hessian(**kwargs)
        self.sample(Nsim, **kwargs)
        if out_dir is not None:
            self.save(out_dir, **kwargs)
