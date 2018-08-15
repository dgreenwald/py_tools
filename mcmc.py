import os
import numpy as np
from scipy.optimize import minimize
# import py_tools.numerical as nm
from py_tools import in_out as io, numerical as nm

def logit(x, lb=0.0, ub=1.0):
    return np.log(x - lb) - np.log(ub - x)

def logistic(x, lb=0.0, ub=1.0):
    return lb + (ub - lb) / (1.0 + np.exp(-x))

def randomize_blocks(nx, nblock):
    """nx is number of entries, nblock is number of blocks"""
    ix_all = np.random.permutation(nx)
    block_size = int(np.ceil(nx / nblock))

    blocks = []
    cutoffs = np.arange(0, nx, block_size)

    for ii in range(nblock):
        start_ix = cutoffs[ii]
        if ii < nblock - 1:
            end_ix = cutoffs[ii + 1]
        else:
            end_ix = nx
        blocks.append(ix_all[start_ix:end_ix])

    return numerical_to_bool_blocks(blocks, nx)

def numerical_to_bool_blocks(blocks, nx):
    
    bool_blocks = []
    for block in blocks:
        this_block = np.zeros(nx, dtype=bool)
        this_block[block] = True
        bool_blocks.append(this_block)
        
    return bool_blocks

def check_bounds(x, bounds):

    return (np.all(x >= lb) and np.all(x <= ub))

def print_mesg(mesg, fid=None):
    
    if fid is not None:
        fid.write(mesg + '\n')
        fid.flush()
    else:
        print(mesg)

def save_file(x, out_dir, name, suffix=None, pickle=False):

    if out_dir[-1] != '/':
        out_dir += '/'

    outfile = out_dir + name
    if suffix is not None:
        outfile += '_' + suffix

    if pickle:
        outfile += '.pkl'
        io.save_pickle(x, outfile)
    else:
        outfile += '.npy'
        np.save(outfile, x)

def load_file(out_dir, name, suffix=None, pickle=False):

    if out_dir[-1] != '/':
        out_dir += '/'

    outfile = out_dir + name
    if suffix is not None:
        outfile += '_' + suffix
        
    if pickle:
        outfile += '.pkl'
    else:
        outfile += '.npy'
        
    if not os.path.exists(outfile):
        return None

    if pickle:
        return io.load_pickle(outfile)  
    else:
        return np.load(outfile)

def metropolis_step(fcn, x, x_try, L=None, log_u=None, args=()):

    if L is None:
        L = fcn(x, *args)

    L_try = fcn(x_try, *args)
    
    if log_u is None:
        log_u = np.log(np.random.rand())

    if log_u < L_try - L:
        return (x_try, L_try, True)
    else:
        return (x, L, False)

class MonteCarlo:
    """Master class for Monte Carlo samplers"""

    def __init__(self, log_like=None, args=(), lb=None, ub=None, names=None,
                 bounds_dict={}, out_dir=None, suffix=None, Nx=None):

        self.log_like = log_like
        self.args = args
        self.names = names
        self.x_mode = None
        self.L_mode = None
        self.CH_inv = None

        if lb is not None:
            self.Nx = len(lb)
        elif ub is not None:
            self.Nx = len(ub)
        elif names is not None:
            self.Nx = len(names)
        elif Nx is not None:
            self.Nx = Nx
        else:
            print("Length of parameter vector unknown")
            raise Exception

        if lb is None or ub is None:

            if lb is None:
                self.lb = np.inf * np.ones(self.Nx)
            if ub is None:
                self.ub = np.inf * np.ones(self.Nx)

            if self.names is not None:
                for ii, name in enumerate(self.names):
                    lb_i, ub_i = bounds_dict.get(name, (-np.inf, np.inf))

                    if lb_i is None: lb_i = -np.inf
                    if ub_i is None: ub_i = np.inf

                    if lb is None: self.lb[ii] = lb_i
                    if ub is None: self.ub[ii] = ub_i

        self.out_dir = out_dir
        self.suffix = suffix

    def log_like_args(self, params):

        if check_bounds(params, self.bounds):
            return self.log_like(params, *self.args)
        else:
            return -1e+10

    def min_objfcn(self, unbdd_params):

        params = self.transform(unbdd_params, to_bdd=True)
        return -self.log_like(params, *self.args)

    def find_mode(self, x0, method='bfgs', tol=1e-8, **kwargs):
        
        self.tol = tol

        x0_u = self.transform(x0, to_bdd=False)
        res = minimize(self.objfcn, x0_u, tol=self.tol)
        self.x_mode = self.transform(res.x, to_bdd=True)
        self.L_mode = -res.fun
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
            x0 = self.x_mode.copy()

        self.H = -nm.hessian(self.log_like_args, x0)

        self.H_inv = np.linalg.pinv(self.H)
        self.CH_inv = np.linalg.cholesky(self.H_inv)

        return None

    def metro(self, x, L, x_try, **kwargs):

        return metropolis_step(self.log_like_args, x, x_try, L=L, **kwargs)

    def open_log(self, title='log'):

        if self.out_dir[-1] != '/':
            self.out_dir += '/'

        log_file = self.out_dir + 'log'
            
        if self.suffix is not None:
            log_file += '_' + self.suffix

        log_file += '.txt'

        self.fid = open(log_file, 'wt')

    def print_log(self, mesg):
        print_mesg(mesg, fid=self.fid)

    def close_log(self):
        self.fid.close()

    def save_item(self, name, **kwargs):

        assert(self.out_dir is not None)
        obj = getattr(self, name)
        if obj is not None:
            save_file(obj, self.out_dir, name, self.suffix, **kwargs)

        return None

    def load_item(self, name, **kwargs):

        assert(self.out_dir is not None)
        setattr(self, name, load_file(self.out_dir, name, self.suffix, **kwargs))

        return None

class RWMC(MonteCarlo):
    """Class for Markov Chain Monte Carlo sampler"""

    def __init__(self, *args, **kwargs):
        """Constructor -- need to finish"""

        MonteCarlo.__init__(self, *args, **kwargs)

        self.draws = None
        self.acc_rate = None

        self.out_dir = out_dir
        self.suffix = suffix

    def initialize(self, x0=None, jump_scale=None, jump_mult=1.0, stride=1, 
                   C=None, blocks='none', bool_blocks=False, n_blocks=None):

        self.stride = stride

        if x0 is None:
            self.x0 = self.x_mode
        else:
            self.x0 = x0
            
        if self.Nx is None:
            self.Nx = len(self.x0)
        else:
            assert(self.Nx == len(self.x0))

        if jump_scale is None:
            self.jump_scale = jump_mult * 2.4 / np.sqrt(self.Nx)
        else:
            self.jump_scale = jump_scale

        if blocks == 'none':
            self.blocks = [np.ones(self.Nx, dtype=bool)]
        elif blocks == 'random':
            assert(n_blocks is not None)
            self.blocks = randomize_blocks(self.Nx, n_blocks)
        elif bool_blocks:
            # Boolean blocks, i.e., [False, True, False, False, True, True]
            self.blocks = blocks
        else:
            # Numerical blocks, i.e. [1, 4, 5]
            self.blocks = numerical_to_bool_blocks(blocks, self.Nx)

        # Make sure every parameter is in some block
        assert (sum([np.sum(block) for block in self.blocks])) == len(self.x0)

        if C is None:
            self.C = []
            for iblock, block in enumerate(self.blocks):
                self.C += [self.CH_inv[block, :][:, block]]
        else:
            self.C = C

        self.Nblock = len(self.blocks)

    def sample(self, Nsim, n_print=None, n_recov=None, n_save=None, log=True,
               *args, **kwargs):

        self.Nsim = Nsim

        x = self.x0.copy()
        L = self.log_like_args(x)

        if self.Nx is None:
            self.Nx = len(x)

        if log:
            self.open_log()
        else:
            self.fid = None

        Nstep = Nsim * self.stride
        Ntot = Nstep * self.Nblock

        self.draws = np.zeros((self.Nsim, self.Nx))
        self.L_sim = np.zeros(self.Nsim)
        self.acc_rate = 0.0
        
        e = [np.random.randn(Nstep, np.sum(block)) for block in self.blocks]
        log_u = np.log(np.random.rand(Nstep, self.Nblock))

        self.max_x = 1.0 * x
        self.max_L = 1.0 * L

        self.print_log("Jump scale is {}".format(self.jump_scale))

        for istep in range(Nstep):

            for iblock, block in enumerate(self.blocks):

                x_try = x.copy()
                x_try[block] += self.jump_scale * np.dot(self.C[iblock], e[iblock][istep, :])
                x, L, acc = self.metro(x, L, x_try, log_u=log_u[istep, iblock])

                if L > self.max_L:
                    self.max_L = 1.0 * L
                    self.max_x = 1.0 * x

                self.acc_rate += acc

            if (istep + 1) % self.stride == 0:
                self.draws[istep // self.stride, :] = x
                self.L_sim[istep // self.stride] = L

                if n_print is not None:
                    if (istep // self.stride + 1) % n_print == 0:
                        self.print_log("Draw {0:d}. Acceptance rate: {1:4.3f}. Max L = {2:4.3f}".format(
                            (istep + 1) // self.stride, self.acc_rate / istep, self.max_L
                        ))

                # if n_recov is not None:
                    # if ((istep // self.stride + 1) - n_burn) % n_recov == 0:
                        # self.print_log("Recomputing covariance")
                        # self.C = np.linalg.cholesky(np.cov(self.draws[n_burn : (istep // self.stride) + 1, :], rowvar=False))

                if n_save is not None:
                    if ((istep // self.stride + 1) % n_save == 0) and istep < Nstep - 1:
                        self.print_log("Saving intermediate output")
                        self.save_all()

        self.acc_rate /= Ntot

        if self.fid is not None:
            self.fid.close()

        return None

    def save_all(self, **kwargs):

        for name in ['x_mode', 'L_mode', 'CH_inv', 'draws', 'L_sim', 'acc_rate']:
            self.save_item(name)

        # Pickled items
        for name in ['names']:
            self.save_item(name, pickle=True)

        return None

    def load_all(self, **kwargs):

        for name in ['x_mode', 'L_mode', 'CH_inv', 'draws', 'L_sim', 'acc_rate']:
            self.load_item(name)

        # Pickled items
        for name in ['names']:
            self.load_item(name, pickle=True)

        return None

    def run_all(self, x0, Nsim, **kwargs):
        """Find mode, run MCMC chain, and save"""

        self.find_mode(x0, **kwargs)
        self.compute_hessian(**kwargs)
        self.sample(Nsim, **kwargs)
        if self.out_dir is not None:
            self.save(self.out_dir, **kwargs)

class SMC:
    """Sequential Monte Carlo Sampler"""

    def __init__(self, log_like, prior, args=()):

        self.like = like
        self.prior = prior
        self.Nx = len(prior.dists)

    def initialize(self, Npts, Nsteps, init_jump_scale=0.25):

        self.Npts = Npts
        self.Nsteps = Nsteps

        self.jump_scales = np.zeros(self.Nsteps)
        self.jump_scales[0] = init_jump_scale

        self.draws = np.zeros((self.Nsteps, self.Npts, self.Nx))
        self.draws[0, :, :] = self.prior.sample(self.Npts).T 

        self.W = np.zeros((self.Nsteps, self.Npts))
        self.W[0, :] = 1.0

    def adapt(self, istep):

        self.the_star = np.mean(self.draws[istep-1, :, :], axis=0)
        self.Sig_star = np.cov(self.draws[istep-1, :, :], rowvar=False)
        self.C_star = np.linalg.cholesky(self.Sig_star)

        if istep > 1:
            raise Exception
        else:
            self.jump_scales[istep] = self.jump_scales[istep-1]
