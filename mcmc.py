import os
import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
# import py_tools.numerical as nm
from py_tools import in_out as io, numerical as nm
from py_tools.prior import Prior
from py_tools.mpi_array import MPIArray

from mpi4py import MPI

def logit(x, lb=0.0, ub=1.0):
    return np.log(x - lb) - np.log(ub - x)

def logistic(x, lb=0.0, ub=1.0):
    return lb + (ub - lb) / (1.0 + np.exp(-x))

def transform(vals, lb, ub, to_bdd=True):

    trans_vals = vals.copy()

    # Indices
    ix_lb = lb > -np.inf
    ix_ub = ub < np.inf

    ix_both = ix_lb & ix_ub
    ix_lb_only = ix_lb & (~ix_ub)
    ix_ub_only = (~ix_lb) & ix_ub

    if to_bdd:
        trans_vals[ix_both] = logistic(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = lb[ix_lb_only] + np.exp(vals[ix_lb_only])
        trans_vals[ix_ub_only] = ub[ix_ub_only] - np.exp(vals[ix_ub_only])
    else:
        trans_vals[ix_both] = logit(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = np.log(vals[ix_lb_only] - lb[ix_lb_only])
        trans_vals[ix_ub_only] = -np.log(ub[ix_ub_only] - vals[ix_ub_only])

    return trans_vals

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

def check_bounds(x, lb, ub):

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

def metropolis_step(fcn, x, x_try, post=None, log_u=None, args=()):

    if post is None:
        post = fcn(x, *args)

    post_try = fcn(x_try, *args)
    
    if log_u is None:
        log_u = np.log(np.random.rand())

    if log_u < post_try - post:
        return (x_try, post_try, True)
    else:
        return (x, post, False)

def rwmh(posterior, x_init, jump_scale=1.0, C=None, Nstep=1, blocks=None,
              block_sizes=None, post_init=None, e=None, log_u=None):

    Nx = len(x_init)
    Nblock = len(blocks)

    if blocks is None:
        blocks = [np.ones(Nx, dtype=bool)]

    if block_sizes is None:
        block_sizes = [np.sum(block) for block in blocks]

    if e is None:
        e = np.random.randn(Nstep, Nx)

    if C is None:
        C = [np.eye(block_size) for block_size in block_sizes]

    if log_u is None:
        log_u = np.log(np.random.rand(Nstep, Nblock))

    if post_init is None:
        post_init = posterior(x_init)

    x_store = np.zeros((Nstep, Nx))
    post_store = np.zeros(Nstep)
    acc_rate = 0

    x = x_init.copy()
    post = post_init
    
    for istep in range(Nstep):

        for iblock, block in enumerate(blocks):
            x_try = x.copy()
            x_try[block] += jump_scale * np.dot(C[iblock], e[istep, block])
            x, post, acc = metropolis_step(posterior, x, x_try, post=post, log_u=log_u[istep, iblock])
            acc_rate += acc
        
        x_store[istep, :] = x
        post_store[istep] = post

    acc_rate /= (Nstep * Nblock)

    return (x_store, post_store, acc_rate)

class MonteCarlo:
    """Master class for Monte Carlo samplers"""

    def __init__(self, log_like=None, prior=Prior(), args=(), lb=None, ub=None,
                 names=None, bounds_dict={}, out_dir=None, suffix=None,
                 Nx=None):

        self.log_like = log_like
        self.prior = prior
        self.args = args
        self.names = names
        self.x_mode = None
        self.post_mode = None
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

    def posterior(self, params):

        if check_bounds(params, self.lb, self.ub):
            return (self.log_like(params, *self.args) + self.prior.logpdf(params))
        else:
            return -1e+10

    def min_objfcn(self, unbdd_params):

        params = self.transform(unbdd_params, to_bdd=True)
        return -self.posterior(params)

    def find_mode(self, x0, method='bfgs', tol=1e-8, **kwargs):
        
        self.tol = tol

        x0_u = self.transform(x0, to_bdd=False)
        res = minimize(self.min_objfcn, x0_u, tol=self.tol)
        self.x_mode = self.transform(res.x, to_bdd=True)
        self.post_mode = -res.fun
        return res

    def transform(self, vals, *args, **kwargs):

        return transform(vals, self.lb, self.ub, *args, **kwargs)

    def compute_hessian(self, x0=None, **kwargs):

        if x0 is None:
            x0 = self.x_mode.copy()

        self.H = -nm.hessian(self.posterior, x0)

        self.H_inv = np.linalg.pinv(self.H)
        self.CH_inv = np.linalg.cholesky(self.H_inv)

        return None

    def metro(self, x, post, x_try, **kwargs):

        return metropolis_step(self.posterior, x, x_try, post=post, **kwargs)

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
        post = self.posterior(x)

        if self.Nx is None:
            self.Nx = len(x)

        if log:
            self.open_log()
        else:
            self.fid = None

        Nstep = Nsim * self.stride
        Ntot = Nstep * self.Nblock

        self.draws = np.zeros((self.Nsim, self.Nx))
        self.post_sim = np.zeros(self.Nsim)
        self.acc_rate = 0.0
        
        e = [np.random.randn(Nstep, np.sum(block)) for block in self.blocks]
        log_u = np.log(np.random.rand(Nstep, self.Nblock))

        self.max_x = 1.0 * x
        self.max_post = 1.0 * post

        self.print_log("Jump scale is {}".format(self.jump_scale))

        for istep in range(Nstep):

            for iblock, block in enumerate(self.blocks):

                x_try = x.copy()
                x_try[block] += self.jump_scale * np.dot(self.C[iblock], e[iblock][istep, :])
                x, post, acc = self.metro(x, post, x_try, log_u=log_u[istep, iblock])

                if post > self.max_post:
                    self.max_post = 1.0 * post
                    self.max_x = 1.0 * x

                self.acc_rate += acc

            if (istep + 1) % self.stride == 0:
                self.draws[istep // self.stride, :] = x
                self.post_sim[istep // self.stride] = post

                if n_print is not None:
                    if (istep // self.stride + 1) % n_print == 0:
                        self.print_log("Draw {0:d}. Acceptance rate: {1:4.3f}. Max posterior = {2:4.3f}".format(
                            (istep + 1) // self.stride, self.acc_rate / istep, self.max_post
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

        for name in ['x_mode', 'post_mode', 'CH_inv', 'draws', 'post_sim', 'acc_rate']:
            self.save_item(name)

        # Pickled items
        for name in ['names']:
            self.save_item(name, pickle=True)

        return None

    def load_all(self, **kwargs):

        for name in ['x_mode', 'post_mode', 'CH_inv', 'draws', 'post_sim', 'acc_rate']:
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

class SMC(MonteCarlo):
    """Sequential Monte Carlo Sampler"""

    def __init__(self, *args, **kwargs):
        """Constructor -- need to finish"""

        MonteCarlo.__init__(self, *args, **kwargs)

    def initialize(self, Npt, Nstep, Nmut=1, Nblock=1, blocks=None,
                   init_jump_scale=0.25, lam=2.0, adapt_sens=16.0,
                   adapt_range=0.1, adapt_target=0.25, parallel=False):

        self.Npt = Npt
        self.Nstep = Nstep
        self.Nmut = Nmut

        # Adaptive proposal
        self.adapt_sens = adapt_sens
        self.adapt_range = adapt_range
        self.adapt_target = adapt_target

        # Random walk jump scaling (c_n)
        self.jump_scales = np.zeros(self.Nstep)
        self.jump_scales[0] = init_jump_scale

        # Acceptance rate
        self.acc_rate = np.zeros(self.Nstep)

        self.draws = np.zeros((self.Nstep, self.Npt, self.Nx))
        self.draws[0, :, :] = self.prior.sample(self.Npt).T 

        self.W = np.ones((self.Nstep, self.Npt))
        self.post = np.zeros((self.Nstep, self.Npt))

        # Posterior weighting schedule
        self.lam = lam
        self.phi = (np.arange(self.Nstep + 1) / self.Nstep) ** self.lam

        # Blocks
        if blocks is None:
            self.blocks = randomize_blocks(self.Nx, Nblock)
        else:
            self.blocks = blocks

        self.Nblock = len(self.blocks)
        self.block_sizes = [np.sum(block) for block in self.blocks]
        
        # Other drawing parameters
        self.the_star = None
        self.C_star = None
        
        # Temporary (for diagnostics)
        self.ess = np.zeros(self.Nstep)
        self.draws_pre_mut = self.draws.copy()
        self.post_pre_mut = self.post.copy()

        # Parallelization
        self.parallel = parallel
        if self.parallel:
            comm = MPI.COMM_WORLD
            self.rank = comm.Get_rank()

    def sample(self):

        for istep in range(1, self.Nstep):
            self.correct(istep)
            self.adapt(istep)
            self.mutate(istep)

    def correct(self, istep):

        self.draws[istep, :, :] = self.draws[istep-1, :, :]
        
        if self.parallel:
#        if True:
             
            # Create MPI arrays and scatter to nodes
            mpi_draws = MPIArray(root_data=self.draws[istep, :, :])
            mpi_post = MPIArray(root_data=self.post[istep, :])

            # Get local data
            local_draws = mpi_draws.get_local_data()
            local_post = mpi_post.get_local_data()

            # Loop
            Nloc = local_draws.shape[0]
            for ipt in range(Nloc):
                local_post[ipt] = self.posterior(local_draws[ipt, :])

            mpi_post.set_local_data(local_post)
            
            if self.rank > 0: return
#                raise Exception
            
#            this_post2 = mpi_post.get_root_data()
            this_post = mpi_post.get_root_data()
            
        else:
#        if True:
             
            this_post = np.zeros(self.Npt)
            for ipt in range(self.Npt):
                this_post[ipt] = self.posterior(self.draws[istep, ipt, :])
                
#        print("correct error:")
#        print(this_post - this_post2)
        
#        raise Exception

        # Turn into weight using incremental
        w = np.exp((self.phi[istep] - self.phi[istep-1]) * this_post)
        W_til = w * self.W[istep-1, :]
        W_til /= np.mean(W_til)

        # Resample if effective sample size too small
        ess = self.Npt / np.mean(W_til ** 2)
        if ess < self.Npt / 2:
            ix = np.random.choice(self.Npt, size=self.Npt, p=W_til / self.Npt)
            self.draws[istep, :, :] = self.draws[istep, ix, :]
            
            self.W[istep, :] = 1.0
            self.post[istep, :] = this_post[ix]
        else:
            self.W[istep, :] = W_til
            self.post[istep, :] = this_post
            
        self.draws_pre_mut[istep, :, :] = self.draws[istep, :, :]
        self.post_pre_mut[istep, :] = self.post[istep, :]
        self.ess[istep] = ess

    def adapt(self, istep):
        
        if self.rank > 0: return
        
        # Weighted mean
        weights = self.W[istep, :]
        self.the_star = np.dot(weights, self.draws[istep, :, :]) / self.Npt
        
        # Weighted covariance
        the_til = self.draws[istep, :, :] - self.the_star[np.newaxis, :]
        if np.any(self.the_star > 0.0):
            w_the_til = weights[:, np.newaxis] * the_til
            self.Sig_star = np.dot(w_the_til.T, the_til) / self.Npt
#            print(self.Sig_star)
            if not np.all(np.linalg.eigvals(self.Sig_star) > 0):
                print("Bad Sig_star:")
                print(self.Sig_star)
            self.C_star = np.linalg.cholesky(self.Sig_star)
        elif self.C_star is None:
            print("No valid value for C_star")
            raise Exception

        self.jump_scales[istep] = self.jump_scales[istep-1]
        if istep > 1:
            e_term = np.exp(self.adapt_sens * (self.acc_rate[istep-1] - self.adapt_target))
            adj = (1.0 - 0.5 * self.adapt_range) + self.adapt_range * (e_term / (1.0 + e_term))
            self.jump_scales[istep] *= adj

    def mutate(self, istep):
        
        if self.rank > 0: return

#        if self.parallel:
#            
#            # Set arrays and scatter
#            mpi_e = MPIArray(root_data=np.random.randn(self.Npt, self.Nmut * self.Nx))
#            mpi_log_u = MPIArray(root_data=np.random.rand(self.Npt, self.Nmut * self.Nblock)
#            mpi_draws = MPIArray(root_data=self.draws[istep, :, :])
#            mpi_post = MPIArray(root_data=self.post[istep, :][:, np.newaxis])
#            
#        else:
        if True:
            
            e = np.random.randn(self.Npt, self.Nmut, self.Nx)
            log_u = np.log(np.random.rand(self.Npt, self.Nmut, self.Nblock))
    
            for ipt in range(self.Npt):
                
                x_i, post_i, acc_rate_i = rwmh(
                    self.posterior, self.draws[istep, ipt, :],
                    jump_scale=self.jump_scales[istep], C=self.C_star,
                    Nstep=self.Nmut, blocks=self.blocks,
                    block_sizes=self.block_sizes, post_init=self.post[istep, ipt], 
                    e=e[ipt, :, :], log_u=log_u[ipt, :, :],
                )
                
                self.draws[istep, ipt, :] = x_i[-1, :]
                self.post[istep, ipt] = post_i[-1]
                self.acc_rate[istep] += acc_rate_i
                
            self.acc_rate[istep] /= self.Npt

        return None
