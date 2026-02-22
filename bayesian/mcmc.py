import os
import time
import numpy as np
import scipy.optimize as opt
from scipy.stats import multivariate_normal as mv
from py_tools import in_out as io, numerical as nm
from py_tools.bayesian.prior import Prior


def _load_parallel_tools():
    try:
        import py_tools.compute.mpi_array as mp
        from py_tools.compute.mpi_array import MPIArray
        from mpi4py import MPI
    except Exception as exc:
        raise RuntimeError(
            "MPI parallel features are unavailable. "
            "Use parallel=False or configure a working MPI runtime."
        ) from exc
    return mp, MPIArray, MPI

def adapt_jump_scale(acc_rate, adapt_sens, adapt_target, adapt_range):

    e_term = np.exp(adapt_sens * (acc_rate - adapt_target))
    return (1.0 - 0.5 * adapt_range) + adapt_range * (e_term / (1.0 + e_term))

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

def partition_C(C, blocks):
    
    C_list = []
    for iblock, block in enumerate(blocks):
        C_list += [C[block, :][:, block]]
        
    return C_list

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
        print(mesg, flush=True)

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

def importance_sample(fcn, dist, Nsim, Nx, args=(), parallel=False):

    if not parallel:
        draws = np.atleast_2d(dist.rvs(Nsim))
        if draws.shape[0] != Nsim:
            draws = draws.reshape(Nsim, Nx)
        post = np.zeros(Nsim)
        for jj in range(Nsim):
            post[jj] = fcn(draws[jj, :], *args)
    else:
        mp, _, _ = _load_parallel_tools()
        draws = np.zeros((Nsim, Nx))
        post = np.zeros(Nsim)

        draws_mpi, draws_loc = mp.initialize(draws, fake=False)
        post_mpi, post_loc = mp.initialize(post, fake=False)

        Nloc = len(post_loc)
        draws_loc = np.atleast_2d(dist.rvs(Nloc))
        if draws_loc.shape[0] != Nloc:
            draws_loc = draws_loc.reshape(Nloc, Nx)
        mp.disp("Draws per task: {:d}".format(Nloc))
        for jj in range(Nloc):
            post_loc[jj] = fcn(draws_loc[jj, :], *args)

        draws = mp.finalize(draws_mpi, draws_loc)
        post = mp.finalize(post_mpi, post_loc)
        
        if mp.rank() != 0:
            return None, None
    
    p_proposal = dist.logpdf(draws)
    log_weights = post - p_proposal
    return draws, log_weights

def rwmh(posterior, x_init, jump_scale=1.0, C_list=None, Nstep=1, blocks=None,
              block_sizes=None, post_init=None, e=None, log_u=None, quiet=True):

    Nx = len(x_init)

    if blocks is None:
        blocks = [np.ones(Nx, dtype=bool)]
    Nblock = len(blocks)

    if block_sizes is None:
        block_sizes = [np.sum(block) for block in blocks]

    if e is None:
        e = np.random.randn(Nstep, Nx)

    if C_list is None:
        C_list = [np.eye(block_size) for block_size in block_sizes]

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
            x_try[block] += jump_scale * np.dot(C_list[iblock], e[istep, block])
            x, post, acc = metropolis_step(posterior, x, x_try, post=post, log_u=log_u[istep, iblock])
            acc_rate += acc
            
            if not quiet:
                print("x: " + repr(x))
                print("x_try: " + repr(x_try))
                print("post: " + repr(post))
        
        x_store[istep, :] = x
        post_store[istep] = post

    acc_rate /= (Nstep * Nblock)

    return (x_store, post_store, acc_rate)

class MonteCarlo:
    """Master class for Monte Carlo samplers"""

    def __init__(self, log_like=None, prior=None, args=(), lb=None, ub=None,
                 names=None, bounds_dict=None, out_dir=None, suffix=None,
                 Nx=None):
        """
        Parameters
        ----------
        log_like : function
            Function of (vals, *args) that evaluates the log likelihood.
        prior : Prior object
            Bayesian prior
        args : tuple
            Additional arguments of log_like
        lb : ndarray
            Lower bounds of parameters
        ub : ndarray
            Upper bounds of parameters
        names : list
            Names of parameters
        bounds_dict: dict
            Alternative bounds input, maps param names to (lb, ub) pairs
        out_dir: string
            Directory where output should be saved
        suffix: string
            Suffix for labeling
        Nx : int
            number of parameters
        """

        if bounds_dict is None: bounds_dict = {}

        self.log_like = log_like
        if prior is None:
            prior = Prior()
        self.prior = prior
        self.args = args
        self.names = names
        self.x_mode = None
        self.post_mode = None
        self.H_inv = None
        self.CH_inv = None
        self.lb = None
        self.ub = None

        if lb is not None:
            self.Nx = len(lb)
        elif ub is not None:
            self.Nx = len(ub)
        elif names is not None:
            self.Nx = len(names)
        elif Nx is not None:
            self.Nx = Nx
        else:
            self.Nx = None
            
        if self.Nx is not None:

            lb_missing = (lb is None)
            ub_missing = (ub is None)

            if lb_missing:
                self.lb = -np.inf * np.ones(self.Nx)
            else:
                self.lb = np.array(lb, copy=True)

            if ub_missing:
                self.ub = np.inf * np.ones(self.Nx)
            else:
                self.ub = np.array(ub, copy=True)

            if self.names is not None and (lb_missing or ub_missing):
                for ii, name in enumerate(self.names):
                    lb_i, ub_i = bounds_dict.get(name, (-np.inf, np.inf))

                    if lb_i is None: lb_i = -np.inf
                    if ub_i is None: ub_i = np.inf

                    if lb_missing: self.lb[ii] = lb_i
                    if ub_missing: self.ub[ii] = ub_i

        self.out_dir = out_dir
        self.suffix = suffix

    def posterior(self, params):

        if (self.lb is None) or (self.ub is None) or check_bounds(params.ravel(), self.lb, self.ub):
            return (self.log_like(params, *self.args) + self.prior.logpdf(params))
        else:
            return -1e+10

    def min_objfcn(self, unbdd_params):

        params = self.bound_transform(unbdd_params, to_bdd=True)
        return -self.posterior(params)

    def find_mode(self, x0, tol=1e-8, basinhopping=False, method='bfgs',
                  iterate=False, iter_tol=1e-6, disp_iterate=True, **kwargs):
        
        x0 = x0.ravel()

        post_start = None
        done = False
        count = 0
        while not done:

            count += 1
            if iterate and (post_start is None):
                post_start = self.posterior(x0)

            x0_u = self.bound_transform(x0, to_bdd=False)
            if basinhopping:
                minimizer_kwargs = kwargs.get('minimizer_kwargs', {})
                if 'method' not in minimizer_kwargs:
                    minimizer_kwargs['method'] = method
                res = opt.basinhopping(self.min_objfcn, x0_u,
                                       minimizer_kwargs=minimizer_kwargs, **kwargs)
            else:
                res = opt.minimize(self.min_objfcn, x0_u, method=method, tol=tol, **kwargs)
                
            self.x_mode = self.bound_transform(res.x, to_bdd=True)
            self.post_mode = -res.fun

            if iterate:
                if disp_iterate:
                    print("Iteration {0:d}: starting posterior = {1:g}, "
                          "ending posterior = {2:g}".format(count, post_start, self.post_mode), flush=True)

                    if self.names is not None:
                        these_params = {self.names[ii]: self.x_mode[ii] for ii in range(len(self.names))}
                    else:
                        these_params = {'param{:d}'.format(ii): self.x_mode[ii] for ii in range(len(self.x_mode))}
                    print("Params: " + repr(these_params), flush=True)
                done = np.abs(self.post_mode - post_start) < iter_tol
                if not done:
                    x0 = self.x_mode
                    post_start = self.post_mode
            else:
                done = True

        return res

    def find_mode_de(self, bounds, **kwargs):

        res = opt.differential_evolution(self.min_objfcn, bounds, **kwargs)
        self.x_mode = res.x
        self.post_mode = -res.fun
        return res

    def bound_transform(self, vals, *args, **kwargs):

        return nm.bound_transform(vals, self.lb, self.ub, *args, **kwargs)

    def compute_hessian(self, x0=None, cholesky=True, robust=True,
                        **kwargs):

        if x0 is None:
            x0 = self.x_mode.copy()

        self.H = -nm.hessian(self.posterior, x0, **kwargs)

        self.H_inv = np.linalg.pinv(self.H)
        
        if cholesky:

            if robust:

                self.CH_inv = nm.robust_cholesky(self.H_inv)

            else:

                self.CH_inv = np.linalg.cholesky(self.H_inv)

        return None

    def set_CH_inv(self, CH_inv):

        self.CH_inv = CH_inv
        return None

    def metro(self, x, post, x_try, **kwargs):

        return metropolis_step(self.posterior, x, x_try, post=post, **kwargs)

    def open_log(self, title='log', suffix=None):

        if self.out_dir is None:
            self.fid = None
            return None

        if suffix is None:
            suffix = self.suffix
        
        if self.out_dir[-1] != '/':
            self.out_dir += '/'

        log_file = self.out_dir + title
            
        if suffix is not None:
            log_file += '_' + suffix

        log_file += '.txt'

        self.fid = open(log_file, 'wt')

    def print_log(self, mesg):
        print_mesg(mesg, fid=self.fid)

    def close_log(self):
        if self.fid is not None:
            self.fid.close()

    def save_item(self, name, suffix=None, **kwargs):

        if suffix is None:
            suffix = self.suffix
        
        assert(self.out_dir is not None)
        obj = getattr(self, name)
        if obj is not None:
            save_file(obj, self.out_dir, name, suffix, **kwargs)

        return None

    def load_item(self, name, suffix=None, **kwargs):
        
        if suffix is None:
            suffix = self.suffix

        assert(self.out_dir is not None)
        setattr(self, name, load_file(self.out_dir, name, suffix, **kwargs))

        return None
    
    def save_list(self, np_list=None, pkl_list=None, **kwargs):

        if np_list is None: np_list = []
        if pkl_list is None: pkl_list = []

        for var in np_list:
            self.save_item(var, **kwargs)
            
        for var in pkl_list:
            self.save_item(var, pickle=True, **kwargs)

        return None
    
    def load_list(self, np_list=None, pkl_list=None, **kwargs):

        if np_list is None: np_list = []
        if pkl_list is None: pkl_list = []

        for var in np_list:
            try:
                self.load_item(var, **kwargs)
            except:
                print("Warning: could not load " + var)
            
        for var in pkl_list:
            try:
                self.load_item(var, pickle=True, **kwargs)
            except:
                print("Warning: could not load " + var)

        return None

    def importance_sample(self, Nsim, resample=True, offset=None, **kwargs):

        assert self.x_mode is not None
        assert self.H_inv is not None
        
        cov = self.H_inv.copy()
        if offset is not None:
            cov += np.diag(offset * np.ones(self.Nx))

        dist = mv(mean=self.x_mode, cov=cov)
        draws, log_weights = importance_sample(self.posterior, dist, Nsim, self.Nx, **kwargs)

        if draws is None:
            
            return None, None, None
            
        probs = np.exp(log_weights - np.amax(log_weights))
        probs /= np.sum(probs)
        
        W_til = Nsim * probs
        ess = len(W_til) / np.mean(W_til ** 2)

        if resample:
            probs = np.exp(log_weights - np.amax(log_weights))
            probs /= np.sum(probs)
            ix = np.random.choice(Nsim, size=Nsim, p=probs)
            draws = draws[ix, :]
            log_weights = np.zeros(log_weights.shape)

        return draws, log_weights, ess

class RWMC(MonteCarlo):
    """Class for Markov Chain Monte Carlo sampler"""

    def __init__(self, rwmc_chains=None, *args, **kwargs):
        """Constructor -- need to finish"""

        MonteCarlo.__init__(self, *args, **kwargs)
        
        self.np_list = ['x_mode', 'post_mode', 'CH_inv', 'draws', 'post_sim', 'acc_rate']
        self.pkl_list = ['names']

        if rwmc_chains is not None:
            self.post_sim = np.hstack([chain.post_sim for chain in rwmc_chains])
            self.draws = np.vstack([chain.draws for chain in rwmc_chains])

    def initialize(self, x0=None, jump_scale=None, jump_mult=1.0, stride=1,
                   C=None, C_list=None, blocks='none', bool_blocks=False,
                   n_blocks=None, adapt_sens=16.0, adapt_range=0.1,
                   adapt_target=0.25):

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

        if C_list is not None:
            self.C_list = C_list
        else:
            if C is None:
                C = self.CH_inv.copy()
            self.C_list = []
            for iblock, block in enumerate(self.blocks):
                self.C_list += [C[block, :][:, block]]
                
        self.adapt_sens = adapt_sens
        self.adapt_range = adapt_range
        self.adapt_target = adapt_target

        self.Nblock = len(self.blocks)

    def sample(self, Nsim, n_print=None, n_recov=None, n_save=None, log=True,
               cov_offset=0.0, min_recov=0, n_retune=None, chain_no=0,
               *args, **kwargs):

        if (n_save is not None) and (self.out_dir is None):
            raise ValueError("RWMC.sample requires out_dir when n_save is set.")

        self.Nsim = Nsim

        x = self.x0.copy()
        post = self.posterior(x)

        full_suffix = self.chain_suffix(chain_no)

        if self.Nx is None:
            self.Nx = len(x)

        if log:
            self.open_log(suffix=full_suffix)
        else:
            self.fid = None

        Nstep = Nsim * self.stride
        Ntot = Nstep * self.Nblock

        self.draws = np.zeros((self.Nsim, self.Nx))
        self.post_sim = np.zeros(self.Nsim)
        self.acc = 0

        acc_last_retune = 0
        istep_last_retune = 0
        
        e = [np.random.randn(Nstep, np.sum(block)) for block in self.blocks]
        log_u = np.log(np.random.rand(Nstep, self.Nblock))

        self.max_x = 1.0 * x
        self.max_post = 1.0 * post

        self.print_log("Jump scale is {}".format(self.jump_scale))

        for istep in range(Nstep):

            for iblock, block in enumerate(self.blocks):

                x_try = x.copy()
                x_try[block] += self.jump_scale * np.dot(self.C_list[iblock], e[iblock][istep, :])
                x, post, acc = self.metro(x, post, x_try, log_u=log_u[istep, iblock])

                if post > self.max_post:
                    self.max_post = 1.0 * post
                    self.max_x = 1.0 * x

                self.acc += acc

            if (istep + 1) % self.stride == 0:

                self.acc_rate = self.acc / ((istep + 1) * self.Nblock)
                jstep = (istep + 1) // self.stride - 1

                self.draws[jstep, :] = x
                self.post_sim[jstep] = post

                if n_print is not None:
                    if (jstep + 1) % n_print == 0:
                        self.print_log("Draw {0:d}. Acceptance rate: {1:4.3f}. Max posterior = {2:4.3f}".format(
                            jstep + 1, self.acc_rate, self.max_post
                        ))

                if n_recov is not None:
                    if (jstep + 1 >= min_recov) and (((jstep + 1) - min_recov) % n_recov == 0):

                        self.print_log("Recomputing covariance")
                        for iblock, block in enumerate(self.blocks):
                            sample_cov = (np.cov(self.draws[:jstep+1, block], rowvar=False) 
                                          + cov_offset * np.eye(len(x)))
                            self.C_list[iblock] = np.linalg.cholesky(sample_cov)

                if n_retune is not None:
                    if (jstep + 1) % n_retune == 0:

                        # Compute acceptance rate since last retuning
                        acc_since_retune = self.acc - acc_last_retune
                        steps_since_retune = (istep + 1) - istep_last_retune
                        acc_rate_since_retune = acc_since_retune / (steps_since_retune * self.Nblock)

                        self.print_log("Acceptance rate for last {0:d} draws: {1:4.3f}".format(steps_since_retune, acc_rate_since_retune))
                        self.print_log("Retuning: old jump scale = {:7.6f}".format(self.jump_scale))
                        self.jump_scale *= adapt_jump_scale(
                            acc_rate_since_retune, self.adapt_sens, self.adapt_target, self.adapt_range
                        )
                        self.print_log("Retuning: new jump scale = {:7.6f}".format(self.jump_scale))

                        # Reset acceptance counter since retuning
                        acc_last_retune = self.acc
                        istep_last_retune = istep

                if n_save is not None:
                    if (jstep + 1) % n_save == 0:
                        self.print_log("Saving intermediate output")
                        self.save_chain(chain_no=chain_no)


        self.acc_rate = self.acc / Ntot

        self.close_log()

        return None

    def chain_suffix(self, chain_no=0):
        if self.suffix is None:
            return 'chain{:d}'.format(chain_no)
        return self.suffix + '_chain{:d}'.format(chain_no)

    def save_chain(self, chain_no=0):

        full_suffix = self.chain_suffix(chain_no)
        self.save_list(np_list=['draws', 'post_sim', 'acc_rate'], suffix=full_suffix)
        self.save_list(np_list=['jump_scale'], suffix=full_suffix)

    def load_chain(self, chain_no=0):

        full_suffix = self.chain_suffix(chain_no)
        self.load_list(np_list=['draws', 'post_sim', 'acc_rate'], suffix=full_suffix)
        self.load_list(np_list=['jump_scale'], suffix=full_suffix)

    def load_chains(self, chains):

        self.draws_list = []
        self.post_sim_list = []
        self.acc_rate_list = []

        for chain_no in chains:
            self.load_chain(chain_no)
            self.draws_list.append(self.draws)
            self.post_sim_list.append(self.post_sim)
            self.acc_rate_list.append(self.acc_rate)    
            
        # Reset to not keep any chain loaded
        self.draws = None
        self.post_sim = None
        self.acc_rate = None

    def stack_chains(self, burn_in=0, stride=1):

        draws_all = np.vstack([draws[burn_in::stride, :] for draws in self.draws_list])
        post_sim_all = np.hstack([post_sim[burn_in::stride] for post_sim in self.post_sim_list])
        
        return draws_all, post_sim_all

    def save_all(self, **kwargs):

        self.save_list(np_list=self.np_list, pkl_list=self.pkl_list)

    def load_all(self, **kwargs):

        self.load_list(np_list=self.np_list, pkl_list=self.pkl_list)

    def run_all(self, x0, Nsim, mode_kwargs=None, hess_kwargs=None,
                init_kwargs=None, sample_kwargs=None, **kwargs):
        """Find mode, run MCMC chain, and save"""

        if mode_kwargs is None:
            mode_kwargs = {}
        if hess_kwargs is None:
            hess_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        self.find_mode(x0, **mode_kwargs)
        self.compute_hessian(**hess_kwargs)
        self.initialize(**init_kwargs)
        self.sample(Nsim, **sample_kwargs)
        if self.out_dir is not None:
            self.save_all(**kwargs)

class SMC(MonteCarlo):
    """Sequential Monte Carlo Sampler"""

    def __init__(self, *args, **kwargs):
        """Constructor -- need to finish"""

        MonteCarlo.__init__(self, *args, **kwargs)
        
        # Lists for saving    
        self.np_list = ['C_star', 'Sig_star', 'W', 'acc_rate', 'draws', 'ess',
                        'jump_scales', 'lb', 'phi', 'post', 'the_star', 'ub']

        self.pkl_list = ['C_list', 'Nblock', 'Nmut', 'Npt', 'Nstep', 'Nx',
                         'args', 'blocks', 'block_sizes', 'fixed_blocks',
                         'lam', 'log_like', 'names', 'parallel', 'post_mode',
                         'prior', 'save_intermediate', 'x_mode']

    def initialize(self, Npt, Nstep, Nmut=1, Nblock=1, blocks=None,
                   init_jump_scale=0.25, lam=2.0, adapt_sens=16.0,
                   adapt_range=0.1, adapt_target=0.25, parallel=False, 
                   save_intermediate=True, test_flag=False):

        if save_intermediate and (self.out_dir is None):
            raise ValueError("SMC.initialize requires out_dir when save_intermediate=True.")

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

        self.W = np.ones((self.Nstep, self.Npt))
        self.post = np.zeros((self.Nstep, self.Npt))

        # Posterior weighting schedule
        self.lam = lam
        self.phi = (np.arange(self.Nstep + 1) / self.Nstep) ** self.lam

        # Blocks
        if blocks is None:
            self.blocks = randomize_blocks(self.Nx, Nblock)
            self.fixed_blocks = False
        else:
            self.blocks = blocks
            self.fixed_blocks = True

        self.Nblock = len(self.blocks)
        self.block_sizes = [np.sum(block) for block in self.blocks]
        
        # Other drawing parameters
        self.the_star = None
        self.Sig_star = None
        self.C_star = None
        self.C_list = None
        
        # Temporary (for diagnostics)
        self.ess = np.zeros(self.Nstep)

        # Parallelization
        self.set_rank(parallel) # MPI rank

        # Initialize draws
        if self.rank == 0:
            self.draws[0, :, :] = self.prior.sample(self.Npt).T 
            
        # Save output?
        self.save_intermediate = save_intermediate

        # Testing mode?
        self.test_flag = test_flag
        if (self.rank == 0) and self.test_flag: print("TEST FLAG ON")
        
    def set_rank(self, parallel):
        """ Set MPI rank and save parallel flag """
        
        self.parallel = parallel
        
        if self.parallel:
            _, self.MPIArray, self.MPI = _load_parallel_tools()
            self.comm = self.MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.MPIArray = None
            self.MPI = None
            self.rank = 0

    def sample(self, quiet=False):
        
        self.quiet = quiet

        for istep in range(1, self.Nstep):

            # On last step, force resampling
            last_step = (istep == self.Nstep - 1)
            
            if self.parallel:
                start = self.MPI.Wtime()
            else:
                start = time.perf_counter()
            
            self.correct(istep, force_resample=last_step)

            if self.parallel and self.test_flag:
                end = self.MPI.Wtime()
                self.rank_print("Step {0:d} time elapsed: {1:g} seconds".format(istep, end - start))
                raise Exception

            self.adapt(istep)
            self.mutate(istep)
                
            if self.parallel:
                end = self.MPI.Wtime()
            else:
                end = time.perf_counter()
            self.rank_print("Step {0:d} time elapsed: {1:g} seconds".format(istep, end - start))
            
            # End-of-iteration tasks
            if self.rank == 0:

                # Save key output
                if self.save_intermediate:
                    self.save_list(np_list=['draws', 'W', 'post'])
                
                # Re-randomize blocks
                if not self.fixed_blocks:
                    self.update_blocks()

                ix_max = np.argmax(self.post[istep, :])
                post_max = self.post[istep, ix_max]

                self.rank_print("Current mode: " + repr(self.post_mode))
                self.rank_print("Max posterior: {:g}".format(post_max))

                if (self.post_mode is None) or (post_max > self.post_mode):

                    self.x_mode = self.draws[istep, ix_max]
                    self.post_mode = post_max

                    self.rank_print("New mode found: {:g}".format(self.post_mode))

    def correct(self, istep, force_resample=False):

        if self.rank == 0:
            self.draws[istep, :, :] = self.draws[istep-1, :, :]
        
        # Only time we don't have posterior from the mutation step
        if istep == 1:
            if self.parallel:
                # Create MPI arrays and scatter to nodes
                mpi_draws = self.MPIArray(root_data=self.draws[istep, :, :])
                mpi_post = self.MPIArray(root_data=self.post[istep, :])
    
                # Get local data
                local_draws = mpi_draws.get_local_data()
                local_post = mpi_post.get_local_data()
    
                # Loop
                Nloc = local_draws.shape[0]
                for ipt in range(Nloc):
                    local_post[ipt] = self.posterior(local_draws[ipt, :])
    
                mpi_post.set_local_data(local_post)
                
                if self.rank > 0: return None
                this_post = mpi_post.get_root_data()
                
            else:
                this_post = np.zeros(self.Npt)
                for ipt in range(self.Npt):
                    this_post[ipt] = self.posterior(self.draws[istep, ipt, :])
        else:
            
            this_post = self.post[istep-1, :]

        # Turn into weight using incremental, ignoring non-finite values
        ix_good = np.isfinite(this_post)
        w = np.zeros(this_post.shape)
        w[ix_good] = np.exp((self.phi[istep] - self.phi[istep-1]) * this_post[ix_good])

        W_til = w * self.W[istep-1, :]
        W_til /= np.mean(W_til)

        # Resample if effective sample size too small
        ess = self.Npt / np.mean(W_til ** 2)
        self.rank_print("Step {:d}, ESS = {:g}".format(istep, ess))
            
        if force_resample or (ess < self.Npt / 2):
            ix = np.random.choice(self.Npt, size=self.Npt, p=W_til / self.Npt)
            self.draws[istep, :, :] = self.draws[istep, ix, :]
            
            self.W[istep, :] = 1.0
            self.post[istep, :] = this_post[ix]
        else:
            self.W[istep, :] = W_til
            self.post[istep, :] = this_post
            
        self.ess[istep] = ess

    def adapt(self, istep):
        
        if self.parallel: 
            if self.rank > 0: return None

        # Weighted mean
        weights = self.W[istep, :]
        self.the_star = np.dot(weights, self.draws[istep, :, :]) / self.Npt
        
        # Weighted covariance
        the_til = self.draws[istep, :, :] - self.the_star[np.newaxis, :]
        w_the_til = weights[:, np.newaxis] * the_til
        self.Sig_star = np.dot(w_the_til.T, the_til) / self.Npt

        if not np.all(np.isfinite(self.Sig_star)):
            raise ValueError("Sig_star contains non-finite values.")

        self.C_star = nm.robust_cholesky(self.Sig_star, min_eig=0.0)
            
        # Update C_list as list by block
        self.C_list = partition_C(self.C_star, self.blocks)

        self.jump_scales[istep] = self.jump_scales[istep-1]
        if istep > 1:
            self.jump_scales[istep] *= adapt_jump_scale(
                self.acc_rate[istep-1], self.adapt_sens, self.adapt_target, self.adapt_range
            )

    def mutate(self, istep):

        e = np.random.randn(self.Npt, self.Nmut, self.Nx)
        log_u = np.log(np.random.rand(self.Npt, self.Nmut, self.Nblock))
        
        if self.parallel:
            
            local_C_list = self.comm.bcast(self.C_list, root=0)
            local_jump_scale = self.comm.bcast(self.jump_scales[istep], root=0)
            
            # Set arrays and scatter
            mpi_draws = self.MPIArray(root_data=self.draws[istep, :, :])
            mpi_post = self.MPIArray(root_data=self.post[istep, :])
            mpi_e = self.MPIArray(root_data=e)
            mpi_log_u = self.MPIArray(root_data=log_u)
            
            local_draws = mpi_draws.get_local_data()
            local_post = mpi_post.get_local_data()
            local_e = mpi_e.get_local_data()
            local_log_u = mpi_log_u.get_local_data()
            
            local_acc = np.zeros(1)
        
            Nloc = local_draws.shape[0]
            for ipt in range(Nloc):
                x_i, post_i, acc_rate_i = rwmh(
                    self.posterior, local_draws[ipt, :],
                    jump_scale=local_jump_scale, C_list=local_C_list,
                    Nstep=self.Nmut, blocks=self.blocks,
                    block_sizes=self.block_sizes, post_init=local_post[ipt], 
                    e=local_e[ipt, :, :], log_u=local_log_u[ipt, :, :],
                )
                local_draws[ipt, :] = x_i[-1, :]
                local_post[ipt] = post_i[-1]
                local_acc += acc_rate_i
                
            mpi_draws.set_local_data(local_draws)
            mpi_post.set_local_data(local_post)
                
            root_acc = np.zeros(1)
            self.comm.Reduce(local_acc, root_acc, op=self.MPI.SUM, root=0)
            self.acc_rate[istep] = root_acc / self.Npt
            
            
            if self.rank > 0: return None
            
            self.draws[istep, :, :] = mpi_draws.get_root_data()
            self.post[istep, :] = mpi_post.get_root_data()

        else:
            for ipt in range(self.Npt):
                
                x_i, post_i, acc_rate_i = rwmh(
                    self.posterior, self.draws[istep, ipt, :],
                    jump_scale=self.jump_scales[istep], C_list=self.C_list,
                    Nstep=self.Nmut, blocks=self.blocks,
                    block_sizes=self.block_sizes, post_init=self.post[istep, ipt], 
                    e=e[ipt, :, :], log_u=log_u[ipt, :, :],
                )
                
                self.draws[istep, ipt, :] = x_i[-1, :]
                self.post[istep, ipt] = post_i[-1]
                self.acc_rate[istep] += acc_rate_i
                
            self.acc_rate[istep] /= self.Npt
            
        self.rank_print("Acceptance rate: {:g}".format(self.acc_rate[istep]))

        return None
    
    def update_blocks(self):
        
        self.blocks = randomize_blocks(self.Nx, self.Nblock)
        self.C_list = partition_C(self.C_star, self.blocks)
        
    def rank_print(self, mesg):
        if (not self.quiet) and (self.rank == 0):
            print(mesg, flush=True)
            
    def save(self):

        self.save_list(np_list=self.np_list, pkl_list=self.pkl_list)

    def load(self, parallel=False):

        self.load_list(np_list=self.np_list, pkl_list=self.pkl_list)
        
        # Re-set parallel flag and rank
        self.set_rank(parallel)
