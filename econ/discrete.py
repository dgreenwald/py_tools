# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:09:18 2016

@author: dan
"""

import numpy as np
import scipy.sparse as sp

import py_tools.econ as ec

def to_2d(x):
    """Convert a 1D array to a 2D column vector, or pass through a 2D array.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape ``(N,)`` or ``(N, k)``.

    Returns
    -------
    numpy.ndarray or None
        - If ``x`` is 1-D, returns ``x`` reshaped to ``(N, 1)``.
        - If ``x`` is 2-D, returns ``x`` unchanged.
        - If ``x`` has more than 2 dimensions, returns ``None``.
    """
    if len(x.shape) == 1:
        return x[:, np.newaxis]
    elif len(x.shape) == 2:
        return x
    else:
        return None

def combine_grids(x, y):
    """Compute the Cartesian product of two grids.

    Each row of the returned array is one combination ``(x_i, y_j)`` for all
    ``i`` in ``x`` and all ``j`` in ``y``, enumerated in row-major order
    (``x`` varies slowest).

    Parameters
    ----------
    x : numpy.ndarray
        First grid, shape ``(Nx,)`` or ``(Nx, kx)``.
    y : numpy.ndarray
        Second grid, shape ``(Ny,)`` or ``(Ny, ky)``.

    Returns
    -------
    numpy.ndarray
        Combined grid of shape ``(Nx * Ny, kx + ky)``.
    """
    x_2d = to_2d(x)
    y_2d = to_2d(y)
    
    assert (x_2d is not None) and (y_2d is not None)
    
    x_stack = np.repeat(x_2d, len(y), axis=0)
    y_stack = np.tile(y_2d, (len(x), 1))
    
    return np.hstack((x_stack, y_stack))

def combine_grids_from_list(grids_list):
    """Compute the Cartesian product of a list of grids.

    Iteratively applies :func:`combine_grids` to every grid in *grids_list*,
    so the result contains one row for every combination of grid points across
    all grids.

    Parameters
    ----------
    grids_list : list of numpy.ndarray
        Each element is a 1-D or 2-D grid array.  The first element must be
        convertible to 2-D by :func:`to_2d`.

    Returns
    -------
    numpy.ndarray
        Combined grid of shape ``(prod(Ni), sum(ki))``, where ``Ni`` is the
        number of points and ``ki`` the number of columns in the *i*-th grid.
    """
    stack = to_2d(grids_list[0])
    assert stack is not None
    
    for ii in range(1, len(grids_list)):
        stack = combine_grids(stack, grids_list[ii])
        
    return stack

def combine_markov_chains(grids_list, P_list):
    """Combine multiple independent Markov chains into a single joint chain.

    The joint state space is the Cartesian product of the individual grids and
    the joint transition matrix is the Kronecker product of the individual
    transition matrices, which is the correct construction for independent
    chains.

    Parameters
    ----------
    grids_list : list of numpy.ndarray
        State-space grids for each individual chain.  Each element may be 1-D
        or 2-D; see :func:`combine_grids` for shape conventions.
    P_list : list of numpy.ndarray
        Transition matrices for each individual chain.  ``P_list[i]`` must be
        square with side length equal to the number of rows in
        ``grids_list[i]``.

    Returns
    -------
    stack : numpy.ndarray
        Joint state-space grid produced by :func:`combine_grids_from_list`.
    P_stack : numpy.ndarray
        Joint transition matrix formed by successive Kronecker products of the
        matrices in *P_list*.
    """
    stack = grids_list[0]
    P_stack = P_list[0]
    
    for ii in range(1, len(grids_list)):
        stack = combine_grids(stack, grids_list[ii])
        P_stack = np.kron(P_stack, P_list[ii])
        
    return stack, P_stack

def drop_low_probs(P, tol=1e-6):
    """Zero out small transition probabilities and renormalize rows.

    Entries in *P* that are strictly below *tol* are set to zero, then each
    row is divided by its new sum so that rows still sum to one.

    Parameters
    ----------
    P : numpy.ndarray
        Transition probability matrix of shape ``(N, N)``.  Rows are assumed
        to sum to one before the operation.
    tol : float, optional
        Threshold below which probabilities are zeroed out.  Default is
        ``1e-6``.

    Returns
    -------
    numpy.ndarray
        A copy of *P* with sub-threshold entries removed and rows renormalized.
    """
    P_new = P.copy()
    ix = P_new < tol
    P_new[ix] = 0
    P_sum = np.sum(P_new, axis=1)
    P_new = P_new / P_sum[:, np.newaxis]
    
    return P_new

class DiscreteModel:
    """Infinite-horizon discrete-choice dynamic programming model.

    Solves a stationary Bellman equation over a finite grid of endogenous
    states *x* and exogenous states *z* using Howard's policy-improvement
    algorithm.

    Parameters
    ----------
    bet : array-like of shape ``(Nz,)`` or scalar
        Discount factor(s), one per exogenous state.  A scalar is broadcast
        to all exogenous states.
    flow_list : list of numpy.ndarray
        Per-period flow payoff matrices.  ``flow_list[iz]`` has shape
        ``(Nx, Nx_choices)`` and gives the payoff for each endogenous state
        and each feasible next-period choice when the exogenous state is
        ``iz``.
    x_grid : numpy.ndarray
        Endogenous state grid of shape ``(Nx, kx)``.
    z_grid : numpy.ndarray
        Exogenous state grid of shape ``(Nz, kz)``.
    Pz : numpy.ndarray
        Transition probability matrix for the exogenous states, shape
        ``(Nz, Nz)``.  Rows are renormalized after dropping probabilities
        below *P_tol*.
    index_list : list of numpy.ndarray, optional
        Initial policy indices.  ``index_list[iz]`` is an integer array of
        length ``Nx`` giving the index of the chosen next-period endogenous
        state for each current state when the exogenous state is ``iz``.
        If ``None``, the policy is initialized to the identity (stay in
        place).
    P_tol : float, optional
        Tolerance passed to :func:`drop_low_probs` when cleaning *Pz*.
        Default is ``1e-6``.
    """

    def __init__(self, bet, flow_list, x_grid, z_grid, Pz, index_list=None,
                 P_tol=1e-6):
        """Initialize a :class:`DiscreteModel` instance.

        Parameters
        ----------
        bet : array-like of shape ``(Nz,)`` or scalar
            Discount factor(s).  Converted to a numpy array and broadcast
            against the exogenous-state dimension.
        flow_list : list of numpy.ndarray
            Flow payoff matrices; see class docstring for details.
        x_grid : numpy.ndarray
            Endogenous state grid, shape ``(Nx, kx)``.
        z_grid : numpy.ndarray
            Exogenous state grid, shape ``(Nz, kz)``.
        Pz : numpy.ndarray
            Exogenous-state transition matrix, shape ``(Nz, Nz)``.
        index_list : list of numpy.ndarray, optional
            Initial policy indices.  Defaults to identity policy.
        P_tol : float, optional
            Probability threshold for :func:`drop_low_probs`.  Default
            ``1e-6``.
        """
        
        self.bet = np.array(bet)
        self.flow_list = flow_list
        self.x_grid = x_grid
        self.z_grid = z_grid
        self.Pz = drop_low_probs(Pz, tol=P_tol)

        # Sizes
        self.Nx, self.kx = self.x_grid.shape
        self.Nz, self.kz = self.z_grid.shape
        
        # Sparse identity  matrix
        self.Ixz = sp.eye(self.Nx * self.Nz)

        # Discounted transition probs for exogenous states
        self.bP = self.bet[:, np.newaxis] * self.Pz
        
        # Create sparse matrices
        self.Pzs = sp.csr_matrix(self.Pz)
        self.bPs = sp.csr_matrix(self.bP)
        
        # Discounted transition probs for combined states
        self.bP1 = np.kron(self.bP, np.ones((self.Nx, 1)))

        if index_list is None:
            self.index_list = self.Nz * [np.arange(self.Nx).astype(int)]
        else:
            self.index_list = index_list
            
        # Check for nans
        opt_flow = self.get_opt_flow()
        assert np.all(np.isfinite(opt_flow))
        
    def get_opt_flow(self):
        """Return the flow payoffs under the current policy.

        For each exogenous state ``iz`` and each endogenous state ``ix``,
        looks up ``flow_list[iz][ix, index_list[iz][ix]]`` and stacks the
        results into a single column vector over all ``(iz, ix)`` pairs.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(Nz * Nx, 1)`` containing the optimal flow value
            for every joint state under the current policy.
        """
        opt_flow_list = [self.flow_list[ii][np.arange(self.Nx), 
                                            np.squeeze(self.index_list[ii])][:, np.newaxis] 
                         for ii in range(self.Nz)]
        
        opt_flow = np.vstack(opt_flow_list)
        return opt_flow

    def solve(self):
        """Solve the model via Howard's policy-improvement algorithm.

        Iterates between a *policy evaluation* step (solving the linear system
        ``(I - β P_π) v = r_π`` for the current policy ``π``) and an
        *improvement* step (taking the greedy policy with respect to *v*) until
        the policy indices stop changing.

        After convergence the following attributes are set:

        Attributes
        ----------
        V : numpy.ndarray
            Value function, shape ``(Nz, Nx)``.
        I : numpy.ndarray
            Optimal policy indices, shape ``(Nz, Nx)``.  ``I[iz, ix]`` is the
            index into ``x_grid`` of the optimal next-period endogenous state.

        Returns
        -------
        None
        """

        it = 0
        done = False

        R = np.vstack(self.flow_list)
        indices = np.concatenate(self.index_list)


        while not done:
            
            it += 1
            
            indices_old = indices
            
            # Howard improvement step
            opt_flow = self.get_opt_flow()
            
            bP_trans_sparse = self.get_P_trans(discount=True, sparse=True)
            v = sp.linalg.spsolve(self.Ixz - bP_trans_sparse, opt_flow)
            
            # Update step
            V = v.reshape((self.Nz, self.Nx))
            W = np.dot(self.bP1, V)
            Q = R + W
            indices, v = ec.update_value(Q)
            self.index_list = np.split(indices, self.Nz)
            
            done = np.all(indices_old == indices)
            
        print('Converged in {} iterations'.format(it))

        self.V = v.reshape((self.Nz, self.Nx))
        self.I = indices.reshape((self.Nz, self.Nx))
        
        return None
    
    def get_P_trans(self, discount=False, sparse=True):
        """Compute the combined state transition matrix under the current policy.

        Builds the joint transition matrix over all ``(iz, ix)`` pairs by
        combining the policy-implied endogenous transitions with the exogenous
        transition matrix ``Pz`` (or its discounted counterpart ``bP``) via
        Kronecker products.

        Parameters
        ----------
        discount : bool, optional
            If ``True``, weight transitions by the discount factor (use
            ``β * Pz`` instead of ``Pz``).  Default is ``False``.
        sparse : bool, optional
            If ``True``, return a ``scipy.sparse`` matrix.  If ``False``,
            return a dense ``numpy.ndarray``.  Default is ``True``.

        Returns
        -------
        scipy.sparse matrix or numpy.ndarray
            Joint transition matrix of shape ``(Nz * Nx, Nz * Nx)``.
        """
        transition_list = [ec.get_transition(index, sparse=False) for index in self.index_list]
        
        if sparse:
            
            if discount:
                this_P = self.bPs
            else:
                this_P = self.Pzs
            
            P_trans_list = [sp.kron(this_P[ii, :], transition_list[ii]) for ii in range(self.Nz)]
            P_trans = sp.vstack(P_trans_list)
            
        else:
            
            if discount:
                this_P = self.bP
            else:
                this_P = self.Pz
            
            P_trans_list = [np.kron(this_P[ii, :], transition_list[ii]) for ii in range(self.Nz)]
            P_trans = np.vstack(P_trans_list)
            
        return P_trans
    
    def compute_stationary_dist(self, tol=1e-6):
        """Compute and store the stationary distribution of the model.

        Finds the stationary distribution by computing the left eigenvector of
        the joint transition matrix (i.e., the eigenvector of its transpose)
        corresponding to eigenvalue 1, then normalizes it to sum to one.

        Parameters
        ----------
        tol : float, optional
            Tolerance used to assert that the leading eigenvalue is
            sufficiently close to 1.  Default is ``1e-6``.

        Returns
        -------
        None

        Attributes
        ----------
        pi_star : numpy.ndarray
            Stationary distribution over joint states ``(iz, ix)``, shape
            ``(Nz * Nx,)``, normalized to sum to one.
        """
        P_trans = self.get_P_trans(discount=False, sparse=True)
        vals, vecs = sp.linalg.eigs(P_trans.T)
        
        assert np.abs(np.real(vals[0]) - 1.0) < tol
        
        self.pi_star = np.real(vecs[:, 0])
        self.pi_star = self.pi_star / np.sum(self.pi_star)
        
        return None

    def sim(self, Nsim, ix0=0, iz0=0):
        """Simulate the model from its solution.

        Draws a path of exogenous states using the transition matrix ``Pz``
        and then applies the optimal policy to trace out the endogenous state
        path.  :meth:`solve` must be called before :meth:`sim`.

        Parameters
        ----------
        Nsim : int
            Number of time periods to simulate.
        ix0 : int, optional
            Initial index into ``x_grid`` for the endogenous state.  Default
            is ``0``.
        iz0 : int, optional
            Initial index into ``z_grid`` for the exogenous state.  Default
            is ``0``.

        Returns
        -------
        x_sim : numpy.ndarray
            Simulated endogenous state path, shape ``(Nsim, kx)``.
        z_sim : numpy.ndarray
            Simulated exogenous state path, shape ``(Nsim, kz)``.
        """
        z_ix_sim = ec.sim_discrete(self.Pz, Nsim, i0=iz0)
        z_sim = self.z_grid[z_ix_sim, :]

        x_ix_sim = ec.sim_policy(self.index_list, z_ix_sim, i0=ix0)
        x_sim = self.x_grid[x_ix_sim, :]
        return (x_sim, z_sim)

class LifeCycleModel:
    """Finite-horizon (life-cycle) discrete-choice dynamic programming model.

    Solves a backward-induction Bellman equation over ``T`` periods on a
    finite grid of endogenous states *x* and exogenous states *z*.

    Parameters
    ----------
    bet : float
        Discount factor applied uniformly across all periods.
    flow_lists : list of list of numpy.ndarray
        Period-specific flow payoff matrices.  ``flow_lists[t][iz]`` has shape
        ``(Nx, Nx_choices)`` and gives the flow payoffs at period *t* for
        exogenous state ``iz``.  Length of the outer list is ``Nt``.
    terminal_list : list of numpy.ndarray
        Terminal (period ``Nt``) value arrays.  ``terminal_list[iz]`` has
        shape ``(1, Nx)`` and gives the terminal value for each endogenous
        state when the exogenous state is ``iz``.
    x_grid : numpy.ndarray
        Endogenous state grid, shape ``(Nx, kx)``.
    z_grid : numpy.ndarray
        Exogenous state grid, shape ``(Nz, kz)``.
    Pz : numpy.ndarray
        Transition probability matrix for the exogenous states, shape
        ``(Nz, Nz)``.
    """

    def __init__(self, bet, flow_lists, terminal_list, x_grid, z_grid, Pz):
        """Initialize a :class:`LifeCycleModel` instance.

        Parameters
        ----------
        bet : float
            Discount factor.
        flow_lists : list of list of numpy.ndarray
            Period-specific flow payoff matrices; see class docstring.
        terminal_list : list of numpy.ndarray
            Terminal value arrays; see class docstring.
        x_grid : numpy.ndarray
            Endogenous state grid, shape ``(Nx, kx)``.
        z_grid : numpy.ndarray
            Exogenous state grid, shape ``(Nz, kz)``.
        Pz : numpy.ndarray
            Exogenous-state transition matrix, shape ``(Nz, Nz)``.
        """
        self.bet = bet
        self.flow_lists = flow_lists
        self.terminal_list = terminal_list
        self.x_grid = x_grid
        self.z_grid = z_grid
        self.Pz = Pz

        # Initial calculations
        self.Nx, self.kx = self.x_grid.shape
        self.Nz, self.kz = self.z_grid.shape
        self.Nt = len(self.flow_lists)

        self.bP = self.bet * self.Pz
        self.bP1 = np.kron(self.bP, np.ones((self.Nx, 1)))

        # Initializations
        self.v_lists = [
            [np.zeros((self.Nx, 1)) for _ in range(self.Nz)]
            for _ in range(self.Nt)
        ]
        self.index_lists = [
            [np.zeros((self.Nx, 1), dtype=int) for _ in range(self.Nz)]
            for _ in range(self.Nt)
        ]

    def solve(self):
        """Solve the life-cycle model by backward induction.

        Iterates backward from period ``Nt - 1`` to ``0``.  At each period
        the Bellman operator is applied once using the value function from the
        next period (initialized with *terminal_list* at the final period).

        After completion the following attributes are populated for every
        period:

        Attributes
        ----------
        v_lists : list of list of numpy.ndarray
            ``v_lists[t][iz]`` is the value function at period *t* and
            exogenous state ``iz``, shape ``(Nx, 1)``.
        index_lists : list of list of numpy.ndarray
            ``index_lists[t][iz]`` contains the optimal policy indices at
            period *t* and exogenous state ``iz``, shape ``(Nx,)``.

        Returns
        -------
        None
        """

        v_next = np.vstack([vi.T for vi in self.terminal_list])

        for tt in range(self.Nt-1, -1, -1):

            R = np.vstack(self.flow_lists[tt])
            W = np.dot(self.bP1, v_next)
            V = R + W
            indices, v = ec.update_value(V)

            self.v_lists[tt] = np.split(v, self.Nz)
            self.index_lists[tt] = np.split(indices, self.Nz)
            
            v_next = np.vstack([vi.T for vi in self.v_lists[tt]])
        
        return None

    def sim(self, Nsim):
        """Simulate the life-cycle model from its solution.

        For each of the *Nsim* agents draws a length-``Nt`` exogenous-state
        path and applies the period-specific optimal policies to generate the
        corresponding endogenous-state path.  :meth:`solve` must be called
        before :meth:`sim`.

        Parameters
        ----------
        Nsim : int
            Number of agents (independent simulations) to draw.

        Returns
        -------
        x_sim : numpy.ndarray
            Simulated endogenous state paths, shape ``(kx, Nt, Nsim)``.
        z_sim : numpy.ndarray
            Simulated exogenous state paths, shape ``(kz, Nt, Nsim)``.
        """
        z_sim = np.zeros((self.kz, self.Nt, Nsim))
        x_sim = np.zeros((self.kx, self.Nt, Nsim))
        
        for ii in range(Nsim):
            z_ix_sim = ec.sim_discrete(self.Pz, self.Nt)
            z_sim[:, :, ii] = self.z_grid[z_ix_sim, :].T  
            
            x_ix_sim = ec.sim_life_cycle(self.index_lists, z_ix_sim)
            x_sim[:, :, ii] = self.x_grid[x_ix_sim, :].T
            
        return (x_sim, z_sim)
