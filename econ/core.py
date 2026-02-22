# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:46:14 2016

@author: dan
"""

import numpy as np
import scipy.sparse as sp
from py_tools.stats import walker

def get_unit_vecs(P, tol=1e-8, normalize=False, **kwargs):
    """Compute eigenvectors of a matrix corresponding to eigenvalues near 1.

    Parameters
    ----------
    P : array_like of shape (n, n)
        Square matrix whose eigenvectors are sought.
    tol : float, optional
        Tolerance for identifying unit eigenvalues; eigenvalues satisfying
        ``|lambda - 1| < tol`` are selected. Default is 1e-8.
    normalize : bool, optional
        If True, each selected eigenvector is rescaled so its elements sum
        to 1 (useful for obtaining probability distributions). Default is
        False.
    **kwargs
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    unit_vecs : ndarray of shape (n, k)
        Real parts of the eigenvectors whose corresponding eigenvalues are
        within *tol* of 1.0. Columns correspond to distinct unit
        eigenvectors.
    """
    
    vals, vecs = np.linalg.eig(P)
    unit = np.abs(vals - 1.0) < tol
    unit_vecs = np.real(vecs[:, unit])
    
    if normalize:
        unit_vecs /= np.sum(unit_vecs, axis=0)
        
    return unit_vecs

def stationary_doubling(P, tol=1e-12, pi_seed=None, maxit=500, normalize=True, **kwargs):
    """Find the invariant distribution of a Markov chain by forward iteration.

    Uses a doubling algorithm: at each step the candidate distribution is
    propagated through the current power of *P*, and *P* is squared, so
    convergence is super-linear in practice.

    Parameters
    ----------
    P : array_like of shape (n, n)
        Row-stochastic transition matrix of the Markov chain.
    tol : float, optional
        Convergence tolerance on the sup-norm change in the distribution
        between successive iterates. Default is 1e-12.
    pi_seed : array_like of shape (n,) or None, optional
        Initial distribution.  If None a uniform distribution is used.
    maxit : int, optional
        Maximum number of iterations before raising a ``ValueError``.
        Default is 500.
    normalize : bool, optional
        If True, the converged distribution is rescaled to sum exactly to 1.
        Default is True.
    **kwargs
        Additional keyword arguments are accepted but ignored.

    Returns
    -------
    pi : ndarray of shape (n,)
        Stationary (invariant) distribution of the Markov chain.

    Raises
    ------
    ValueError
        If convergence is not achieved within *maxit* iterations.
    """
    
    if pi_seed is None:
        pi = np.ones(P.shape[0]) / P.shape[0]
    else:
        pi = pi_seed
        
    Pj = P.copy()

    for it in range(maxit):
        pi_new = pi @ Pj
        Pj = Pj @ Pj
        dist = np.max(np.abs(pi_new - pi))
        if dist < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new
    
    if normalize:
        pi = pi / np.sum(pi)

    return pi

def ergodic_dist(P, doubling=False, **kwargs):
    """Compute the ergodic (stationary) distribution of a Markov chain.

    Parameters
    ----------
    P : array_like of shape (n, n)
        Row-stochastic transition matrix of the Markov chain.
    doubling : bool, optional
        If True, use :func:`stationary_doubling` (forward iteration /
        doubling algorithm).  If False (default), compute the left unit
        eigenvector of *P* via :func:`get_unit_vecs` applied to ``P.T``.
    **kwargs
        Additional keyword arguments are forwarded to the selected solver.

    Returns
    -------
    pi : ndarray of shape (n,)
        Ergodic distribution of the Markov chain, normalised to sum to 1.
    """
    
    if doubling:
        return stationary_doubling(P, **kwargs)
    else:
        return get_unit_vecs(P.T, normalize=True, **kwargs) 

def check_ergodic(invariant, tol=1e-8):
    """Check whether a distribution matrix is ergodic.

    An invariant distribution matrix is ergodic if all rows are identical,
    i.e. the long-run distribution is independent of the starting state.

    Parameters
    ----------
    invariant : ndarray of shape (m, n)
        Matrix of candidate invariant distributions, one per row.
    tol : float, optional
        Tolerance for comparing rows to the first row.  Default is 1e-8.

    Returns
    -------
    is_ergodic : bool
        True if all rows of *invariant* agree with the first row to within
        *tol* (sup-norm); False otherwise.
    """
    
    invariant_test = invariant - invariant[0, :]
    is_ergodic = np.amax(np.abs(invariant_test)) < tol
    return is_ergodic

def markov_std(P, vals):
    """Compute the conditional standard deviation under a Markov transition.

    For each state *i*, computes the standard deviation of *vals* under the
    conditional distribution given by row *i* of *P*.

    Parameters
    ----------
    P : array_like of shape (n, n)
        Row-stochastic transition matrix of the Markov chain.
    vals : array_like of shape (n,)
        Values associated with each state.

    Returns
    -------
    sig : ndarray of shape (n,)
        Conditional standard deviation ``sqrt(E[v^2] - (E[v])^2)`` for
        each state, where expectations are taken under the corresponding
        row of *P*.
    """

    Ev = np.dot(P, vals)
    Ev2 = np.dot(P, vals ** 2)
    V = Ev2 - (Ev ** 2)
    sig = np.sqrt(V)

    return sig
    
def update_value(V):
    """Extract the optimal policy indices and values from a value function matrix.

    Parameters
    ----------
    V : ndarray of shape (n_states, n_choices)
        Value function array where entry ``V[i, j]`` is the value of taking
        choice *j* in state *i*.

    Returns
    -------
    indices : ndarray of int, shape (n_states,)
        Index of the maximising choice for each state.
    v : ndarray of shape (n_states,)
        Maximum value in each state, i.e. ``V[i, indices[i]]`` for each *i*.
    """
    
    indices = np.argmax(V, axis=1)
    v = V[np.arange(V.shape[0]), indices]
    return indices, v
    
def get_transition(indices, sparse=False):
    """Build a deterministic transition matrix from an array of policy indices.

    Constructs an ``(n, n)`` matrix where row *i* has a single 1 in column
    ``indices[i]`` and zeros elsewhere, representing a deterministic Markov
    transition implied by a policy function.

    Parameters
    ----------
    indices : array_like of int, shape (n,)
        Target state index for each current state.
    sparse : bool, optional
        If True, return a ``scipy.sparse.csr_matrix``; otherwise return a
        dense ``numpy.ndarray``. Default is False.

    Returns
    -------
    transition : ndarray or csr_matrix of shape (n, n)
        Row-stochastic transition matrix with exactly one 1 per row.
    """
    
    n = len(indices)
    
    if sparse:
        transition = sp.lil_matrix((n, n))
        transition = transition.tocsr()
    else:
        transition = np.zeros((n, n))
        
    transition[np.arange(n), indices] = 1
    return transition
    
def sim_discrete(P, N, i0=0):
    """Simulate a discrete Markov chain for N periods.

    Uses the Walker alias method for O(1) sampling at each step.

    Parameters
    ----------
    P : array_like of shape (n, n)
        Row-stochastic transition matrix of the Markov chain.
    N : int
        Number of periods to simulate (length of the output sequence).
    i0 : int, optional
        Initial state index. Default is 0.

    Returns
    -------
    ix : ndarray of int, shape (N,)
        Sequence of state indices, with ``ix[0] == i0``.
    """
    
    samplers = [walker.WalkerRandomSampling(P[ii, :]) for ii in range(P.shape[0])]
    
    ix = np.zeros(N).astype(int)
    ix[0] = i0
    
    for ii in range(1, N):
        ix[ii] = samplers[ix[ii-1]].random()
    
    return ix

def sim_iid(p, N):
    """Simulate N i.i.d. draws from a discrete distribution.

    Parameters
    ----------
    p : array_like of shape (n,)
        Probability vector (must sum to 1).
    N : int
        Number of draws.

    Returns
    -------
    draws : ndarray of int, shape (N,)
        Sampled indices drawn independently from distribution *p*.
    """

    sampler = walker.WalkerRandomSampling(p)
    return sampler.random(N)

def sim_discrete_from_ergodic(P, N, pi_star=None):
    """Simulate a Markov chain starting from its ergodic distribution.

    The initial state is drawn at random from the ergodic (stationary)
    distribution, so the simulated path is stationary from period 0.

    Parameters
    ----------
    P : array_like of shape (n, n)
        Row-stochastic transition matrix of the Markov chain.
    N : int
        Number of periods to simulate.
    pi_star : array_like of shape (n,) or None, optional
        Pre-computed ergodic distribution.  If None it is computed from *P*
        via :func:`ergodic_dist`.

    Returns
    -------
    ix : ndarray of int, shape (N,)
        Sequence of state indices simulated from the stationary distribution.
    """

    if pi_star is None:
        pi_star = ergodic_dist(P)
        
    i0 = np.random.choice(len(pi_star), p=pi_star.ravel())

    return sim_discrete(P, N, i0)

def multi_choice(p):
    """Draw one sample from each of many probability vectors in parallel.

    Parameters
    ----------
    p : ndarray of shape (m, n)
        Matrix of probability vectors; each row must sum to 1 and represent
        one discrete distribution over *n* outcomes.

    Returns
    -------
    choices : ndarray of int, shape (m,)
        Index of the sampled outcome for each row of *p*.

    Notes
    -----
    Sampling is done via the inverse-CDF method using a single batch of
    uniform draws, making this efficient for large *m*.
    """

    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)

    return choices
    
def sim_policy(index_list, z_ix_sim, i0=0):
    """Simulate the endogenous state by following a policy function.

    At each period the policy maps the current endogenous state and the
    contemporaneous exogenous state to the next endogenous state index.

    Parameters
    ----------
    index_list : list of array_like
        Policy function stored as a list indexed by the exogenous state.
        ``index_list[z][x]`` gives the next endogenous state index when the
        exogenous state is *z* and the current endogenous state is *x*.
    z_ix_sim : array_like of int, shape (Nsim,)
        Simulated exogenous state indices for each period.
    i0 : int, optional
        Initial endogenous state index. Default is 0.

    Returns
    -------
    ix : ndarray of int, shape (Nsim,)
        Simulated endogenous state indices.
    """
    
    Nsim = len(z_ix_sim)
    ix = np.zeros(Nsim).astype(int)
    ix[0] = i0
    
    for ii in range(0, Nsim):
        if ii > 0:
            ix_old = ix[ii-1]
        else:
            ix_old = i0

        ix[ii] = index_list[z_ix_sim[ii]][ix_old]
        
    return ix

def sim_life_cycle(index_lists, z_ix_sim, i0=0):
    """Simulate the endogenous state for a life-cycle model.

    Like :func:`sim_policy` but the policy function varies with calendar
    time (age), so a different policy is applied at each period.

    Parameters
    ----------
    index_lists : list of list of array_like, length Nt
        Time-varying policy functions. ``index_lists[t][z][x]`` gives the
        next endogenous state index at time *t* when the exogenous state is
        *z* and the current endogenous state is *x*.
    z_ix_sim : array_like of int, shape (Nt,)
        Simulated exogenous state indices for each period.
    i0 : int, optional
        Initial endogenous state index. Default is 0.

    Returns
    -------
    ix : ndarray of int, shape (Nt,)
        Simulated endogenous state indices over the life cycle.
    """

    Nt = len(z_ix_sim)
    ix = np.zeros(Nt).astype(int)

    for tt in range(0, Nt):
        if tt > 0:
            ix_old = ix[tt-1]
        else:
            ix_old = i0

        ix[tt] = index_lists[tt][z_ix_sim[tt]][ix_old]

    return ix

def sim_ar1(rho, sig, mu=0.0, Nsim=100, e=None, x0=None):
    """Simulate an AR(1) process.

    The process follows ``x[t] = mu + rho * (x[t-1] - mu) + sig * e[t]``,
    or equivalently in demeaned form ``x[t] - mu = rho * (x[t-1] - mu) + sig * e[t]``.
    The initial value is drawn from the stationary distribution when *x0*
    is not provided.

    Parameters
    ----------
    rho : float
        Autoregressive coefficient. Must satisfy ``|rho| < 1`` for
        stationarity (required when *x0* is None).
    sig : float
        Standard deviation of the innovations.
    mu : float, optional
        Unconditional mean of the process. Default is 0.0.
    Nsim : int, optional
        Number of periods to simulate. Default is 100.
    e : array_like of shape (Nsim,) or None, optional
        Pre-drawn standard-normal innovations.  If None, drawn internally.
    x0 : float or None, optional
        Initial demeaned value of the process (i.e., the initial offset from
        *mu*).  If None, drawn from the stationary distribution.

    Returns
    -------
    x : ndarray of shape (Nsim,)
        Simulated AR(1) path including the mean *mu*.
    """

    x = np.zeros(Nsim)
    if e is None:
        e = np.random.randn(Nsim)

    if x0 is None:
        sig0 = sig / np.sqrt(1.0 - rho ** 2)
        x[0] = sig0 * e[0] 
    else:
        x[0] = x0

    for jj in range(1, Nsim):
        x[jj] = rho * x[jj-1] + sig * e[jj]

    x += mu
    return x

def sim_ar1_multi(rho, sig, Nper, Nsim=1, mu=0.0, e=None, x0=None):
    """Simulate multiple independent AR(1) processes simultaneously.

    Each of the *Nsim* processes follows ``x[t] = mu + rho*(x[t-1]-mu) + sig*e[t]``.

    Parameters
    ----------
    rho : float
        Autoregressive coefficient shared across all processes.
    sig : float
        Innovation standard deviation shared across all processes.
    Nper : int
        Number of periods per simulation.
    Nsim : int, optional
        Number of independent simulations. Default is 1.
    mu : float, optional
        Unconditional mean of the process. Default is 0.0.
    e : ndarray of shape (Nsim, Nper) or None, optional
        Pre-drawn standard-normal innovations.  If None, drawn internally.
    x0 : array_like of shape (Nsim,) or None, optional
        Initial values for each simulation.  If None, each process starts
        from the stationary distribution.

    Returns
    -------
    x : ndarray of shape (Nsim, Nper)
        Simulated AR(1) paths including the mean *mu*.
    """
    
    x = np.zeros((Nsim, Nper))
    if e is None:
        e = np.random.randn(Nsim, Nper)
    else:
        assert e.shape == (Nsim, Nper)
        
    if x0 is None:
        sig0 = sig / np.sqrt(1.0 - rho ** 2)
        x[:, 0] = sig0 * e[:, 0]
    else:
        assert len(x0) == Nsim
        
    for jj in range(1, Nper):
        x[:, jj] = rho * x[:, jj-1] + sig * e[:, jj]
        
    x += mu
    return x

def sim_cir(rho, sig, mu=0.0, Nsim=100, e=None, x0=None, bound=False):
    """Simulate a Cox-Ingersoll-Ross (CIR) process.

    The discrete-time CIR recursion is
    ``x[t] = (1 - rho)*mu + rho*x[t-1] + sqrt(|x[t-1]|) * sig * e[t]``,
    where the diffusion term scales with the square root of the current
    level.

    Parameters
    ----------
    rho : float
        Mean-reversion (persistence) coefficient.
    sig : float
        Volatility scaling parameter.
    mu : float, optional
        Long-run mean of the process. Default is 0.0.
    Nsim : int, optional
        Number of periods to simulate. Default is 100.
    e : array_like of shape (Nsim,) or None, optional
        Pre-drawn standard-normal innovations.  If None, drawn internally.
    x0 : float or None
        Initial value of the process.  **Required** (pass ``None`` only to
        trigger an informative exception); raises ``Exception`` if not
        provided.
    bound : bool, optional
        If True, take the absolute value of the process at each step to
        prevent negative values. Default is False.

    Returns
    -------
    x : ndarray of shape (Nsim,)
        Simulated CIR path.

    Raises
    ------
    Exception
        If *x0* is not provided.
    """

    x = np.zeros(Nsim)
    if e is None:
        e = np.random.randn(Nsim)

    if x0 is None:
        raise Exception("x0 must be provided for sim_cir")
    else:
        x[0] = x0

    if bound:
        x[0] = np.abs(x[0])

    for jj in range(1, Nsim):
        x[jj] = (1.0 - rho) * mu + rho * x[jj-1] + np.sqrt(np.abs(x[jj-1])) * sig * e[jj]
        if bound:
            x[jj] = np.abs(x[jj])

    return x

# Rouwenhorst approximation
def discrete_approx(rho, sig_e, N, cons=0.0):
    """Compute a Rouwenhorst N-point discrete approximation to an AR(1) process.

    Uses the Rouwenhorst (1995) method to construct a discrete Markov chain
    that matches the first two moments of the AR(1) process
    ``z[t] = rho * z[t-1] + sig_e * eps[t]``.

    Parameters
    ----------
    rho : float
        Autoregressive coefficient of the AR(1) process.
    sig_e : float
        Standard deviation of the i.i.d. innovation ``eps[t]``.
    N : int
        Number of grid points (states) in the approximation.
    cons : float, optional
        Constant added to all grid points (shifts the grid). Default is 0.0.

    Returns
    -------
    y : ndarray of shape (N,)
        Equally-spaced grid of state values, symmetric around *cons*.
    P : ndarray of shape (N, N)
        Row-stochastic transition matrix of the approximating Markov chain.

    Notes
    -----
    The grid half-width is ``sqrt(N-1) * sig_z`` where
    ``sig_z = sig_e / sqrt(1 - rho**2)`` is the unconditional standard
    deviation of the AR(1) process.
    """

    q = 0.5 * (1.0 + rho)
    sig_z = np.sqrt((sig_e ** 2)/(1 - rho ** 2))
    psi = np.sqrt(N - 1.0) * sig_z

    y = np.linspace(-psi, psi, N) + cons

    P = np.array([[q, 1.0 - q], [1.0 - q, q]])

    for ii in range(2, N):
        P_old = P
        P = np.zeros((ii+1, ii+1))

        P[:-1, :-1] += q * P_old
        P[:-1, 1:] += (1.0 - q) * P_old
        P[1:, :-1] += (1.0 - q) * P_old
        P[1:, 1:] += q * P_old

        P[1:-1, :] *= 0.5

    P /= np.sum(P, axis=1)

    return (y, P)
