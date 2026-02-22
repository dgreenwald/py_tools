import numpy as np
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

from py_tools import econ


def make_2d(x):
    """Ensure an array is at least 2-dimensional.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    x : np.ndarray
        Array with at least 2 dimensions; if 1-D, a new axis is prepended.
    """

    if len(x.shape) < 2:
        x = x[np.newaxis, :]

    return x


class HiddenMarkov:
    """Hidden Markov model with forward filtering and backward smoothing.

    Parameters
    ----------
    P : np.ndarray, shape (Nx, Nx)
        Transition probability matrix where P[i, j] is the probability of
        transitioning from state i to state j.
    log_err_density : callable
        Function that returns log measurement error densities given observed
        values and a time index.
    y_vals : np.ndarray
        Observed data values, shape (Nt,) or (Nt, Ny).
    sparse : bool, optional
        If True, use a sparse representation of the transition matrix.
        Default is False.
    tol : float, optional
        Threshold below which transition probabilities are set to zero when
        building the sparse matrix. Default is 1e-8.

    Attributes
    ----------
    Nx : int
        Number of hidden states.
    Nt : int
        Number of time periods.
    Ny : int
        Number of observable variables.
    px_filt : np.ndarray, shape (Nt, Nx)
        Filtered state probabilities.
    px_pred : np.ndarray, shape (Nt, Nx)
        Predicted state probabilities.
    px_smooth : np.ndarray, shape (Nt, Nx)
        Smoothed state probabilities (set after calling smooth()).
    L : float
        Log-likelihood (set after calling filter()).
    """

    def __init__(self, P, log_err_density, y_vals, sparse=False, tol=1e-8):
        """Initialize parameters of the hidden Markov model.

        Parameters
        ----------
        P : np.ndarray, shape (Nx, Nx)
            Transition probability matrix.
        log_err_density : callable
            Log measurement error density function.
        y_vals : np.ndarray
            Observed data values, shape (Nt,) or (Nt, Ny).
        sparse : bool, optional
            If True, use sparse transition matrix. Default is False.
        tol : float, optional
            Sparsity threshold. Default is 1e-8.
        """

        self.P = P

        self.sparse = sparse
        if self.sparse:
            self.Ps = P.copy()
            self.Ps[self.Ps < tol] = 0
            self.Ps = self.Ps / np.sum(self.Ps, axis=1)[:, np.newaxis]
            self.Ps = csr_matrix(self.Ps)
        else:
            self.Ps = self.P

        self.log_err_density = log_err_density  # Measurement error density function
        self.y_vals = make_2d(y_vals.copy())  # Data values

        self.Nx = self.P.shape[0]
        self.Nt, self.Ny = self.y_vals.shape

        self.px_filt = np.zeros((self.Nt, self.Nx))
        self.log_p_err = np.zeros((self.Nt, self.Nx))

    def set_px_init(self, px_init):
        """Set the initial state probability distribution.

        Parameters
        ----------
        px_init : np.ndarray, shape (Nx,)
            Initial distribution over hidden states.
        """

        self.px_init = px_init
        return None

    def init_stationary(self):
        """Set the initial distribution to the stationary distribution of P."""

        self.px_init = np.squeeze(econ.ergodic_dist(self.P))
        return None

    def filter(self):
        """Run the forward filter to compute filtered state probabilities.

        Computes px_filt, px_pred, and the log-likelihood L.
        """

        self.L = 0.0
        self.px_filt = np.zeros((self.Nt, self.Nx))
        self.px_pred = np.zeros((self.Nt, self.Nx))

        px_pred_t = self.px_init
        for tt in range(self.Nt):
            log_p_err_t = self.log_err_density(self.y_vals[tt, :], tt)
            log_py_all = log_p_err_t + np.log(px_pred_t)
            log_py_marg = logsumexp(log_py_all)

            px_filt_t = np.exp(log_py_all - log_py_marg)
            px_pred_t = self.Ps.T.dot(px_filt_t)

            self.log_p_err[tt, :] = log_p_err_t
            self.px_filt[tt, :] = px_filt_t
            self.px_pred[tt, :] = px_pred_t
            self.L += log_py_marg

        return None

    def smooth(self):
        """Run the backward smoother to compute smoothed state probabilities.

        Requires filter() to have been called first.
        Computes px_smooth.
        """

        self.px_smooth = np.zeros((self.Nt, self.Nx))

        for tt in range(self.Nt - 1, -1, -1):
            if tt < self.Nt - 1:
                px_smooth_t = self.px_filt[tt, :] * (
                    self.Ps.dot(self.px_smooth[tt + 1, :] / self.px_pred[tt, :])
                )
            else:
                px_smooth_t = self.px_filt[-1, :]

            self.px_smooth[tt, :] = px_smooth_t

        return None

    def smoothed_vals(self, grid):
        """Compute expected values of grid points under smoothed state probabilities.

        Parameters
        ----------
        grid : np.ndarray
            Grid of values indexed by hidden state.

        Returns
        -------
        vals : np.ndarray
            Expected values, shape (Nt, Nvars).
        """

        grid_2d = make_2d(grid.copy())
        return np.dot(self.px_smooth, grid_2d.T)

    def filtered_vals(self, grid):
        """Compute expected values of grid points under filtered state probabilities.

        Parameters
        ----------
        grid : np.ndarray
            Grid of values indexed by hidden state.

        Returns
        -------
        vals : np.ndarray
            Expected values, shape (Nt, Nvars).
        """

        grid_2d = make_2d(grid.copy())
        return np.dot(self.px_filt, grid_2d.T)

    def sample(self, Nsim):
        """Draw state path samples using backward simulation.

        Parameters
        ----------
        Nsim : int
            Number of simulations to draw.
        """

        self.Nsim = Nsim
        self.ix_sample = np.zeros((self.Nt, self.Nsim), dtype=int)

        # Last period: draw from marginal
        self.ix_sample[-1, :] = np.random.choice(
            self.Nx, size=self.Nsim, p=self.px_smooth[-1, :]
        )

        # Now iterate backwards
        for tt in range(self.Nt - 2, -1, -1):
            ix_next = self.ix_sample[tt + 1, :]
            Pk = self.P[:, ix_next] / self.px_pred[tt, ix_next][np.newaxis, :]
            new_probs = self.px_filt[tt, :][:, np.newaxis] * Pk

            self.ix_sample[tt, :] = econ.multi_choice(new_probs.T)

        return None

    def sampled_vals(self, grid):
        """Compute grid values at sampled state indices.

        Parameters
        ----------
        grid : np.ndarray
            Grid of values indexed by hidden state.

        Returns
        -------
        vals : np.ndarray, shape (Nt, Nvars, Nsim)
            Grid values for each simulated path.
        """

        grid_2d = make_2d(grid.copy())

        Nvars = grid_2d.shape[0]
        vals = np.zeros((self.Nt, Nvars, self.Nsim))

        for tt in range(self.Nt):
            vals[tt, :, :] = grid_2d[:, self.ix_sample[tt, :]]

        return vals

    def smoothed_quantiles(self, grid, q):
        """Compute quantiles of a grid variable under the smoothed distribution.

        Parameters
        ----------
        grid : np.ndarray
            Grid of values indexed by hidden state.
        q : array-like
            Quantile levels to compute, each in [0, 1].

        Returns
        -------
        vals : np.ndarray, shape (len(q), Nvars, Nt)
            Quantile values at each time period.
        """

        grid_2d = make_2d(grid.copy())
        Nvars = grid_2d.shape[0]

        Fx_smooth = np.cumsum(self.px_smooth, axis=1)
        vals = np.zeros((len(q), Nvars, self.Nt))
        for ii in range(Nvars):
            for tt in range(self.Nt):
                vals[:, ii, tt] = np.interp(q, Fx_smooth[tt, :], grid_2d[ii, :])

        return vals
