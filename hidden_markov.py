# import ipdb
import numpy as np
from scipy.misc import logsumexp

from . import econ

def make_2d(x):

    if len(x.shape) < 2:
        x = x[np.newaxis, :]

    return x

class HiddenMarkov:

    def __init__(self, P, log_err_density, y_vals):
        """Initialize parameters of the HM model"""

        self.P = P  # Transition matrix
        self.log_err_density = log_err_density  # Measurement error density function
        self.y_vals = make_2d(y_vals.copy())    # Data values

        self.Nx = self.P.shape[0]
        self.Nt, self.Ny = self.y_vals.shape 

        self.px_filt = np.zeros((self.Nt, self.Nx))
        self.log_p_err = np.zeros((self.Nt, self.Nx))

    def set_px_init(self, px_init):
        """Setter function for initial distribution"""

        self.px_init = px_init
        return None

    def init_stationary(self):
        """Set initial distribution to stationary distribution"""

        self.px_init = np.squeeze(econ.ergodic_dist(self.P))
        return None

    def filter(self):
        """Filter data"""

        self.L = 0.0
        self.px_filt = np.zeros((self.Nt, self.Nx))
        self.px_pred = np.zeros((self.Nt, self.Nx))

        px_pred_t = self.px_init
        for tt in range(self.Nt):
            # p_err = self.log_err_density(self.y_vals[tt, :], self.x_bar, tt)
            log_p_err_t = self.log_err_density(self.y_vals[tt, :], tt)
            log_py_all = log_p_err_t + np.log(px_pred_t)
            # log_py_marg = econ.lse(log_py_all)
            log_py_marg = logsumexp(log_py_all)
            
            px_filt_t = np.exp(log_py_all - log_py_marg)
            # px_pred_t = np.dot(px_filt_t, self.P.T)
            px_pred_t = np.dot(px_filt_t, self.P)
            
            self.log_p_err[tt, :] = log_p_err_t
            self.px_filt[tt, :] = px_filt_t
            self.px_pred[tt, :] = px_pred_t
            self.L += log_py_marg

        return None

    def smooth(self):

        self.px_smooth = np.zeros((self.Nt, self.Nx))
        # self.px_joint = np.zeros((self.Nt, self.Nx, self.Nx))

        for tt in range(self.Nt - 1, -1, -1):

            if tt < self.Nt - 1:
                px_joint_t = (self.P * self.px_filt[tt, :][:, np.newaxis]
                              * (self.px_smooth[tt+1, :] / self.px_pred[tt, :])[np.newaxis, :])
                px_smooth_t = np.sum(px_joint_t, axis=1)
                # self.px_joint[tt, :, :] = px_joint_t
            else:
                px_smooth_t = self.px_filt[-1, :]

            self.px_smooth[tt, :] = px_smooth_t 

        return None

    def smoothed_vals(self, grid):

        grid_2d = make_2d(grid.copy())
        return np.dot(self.px_smooth, grid_2d.T)
    
    def filtered_vals(self, grid):

        grid_2d = make_2d(grid.copy())
        return np.dot(self.px_filt, grid_2d.T)

    def sample(self, Nsim):

        self.Nsim = Nsim
        self.ix_sample = np.zeros((self.Nt, self.Nsim), dtype=np.int) 

        # Last period: draw from marginal
        self.ix_sample[-1, :] = np.random.choice(self.Nx, size=self.Nsim,
                                                p=self.px_smooth[-1, :])
        
        # Now iterate backwards
        for tt in range(self.Nt - 2, -1, -1):
            px_joint_t = (self.P * self.px_filt[tt, :][:, np.newaxis]
                          * (self.px_smooth[tt+1, :] / self.px_pred[tt, :])[np.newaxis, :])
            new_probs = (px_joint_t[:, self.ix_sample[tt+1, :]] / 
                         self.px_smooth[tt+1, self.ix_sample[tt+1, :]])

            self.ix_sample[tt, :] = econ.multi_choice(new_probs.T)

        return None

    def sampled_vals(self, grid):

        grid_2d = make_2d(grid.copy())

        Nvars = grid_2d.shape[0]
        vals = np.zeros((self.Nt, Nvars, self.Nsim))

        for tt in range(self.Nt):
            vals[tt, :, :] = grid_2d[:, self.ix_sample[tt, :]]

        return vals

    def smoothed_quantiles(self, grid, q):

        grid_2d = make_2d(grid.copy())
        Nvars = grid_2d.shape[0]

        Fx_smooth = np.cumsum(self.px_smooth, axis=0)
        vals = np.zeros((len(q), Nvars, self.Nt))
        for ii in range(Nvars):
            for tt in range(self.Nt):
                vals[:, ii, tt] = np.interp(q, Fx_smooth[tt, :], grid_2d[ii, :])

        return vals 
