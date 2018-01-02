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
        # self.x_bar = x_bar  # State values
        self.log_err_density = log_err_density  # Measurement error density function
        self.y_vals = make_2d(y_vals.copy())    # Data values

        # if x_grid is not None:
            # self.set_grid(x_grid)
        # else:
            # self.x_grid = None

        # self.Nx = len(self.x_bar)   # No. of states
        self.Nx = self.P.shape[0]
        self.Ny = self.y_vals.shape[0]  # No. of data series
        self.Nt = self.y_vals.shape[1]  # No. of observations

        self.px_filt = np.zeros((self.Nx, self.Nt))
        self.log_p_err = np.zeros((self.Nx, self.Nt))

    # def set_grid(x_grid):

        # self.x_grid = make_2d(x_grid.copy())

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
        self.px_filt = np.zeros((self.Nx, self.Nt))
        self.px_pred = np.zeros((self.Nx, self.Nt))

        px_pred_t = self.px_init
        for tt in range(self.Nt):
            # p_err = self.log_err_density(self.y_vals[:, tt], self.x_bar, tt)
            log_p_err_t = self.log_err_density(self.y_vals[:, tt], tt)
            log_py_all = log_p_err_t + np.log(px_pred_t)
            # log_py_marg = econ.lse(log_py_all)
            log_py_marg = logsumexp(log_py_all)
            
            px_filt_t = np.exp(log_py_all - log_py_marg)
            # px_pred_t = np.dot(px_filt_t, self.P.T)
            px_pred_t = np.dot(px_filt_t, self.P)
            
            self.log_p_err[:, tt] = log_p_err_t
            self.px_filt[:, tt] = px_filt_t
            self.px_pred[:, tt] = px_pred_t
            self.L += log_py_marg

        return None

    def smooth(self):

        self.px_smooth = np.zeros((self.Nx, self.Nt))
        # self.px_joint = np.zeros((self.Nx, self.Nx, self.Nt))

        for tt in range(self.Nt - 1, -1, -1):

            if tt < self.Nt - 1:
                px_joint_t = (self.P * self.px_filt[:, tt][:, np.newaxis]
                              * (px_smooth_old / self.px_pred[:, tt])[np.newaxis, :])
                px_smooth_t = np.sum(px_joint_t, axis=1)
                # self.px_joint[:, :, tt] = px_joint_t.copy()
            else:
                px_smooth_t = self.px_filt[:, -1]

            self.px_smooth[:, tt] = px_smooth_t 
            px_smooth_old = px_smooth_t.copy()

        return None

    def smoothed_vals(self, x_grid):

        x_grid_2d = make_2d(x_grid.copy())
        return np.dot(x_grid_2d, self.px_smooth)

    # def draw_smoothed(self, Nsim):

        # self.Nsim = Nsim
        # self.x_draws = np.zeros((self.Nx, self.Nt, self.Nsim)) 
        
        
