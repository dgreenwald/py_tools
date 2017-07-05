import ipdb
import numpy as np
from scipy.misc import logsumexp

from . import econ

class HiddenMarkov:

    def __init__(self, P, log_err_density, y_vals):
        """Initialize parameters of the HM model"""

        self.P = P  # Transition matrix
        # self.x_bar = x_bar  # State values
        self.log_err_density = log_err_density  # Measurement error density function
        self.y_vals = y_vals    # Data values

        if len(self.y_vals.shape) < 2:
            self.y_vals = self.y_vals[np.newaxis, :]

        # self.Nx = len(self.x_bar)   # No. of states
        self.Nx = self.P.shape[0]
        self.Ny = self.y_vals.shape[0]  # No. of data series
        self.Nt = self.y_vals.shape[1]  # No. of observations

        self.px_filt_storage = np.zeros((self.Nx, self.Nt))
        self.log_p_err_storage = np.zeros((self.Nx, self.Nt))

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
        self.px_filt_storage = np.zeros((self.Nx, self.Nt))

        px_pred = self.px_init
        for tt in range(self.Nt):
            # p_err = self.log_err_density(self.y_vals[:, tt], self.x_bar, tt)
            log_p_err = self.log_err_density(self.y_vals[:, tt], tt)
            log_py_all = log_p_err + np.log(px_pred)
            # log_py_marg = econ.lse(log_py_all)
            log_py_marg = logsumexp(log_py_all)
            
            px_filt = np.exp(log_py_all - log_py_marg)
            px_pred = np.dot(px_filt, self.P.T)
            
            self.log_p_err_storage[:, tt] = log_p_err
            self.px_filt_storage[:, tt] = px_filt
            self.L += log_py_marg
            
        return None
