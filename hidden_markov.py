import numpy as np

from . import econ

class HiddenMarkov:

    def __init__(self, P, x_bar, err_density, y_vals):
        """Initialize parameters of the HM model"""

        self.P = P  # Transition matrix
        self.x_bar = x_bar  # State values
        self.err_density = err_density  # Measurement error density function
        self.y_vals = y_vals    # Data values

        if len(self.y_vals.shape) < 2:
            self.y_vals = self.y_vals[np.newaxis, :]

        self.Nx = len(self.x_bar)   # No. of states
        self.Ny = self.y_vals.shape[0]  # No. of data series
        self.Nt = self.y_vals.shape[1]  # No. of observations

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
            p_err = self.err_density(self.y_vals[:, tt], self.x_bar, tt)
            py_all = p_err * px_pred
            py_marg = np.sum(py_all)
            
            px_filt = py_all / py_marg
            px_pred = np.dot(px_filt, self.P.T)
            
            self.px_filt_storage[:, tt] = px_filt
            self.L += np.log(py_marg)
            
        return None
