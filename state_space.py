import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn

import copy

from py_tools import numerical as nm, stats as st
from py_tools.utilities import tic, toc

def init_to_val(shape, val):

    x = np.empty(shape)
    x[:] = val
    return x

class StateSpaceModel:
    """State space model.
    
    Measurement equation: y_t = b + Z x_t + e_t, e_t ~ N(0, H_t)
    Transition equation: x_t = c + A x_{t-1} + R eps_t, eps_t ~ N(0, Q_t)

    Definitions (where y_t is an Ny x 1 vector):
        A: (Nx x Nx) transition matrix
        R: (Nx x Ne) shock impact matrix
        Q: (Ne x Ne) shock covariance matrix
        c: Length Nx transition equation constant vector
        Z: (Ny x Nx) measurement matrix
        b: Length Ny measurement equation constant vector 
    """

    def __init__(self, A, R, Q, Z, H, c=None, b=None):

        # Copy data
        self.A = A
        self.R = R
        self.Q = Q
        self.Z = Z
        self.H = H

        # Set sizes
        self.Nx, self.Ne = R.shape
        
        if self.Z is not None:
            self.Ny, _ = Z.shape
        else:
            self.Ny = 0

        # Check sizes
        assert all([var == self.Nx for var in [self.A.shape[0], self.A.shape[1]]])
        assert all([var == self.Ne for var in [self.Q.shape[0], self.Q.shape[1]]])

        # Set constant terms if needed
        if b is None:
            self.b = np.zeros(self.Ny)
        else:
            assert len(b) == self.Ny
            self.b = b

        if c is not None:
            assert len(c) == self.Nx
            x_bar = np.linalg.solve(np.eye(self.Nx) - self.A, c)
            self.b -= np.dot(Z, x_bar) 

        self.QR = self.Q @ self.R.T
        self.RQR = self.R @ self.QR
        
        self.CQT = nm.robust_cholesky(self.Q, min_eig=0.0).T
        if H is not None:
            self.CHT = nm.robust_cholesky(self.H, min_eig=0.0).T
        
    def unconditional_cov(self, fixed_init=None):

        if fixed_init is None: fixed_init = []
        
        if not fixed_init:

            try: 
                return sp.linalg.solve_discrete_lyapunov(self.A, self.RQR)
            except:
                return None
            
        else:
            
            free_init = [ii for ii in range(self.Nx) if ii not in fixed_init]
            A_trunc = self.A[free_init, :][:, free_init]
            RQR_trunc = self.RQR[free_init, :][:, free_init]
            
            V_trunc = sp.linalg.solve_discrete_lyapunov(A_trunc, RQR_trunc)
            V_full = np.zeros((self.Nx, self.Nx))
            for itrunc, ifull in enumerate(free_init):
                V_full[ifull, free_init] = V_trunc[itrunc, :]
            
            return V_full

    def simulate(self, x_1=None, Nt=None, shocks=None, meas_err=None, ix=None,
                 use_b=True):

        if shocks is None:
            if meas_err is None:
                assert Nt is not None
            else:
                Nt = meas_err.shape[0]
        else:
            Nt = shocks.shape[0] + 1

        if shocks is None:
            # CQ = np.linalg.cholesky(self.Q)
            # shocks = np.dot(CQ, np.random.randn(self.Ne, Nt)).T
            shocks = self.draw_shocks(Nt - 1)
        
        if meas_err is None:
            # CH = np.linalg.cholesky(self.H)
            # meas_err = np.dot(CH, np.random.randn(self.Ny, Nt)).T
            meas_err = self.draw_meas_err(Nt)

        assert shocks.shape == (Nt - 1, self.Ne)
        assert meas_err.shape == (Nt, self.Ny)

        if x_1 is None:
            Sig0 = self.unconditional_cov()
            C = np.linalg.cholesky(Sig0)
            x_1 = np.dot(C, np.random.randn(self.Nx))

        if ix is None:
            ix = np.ones((Nt, self.Ny), dtype=bool)
        else:
            assert ix.shape == (Nt, self.Ny)
            
        y_sim = init_to_val((Nt, self.Ny), np.nan)
        x_sim = np.zeros((Nt, self.Nx))

        x_t = x_1

        for tt in range(Nt):

            ix_t = ix[tt, :]
            y_sim[tt, ix_t] = np.dot(self.Z[ix_t, :], x_t) + meas_err[tt, ix_t]
            if use_b: 
                y_sim[tt, ix_t] += self.b[ix_t]
            x_sim[tt, :] = x_t

            if tt < Nt - 1:
                x_t = np.dot(self.A, x_t) + np.dot(self.R, shocks[tt, :])

        return (y_sim, x_sim)

    def draw_shocks(self, Nt):
        return np.dot(np.random.randn(Nt, self.Ne), self.CQT)

    def draw_meas_err(self, Nt):
        return np.dot(np.random.randn(Nt, self.Ny), self.CHT)

    def decompose_by_shock(self, shocks, states, start_ix=0):

        shock_components_samp, det_component_samp = self.decompose_by_shock_init(
            shocks[start_ix:, :], states[start_ix, :]
        )

        if start_ix > 0:
            shock_components = np.concatenate((np.zeros((self.Ne, start_ix, self.Nx)),
                                          shock_components_samp), axis=1)
            det_component = np.concatenate((states[:start_ix, :], 
                                          det_component_samp), axis=0)
            return shock_components, det_component
        else:
            return shock_components_samp, det_component_samp

    def decompose_by_shock_init(self, shocks, x1):

        Nt = shocks.shape[0] + 1

        # Compute deterministic component
        det_component = np.zeros((Nt, self.Nx))
        det_component[0, :] = x1
            
        for tt in range(1, Nt):
            det_component[tt, :] = self.A @ det_component[tt-1, :]

        # Compute shock components
        shock_components = np.zeros((Nt, self.Nx, self.Ne))
        Ri_shock_list = [shocks[:, ishock][:, np.newaxis] @ self.R[:, ishock][np.newaxis, :] 
                         for ishock in range(self.Ne)]
        R_shocks = np.concatenate([x[:, :, np.newaxis] for x in Ri_shock_list], axis=2)

        for tt in range(Nt - 1):
            shock_components[tt + 1, :, :] = self.A @ shock_components[tt, :, :] + R_shocks[tt, :, :]

        shock_components = np.moveaxis(shock_components, [0, 1, 2], [1, 2, 0])
        
        return shock_components, det_component

    def decompose_y_by_shock(self, shocks, states, y=None, start_ix=0):

        if y is None:
            y = states @ self.Z.T + self.b[np.newaxis, :]

        shock_components_samp, det_component_samp = self.decompose_by_shock_init(
            shocks[start_ix:, :], states[start_ix, :]
        )

        y_shock_components_samp = np.concatenate([
            shock_components_samp[ishock, :, :] @ self.Z.T
            for ishock in range(self.Ne)
        ])

        y_det_component_samp = det_component @ self.Z.T

        y_shock_only_samp = y_shock_components_samp + y_det_component_samp
        y_shock_removed_samp = y - y_shock_components_samp

        if start_ix > 0:

            this_y = y[np.newaxis, :start_ix, :]

            y_shock_only = np.concatenate((this_y, y_shock_only_samp), axis=0)
            y_shock_removed = np.concatenate((this_y, y_shock_removed_samp), axis=0)

            return y_shock_only, y_shock_removed

        else:

            return y_shock_only_samp, y_shock_removed_samp

    def decompose_y_by_state(self, states, y=None, start_ix=0):

        if y is None:
            y = states @ self.Z.T + self.b[np.newaxis, :]

        y_state_components_samp = np.concatenate([
            (states[start_ix:, istate] @ self.Z[:, istate].T)[np.newaxis, :, :]
            for istate in range(self.Nx)
        ], axis=0)

        y_state_only_samp = y_state_components_samp + self.b[np.newaxis, np.newaxis, :]
        y_state_removed_samp = y[np.newaxis, start_ix:, :] - y_state_components_samp

        if start_ix > 0:

            this_y = y[np.newaxis, :start_ix, :]

            y_state_only = np.concatenate((this_y, y_state_only_samp), axis=0)
            y_state_removed = np.concatenate((this_y, y_state_removed_samp), axis=0)

            return y_state_only, y_state_removed

        else:

            return y_state_only_samp, y_state_removed_samp

class StateSpaceEstimates:
    """Estimated states from applying state space model to particular dataset
    
    Associated with StateSpaceModel ssm
    """

    def __init__(self, ssm, y, x_init=None, P_init=None, fixed_init=None):

        if fixed_init is None: fixed_init = []

        # Fixed vars for initial condition
        self.fixed_init = fixed_init        

        self.set_data(y)
        self.set_ssm(ssm)

        # Data net of constant
        # self.y_til = self.y - self.ssm.b[np.newaxis, :]

        # Set initializations
        if x_init is None:
            x_init = np.zeros(self.Nx)

        if P_init is None:
            P_init = self.ssm.unconditional_cov(self.fixed_init)
            if P_init is None:
                self.valid = False # bad model
            else:
                self.valid = True

        self.x_init = x_init
        self.P_init = P_init
        
        # Item for smoother
        self.r = None

    def set_data(self, y):
        """Read in a new set of observables"""

        self.y = y
        self.Nt, self.Ny = self.y.shape
        self.ix = np.isfinite(self.y)

    def set_ssm(self, ssm):

        self.ssm = ssm
        self.Nx, self.Ne = self.ssm.R.shape
        assert self.Ny == ssm.Ny

        # Observables net of constant
        self.y_til = self.y - self.ssm.b[np.newaxis, :]

    def kalman_filter(self, x_init=None, P_init=None, overwrite_r=True,
                      y_til=None):
        """Run the Kalman filter on the data y.
        
        Inputs:

            y: (Nt x Ny) data matrix
            x_init: Length Nx mean of initial x distribution
            P_init: (Nx x Nx) covariance matrix of initial x distribution
        """

        if overwrite_r:
            self.r = None

        if x_init is not None:
            self.x_init = x_init
        if P_init is not None:
            self.P_init = P_init

        if y_til is None:
            self.base_data_results = True
            y_til = self.y_til
        else:
            self.base_data_results = False
            
        x_pred_t = self.x_init
        P_pred_t = self.P_init

        self.err = np.zeros((self.Nt, self.Ny))

#        self.x_filt = np.zeros((self.Nt, self.Nx))
#        self.P_filt = np.zeros((self.Nt, self.Nx, self.Nx))

        self.x_pred = np.zeros((self.Nt, self.Nx))
        self.P_pred = np.zeros((self.Nt, self.Nx, self.Nx))

        self.ZFi = np.zeros((self.Nt, self.Nx, self.Ny))
        self.K = np.zeros((self.Nt, self.Nx, self.Ny))
        self.G = np.zeros((self.Nt, self.Nx, self.Nx))

        self.log_like = 0.0 # log likelihood

        for tt in range(self.Nt):

            self.x_pred[tt, :] = x_pred_t
            self.P_pred[tt, :, :] = P_pred_t

            # Get error and update likelihood
            ix_t = self.ix[tt, :]
            Z_t = self.ssm.Z[ix_t, :]
            err_t = y_til[tt, ix_t] - np.dot(Z_t, x_pred_t)

            H_t = self.ssm.H[ix_t, :][:, ix_t]
            PZ = np.dot(P_pred_t, Z_t.T)
            F_t = np.dot(Z_t, PZ) + H_t
            
            try:
                self.log_like += mvn.logpdf(err_t, mean=np.zeros(np.sum(ix_t)), cov=F_t) 
            except:
                self.log_like = -1e+10
                return None
            
            # Update step (DK style)
            ZFi_t = nm.rsolve(Z_t.T, F_t)
            AP_t = np.dot(self.ssm.A, P_pred_t)
            K_t = np.dot(AP_t, ZFi_t)
            G_t = self.ssm.A - np.dot(K_t, Z_t)
            
            x_pred_t = np.dot(self.ssm.A, x_pred_t) + np.dot(K_t, err_t)
            P_pred_t = np.dot(AP_t, G_t.T) + self.ssm.RQR

            """OLD VERSION: STORE FILTERED RESULTS SEPARATELY"""            
            # Note: "K" in DK notation is AK. "L" is AG.
#            K_t = np.dot(P_pred_t, ZFi_t)
#            G_t = np.eye(self.Nx) - np.dot(K_t, self.ssm.Z[ix_t, :])

#            x_filt = x_pred_t + np.dot(K_t, err_t)
#            P_filt = np.dot(P_pred_t, G_t.T)

            # Save values
            self.ix[tt, :] = ix_t
            self.err[tt, ix_t] = err_t

#            self.x_filt[tt, :] = x_filt
#            self.P_filt[tt, :, :] = P_filt

            self.ZFi[tt, :, ix_t] = ZFi_t.T
            self.K[tt, :, ix_t] = K_t.T
            self.G[tt, :, :] = G_t

            # Update for next period
#            x_pred_t = np.dot(self.ssm.A, x_filt)
#            P_pred_t = np.dot(self.ssm.A, np.dot(P_filt, self.ssm.A.T)) + self.ssm.RQR

        return None

    def disturbance_smoother(self):

        self.r = np.zeros((self.Nt, self.Nx))

        r_t = np.zeros(self.Nx)

        for tt in range(self.Nt - 1, -1, -1):

            ix_t = self.ix[tt, :]
            r_t = (np.dot(self.ZFi[tt, :, ix_t].T, self.err[tt, ix_t]) 
                   + np.dot(self.G[tt, :, :].T, r_t))
            
            self.r[tt, :] = r_t

        return None

    def state_smoother(self, disturbance_smooth=False):
        
        if (self.r is None) or (disturbance_smooth):
            self.disturbance_smoother()

        self.x_smooth = np.zeros((self.Nt, self.Nx))

        # Initial value
        self.x_smooth[0, :] = self.x_init + np.dot(self.P_init, self.r[0, :])

        # Recursively compute later values
        for tt in range(1, self.Nt):

            self.x_smooth[tt, :] = (np.dot(self.ssm.A, self.x_smooth[tt-1, :]) 
                                          + np.dot(self.ssm.RQR, self.r[tt, :]))

        return None

    def shock_smoother(self, disturbance_smooth=False):

        if (self.r is None) or disturbance_smooth:
            self.disturbance_smoother()

        # empty_row = np.empty((1, self.Ne))
        # empty_row.fill(np.nan)
        # self.shocks_smooth = np.vstack((empty_row, self.r[1:, :] @ self.ssm.QR.T))
            
        self.shocks_smooth = self.r[1:, :] @ self.ssm.QR.T
        return None

    def meas_err_smoother(self, disturbance_smooth=False):
        """NEED TO TEST THIS"""
        
        if (self.r is None) or disturbance_smooth:
            self.disturbance_smoother()

        self.meas_err_smooth = init_to_val((self.Nt, self.Ny), np.nan)

        # Recursively compute values
        for tt in range(self.Nt):

            ix_t = self.ix[tt, :]
            H_t = self.ssm.H[ix_t, :][:, ix_t]
            Z_t = self.ssm.Z[ix_t, :]
            F_t = (Z_t @ (self.P_pred[tt, :, :] @ Z_t.T))

            HFi_t = nm.rsolve(H_t, F_t)

            self.meas_err_smooth[tt, ix_t] = HFi_t @ self.err[tt, ix_t] - (H_t @ self.K[tt, :, ix_t]) @ self.r[tt, :]

        return None

    def draw_states(self, draw_shocks=False, draw_meas_err=False):

        # Draw shocks
        shocks = self.ssm.draw_shocks(self.Nt - 1)
        meas_err = self.ssm.draw_meas_err(self.Nt)

        # shocks = st.draw_norm_multi(self.ssm.Q, self.Nt)
        # meas_err = st.draw_norm_multi(self.ssm.H, self.Nt)
        x_1 = self.x_init + st.draw_norm(self.P_init)

        # Simulate using random draws
        y_plus, x_plus = self.ssm.simulate(
            x_1, shocks=shocks, meas_err=meas_err, ix=self.ix, use_b=False
        )

        # Create "star" observables as diff between old and random
        y_star = self.y_til - y_plus

        # Run Kalman filter and smoother
        self.kalman_filter(x_init=self.x_init, P_init=self.P_init, y_til=y_star)
        self.state_smoother(disturbance_smooth=True)
        
        # State draw is difference between random states and smoothed "star" states
        self.state_draw = x_plus + self.x_smooth

        if draw_shocks:
            self.shock_smoother()
            self.shock_draw = shocks + self.shocks_smooth

        if draw_meas_err:
            self.meas_err_smoother()
            self.meas_err_draw = meas_err + self.meas_err_smooth

        return None

    def get_shock_components(self, redraw_shocks=False, start_ix=0):

        # Set shock series to use
        if redraw_shocks or (self.shock_draw is None):
            self.state_smoother(draw_shocks=True)

        return self.ssm.decompose_by_shock(self.shock_draw, self.state_draw,
                                           start_ix=start_ix)

