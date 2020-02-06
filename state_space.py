import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn

import copy

from py_tools import numerical as nm, stats as st

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
        self.Ny, _ = Z.shape

        # Set constant terms if needed
        if b is None:
            self.b = np.zeros(self.Ny)
        else:
            self.b = b

        if c is not None:
            x_bar = np.linalg.solve(np.eye(self.Nx) - self.A, c)
            self.b -= np.dot(Z, x_bar) 

        self.QR = self.Q @ self.R.T
        self.RQR = self.R @ self.QR

        self.CQT = np.linalg.cholesky(self.Q).T
        self.CHT = nm.robust_cholesky(self.H, min_eig=0.0).T
        
    def unconditional_cov(self):
        
        try: 
            return sp.linalg.solve_discrete_lyapunov(self.A, self.RQR)
        except:
            return None

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

class StateSpaceEstimates:
    """Estimated states from applying state space model to particular dataset
    
    Associated with StateSpaceModel ssm
    """

    def __init__(self, ssm, y, x_init=None, P_init=None):

        self.y = y
        self.Nt, self.Ny = self.y.shape
        self.ix = np.isfinite(self.y)

        self.ssm = ssm
        self.Nx, _ = self.ssm.A.shape
        
        # Create version with no constant
        self.ssm_til = copy.deepcopy(ssm)
        self.ssm_til.b = np.zeros(ssm.b.shape)
        
        # Data net of constant
        self.y_til = self.y - self.ssm.b[np.newaxis, :]
        
        # Set initializations
        if x_init is None:
            x_init = np.zeros(self.Nx)

        if P_init is None:
            P_init = self.ssm.unconditional_cov()
            if P_init is None:
                self.valid = False # bad model
            else:
                self.valid = True

        self.x_init = x_init
        self.P_init = P_init
        
        # Item for smoother
        self.r = None

    def kalman_filter(self, x_init=None, P_init=None, overwrite_r=True):
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
            err_t = self.y_til[tt, ix_t] - np.dot(Z_t, x_pred_t)

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

        # empty_row = np.empty((1, self.ssm.Ne))
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

        # Create artificial observations
        y_star = self.y_til - y_plus

        # Get smoothed values
        sse_til = StateSpaceEstimates(self.ssm_til, y_star)
        sse_til.kalman_filter(x_init=self.x_init, P_init=self.P_init)
        # sse_til.disturbance_smoother()
        sse_til.state_smoother(disturbance_smooth=True)

        self.x_draw = x_plus + sse_til.x_smooth

        if draw_shocks:
            sse_til.shock_smoother()
            self.shock_draw = shocks + sse_til.shocks_smooth

        if draw_meas_err:
            sse_til.meas_err_smoother()
            self.meas_err_draw = meas_err + sse_til.meas_err_smooth

        return None
