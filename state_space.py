import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn
import sys

def init_to_val(shape, val):

    x = np.empty(shape)
    x[:] = val
    return x

def draw_norm(Sig):

    C = np.linalg.cholesky(Sig)
    return np.dot(C, np.random.randn(Sig.shape[0]))

def draw_norm_multi(Sig, n):

    C = np.linalg.cholesky(Sig)
    return np.dot(np.random.randn(n, Sig.shape[0]), C.T)

def rsolve(b, A):
    return np.linalg.solve(A.T, b.T).T

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

        self.RQR = np.dot(self.R, np.dot(self.Q, self.R.T))
        
    def unconditional_cov(self):
        
        return sp.linalg.solve_discrete_lyapunov(self.A, self.RQR)

    def simulate(self, x_1=None, Nt=None, shocks=None, ix=None):

        if shocks is None:
            assert Nt is not None
            CQ = np.linalg.cholesky(self.Q)
            shocks = np.dot(CQ, np.random.randn(self.Ne, Nt)).T
        else:
            assert Nt is None
            Nt, _ = shocks.shape
        
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
            y_sim[tt, ix_t] = np.dot(self.Z[ix_t, :], x_t) + self.b[ix_t]
            x_sim[tt, :] = x_t

            x_t = np.dot(self.A, x_t) + np.dot(self.R, shocks[tt, :])

        return (y_sim, x_sim)

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
        
        # Set initializations
        if x_init is None:
            x_init = np.zeros(self.Nx)

        if P_init is None:
            P_init = self.ssm.unconditional_cov()
##            if np.linalg.cond(self.ssm.Q) < 1e+16:
#            if True:
#                P_init = sp.linalg.solve_discrete_lyapunov(self.ssm.A, self.ssm.Q)
##            else:
##                print("Bad P_init")
##                raise Exception

        self.x_init = x_init
        self.P_init = P_init
        
        # Item for smoother
        self.r = None

    def kalman_filter(self):
        """Run the Kalman filter on the data y.
        
        Inputs:

            y: (Nt x Ny) data matrix
            x_init: Length Nx mean of initial x distribution
            P_init: (Nx x Nx) covariance matrix of initial x distribution
        """

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
            err_t = self.y[tt, ix_t] - np.dot(Z_t, x_pred_t) - self.ssm.b[ix_t]

            PZ = np.dot(P_pred_t, Z_t.T)
            F_t = np.dot(Z_t, PZ) + self.ssm.H[ix_t, ix_t]
            
            try:
                self.log_like += mvn.logpdf(err_t, mean=np.zeros(self.Ny), cov=F_t) 
            except:
                self.log_like = -1e+10
                return None
            
            # Update step (DK style)
            ZFi_t = rsolve(Z_t.T, F_t)
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
            self.err[tt, :] = err_t

#            self.x_filt[tt, :] = x_filt
#            self.P_filt[tt, :, :] = P_filt

            self.ZFi[tt, :, :] = ZFi_t
            self.K[tt, :, :] = K_t
            self.G[tt, :, :] = G_t

            # Update for next period
#            x_pred_t = np.dot(self.ssm.A, x_filt)
#            P_pred_t = np.dot(self.ssm.A, np.dot(P_filt, self.ssm.A.T)) + self.ssm.RQR

        return None

    def disturbance_smoother(self):

        self.r = np.zeros((self.Nt, self.Nx))

        r_t = np.zeros(self.Nx)

        for tt in range(self.Nt - 1, -1, -1):

            r_t = (np.dot(self.ZFi[tt, :, :], self.err[tt, :]) 
                   + np.dot(self.G[tt, :, :].T, r_t))
            
            self.r[tt, :] = r_t

        return None

    def state_smoother(self):
        
        if self.r is None:
            self.disturbance_smoother()

        self.x_smooth = np.zeros((self.Nt, self.Nx))

        # Initial value
        self.x_smooth[0, :] = self.x_init + np.dot(self.P_init, self.r[0, :])

        # Recursively compute later values
        for tt in range(1, self.Nt):

            self.x_smooth[tt, :] = (np.dot(self.ssm.A, self.x_smooth[tt-1, :]) 
                                          + np.dot(self.ssm.RQR, self.r[tt, :]))

        return None
    
    def draw_states(self):

        # Draw shocks
        shocks = draw_norm_multi(self.Q, self.Nt)
        x_1 = self.x_init + draw_norm(self.P_init)

        # Simulate using random draws
        y_plus, x_plus = self.ssm.simulate(x_1, shocks, self.ix)

        # Create artificial observations
        y_star = self.y - y_plus

        # Get smoothed values
        sse = StateSpaceEstimates(self, y_star)
        sse.kalman_filter(self.x_init, self.P_init)
        sse.disturbance_smoother()
        sse.state_smoother()

        return x_plus + sse.x_smooth