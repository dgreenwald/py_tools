import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal as mvn

import numerical as nm

class StateSpace:

    def __init__(self, A, Q, Z, b, R=None):
        self.A = A # Transition matrix
        self.Q = Q # Shock covariance
        self.Z = Z # Measurement matrix
        self.b = b # Measurement mean
        self.R = R # Measurement err covariance

        self.nx = self.A.shape[0]
        self.ne = self.Q.shape[0]
        self.ny = self.Z.shape[0]

        if self.R is None:
            self.R = np.zeros((self.ny, self.ny))

    def filter(self, data, x0=None, P0=None):

        if x0 is None:
            x0 = np.zeros((self.nx, 1))

        if P0 is None:
            P0 = solve_discrete_lyapunov(self.A, self.Q)

        assert (data.shape[1] == self.ny)
        self.nt = data.shape[0]

        self.xpred = np.zeros((self.nt, self.nx))
        self.xfilt = np.zeros((self.nt, self.nx))

        self.Pfilt = np.zeros((self.nt, self.nx, self.nx))
        self.Ppred = np.zeros((self.nt, self.nx, self.nx))

        self.L = 0.0

        xfilt_t = x0.copy()
        Pfilt_t = P0.copy()

        for tt in range(self.nt):

            # Prediction step
            xpred_t = np.dot(self.A, x0)
            Ppred_t = quad_form(self.A, Pfilt_t) + self.Q

            ypred_t = np.dot(self.Z, xpred_t) + self.b
            Vpred_t = quad_form(self.Z, Ppred_t) + self.R

            # Update likelihood
            err_t = data[tt, :] - ypred_t
            self.L += mvn.logpdf(err_t, mean=ypred_t, cov=Vpred_t)

            # Filtering step
            K = np.dot(Ppred_t, np.linalg.solve(Vpred_t, self.Z).T)
            xfilt_t = xpred_t + np.dot(K, err_t)

            IKZ = np.eye(self.nx) - np.dot(K, self.Z)
            Pfilt_t = quad_form(IKZ, Ppred_t) + quad_form(K, self.R)

            self.xpred[tt, :] = xpred_t.ravel()
            self.xfilt[tt, :] = xfilt_t.ravel()

            self.Ppred[tt, :, :] = Ppred_t
            self.Pfilt[tt, :, :] = Pfilt_t

        return None

def quad_form(A, X):
    return np.dot(A, np.dot(X, A.T))
