import numpy as np
from scipy.linalg import ordqz

class KleinObj:
    """Solver for Klein system"""

    def __init__(self, A, B, C, Phi, Sig, n_pre):

        self.A = A.asmatrix()
        self.B = B.asmatrix()
        self.C = C.asmatrix()
        self.Phi = Phi.asmatrix()
        self.Sig = Sig.asmatrix()
        self.n_endog = self.A.shape[0]
        self.n_pre = n_pre
        self.n_uns = self.n_endog - self.n_pre
        self.n_exog = self.C.shape[1]

    def solve(self):

        S, T, alp, bet, Q, Z = ordqz(A, B, sort='ouc', output='complex')  

        # TODO: checks here

        Q_star = Q.H 
        Q1 = Q_star[:n_pre, :].asmatrix()
        Q2 = Q_star[n_pre:, :].asmatrix()

        Z11 = Z[:n_pre, :n_pre].asmatrix()
        Z12 = Z[:n_pre, n_pre:].asmatrix()
        Z21 = Z[n_pre:, :n_pre].asmatrix()
        Z22 = Z[n_pre:, n_pre:].asmatrix()

        S11 = S[:n_pre, :n_pre].asmatrix()
        S12 = S[:n_pre, n_pre:].asmatrix()
        S22 = S[n_pre:, n_pre:].asmatrix()

        T11 = T[:n_pre, :n_pre].asmatrix()
        T12 = T[:n_pre, n_pre:].asmatrix()
        T22 = T[n_pre:, n_pre:].asmatrix()

        G_xc = Z11 * (Z12.inverse())
        H_xc = Z11 * (np.solve(S11, T11)) * (Z11.inverse())

        self.G_x = np.real(G_xc)
        self.H_x = np.real(H_xc)

        PhiST = np.kron(Phi.T, S22) - np.kron(np.eye(self.n_exog), T22)
        q2C = (Q2 * C).flatten(order='F')
        M = np.solve(PhiST, q2C).reshape((self.n_uns, self.n_exog), order='F')

        N = (Z22 - Z21 * np.solve(Z11, Z12)) * M
        L = (-Z11 * np.solve(S11, T11) * np.solve(Z11, Z12) * M 
             + Z11 * np.solve(S11, T12 * M - S12 * M * Phi + Q1 - C)
             + Z12 * M * Phi)

        self.G_f = np.real(N)
        self.H_f = np.real(L)

        return None

class AimObj:
    """Solver for AIM system"""

    def __init__(self, H, nlead=1, tol=1e-10):
       
        self.H = H
        self.neq, self.hcols = H.shape
        self.periods = H.shape[1] / self.neq
        self.nlead = nlead
        self.nlag = self.periods - self.nlead - 1
        self.tol = tol
        # self.eig_bnd = 1.0

        self.left = range(self.hcols - self.neq)
        self.right = range(self.hcols - self.neq, self.hcols)

        self.iz = 0
        self.zrows = int(self.neq * self.nlead)
        self.zcols = int(self.neq * (self.periods - 1))
        self.Z = np.zeros((self.zrows, self.zcols))

    def solve(self):

        self.exact_shift()
        self.numeric_shift()
        self.build_companion()
        self.eigensystem()
        self.reduced_form()

        return None

    def shift_right(self, x):

        x_shift = np.zeros(x.shape)
        x_shift[:, self.neq:] = x[:, :-self.neq]

        return x_shift

    def shuffle(self, ix):

        nz = len(ix)
        self.Z[self.iz:self.iz + nz, :] = self.H[ix, self.left]
        self.H[ix, :] = self.shift_right(self.H[ix, :])
        self.iz += nz

        return None

    def exact_shift(self):

        zerorows = np.sum(np.abs(self.H[:, self.right]), axis=1) < self.tol
        while (np.any(zerorows) and self.iz < self.zrows):
            ix = np.arange(self.neq)[zerorows]
            self.shuffle(ix)
            # nz = np.sum(zerorows)
            # self.Z[self.iz:self.iz + nz, :] = H[zerorows, self.right]
            # self.H[zerorows, :] = self.shift_right(self.H[zerorows, :])
            # self.iz += nz
            zerorows = np.sum(np.abs(self.H[:, self.right]), axis=1) < self.tol

        return None

    def numeric_shift(self):

        q, r = np.linalg.qr(self.H[:, self.right])
        zerorows = np.abs(np.diag(r)) < self.tol
        while (np.any(zerorows) and self.iz < self.zrows):
            ix = np.arange(self.neq)[zerorows]
            self.H = np.dot(q.T, self.H)
            self.shuffle(ix)
            q, r = np.linalg.qr(self.H[:, self.right])
            zerorows = np.abs(np.diag(r)) < self.tol

        return None

    def build_companion(self):

        self.A = np.zeros((self.zcols, self.zcols))
        if self.zcols > self.neq:
            self.A[:-self.neq, self.neq:] = np.eye(self.zcols - self.neq)

        Gam = -np.linalg.solve(self.H[:, self.right], self.H[:, self.left])
        self.A[-self.neq:, :] = Gam

        # Delete inessential lags
        self.js = list(range(self.zcols))
        drop = np.sum(np.abs(self.A), axis=0) < self.tol
        while np.any(drop):
            ix = np.arange(len(drop))[drop]
            self.A = np.delete(self.A, ix, axis=0)
            self.A = np.delete(self.A, ix, axis=1)
            self.js = np.delete(self.js, ix)
            drop = np.sum(np.abs(self.A), axis=0) < self.tol

        return None

    def eigensystem(self):

        vals, vecs = np.linalg.eig(self.A.T)
        ix = np.flipud(np.argsort(np.abs(vals)))

        sorted_vals = vals[ix]
        sorted_vecs = vecs[:, ix]

        # self.lgroots = np.sum(np.abs(sorted_vals) > self.eig_bnd)

        self.Z[self.iz:, self.js] =  sorted_vecs[:, :(self.zrows - self.iz)].T

        return None

    def reduced_form(self):

        self.Z = -np.linalg.solve(self.Z[:, self.zcols - self.zrows:], self.Z[:, :self.zcols - self.zrows])
        self.B = self.Z[:self.neq, :self.neq * self.nlag]

        return None

