import ipdb
import numpy as np
from scipy.special import gammaln
from scipy.stats import multivariate_normal, invwishart
import pandas as pd

import py_tools.time_series as ts

# def ols_likelihood(X, Y, S):

    # Nt, Nx = X.shape
    # _, Ny = Y.shape

    # L = (
        # - 0.5 * Ny * Nt * np.log(2.0 * np.pi)
        # - 0.5 * Ny * np.log(np.abs(np.linalg.det(xtx(X))))
        # - 0.5 * (Nt - Nx) * np.log(np.abs(np.linalg.det(S)))
        # + 0.5 * Ny * (Nt - Nx) * np.log(2.0)
        # + np.sum(gammaln(0.5 * (Nt - Nx - np.arange(Ny))))
    # )

    # return L

def fit_ols(X, Y):

    Phi = ts.least_sq(X, Y)
    e = Y - np.dot(X, Phi)
    Sig = xtx(e) / X.shape[0]

    return (Phi, Sig)

def post_mode(X, Y, b_bar, Om_inv_bar, Psi_bar):

    Nt, Nx = X.shape
    _, Ny = Y.shape

    p = (Nx - 1) / Ny

    # Posterior for B
    B_bar = np.reshape(b_bar, (Ny, Nx)).T
    XX = np.dot(X.T, X) + Om_inv_bar
    Xy = np.dot(X.T, Y) + np.dot(Om_inv_bar, B_bar)

    XX_inv = np.linalg.inv(XX)
    # b_hat = np.linalg.solve(XX, Xy)
    B_hat = np.dot(XX_inv, Xy)

    # Posterior for Sig
    eps_hat = (Y - np.dot(X, B_hat))
    B_diff = B_hat - B_bar
    Psi_hat = Psi_bar + np.dot(eps_hat.T, eps_hat) + ts.quad_form(B_diff, Om_inv_bar)
    df_hat = Nt - p + Ny + 2

    # Marginal likelihood
    ipdb.set_trace()
    return (B_hat, XX_inv, Psi_hat, df_hat)

def draw_mniw(b_hat, XX_inv, Psi_hat, df_hat, Ny, Nx):

    Sig = invwishart.rvs(df_hat, scale=Psi_hat)
    V_b = np.kron(Sig, XX_inv)
    b = multivariate_normal.rvs(b_hat, V_b)
    # B = np.reshape(b, (Ny, Nx))

    return (b, Sig)

# def ols(X, y):

    # return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

def xtx(x):

    return np.dot(x.T, x)

def compute_irfs(B, Ny, p, Nirf, impact):

    B_comp = np.vstack(( 
        B[:p * Ny, :].T, 
        np.hstack((np.eye(Ny * (p - 1)), np.zeros((Ny * (p-1), Ny))))                 
    ))

    cc_dy = np.eye(Ny * p)

    msel = np.vstack((
        np.eye(Ny), np.zeros(((p-1)*Ny, Ny))
    ))

    virf = np.zeros((Nirf, Ny, Ny))

    for tt in range(Nirf):
        virf[tt, :, :] = np.dot(ts.quad_form(msel, cc_dy), impact)
        cc_dy = np.dot(B_comp, cc_dy)

    return virf # tt x var x shock

def mniw_prior(params, Ny, Nx, p):

    lam = params[0]
    psi = params[1:]

    B_bar_T = np.zeros((Ny, Nx))
    B_bar_T[:Ny, :Ny] = np.eye(Ny)
    b_bar = B_bar_T.flatten()

    om_inv_diag_1 = psi / (lam ** 2)
    om_inv_diag_mat = np.zeros((p, Ny))
    for ss in range(p):
        om_inv_diag_mat[ss, :] = om_inv_diag_1 * ((ss + 1) ** 2)
    
    om_inv_diag = np.hstack((om_inv_diag_mat.flatten(), 0.0))
    Om_inv_bar = np.diag(om_inv_diag)

    return (b_bar, Om_inv_bar)

def co_persistence_prior(params, Nx, Ny, p, ybar, sbar):

    mu, delta = params

    Nt_star = Ny + 1
    X_star = np.zeros((Nt_star, Nx))
    Y_star = np.zeros((Nt_star, Ny))

    for ii in range(Ny):

        ix = Ny * np.arange(p) + ii
        Y_star[ii, ii] = ybar[ii] / mu
        X_star[ii, ix] = ybar[ii] / mu

        # Co-persistence observations
        ix = Ny * np.arange(p) + ii
        Y_star[Ny, ii] = ybar[ii] / delta
        X_star[Ny, ix] = ybar[ii] / delta

    X_star[-1, Ny * p] = 1.0 / delta

    return (X_star, Y_star)

def mn_prior(lam, Nx, Ny, p, rwlist, ybar, sbar):

    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = lam

    lam3 = int(lam3) # Must be integer
    Nt_star = (p + lam3 + 1) * Ny + 1 # number of dummy observations

    X_star = np.zeros((Nt_star, Nx))
    Y_star = np.zeros((Nt_star, Ny))

    for ii in range(Ny):

        rw = rwlist[ii]
        not_rw = 1.0 - rw

        # Prior on first lag: unit root
        Y_star[ii, ii] = sbar[ii] * (rw + lam8 * not_rw) / lam1
        X_star[ii, ii] = sbar[ii] / lam1

        # Prior on other lags: zero
        for jj in range(1, p):
            X_star[jj * Ny + ii, jj * Ny + ii] = sbar[ii] * ((jj+1) ** (lam2 * (1.0 + lam7 * not_rw))) / lam1

        offset = Ny * p

        # Covariance diagonal elements
        if lam3 > 0:
            ix = Ny * np.arange(lam3) + offset + ii
            Y_star[ix, ii] = sbar[ii] * lam6
            offset += Ny * lam3

        # Sum of coefficients observations
        ix = Ny * np.arange(p) + ii
        Y_star[offset + ii, ii] = ybar[ii] / lam4
        X_star[offset + ii, ix] = ybar[ii] / lam4

        offset += Ny

        # Co-persistence observations
        ix = Ny * np.arange(p) + ii
        Y_star[offset, ii] = ybar[ii] / lam5
        X_star[offset, ix] = ybar[ii] / lam5

    X_star[-1, Ny * p] = 1.0 / lam5

    return (X_star, Y_star)

class BVAR:
    """Bayesian VAR"""

    def __init__(self, df_in, y_vars, p=1, hyperparams_init=None, rwlist=None, glp_prior=False):

        # Copy data
        self.y_vars = y_vars
        self.Ny = len(self.y_vars)
        self.p = p 
        self.glp_prior = glp_prior

        self.df = df_in[self.y_vars].copy()
        self.df['const'] = 1

        # Construct lags
        self.x_vars = []
        for lag in range(1, self.p + 1):
            for var in self.y_vars:
                self.x_vars += ts.transform(self.df, [var], lag=lag)

        self.x_vars += ['const']

        # Cut down to common indices
        self.ix = np.all(pd.notnull(self.df[self.y_vars + self.x_vars]), axis=1)
        self.Y = self.df.loc[self.ix, self.y_vars].values
        self.X = self.df.loc[self.ix, self.x_vars].values

        self.Nt, self.Nx = self.X.shape

        # Prepare for prior
        if hyperparams_init is None:

            if self.glp_prior:

                self.hyperparams = np.hstack((
                    np.array((1.0, 1.0, 0.2)), (0.02 ** 2) * np.ones(self.Ny)
                ))

            else:

                self.hyperparams = np.array((0.2, 1.0, 1, 1.0, 1.0, 1.0, 0.0, 1.0))

        else:

            self.hyperparams = hyperparams_init

        self.ybar = np.mean(self.Y, axis=0)
        self.sbar = np.std(self.Y, axis=0)

        # Right now only needed for MN prior
        if not self.glp_prior:

            if rwlist is None:
                self.rwlist = np.ones(self.Ny)
            else:
                self.rwlist = rwlist

    def add_prior(self, compute_ols=True):

        if self.glp_prior:

            self.X_star, self.Y_star = co_persistence_prior(self.hyperparams[:2], self.Nx, self.Ny, 
                                                            self.p, self.ybar, self.sbar)

            self.b_bar, self.Om_inv_bar = mniw_prior(self.hyperparams[2:], self.Ny, self.Nx, self.p)
            self.Psi_bar = np.diag(self.hyperparams[3:])

        else: # standard MN prior

            self.X_star, self.Y_star = mn_prior(self.hyperparams, self.Nx, self.Ny, self.p, 
                                              self.rwlist, self.ybar, self.sbar)

            self.b_bar = np.zeros((self.Nx * self.Ny))
            self.Om_inv_bar = np.zeros((self.Nx, self.Nx))
            self.Psi_bar = np.zeros((self.Ny, self.Ny))

        self.Nt_star = self.X_star.shape[1]

        self.X_all = np.vstack((self.X_star, self.X))
        self.Y_all = np.vstack((self.Y_star, self.Y))

        if compute_ols:
            self.Phi_star, self.Sig_star = fit_ols(self.X_star, self.Y_star)
            self.Phi_hat, self.Sig_hat = fit_ols(self.X_all, self.Y_all)

        self.Nt_all = self.X_all.shape[1]

        return None

    def fit(self):

        self.B_hat, self.XX_inv, self.Psi_hat, self.df_hat = post_mode(
            self.X_all, self.Y_all, self.b_bar, self.Om_inv_bar, self.Psi_bar
        )
        self.b_hat = self.B_hat.T.flatten()

        return None

    def sample(self, Nsim=1000):

        self.Nsim = Nsim
        self.B_sim = np.zeros((self.Nsim, self.Nx, self.Ny))
        self.Sig_sim = np.zeros((self.Nsim, self.Ny, self.Ny))

        for jj in range(self.Nsim):

            b_t, Sig_t = draw_mniw(self.b_hat, self.XX_inv, self.Psi_hat, self.df_hat,
                                   self.Ny, self.Nx)

            self.B_sim[jj, :, :] = np.reshape(b_t, (self.Ny, self.Nx)).T
            self.Sig_sim[jj, :, :] = Sig_t

        return None

    # def sample_glp(self, Nsim=1000, Nburn=1000):

        # for jj in range(self.Nsim + Nburn):

            # b_t, Sig_t = draw_mniw(self.b_hat, self.XX_inv, self.Psi_hat, self.df_hat,
                                   # self.Ny, self.Nx)

            # jj_sim = jj - Nburn

        # return None

    def compute_irfs_sim(self, Nirf=41, impact_type='identity'):
        """ Computes IRFs from sampled parameters.

        impact_type should be 'identity' or 'cholesky'.
        """

        self.irf_sim = np.zeros((self.Nsim, Nirf, self.Ny, self.Ny))

        for jj in range(self.Nsim):

            if impact_type == 'cholesky':
                impact = np.linalg.cholesky(self.Sig_sim[jj, :, :])
            else:
                impact = np.eye(self.Ny)

            self.irf_sim[jj, :, :, :] = compute_irfs(self.B_sim[jj, :, :], self.Ny, self.p, Nirf, impact)

        return None

    # def marg_like(self):

        # L_all = ols_likelihood(self.X_all, self.Y_all, self.Sig_hat)
        # L_star = ols_likelihood(self.X_star, self.Y_star, self.Sig_star)
        
        # return L_all - L_star

    # def grid_search(self):

        # for lam in itertools.product(
                # np.linspace(0.1, 2.1, 5),
                # np.linspace(0.25, 1.75, 4),
                # [1],
                # np.arange(1, 7),
                # np.arange(1, 6),
                # np.linspace(0.25, 1.0, 4),
                # np.linspace(0.0, 3.0, 7),
                # np.linspace(0.0, 1.0, 3),
        # ):
