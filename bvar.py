# import ipdb
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import multivariate_normal, invwishart, gamma, invgamma
import pandas as pd

import py_tools.time_series as ts
from py_tools.data import merge_date
from py_tools import vector_autoregression as vr

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

def log_abs_det(x):

    return np.log(np.abs(np.linalg.det(x)))

def fit_ols(X, Y):

    Phi = ts.least_sq(X, Y)
    e = Y - np.dot(X, Phi)
    Sig = xtx(e) / X.shape[0]

    return (Phi, Sig)

def eval_glp_hyperprior(hyperparams, gam_hyp_shape, gam_hyp_scale,
                        igam_hyp_shape, igam_hyp_scale):

    gam_params = hyperparams[:3]
    L = np.sum(gamma.logpdf(gam_params, gam_hyp_shape, scale=gam_hyp_scale))

    igam_params = hyperparams[3:]
    L += np.sum(invgamma.logpdf(igam_params, igam_hyp_shape, scale=igam_hyp_scale))

    return L

def post_mode(X, Y, b_bar, Om_inv_bar, Psi_bar, df_bar):

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
    ee_B_om = np.dot(eps_hat.T, eps_hat) + ts.quad_form(B_diff, Om_inv_bar)
    Psi_hat = Psi_bar + ee_B_om
    df_hat = Nt - p + df_bar

    # Marginal likelihood
    Om_bar = np.diag(1.0 / np.diagonal(Om_inv_bar)) 
    D_Om = np.linalg.cholesky(Om_bar)
    D_Psi = np.linalg.cholesky(np.linalg.inv(Psi_hat))

    D_Om_term = xtx(np.dot(X, D_Om))
    D_Psi_term = ts.quad_form(D_Psi, ee_B_om)

    evals_D_Om_term, _ = np.linalg.eig(D_Om_term)
    evals_D_Psi_term, _ = np.linalg.eig(D_Psi_term)
    
    L = (
        -0.5 * Ny * (Nt - p) * np.log(np.pi) 
        + gammaln(0.5 * (Nt - p + df_bar)) - gammaln(0.5 * df_bar)
        - 0.5 * (Nt - p) * log_abs_det(Psi_hat) 
        - 0.5 * Ny * np.log(np.abs(1.0 + np.prod(evals_D_Om_term)))
        - 0.5 * (Nt - p + df_bar) * np.log(np.abs(1.0 + np.prod(evals_D_Psi_term)))
    )

    return (B_hat, XX_inv, Psi_hat, df_hat, eps_hat, L)

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

def compute_irfs(B, p, Nirf, impact):

    Ny, Nshock = impact.shape
    B_comp = np.vstack(( 
        B[:p * Ny, :].T, 
        np.hstack((np.eye(Ny * (p - 1)), np.zeros((Ny * (p-1), Ny))))                 
    ))

    cc_dy = np.eye(Ny * p)

    msel = np.vstack((
        np.eye(Ny), np.zeros(((p-1)*Ny, Ny))
    ))

    virf = np.zeros((Nirf, Ny, Nshock))

    for tt in range(Nirf):
        virf[tt, :, :] = np.dot(ts.quad_form(msel, cc_dy), impact)
        cc_dy = np.dot(B_comp, cc_dy)

    return virf # tt x var x shock

def glp_hyperprior(Ny, gam_hyp_modes=None, gam_hyp_stds=None, 
                   igam_hyp_scale=None, igam_hyp_shape=None):

    # Gamma hyperprior
    if gam_hyp_modes is None:
        gam_hyp_modes = np.array((0.2, 1.0, 1.0))

    if gam_hyp_stds is None:
        gam_hyp_stds = np.array((0.4, 1.0, 1.0))

    # Solve quadratic
    b = -(2.0 + (gam_hyp_modes / gam_hyp_stds) ** 2)
    gam_hyp_shape = 0.5 * (-b + np.sqrt((b ** 2) - 4.0))
    gam_hyp_scale = gam_hyp_modes / (gam_hyp_shape - 1.0)

    # Inverse gamma hyperprior 
    if igam_hyp_shape is None:
        igam_hyp_shape = (0.02 ** 2) * np.ones(Ny)

    if igam_hyp_scale is None:
        igam_hyp_scale = (0.02 ** 2) * np.ones(Ny)

    return (gam_hyp_shape, gam_hyp_scale, igam_hyp_shape, igam_hyp_scale) 

def mniw_prior(params, Ny, Nx, p):

    lam = params[0]
    psi = params[1:]

    B_bar_T = np.zeros((Ny, Nx))
    B_bar_T[:Ny, :Ny] = np.eye(Ny)
    b_bar = B_bar_T.flatten()

    df_bar = Ny + 2

    om_inv_diag_1 = psi / ((df_bar - Ny - 1) * (lam ** 2))
    om_inv_diag_mat = np.zeros((p, Ny))
    for ss in range(p):
        om_inv_diag_mat[ss, :] = om_inv_diag_1 * ((ss + 1) ** 2)
    
    om_inv_diag = np.hstack((om_inv_diag_mat.flatten(), 1e-6))
    Om_inv_bar = np.diag(om_inv_diag)

    return (b_bar, Om_inv_bar, df_bar)

def co_persistence_prior(params, Nx, Ny, p, ybar):

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

                (self.gam_hyp_shape, self.gam_hyp_scale, 
                 self.igam_hyp_shape, self.igam_hyp_scale) = glp_hyperprior(self.Ny)

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

    def add_prior(self):

        if self.glp_prior:

            self.X_star, self.Y_star = co_persistence_prior(
                self.hyperparams[:2], self.Nx, self.Ny, self.p, self.ybar
            )

            self.b_bar, self.Om_inv_bar, self.df_bar = mniw_prior(
                self.hyperparams[2:], self.Ny, self.Nx, self.p
            )

            self.Psi_bar = np.diag(self.hyperparams[3:])

        else: # standard MN prior

            self.X_star, self.Y_star = mn_prior(self.hyperparams, self.Nx, self.Ny, self.p, 
                                              self.rwlist, self.ybar, self.sbar)

            self.b_bar = np.zeros((self.Nx * self.Ny))
            self.Om_inv_bar = np.zeros((self.Nx, self.Nx))
            self.Psi_bar = np.zeros((self.Ny, self.Ny))
            self.df_bar = 0.0

        self.Nt_star = self.X_star.shape[0]

        self.X_all = np.vstack((self.X_star, self.X))
        self.Y_all = np.vstack((self.Y_star, self.Y))

        # if compute_ols:
            # self.Phi_star, self.Sig_star = fit_ols(self.X_star, self.Y_star)
            # self.Phi_hat, self.Sig_hat = fit_ols(self.X_all, self.Y_all)

        self.Nt_all = self.X_all.shape[0]

        return None

    def fit(self):

        (self.B_hat, self.XX_inv, self.Psi_hat, self.df_hat, 
         self.eps_hat, _
         ) = self.eval_post_mode(self.X_all, self.Y_all)
        self.b_hat = self.B_hat.T.flatten()

        return None

    def eval_post_mode(self, X, Y):

        return post_mode(X, Y, self.b_bar, self.Om_inv_bar, 
                         self.Psi_bar, self.df_bar)

    def objfcn_glp(self, x):

        self.hyperparams = np.exp(x) 
        self.add_prior()

        _, _, _, _, _, L_like = self.eval_post_mode(self.X_all, self.Y_all)
        _, _, _, _, _, L_dummy = self.eval_post_mode(self.X_star, self.Y_star)

        L_prior = eval_glp_hyperprior(
            self.hyperparams, self.gam_hyp_shape, self.gam_hyp_scale,
            self.igam_hyp_shape, self.igam_hyp_scale
        )

        L_post = L_like - L_dummy + L_prior

        return L_post

    def glp_mode(self):

        fcn = lambda x: -self.objfcn_glp(x)
        x0 = np.log(self.hyperparams)

        res = minimize(fcn, x0, method='Nelder-Mead', options={'maxiter': 10000, 'disp' : True})

        print(res)
        print("Final hyperparameters:")
        print("mu = {0}".format(self.hyperparams[0]))
        print("delta = {0}".format(self.hyperparams[1]))
        print("lambda = {0}".format(self.hyperparams[2]))
        print("psi:")
        print(self.hyperparams[3:])

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

            self.irf_sim[jj, :, :, :] = compute_irfs(self.B_sim[jj, :, :], self.p, Nirf, impact)

        return None

    def add_instrument(self, df_new, policy_var, instrument):

        self.policy_var = policy_var
        self.instrument = instrument
        self.df = merge_date(self.df, df_new[[instrument]], how='outer')
        return None

    def compute_irfs_instrument(self, Nirf=41, exact_sigma=True):
        """Computes IRFs from sampled parameters using instrument
        to define structural shocks.
        """

        self.irf_sim = np.zeros((self.Nsim, Nirf, self.Ny, 1))
        df_sim = self.df.copy()

        for jj in range(self.Nsim):

            eps_hat = (self.Y_all - np.dot(self.X_all, self.B_sim[jj, :, :]))[self.Nt_star:, :]
            for ii, var in enumerate(self.y_vars):
                df_sim['u_' + var] = np.nan
                df_sim.loc[self.ix, ['u_' + var]] = eps_hat[:, ii]

            if exact_sigma:
                impact = vr.instrument_var(df_sim, self.y_vars, self.policy_var, self.instrument, Sig=self.Sig_sim[jj, :, :])
            else:
                impact = vr.instrument_var(df_sim, self.y_vars, self.policy_var, self.instrument)

            self.irf_sim[jj, :, :, :] = compute_irfs(self.B_sim[jj, :, :], self.p, Nirf, impact)

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
