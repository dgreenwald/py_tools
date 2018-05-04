# import ipdb
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import multivariate_normal, invwishart, gamma, invgamma
import pandas as pd
import re

from py_tools import data as dt, numerical as nm, time_series as ts
from py_tools.time_series import merge_date
from py_tools import vector_autoregression as vr
import py_tools.state_space as ss

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

    Phi = dt.least_sq(X, Y)
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
    ee_B_om = np.dot(eps_hat.T, eps_hat) + nm.quad_form(B_diff, Om_inv_bar)
    Psi_hat = Psi_bar + ee_B_om
    df_hat = Nt - p + df_bar

    # Marginal likelihood
    Om_bar = np.diag(1.0 / np.diagonal(Om_inv_bar)) 
    D_Om = np.linalg.cholesky(Om_bar)
    D_Psi = np.linalg.cholesky(np.linalg.inv(Psi_hat))

    D_Om_term = xtx(np.dot(X, D_Om))
    D_Psi_term = nm.quad_form(D_Psi, ee_B_om)

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
        virf[tt, :, :] = np.dot(nm.quad_form(msel, cc_dy), impact)
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
    for lag in range(p):
        om_inv_diag_mat[lag, :] = om_inv_diag_1 * ((lag + 1) ** 2)
    
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

def check_nan_var(var, data_augmentation_vars):

    for da in data_augmentation_vars:
        if re.match('(L\d*_)?' + da, var) is not None:
            return False

    return True

class BVAR:
    """Bayesian VAR"""

    def __init__(self, df_in, y_vars, p=1, hyperparams_init=None, rwlist=None, glp_prior=False,
                 data_augmentation_vars=[]):

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
        drop_nan_vars = [var for var in self.y_vars + self.x_vars 
                         if check_nan_var(var, data_augmentation_vars)]
        self.ix = np.all(pd.notnull(self.df[drop_nan_vars]), axis=1)
        self.Y_data = self.df.loc[self.ix, self.y_vars].values
        self.X_data = self.df.loc[self.ix, self.x_vars].values

        # Sample mean and std
        self.ybar = np.zeros(self.Ny)
        self.sbar = np.zeros(self.Ny)
        for ii in range(self.Ny):
            ix_i = np.isfinite(self.Y_data[:, ii])
            self.ybar[ii] = np.mean(self.Y_data[ix_i, ii], axis=0)
            self.sbar[ii] = np.std(self.Y_data[ix_i, ii], axis=0)

        # Separate values in case of data augmentation
        self.Y = self.Y_data
        self.X = self.X_data

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

    def glp_em(self, n_iter=5):

        Y_sim = np.zeros((n_iter,) + self.Y.shape)

        self.Y[np.isnan(self.Y)] = 0.0
        self.X[np.isnan(self.X)] = 0.0

        for ii in range(n_iter):

            Y_sim[ii, :, :] = self.Y

            print("\n\n\nITERATION {}\n\n\n".format(ii))

            fcn = lambda x: -self.objfcn_glp(x)
            x0 = np.log(self.hyperparams)
            res = minimize(fcn, x0, method='Nelder-Mead', options={'maxiter': 10000, 'disp' : True})

            print(res)
            print("Fitted hyperparameters:")
            print("mu = {0}".format(self.hyperparams[0]))
            print("delta = {0}".format(self.hyperparams[1]))
            print("lambda = {0}".format(self.hyperparams[2]))
            print("psi:")
            print(self.hyperparams[3:])

            B_hat, _, Psi_hat, df_hat, _, _ = self.eval_post_mode(self.X_all, self.Y_all)
            Sig_hat = Psi_hat / (df_hat + self.Ny + 1)

            self.augment_data(B_hat, Sig_hat, sample=False)

        return None

    def sample(self, Nsim=1000):

        self.Nsim = Nsim
        self.B_sim = np.zeros((self.Nsim, self.Nx, self.Ny))
        self.Sig_sim = np.zeros((self.Nsim, self.Ny, self.Ny))

        for jj in range(self.Nsim):

            b_j, Sig_j = draw_mniw(self.b_hat, self.XX_inv, self.Psi_hat, self.df_hat,
                                   self.Ny, self.Nx)

            self.B_sim[jj, :, :] = np.reshape(b_j, (self.Ny, self.Nx)).T
            self.Sig_sim[jj, :, :] = Sig_j

        return None

    def augment_data(self, B, Sig, sample=True):
        """Draw missing data from state space model."""

        # Picks out top Ny observations
        Z = np.zeros((self.Ny, self.Nx))
        Z[:self.Ny, :self.Ny] = np.eye(self.Ny)

        # Shocks only go to top Ny observations
        R = Z.T

        # No measurement error
        H = np.zeros((self.Ny, self.Ny))

        # Initial conditions
        x_init = self.Y_data[0, :]
        P_init = np.zeros((self.Ny, self.Ny))

        for iy in range(self.Ny):
            if np.isnan(x_init[iy]):
                x_init[iy] = 0.0
                P_init[iy, iy] = 1e+6

        A = vr.companion_form(B)
        ssm = ss.StateSpaceModel(A, R, Sig, Z, H)

        if sample:
            X_samp = ssm.draw_states(self.Y_data, x_init, P_init)
        else:
            sse = ss.StateSpaceEstimates(ssm, self.Y_data)
            sse.kalman_filter(x_init, P_init)
            sse.disturbance_smoother()
            sse.state_smoother()
            X_samp = sse.x_smooth

        self.Y = X_samp[:, :self.Ny]

    # def sample_with_data_augmentation(self, Nsim=1000, Nburn=1000):

        # self.Nsim = Nsim

        # self.B_sim = np.zeros((self.Nsim, self.Nx, self.Ny))
        # self.Sig_sim = np.zeros((self.Nsim, self.Ny, self.Ny))
        # self.Y_sim = np.zeros((self.Nsim, self.Nt, self.Ny))

        # for jj in range(self.Nsim): 

            # # Update data
            # self.X_all = np.vstack((self.X_star, self.X))
            # self.Y_all = np.vstack((self.Y_star, self.Y))

            # self.fit()

            # # Coefficient drawing block
            # b_j, Sig_j = draw_mniw(self.b_hat, self.XX_inv, self.Psi_hat, self.df_hat,
                                   # self.Ny, self.Nx)
           
            # B_j = np.reshape(b_j, (self.Ny, self.Nx)).T

            # # DATA AUGMENTATION BLOCK

            # # Compute companion form and set up state space model
            # A = vr.companion_form(B_j)
            # ssm = ss.StateSpaceModel(A, R, Sig_j, Z, H)
            # self.Y = ssm.draw_states(self.Y_data, x_init, P_init)

        # return None 

    # def sample_glp(self, Nsim=1000, Nburn=1000):

        # for jj in range(self.Nsim + Nburn):

            # b_t, Sig_t = draw_mniw(self.b_hat, self.XX_inv, self.Psi_hat, self.df_hat,
                                   # self.Ny, self.Nx)

            # jj_sim = jj - Nburn

        # return None

    def compute_irfs_sim(self, Nirf=41, impact=None, impact_type='identity'):
        """ Computes IRFs from sampled parameters.

        impact_type should be 'identity' or 'cholesky'.
        """

        self.irf_sim = np.zeros((self.Nsim, Nirf, self.Ny, self.Ny))

        for jj in range(self.Nsim):

            if impact is None:
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
                df_sim.loc[self.ix, 'u_' + var] = eps_hat[:, ii]

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
