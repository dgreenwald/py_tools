import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import multivariate_normal, invwishart, gamma, invgamma
import pandas as pd
import re

from py_tools import data as dt, numerical as nm, time_series as ts
from py_tools.time_series import merge_date
import py_tools.time_series.state_space as ss
from py_tools.time_series import var as vr


def log_abs_det(x):
    """Compute the log of the absolute value of the determinant of a matrix.

    Parameters
    ----------
    x : np.ndarray, shape (N, N)
        Square matrix.

    Returns
    -------
    result : float
        log|det(x)|.
    """

    return np.log(np.abs(np.linalg.det(x)))

def fit_ols(X, Y):
    """Estimate OLS coefficients and residual covariance.

    Parameters
    ----------
    X : np.ndarray, shape (Nt, Nx)
        Regressor matrix.
    Y : np.ndarray, shape (Nt, Ny)
        Dependent variable matrix.

    Returns
    -------
    Phi : np.ndarray, shape (Nx, Ny)
        OLS coefficient matrix.
    Sig : np.ndarray, shape (Ny, Ny)
        Residual covariance matrix.
    """

    Phi = dt.least_sq(X, Y)
    e = Y - np.dot(X, Phi)
    Sig = xtx(e) / X.shape[0]

    return (Phi, Sig)

def eval_glp_hyperprior(hyperparams, gam_hyp_shape, gam_hyp_scale,
                        igam_hyp_shape, igam_hyp_scale):
    """Evaluate the log-density of the GLP hyperprior.

    Parameters
    ----------
    hyperparams : np.ndarray
        Hyperparameter vector; first 3 entries use a gamma prior, the rest
        use an inverse-gamma prior.
    gam_hyp_shape : np.ndarray
        Shape parameters for the gamma hyperprior.
    gam_hyp_scale : np.ndarray
        Scale parameters for the gamma hyperprior.
    igam_hyp_shape : np.ndarray
        Shape parameters for the inverse-gamma hyperprior.
    igam_hyp_scale : np.ndarray
        Scale parameters for the inverse-gamma hyperprior.

    Returns
    -------
    L : float
        Log-density of the hyperprior.
    """

    gam_params = hyperparams[:3]
    L = np.sum(gamma.logpdf(gam_params, gam_hyp_shape, scale=gam_hyp_scale))

    igam_params = hyperparams[3:]
    L += np.sum(invgamma.logpdf(igam_params, igam_hyp_shape, scale=igam_hyp_scale))

    return L

def post_mode(X, Y, b_bar, Om_inv_bar, Psi_bar, df_bar):
    """Compute the posterior mode under the MNIW prior.

    Parameters
    ----------
    X : np.ndarray, shape (Nt, Nx)
        Regressor matrix.
    Y : np.ndarray, shape (Nt, Ny)
        Dependent variable matrix.
    b_bar : np.ndarray, shape (Nx * Ny,)
        Prior mean for the vectorized coefficient matrix.
    Om_inv_bar : np.ndarray, shape (Nx, Nx)
        Prior precision matrix for the coefficients.
    Psi_bar : np.ndarray, shape (Ny, Ny)
        Prior scale matrix for the covariance.
    df_bar : float
        Prior degrees of freedom for the covariance.

    Returns
    -------
    B_hat : np.ndarray, shape (Nx, Ny)
        Posterior mode of the coefficient matrix.
    XX_inv : np.ndarray, shape (Nx, Nx)
        Inverse of the posterior precision matrix.
    Psi_hat : np.ndarray, shape (Ny, Ny)
        Posterior scale matrix.
    df_hat : float
        Posterior degrees of freedom.
    eps_hat : np.ndarray, shape (Nt, Ny)
        Posterior residuals.
    L : float
        Log marginal likelihood.
    """

    Nt, Nx = X.shape
    _, Ny = Y.shape

    p = (Nx - 1) / Ny

    # Posterior for B
    B_bar = np.reshape(b_bar, (Ny, Nx)).T
    XX = np.dot(X.T, X) + Om_inv_bar
    Xy = np.dot(X.T, Y) + np.dot(Om_inv_bar, B_bar)

    XX_inv = np.linalg.inv(XX)
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
    """Draw from the matrix normal inverse Wishart (MNIW) posterior.

    Parameters
    ----------
    b_hat : np.ndarray, shape (Nx * Ny,)
        Posterior mean of the vectorized coefficient matrix.
    XX_inv : np.ndarray, shape (Nx, Nx)
        Posterior coefficient covariance factor.
    Psi_hat : np.ndarray, shape (Ny, Ny)
        Posterior scale matrix.
    df_hat : float
        Posterior degrees of freedom.
    Ny : int
        Number of dependent variables.
    Nx : int
        Number of regressors.

    Returns
    -------
    b : np.ndarray, shape (Nx * Ny,)
        Sampled vectorized coefficient matrix.
    Sig : np.ndarray, shape (Ny, Ny)
        Sampled covariance matrix.
    """

    Sig = invwishart.rvs(df_hat, scale=Psi_hat)
    V_b = np.kron(Sig, XX_inv)
    b = multivariate_normal.rvs(b_hat, V_b)
    return (b, Sig)

def xtx(x):
    """Compute x.T @ x.

    Parameters
    ----------
    x : np.ndarray, shape (N, M)
        Input matrix.

    Returns
    -------
    result : np.ndarray, shape (M, M)
        Product x.T @ x.
    """

    return np.dot(x.T, x)

def compute_irfs(B, p, Nirf, impact):
    """Compute impulse response functions from BVAR coefficients.

    Parameters
    ----------
    B : np.ndarray, shape (p*Ny + 1, Ny)
        VAR coefficient matrix (lags stacked, with constant last row).
    p : int
        Number of VAR lags.
    Nirf : int
        Number of IRF periods.
    impact : np.ndarray, shape (Ny, Nshock)
        Impact matrix mapping shocks to variables.

    Returns
    -------
    virf : np.ndarray, shape (Nirf, Ny, Nshock)
        Impulse response functions.
    """

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
    """Compute GLP hyperprior parameters from modes and standard deviations.

    Parameters
    ----------
    Ny : int
        Number of VAR variables.
    gam_hyp_modes : np.ndarray, optional
        Modes for the gamma hyperprior. Default is (0.2, 1.0, 1.0).
    gam_hyp_stds : np.ndarray, optional
        Standard deviations for the gamma hyperprior. Default is (0.4, 1.0, 1.0).
    igam_hyp_scale : np.ndarray, optional
        Scale parameters for the inverse-gamma hyperprior.
        Default is 0.02^2 * ones(Ny).
    igam_hyp_shape : np.ndarray, optional
        Shape parameters for the inverse-gamma hyperprior.
        Default is 0.02^2 * ones(Ny).

    Returns
    -------
    gam_hyp_shape : np.ndarray
        Gamma prior shape parameters.
    gam_hyp_scale : np.ndarray
        Gamma prior scale parameters.
    igam_hyp_shape : np.ndarray
        Inverse-gamma prior shape parameters.
    igam_hyp_scale : np.ndarray
        Inverse-gamma prior scale parameters.
    """

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
    """Construct the MNIW prior from GLP hyperparameters.

    Parameters
    ----------
    params : np.ndarray
        Hyperparameter vector; first element is lambda (overall tightness),
        remaining elements are psi (diagonal prior scale per variable).
    Ny : int
        Number of VAR variables.
    Nx : int
        Number of regressors per equation.
    p : int
        Number of VAR lags.

    Returns
    -------
    b_bar : np.ndarray, shape (Nx * Ny,)
        Prior mean for the vectorized coefficient matrix.
    Om_inv_bar : np.ndarray, shape (Nx, Nx)
        Prior precision matrix.
    df_bar : float
        Prior degrees of freedom.
    """

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
    """Construct co-persistence dummy observations for the BVAR prior.

    Parameters
    ----------
    params : array-like, length 2
        mu: sum-of-coefficients tightness parameter.
        delta: co-persistence tightness parameter.
    Nx : int
        Number of regressors per equation.
    Ny : int
        Number of VAR variables.
    p : int
        Number of VAR lags.
    ybar : np.ndarray, shape (Ny,)
        Sample mean of each variable.

    Returns
    -------
    X_star : np.ndarray, shape (Ny+1, Nx)
        Dummy regressor matrix.
    Y_star : np.ndarray, shape (Ny+1, Ny)
        Dummy dependent variable matrix.
    """

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
    """Construct dummy observations for the Minnesota-style BVAR prior.

    Parameters
    ----------
    lam : array-like, length 8
        Hyperparameter vector (lam1 through lam8) controlling prior tightness.
    Nx : int
        Number of regressors per equation.
    Ny : int
        Number of VAR variables.
    p : int
        Number of VAR lags.
    rwlist : np.ndarray, shape (Ny,)
        Binary indicators for random-walk prior (1 = random walk, 0 = white noise).
    ybar : np.ndarray, shape (Ny,)
        Sample mean of each variable.
    sbar : np.ndarray, shape (Ny,)
        Sample standard deviation of each variable.

    Returns
    -------
    X_star : np.ndarray, shape (Nt_star, Nx)
        Dummy regressor matrix.
    Y_star : np.ndarray, shape (Nt_star, Ny)
        Dummy dependent variable matrix.
    """

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
    """Check whether a variable should be included in the NaN-dropping step.

    Returns True if the variable is not a lag of a data-augmentation variable.

    Parameters
    ----------
    var : str
        Variable name to check.
    data_augmentation_vars : list of str
        Variables whose missing values will be imputed by data augmentation.

    Returns
    -------
    result : bool
        True if the variable should be dropped when NaN, False otherwise.
    """

    for da in data_augmentation_vars:
        if re.match('(L\d*_)?' + da, var) is not None:
            return False

    return True

class BVAR:
    """Bayesian Vector Autoregression (BVAR).

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame containing the VAR variables.
    y_vars : list of str
        Names of the VAR variables.
    p : int, optional
        Number of lags. Default is 1.
    hyperparams_init : np.ndarray, optional
        Initial hyperparameter values. If None, defaults are used.
    rwlist : np.ndarray, optional
        Binary indicators for random-walk prior (Minnesota prior only).
        Default is all ones.
    glp_prior : bool, optional
        If True, use the GLP prior; otherwise use the Minnesota prior.
        Default is False.
    data_augmentation_vars : list of str, optional
        Variables whose missing data will be handled by data augmentation.

    Attributes
    ----------
    Ny : int
        Number of VAR variables.
    Nt : int
        Number of observations.
    B_hat : np.ndarray
        Posterior mode coefficient matrix (set after fit()).
    B_sim : np.ndarray, shape (Nsim, Nx, Ny)
        Sampled coefficient matrices (set after sample()).
    Sig_sim : np.ndarray, shape (Nsim, Ny, Ny)
        Sampled covariance matrices (set after sample()).
    irf_sim : np.ndarray
        Simulated IRFs (set after compute_irfs_sim()).
    """

    def __init__(self, df_in, y_vars, p=1, hyperparams_init=None, rwlist=None, glp_prior=False,
                 data_augmentation_vars=None):
        """Initialize the BVAR model.

        Parameters
        ----------
        df_in : pd.DataFrame
            Input DataFrame containing the VAR variables.
        y_vars : list of str
            Names of the VAR variables.
        p : int, optional
            Number of lags. Default is 1.
        hyperparams_init : np.ndarray, optional
            Initial hyperparameter values. Default is None.
        rwlist : np.ndarray, optional
            Random-walk indicators (Minnesota prior). Default is ones.
        glp_prior : bool, optional
            Use GLP prior if True, else Minnesota prior. Default is False.
        data_augmentation_vars : list of str, optional
            Variables with missing data to impute. Default is None.
        """

        if data_augmentation_vars is None: data_augmentation_vars = []

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
        """Construct prior dummy observations from the current hyperparameters.

        Populates X_star, Y_star, X_all, Y_all, and related prior matrices.
        """

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

        self.Nt_all = self.X_all.shape[0]

        return None

    def fit(self):
        """Compute the posterior mode of the BVAR coefficients.

        Populates B_hat, XX_inv, Psi_hat, df_hat, eps_hat, and b_hat.
        """

        (self.B_hat, self.XX_inv, self.Psi_hat, self.df_hat, 
         self.eps_hat, _
         ) = self.eval_post_mode(self.X_all, self.Y_all)
        self.b_hat = self.B_hat.T.flatten()

        return None

    def eval_post_mode(self, X, Y):
        """Evaluate the posterior mode for a given data matrix.

        Parameters
        ----------
        X : np.ndarray, shape (Nt, Nx)
            Regressor matrix.
        Y : np.ndarray, shape (Nt, Ny)
            Dependent variable matrix.

        Returns
        -------
        result : tuple
            (B_hat, XX_inv, Psi_hat, df_hat, eps_hat, L) as returned by post_mode.
        """

        return post_mode(X, Y, self.b_bar, self.Om_inv_bar, 
                         self.Psi_bar, self.df_bar)

    def objfcn_glp(self, x):
        """Evaluate the GLP posterior objective function.

        Parameters
        ----------
        x : np.ndarray
            Log-transformed hyperparameter vector.

        Returns
        -------
        L_post : float
            Log posterior (marginal likelihood + hyperprior - dummy likelihood).
        """

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
        """Optimize the GLP hyperparameters via Nelder-Mead maximization.

        Updates self.hyperparams to the optimal values and prints results.
        """

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
        """Run EM iterations to jointly optimize hyperparameters and impute missing data.

        Parameters
        ----------
        n_iter : int, optional
            Number of EM iterations. Default is 5.
        """

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
        """Draw posterior samples of VAR coefficients and covariance matrices.

        Parameters
        ----------
        Nsim : int, optional
            Number of posterior draws. Default is 1000.
        """

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
        """Impute missing data using the state space model.

        Parameters
        ----------
        B : np.ndarray, shape (Nx, Ny)
            VAR coefficient matrix.
        Sig : np.ndarray, shape (Ny, Ny)
            Shock covariance matrix.
        sample : bool, optional
            If True, draw states; if False, use smoothed states. Default is True.
        """

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

    def compute_irfs_sim(self, Nirf=41, impact=None, impact_type='identity'):
        """Compute IRFs from sampled VAR parameters.

        Parameters
        ----------
        Nirf : int, optional
            Number of IRF periods. Default is 41.
        impact : np.ndarray, optional
            Impact matrix. If None, determined by impact_type.
        impact_type : {'identity', 'cholesky'}, optional
            Type of impact matrix to use if impact is None. Default is 'identity'.
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
        """Register an external instrument for IV-identified IRFs.

        Parameters
        ----------
        df_new : pd.DataFrame
            DataFrame containing the instrument series.
        policy_var : str
            Name of the policy variable (must be the first VAR variable).
        instrument : str
            Name of the instrument column in df_new.
        """

        self.policy_var = policy_var
        self.instrument = instrument
        self.df = merge_date(self.df, df_new[[instrument]], how='outer')
        return None

    def compute_irfs_instrument(self, Nirf=41, exact_sigma=True):
        """Compute IRFs using an external instrument to identify structural shocks.

        Parameters
        ----------
        Nirf : int, optional
            Number of IRF periods. Default is 41.
        exact_sigma : bool, optional
            If True, use the sampled covariance matrix in the IV regression.
            Default is True.
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
