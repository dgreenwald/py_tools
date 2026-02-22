import numpy as np
from py_tools import data as dt, time_series as ts
from py_tools.utilities import as_list

def instrument_var(df, var_list, policy_var, instrument, Sig=None, resid_prefix='u_'):
    """Estimate the structural shock impact vector using an external instrument (IV).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing residuals and instrument data.
    var_list : list of str
        Ordered list of VAR variable names; the policy variable must be first.
    policy_var : str
        Name of the policy variable (must equal var_list[0]).
    instrument : str
        Name of the instrument column in df.
    Sig : np.ndarray, optional
        (Ny x Ny) residual covariance matrix. If None, estimated from data.
    resid_prefix : str, optional
        Prefix used for residual column names. Default is 'u_'.

    Returns
    -------
    S : np.ndarray, shape (Ny, 1)
        Impact vector for the structural shock.
    """

    assert policy_var == var_list[0] # Is this important?

    ix_samp = np.ones(len(df), dtype=bool)
    for var in var_list:

        # Regress residuals on instrument
        lhs = resid_prefix + var
        rhs = ['const', instrument]
        fr_u = dt.sm_regression(df, lhs, rhs)

        # Store fitted values
        df[resid_prefix + 'hat_' + var] = np.nan
        df.loc[fr_u.ix, resid_prefix + 'hat_' + var] = fr_u.results.fittedvalues

        # Update sample
        ix_samp = np.logical_and(ix_samp, fr_u.ix)

    gam = np.zeros((len(var_list) - 1, 1))
    for ii, var in enumerate(var_list[1:]):

        # Regress residuals on u_hat_policy
        lhs = resid_prefix + var
        rhs = ['const', resid_prefix + 'hat_' + policy_var]
        fr_2s = dt.sm_regression(df, lhs, rhs)

        # Store coefficient on u_hat_policy
        gam[ii] = fr_2s.results.params[1]

    # Compute variance/covariance matrix (if needed)
    if Sig is None:
        u_list = [resid_prefix + var for var in var_list]
        u = df.loc[ix_samp, u_list].values
        Sig = np.dot(u.T, u) / u.shape[0]

    Sig11 = Sig[0, 0]
    Sig21 = Sig[1:, 0][:, np.newaxis]
    Sig22 = Sig[1:, 1:]

    Q = (np.dot(gam, np.dot(Sig11, gam.T))
         - (np.dot(Sig21, gam.T) + np.dot(gam, Sig21.T))
         + Sig22)
    s12s12p = np.dot((Sig21 - np.dot(gam, Sig11)).T,
                     np.linalg.solve(
                         Q, Sig21 - np.dot(gam, Sig11)
                     ))
    s11 = np.sqrt(Sig11 - s12s12p)

    S = np.vstack((s11, s11 * gam))

    return S 

def companion_form(A, use_const=True):
    """Convert a VAR coefficient matrix to companion form.

    Parameters
    ----------
    A : np.ndarray, shape (Ny, Nx)
        VAR coefficient matrix (including constant if use_const is True).
    use_const : bool, optional
        Whether A includes a constant term column. Default is True.

    Returns
    -------
    A_comp : np.ndarray, shape (Nx, Nx)
        Companion-form transition matrix.
    """

    # Get sizes
    Ny, Nx = A.shape
    if use_const:
        Nlags = (Nx - 1) // Ny
    else:
        Nlags = Nx // Ny

    # Pre-allocate
    A_comp = np.zeros((Nx, Nx))

    # Top rows: VAR coefficients
    A_comp[:Ny, :] = A

    # Remaining blocks: identity matrices
    for lag in range(1, Nlags):
        A_comp[Ny*lag : Ny * (lag + 1), 
               Ny * (lag - 1) : Ny * lag] \
            = np.eye(Ny)

    # One for constant term
    if use_const:
        A_comp[-1, -1] = 1.0

    return A_comp

def compute_irfs(A, B, Nt_irf):
    """Compute impulse response functions for a VAR.

    Parameters
    ----------
    A : np.ndarray, shape (Ny, Nx)
        VAR transition matrix.
    B : np.ndarray, shape (Ny, Nshock)
        Impact matrix mapping shocks to variables.
    Nt_irf : int
        Number of periods for the IRFs.

    Returns
    -------
    irfs : np.ndarray, shape (Ny, Nshock, Nt_irf)
        Impulse response functions.
    """

    # Get sizes
    Ny, Nx = A.shape
    Ne = B.shape[1]
    assert(A.shape[0] == B.shape[0])

    # Update to companion form
    A_comp = companion_form(A)
    B_comp = np.zeros((Nx, Ne))
    B_comp[:Ny, :] = B

    # Run IRFs in companion form
    irf_comp_mats = np.zeros((Nx, Ne, Nt_irf))
    irf_comp_mats[:, :, 0] = B_comp
    for tt in range(1, Nt_irf):
        irf_comp_mats[:, :, tt] = np.dot(A_comp, irf_comp_mats[:, :, tt-1])

    # Chop extraneous rows
    return irf_comp_mats[:Ny, :, :]

class VAR:
    """Vector Autoregression (VAR) estimated by OLS.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame containing the series.
    var_list : list of str
        Names of the variables to include in the VAR.
    n_var_lags : int, optional
        Number of lags. Default is 1.
    use_const : bool, optional
        Whether to include a constant term. Default is True.
    copy_df : bool, optional
        Whether to copy the input DataFrame. Default is True.

    Attributes
    ----------
    A : np.ndarray
        Estimated coefficient matrix (set after fit()).
    resid : np.ndarray
        Residuals (set after fit()).
    irfs : np.ndarray or None
        Impulse response functions (set after compute_irfs()).
    A_boot : np.ndarray or None
        Bootstrapped coefficient matrices (set after wild_bootstrap()).
    """

    def __init__(self, df_in, var_list, n_var_lags=1, use_const=True,
                 copy_df=True):
        """Initialize the VAR model.

        Parameters
        ----------
        df_in : pd.DataFrame
            Input DataFrame.
        var_list : list of str
            Variable names to include.
        n_var_lags : int, optional
            Number of lags. Default is 1.
        use_const : bool, optional
            Include constant term. Default is True.
        copy_df : bool, optional
            Copy the DataFrame. Default is True.
        """

        if copy_df:
            self.df = df_in.copy()
        else:
            self.df = df_in

        self.var_list = var_list
        self.n_var_lags = n_var_lags
        self.use_const = use_const

        # IRFs
        self.irfs = None
        self.Nt_irf = None

        # Bootstrap
        self.A_boot = None
        self.y_boot = None

    def fit(self):
        """Estimate the VAR coefficients using OLS."""
        self.fr = ts.lagged_reg(self.df, self.var_list, self.var_list, 
                                self.n_var_lags, use_const=self.use_const)

        # Process results
        self.A = self.fr.results.params.T
        self.resid = self.fr.results.resid
        self.Ny, self.Nx = self.A.shape
        self.Nt = self.fr.Xs.shape[0]

        self.X = self.fr.Xs
        self.y = self.fr.zs
        self.ix = self.fr.ix

    def get_companion_form(self):
        """Return the companion-form transition matrix for the estimated VAR.

        Returns
        -------
        A_comp : np.ndarray
            Companion-form transition matrix.
        """

        return companion_form(self.A)

    def add_series_recursive(self, new_series_in):
        """Add additional series to the VAR using a recursive structure.

        The new series can be affected by existing series but cannot affect them.

        Parameters
        ----------
        new_series_in : str or list of str
            Name(s) of the new series to append recursively.
        """

        new_series = as_list(new_series_in)
        self.var_list += new_series

        # Shift existing parameters
        A_old = self.A.copy()
        Ny_old = self.Ny

        N_new = len(new_series)
        self.Ny += N_new
        self.Nx += N_new * self.n_var_lags
        A1_new = np.zeros((Ny_old, self.Nx))

        for lag in range(self.n_var_lags):
            A1_new[:, self.Ny * lag : self.Ny * lag + Ny_old] \
                = A_old[:, Ny_old * lag : Ny_old * (lag + 1)]

        fr_new = ts.lagged_reg(self.df, new_series, self.var_list, self.n_var_lags, 
                               use_const=self.use_const)

        A2_new = fr_new.results.params.T
        self.A = np.vstack((A1_new, A2_new))

        return None

    def compute_irfs(self, B=None, Nt_irf=20, bootstrap=False):
        """Compute impulse response functions for the estimated VAR.

        Parameters
        ----------
        B : np.ndarray, optional
            Impact matrix (Ny x Nshock). If None, uses the identity matrix.
        Nt_irf : int, optional
            Number of IRF periods. Default is 20.
        bootstrap : bool, optional
            If True, also compute bootstrapped IRFs. Default is False.
        """

        if B is None:
            B = np.eye(self.Ny)

        self.Nt_irf = Nt_irf
        self.irfs = compute_irfs(self.A, B, self.Nt_irf)

        if bootstrap:
            self.irfs_boot = np.zeros((self.irfs.shape + (self.Nboot,)))
            for i_boot in range(self.Nboot):
                self.irfs_boot[:, :, :, i_boot] = compute_irfs(self.A_boot[:, :, i_boot], 
                                                               B, self.Nt_irf)

        return

    def wild_bootstrap(self, Nboot=1000):
        """Run wild bootstrap to generate bootstrapped parameter draws.

        Parameters
        ----------
        Nboot : int, optional
            Number of bootstrap replications. Default is 1000.
        """

        self.Nboot = Nboot

        x_init = self.X[0, :]

        self.A_boot = np.zeros((self.Ny, self.Nx, self.Nboot))
        self.y_boot = np.zeros((self.Nt, self.Ny, Nboot))

        # Draw bootstrapped residuals
        eta = 1.0 - 2.0 * (np.random.rand(self.Nt, Nboot) > 0.5)
        self.resid_boot = self.resid[:, :, np.newaxis] * eta[:, np.newaxis, :]

        y_i = np.zeros((self.Nt, self.Ny))
        X_i = np.zeros((self.Nt, self.Nx))

        A_comp = companion_form(self.A)

        for i_boot in range(Nboot):

            x = x_init
            for tt in range(self.Nt):
                X_i[tt, :] = x
                x = np.dot(A_comp, x)
                x[:self.Ny] += self.resid_boot[tt, :, i_boot]
                y_i[tt, :] = x[:self.Ny]

            # Store results
            self.y_boot[:, :, i_boot] = y_i

            # Re-estimate VAR
            self.A_boot[:, :, i_boot] = dt.least_sq(X_i, y_i).T

        return

    def bootstrap_irfs(self, B, **kwargs):
        """Compute bootstrapped impulse response functions.

        Parameters
        ----------
        B : np.ndarray, shape (Ny, Nshock)
            Impact matrix.
        **kwargs
            Additional keyword arguments passed to wild_bootstrap if needed.
        """

        if self.y_boot is None or self.A_boot is None:
            print("Bootstrap sample not found, rerunning...")
            self.wild_bootstrap(**kwargs)

        if self.Nt_irf is None:
            self.Nt_irf = kwargs.get('Nt_irf', 20) # TODO don't hard code number

        Ne = B.shape[1] # TODO: should be a way to set B
        self.irfs_boot = np.zeros((self.Ny, Ne, self.Nt_irf, self.Nboot))
        for i_boot in range(self.Nboot):
            self.irfs_boot[:, :, :, i_boot] = compute_irfs(self.A_boot[:, :, i_boot], B, self.Nt_irf)
             
        return
