import numpy as np
from scipy.linalg import ordqz

class KleinObj:
    """Solver for the Klein (2000) linear rational expectations system.

    Solves a system of the form ``A E_t[y_{t+1}] = B y_t + C e_t`` using the
    QZ (generalised Schur) decomposition, as described in Klein (2000).

    Parameters
    ----------
    A : array_like, shape (neq, neq)
        Left-hand-side coefficient matrix on expected next-period values.
    B : array_like, shape (neq, neq)
        Right-hand-side coefficient matrix on current-period values.
    C : array_like, shape (neq, n_exog)
        Coefficient matrix on the exogenous shock vector.
    Phi : array_like, shape (n_exog, n_exog)
        Persistence (transition) matrix for the exogenous shock process.
    Sig : array_like, shape (n_exog, n_exog)
        Variance-covariance matrix of the exogenous shocks.
    n_pre : int
        Number of predetermined (state) variables.

    Attributes
    ----------
    G_x : ndarray, shape (n_pre, n_uns)
        Policy function mapping predetermined states to jump variables.
    H_x : ndarray, shape (n_pre, n_pre)
        Law of motion for predetermined states.
    G_f : ndarray, shape (n_uns, n_exog)
        Policy function mapping exogenous states to jump variables.
    H_f : ndarray, shape (n_pre, n_exog)
        Law of motion mapping exogenous states to predetermined states.
    n_endog : int
        Total number of endogenous variables (``A.shape[0]``).
    n_pre : int
        Number of predetermined variables (as supplied).
    n_uns : int
        Number of non-predetermined (jump) variables (``n_endog - n_pre``).
    n_exog : int
        Number of exogenous shock processes (``C.shape[1]``).
    """

    def __init__(self, A, B, C, Phi, Sig, n_pre):
        """Initialize the KleinObj solver with system matrices.

        Parameters
        ----------
        A : array_like, shape (neq, neq)
            Left-hand-side coefficient matrix on expected next-period values.
        B : array_like, shape (neq, neq)
            Right-hand-side coefficient matrix on current-period values.
        C : array_like, shape (neq, n_exog)
            Coefficient matrix on the exogenous shock vector.
        Phi : array_like, shape (n_exog, n_exog)
            Persistence (transition) matrix for the exogenous shock process.
        Sig : array_like, shape (n_exog, n_exog)
            Variance-covariance matrix of the exogenous shocks.
        n_pre : int
            Number of predetermined (state) variables.
        """
        self.A = np.asmatrix(A)
        self.B = np.asmatrix(B)
        self.C = np.asmatrix(C)
        self.Phi = np.asmatrix(Phi)
        self.Sig = np.asmatrix(Sig)
        self.n_endog = self.A.shape[0]
        self.n_pre = n_pre
        self.n_uns = self.n_endog - self.n_pre
        self.n_exog = self.C.shape[1]

    def solve(self):
        """Solve the Klein system using QZ decomposition.

        Applies the generalised Schur (QZ) decomposition to the pencil
        ``(A, B)``, sorts stable eigenvalues first, then partitions the
        resulting matrices to recover the policy and transition functions.

        Returns
        -------
        None
            Results are stored as instance attributes.

        Attributes
        ----------
        G_x : ndarray, shape (n_pre, n_uns)
            Policy function: maps predetermined states to jump variables.
        H_x : ndarray, shape (n_pre, n_pre)
            Law of motion for predetermined states.
        G_f : ndarray, shape (n_uns, n_exog)
            Policy function: maps exogenous states to jump variables.
        H_f : ndarray, shape (n_pre, n_exog)
            Law of motion mapping exogenous states to predetermined states.
        """
        S, T, alp, bet, Q, Z = ordqz(
            self.A, self.B, sort='ouc', output='complex'
        )  

        # Basic existence/uniqueness diagnostics (Blanchard-Kahn style).
        eig_tol = 1e-10
        eig_abs = np.full(alp.shape, np.inf, dtype=float)
        finite_bet = np.abs(bet) > eig_tol
        eig_abs[finite_bet] = np.abs(alp[finite_bet] / bet[finite_bet])
        n_stable = int(np.sum(eig_abs < (1.0 - eig_tol)))
        n_unit = int(np.sum(np.abs(eig_abs - 1.0) <= eig_tol))

        if n_unit > 0:
            raise ValueError(
                "Klein solver encountered generalized eigenvalues on the unit circle."
            )
        if n_stable != self.n_pre:
            raise ValueError(
                "Blanchard-Kahn condition failed: expected {} stable roots, found {}."
                .format(self.n_pre, n_stable)
            )

        Q_star = np.asmatrix(Q).H 
        Q1 = np.asmatrix(Q_star[:self.n_pre, :])
        Q2 = np.asmatrix(Q_star[self.n_pre:, :])

        Z11 = np.asmatrix(Z[:self.n_pre, :self.n_pre])
        Z12 = np.asmatrix(Z[:self.n_pre, self.n_pre:])
        Z21 = np.asmatrix(Z[self.n_pre:, :self.n_pre])
        Z22 = np.asmatrix(Z[self.n_pre:, self.n_pre:])

        S11 = np.asmatrix(S[:self.n_pre, :self.n_pre])
        S12 = np.asmatrix(S[:self.n_pre, self.n_pre:])
        S22 = np.asmatrix(S[self.n_pre:, self.n_pre:])

        T11 = np.asmatrix(T[:self.n_pre, :self.n_pre])
        T12 = np.asmatrix(T[:self.n_pre, self.n_pre:])
        T22 = np.asmatrix(T[self.n_pre:, self.n_pre:])

        if Z12.shape[0] != Z12.shape[1]:
            raise ValueError(
                "Klein solver requires square Z12 block; got shape {}.".format(Z12.shape)
            )

        G_xc = Z11 * np.linalg.inv(Z12)
        H_xc = Z11 * (np.linalg.solve(S11, T11)) * np.linalg.inv(Z11)

        self.G_x = np.real(G_xc)
        self.H_x = np.real(H_xc)

        PhiST = np.kron(self.Phi.T, S22) - np.kron(np.eye(self.n_exog), T22)
        q2C = (Q2 * self.C).flatten(order='F')
        M = np.linalg.solve(PhiST, q2C).reshape((self.n_uns, self.n_exog), order='F')

        N = (Z22 - Z21 * np.linalg.solve(Z11, Z12)) * M
        L = (-Z11 * np.linalg.solve(S11, T11) * np.linalg.solve(Z11, Z12) * M 
             + Z11 * np.linalg.solve(S11, T12 * M - S12 * M * self.Phi + Q1 - self.C)
             + Z12 * M * self.Phi)

        self.G_f = np.real(N)
        self.H_f = np.real(L)

        return None

class AimObj:
    """Solver for the Anderson-Moore (AIM) linear rational expectations system.

    Implements the algorithm of Anderson and Moore (1985) for finding the
    reduced-form solution to a linear rational expectations model expressed as::

        sum_{s=-nlag}^{nlead} H_s y_{t+s} = 0

    where the stacked coefficient matrix ``H`` has shape ``(neq, neq*(nlag+nlead+1))``.

    Parameters
    ----------
    H : ndarray, shape (neq, hcols)
        Model equation matrix.  Columns are ordered from the most lagged to the
        most leading block, each block being ``neq`` wide.
    nlead : int, optional
        Number of lead periods in the model. Default is 1.
    tol : float, optional
        Numerical tolerance for rank detection. Default is 1e-10.

    Attributes
    ----------
    neq : int
        Number of equations / endogenous variables.
    hcols : int
        Total number of columns in ``H``.
    periods : int
        Total number of time periods spanned by ``H`` (``hcols // neq``).
    nlead : int
        Number of lead periods.
    nlag : int
        Number of lag periods (``periods - nlead - 1``).
    tol : float
        Numerical tolerance.
    Z : ndarray, shape (zrows, zcols)
        Accumulator matrix built during the shift steps; overwritten by
        ``reduced_form`` to hold the solution coefficients.
    B : ndarray, shape (neq, neq * nlag)
        Reduced-form solution matrix (set after ``solve`` or ``reduced_form``).
    A : ndarray
        Companion matrix (set after ``build_companion``).
    """

    def __init__(self, H, nlead=1, tol=1e-10):
        """Initialize the AimObj solver.

        Parameters
        ----------
        H : ndarray, shape (neq, hcols)
            Model equation matrix with columns ordered from most lagged to
            most leading, each block of width ``neq``.
        nlead : int, optional
            Number of lead periods. Default is 1.
        tol : float, optional
            Numerical tolerance for rank detection. Default is 1e-10.
        """
        self.H = H
        self.neq, self.hcols = H.shape
        self.periods = self.hcols // self.neq
        self.nlead = nlead
        self.nlag = int(self.periods - self.nlead) - 1
        self.tol = tol
        # self.eig_bnd = 1.0

        self.left = np.arange(self.hcols - self.neq)
        self.right = np.arange(self.hcols - self.neq, self.hcols)

        self.iz = 0
        self.zrows = int(self.neq * self.nlead)
        self.zcols = int(self.neq * (self.periods - 1))
        self.Z = np.zeros((self.zrows, self.zcols))

    def solve(self):
        """Solve the AIM system for the reduced-form solution matrix ``B``.

        Runs the full AIM algorithm in order: exact shift reductions,
        numeric shift reductions, companion matrix construction, eigensystem
        computation, and finally the reduced-form solution.

        Returns
        -------
        None
            The solution is stored in ``self.B``.
        """
        self.exact_shift()
        self.numeric_shift()
        self.build_companion()
        self.eigensystem()
        self.reduced_form()

        return None

    def shift_right(self, x):
        """Shift the columns of ``x`` right by ``neq`` positions.

        Equivalent to multiplying by the lag operator: the last ``neq``
        columns of the input are discarded and ``neq`` zero columns are
        prepended.

        Parameters
        ----------
        x : ndarray, shape (nrows, hcols)
            Matrix whose columns are to be shifted.

        Returns
        -------
        x_shift : ndarray, shape (nrows, hcols)
            Column-shifted copy of ``x``.
        """
        x_shift = np.zeros(x.shape)
        x_shift[:, self.neq:] = x[:, :-self.neq]

        return x_shift

    def shuffle(self, ix):
        """Move rows ``ix`` of ``H`` into ``Z`` and shift them right.

        Copies the left-block columns of the selected rows into the next
        available rows of ``self.Z``, then replaces those rows in ``self.H``
        with their column-shifted versions (via ``shift_right``).

        Parameters
        ----------
        ix : array_like of int
            Row indices of ``self.H`` to be moved.

        Returns
        -------
        None
            ``self.Z`` and ``self.H`` are updated in place; ``self.iz`` is
            incremented by ``len(ix)``.
        """
        nz = len(ix)
        self.Z[self.iz:self.iz + nz, :] = self.H[ix, self.left]
        self.H[ix, :] = self.shift_right(self.H[ix, :])
        self.iz += nz

        return None

    def exact_shift(self):
        """Perform exact shift reductions on rows with zero right-block columns.

        Iteratively identifies rows of ``self.H`` whose rightmost ``neq``
        columns are all zero (within tolerance) and shuffles them into
        ``self.Z``, until no such rows remain or ``self.Z`` is full.

        Returns
        -------
        None
            ``self.H`` and ``self.Z`` are updated in place.
        """
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
        """Perform numeric shift reductions using QR decomposition.

        Uses QR decomposition of the right block of ``self.H`` to detect
        numerically rank-deficient rows and shuffles them into ``self.Z``,
        repeating until the right block has full numerical rank or ``self.Z``
        is full.

        Returns
        -------
        None
            ``self.H`` and ``self.Z`` are updated in place.
        """
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
        """Build the companion matrix ``A`` from the reduced ``H`` matrix.

        Constructs the block-companion matrix whose bottom block of rows is
        ``-H_right^{-1} H_left`` (the autoregressive coefficients), then
        removes columns (and corresponding rows) that are numerically zero to
        eliminate inessential lags.

        Returns
        -------
        None
            Sets ``self.A`` (the pruned companion matrix) and ``self.js``
            (the surviving column indices) in place.
        """
        self.A = np.zeros((self.zcols, self.zcols))
        if self.zcols > self.neq:
            self.A[:-self.neq, self.neq:] = np.eye(self.zcols - self.neq)

        Gam = -np.linalg.solve(self.H[:, self.right], self.H[:, self.left])
        self.A[-self.neq:, :] = Gam

        # Delete inessential lags
        self.js = np.arange(self.zcols)
        drop = np.sum(np.abs(self.A), axis=0) < self.tol
        while np.any(drop):
            ix = np.arange(len(drop))[drop]
            self.A = np.delete(self.A, ix, axis=0)
            self.A = np.delete(self.A, ix, axis=1)
            self.js = np.delete(self.js, ix)
            drop = np.sum(np.abs(self.A), axis=0) < self.tol

        return None

    def eigensystem(self):
        """Compute the eigensystem of the companion matrix and fill ``Z``.

        Computes eigenvectors of ``self.A.T``, sorts them by descending
        absolute eigenvalue, and stores the leading stable eigenvectors
        (transposed) into the remaining rows of ``self.Z``.

        Returns
        -------
        None
            ``self.Z`` is updated in place.
        """
        vals, vecs = np.linalg.eig(self.A.T)
        ix = np.flipud(np.argsort(np.abs(vals)))

        # sorted_vals = vals[ix]
        sorted_vecs = vecs[:, ix]

        # self.lgroots = np.sum(np.abs(sorted_vals) > self.eig_bnd)

        self.Z[self.iz:, self.js] =  sorted_vecs[:, :(self.zrows - self.iz)].T

        return None

    def reduced_form(self):
        """Compute the reduced-form solution matrix ``B`` from ``Z``.

        Solves for the reduced-form coefficient matrix by partitioning
        ``self.Z`` into left and right blocks and solving the linear system,
        then extracts the ``neq x (neq * nlag)`` submatrix as ``self.B``.

        Returns
        -------
        None
            Sets ``self.B`` (shape ``(neq, neq * nlag)``) and overwrites
            ``self.Z`` with the full solution coefficients.
        """
        self.Z = -np.linalg.solve(self.Z[:, self.zcols - self.zrows:], self.Z[:, :self.zcols - self.zrows])
        self.B = self.Z[:self.neq, :self.neq * self.nlag]

        return None
