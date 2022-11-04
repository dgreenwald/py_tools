import numpy as np
import itertools

def quad_form(A, X):
    return np.dot(A.T, np.dot(X, A))

def rsolve(b, A):
    return np.linalg.solve(A.T, b.T).T

def gradient(f, x, args=(), step=1e-5):

    grad = None
    for ii in range(len(x)):

        x[ii] += step
        f_hi = f(x, *args)
        x[ii] -= (2.0 * step)
        f_lo = f(x, *args)
        x[ii] += step

        df_i = np.array(f_hi - f_lo) / (2.0 * step)

        if grad is None:

            if df_i.shape == ():
                ncols = 1 
            else:
                ncols = len(df_i)

            grad = np.zeros((len(x), ncols))

        grad[ii, :] = df_i

    return grad

def hessian(f, x_in, eps=1e-4):
    
    x = x_in.copy()
    n = len(x)
    H = np.zeros((n, n))
    for ii, jj in itertools.product(range(n), repeat=2):

        if ii <= jj:
        
            x[ii] += eps
            x[jj] += eps
            H[ii, jj] += f(x)
            
            x[jj] -= 2.0 * eps
            if ii != jj:
                H[ii, jj] -= f(x)
            else:
                H[ii, jj] -= 2.0 * f(x)
            
            x[ii] -= 2.0 * eps
            H[ii, jj] += f(x)
            
            x[jj] += 2.0 * eps
            if ii != jj:
                H[ii, jj] -= f(x)
            
            x[ii] += eps
            x[jj] -= eps

        else:

            H[ii, jj] = H[jj, ii]

    return H / (4.0 * (eps ** 2))

def svd_inv(A, sv_tol=1e-8, **kwargs):
    
    u, s, vh = np.linalg.svd(A)
    s_inv = np.zeros(s.shape)
    
    ix = np.abs(s) > sv_tol
    s_inv[ix] = 1.0 / s[ix]
    
    S_inv = np.diag(s_inv)
    A_inv = np.dot(vh.T, np.dot(S_inv, u.T))
    return A_inv

def ghquad_norm(degree, mu=0.0, sig=1.0):

    x, w = np.polynomial.hermite.hermgauss(degree) 

    # Re-scale weights (currently add to pi^{1/2})
    w /= np.sum(w)

    # Re-scale nodes to adjust for mu and sigma
    x *= np.sqrt(2.0) * sig
    x += mu

    return x, w

def gauss_legendre_norm(degree, a=-1.0, b=1.0):
    
    x, w = np.polynomial.legendre.leggauss(degree)
    
    # Re-scale weights
    w *= 0.5 * (b - a)
    
    # Re-scale nodes
    x = 0.5 * ((x + 1.0) * (b - a)) + a
    
    return x, w

def logit(x, lb=0.0, ub=1.0):
    return np.log(x - lb) - np.log(ub - x)

def logistic(x, lb=0.0, ub=1.0):
    return lb + (ub - lb) / (1.0 + np.exp(-x))

def bound_transform(vals, lb, ub, to_bdd=True):

    trans_vals = vals.copy()

    # Indices
    ix_lb = lb > -np.inf
    ix_ub = ub < np.inf

    ix_both = ix_lb & ix_ub
    ix_lb_only = ix_lb & (~ix_ub)
    ix_ub_only = (~ix_lb) & ix_ub

    if to_bdd:
        trans_vals[ix_both] = logistic(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = lb[ix_lb_only] + np.exp(vals[ix_lb_only])
        trans_vals[ix_ub_only] = ub[ix_ub_only] - np.exp(vals[ix_ub_only])
    else:
        trans_vals[ix_both] = logit(vals[ix_both], lb=lb[ix_both], ub=ub[ix_both])
        trans_vals[ix_lb_only] = np.log(vals[ix_lb_only] - lb[ix_lb_only])
        trans_vals[ix_ub_only] = -np.log(ub[ix_ub_only] - vals[ix_ub_only])

    return trans_vals

def robust_cholesky(A, min_eig=1e-12):
    
    vals, vecs = np.linalg.eig(A)
    vals = np.maximum(vals, min_eig)
    Dhalf = np.diag(np.sqrt(vals))
    return vecs @ Dhalf

def my_chol(A_in):
    
    A = A_in.copy()
    
    N = A.shape[0]
    L = np.zeros((N, N))
    
    for jj in range(N):
        Dj = np.sqrt(A[jj, jj])
        aj = A[jj+1:, jj]
        Aj = aj[:, np.newaxis]
        L[jj, jj] = Dj
        L[jj+1:, jj] = aj / Dj
        if jj < N - 1:
            A[jj+1:, jj+1:] -= (Aj @ Aj.T) / (Dj**2)

    return L

#def se99(A_in):
#    
#    A = A_in.copy()
#    
#    N = A.shape[0]
#    L = np.zeros((N, N))
#    G = np.zeros(N)
#    
#    for jj in range(N):
#        
#        Dj = np.sqrt(A[jj, jj])
#        aj = A[jj+1:, jj]
#        Aj = aj[:, np.newaxis]
#        
#        for ii in range(jj, N):
#            G[ii] = 2.0 * A[ii, ii] - np.sum(np.abs(A[ii, jj:]))
##            G2 = A[ii, ii] - np.sum(np.abs(A[ii, jj:])) + np.abs(A[ii, ii])
##            G3 = A[ii, ii]
##            for kk in range(jj, N):
##                if kk != ii: G3 -= np.abs(A[ii, kk])
#        
#        L[jj, jj] = Dj
#        L[jj+1:, jj] = aj / Dj
#        if jj < N - 1:
#            A[jj+1:, jj+1:] -= (Aj @ Aj.T) / (Dj**2)
#
#    return L

def se99(A_in):
    
    A = A_in.copy()
    
    N = A.shape[0]
    
    eps = np.finfo(float).eps
    tau = eps**(1.0/3.0)
    tau_bar = tau**2
    mu = 0.1
    
    gam = np.amax(np.abs(A.diagonal()))
    tb_gam = tau_bar * gam
    mu_gam = mu * gam
    
    L = np.zeros(A.shape)
    g = np.zeros(N)
    
    perm = np.arange(N)
    
    # Phase 1 loop
    jj = 0
    while jj < N:
        
        imax = jj + np.argmax(A.diagonal()[jj:])
        a_max = A[imax, imax]
        a_min = A.diagonal()[jj:].min()
        
        if imax != jj:
            A[[imax, jj], jj:] = A[[jj, imax], jj:]
            A[jj:, [imax, jj]] = A[jj:, [jj, imax]]
            temp = perm[jj]
            perm[jj] = perm[imax]
            perm[imax] = temp
        
#        if (a_max < tb_gam) or (a_min < -(mu * a_max)): break
        
#        for ii in range(jj, N):
##            g[ii] = A[ii, ii] - np.sum(np.abs(A[ii, jj:])) + np.abs(A[ii, ii])
#            g[ii] = A[ii, ii] * (1.0 + np.sign(A[ii, ii])) - np.sum(np.abs(A[ii, jj:])) 
#    
#        g_bound = g[jj:].max()
        Dj = np.sqrt(A[jj, jj])
        aj = A[jj+1:, jj]
        Aj = aj[:, np.newaxis]
        
        L[jj, jj] = Dj
        L[jj+1:, jj] = aj / Dj
        if jj < N - 1:
            A[jj+1:, jj+1:] -= (Aj @ Aj.T) / A[jj, jj]
                
        jj += 1
        
#    P = np.zeros((N, N))
#    for ii in range(N):
#        P[ii, perm[ii]] = 1.0
#    
#    PL = np.empty(L.shape)
#    PL[perm, :] = L
#    
#    Pinv = np.linalg.inv(P)
#    PL2 = Pinv @ L
        
    return L, perm
        
    # Phase 2
#    if jj == N - 1:
#        
#        dlt = -A[jj, jj] + np.maximum(-A[jj, jj] * tau / (1.0 - tau), tb_gam)
#        A[jj, jj] += dlt
#        L[jj, jj] = np.sqrt(A[jj, jj])
#        
#    else:
#        
#        kk = jj - 1
#        for ii in range(jj, N):
#            g[ii] = A[ii, ii] - np.sum(np.abs(A[ii, jj:ii])) - np.sum(np.abs(A[ii+1:, ii]))
        
    return L

#def se99(A_in):
#    
#    A = A_in.copy()
#    
#    N = A.shape[0]
#    
#    eps = np.finfo(float).eps
#    tau = eps**(1.0/3.0)
#    tau_bar = tau**2
#    mu = 0.1
#    
##    phase_one = True
#    gam = np.amax(np.abs(A.diagonal()))
#    tb_gam = tau_bar * gam
#    mu_gam = mu * gam
#    L = np.zeros(A.shape)
#    g = np.zeros(N)
#    jj = 0
#    
#    while (jj < N):
#        
#        imax = jj + np.argmax(A.diagonal()[jj:])
#        a_max = A[imax, imax]
#        a_min = A.diagonal()[jj:].min()
#        
##        if (a_max < tb_gam) or (a_min < -(mu * a_max)): break
#        
#        if imax != jj:
#            A[[imax, jj], :] = A[[jj, imax], :]
#            A[:, [imax, jj]] = A[:, [jj, imax]]
#        
#        A_check = A.diagonal()[jj+1:] - A[jj+1:, jj]**2 / A[jj, jj]
##        if A_check.min() < -mu_gam: break
#    
#        Dj = np.sqrt(A[jj, jj])
#        aj = A[jj+1:, jj]
#        Aj = aj[:, np.newaxis]
#        
#        L[jj, jj] = Dj
#        L[jj+1:, jj] = aj / Dj
#        if jj < N - 1:
#            A[jj+1:, jj+1:] -= (Aj @ Aj.T) / (Dj**2)
#                
#        jj += 1
#        
#    # Phase 2
##    if jj == N - 1:
##        
##        dlt = -A[jj, jj] + np.maximum(-A[jj, jj] * tau / (1.0 - tau), tb_gam)
##        A[jj, jj] += dlt
##        L[jj, jj] = np.sqrt(A[jj, jj])
##        
##    else:
##        
##        kk = jj - 1
##        for ii in range(jj, N):
##            g[ii] = A[ii, ii] - np.sum(np.abs(A[ii, jj:ii])) - np.sum(np.abs(A[ii+1:, ii]))
#        
#    return L

def gchol(A):

    """Return `(L, e)`: the Schnabel-Eskow generalized Cholesky
    decomposition of `M = A + diag(e) = dot(L, L.T)` where

    1) `M` is safely symmetric positive definite (SPD) and well
       conditioned.
    2) `e` is small (zero if `A` is already SPD and not much larger
       than the most negative eigenvalue of `A`)

    .. math::
       \mat{A} + \diag{e} = \mat{L}\mat{L}^{T}

    Parameters
    ----------
    A : array
       Must be a non-singular and symmetric matrix
    test : bool
       If `True`, use the directly translated iterative code for
       testing.

    Returns
    -------
    L : 2d array
       Lower triangular Cholesky factor.
    e : 1d array
       Diagonals of correction matrix `E`.


    Examples
    --------
    >>> np.random.seed(3)
    >>> A = np.random.rand(100,100)*2 - 1
    >>> A = A + A.T
    >>> L, e = sechol(A)
    >>> np.allclose(np.dot(L,L.T), (A + np.diag(e)))
    True
    >>> np.linalg.eigvalsh(A).min()
    ()

    >>> A2 =  np.array([[-0.451, -0.041, 0.124],
    ...                 [-0.041, -0.265, 0.061],
    ...                 [ 0.124, 0.061, -0.517]])
    >>> L, e = sechol(A2, test=True)
    >>> e.max()/np.linalg.eigvalsh(A).min()
    
    """
    _EPS = np.finfo(float).eps
    
    n = A.shape[0]
    phase_one = True
    gamma = abs(A.diagonal()).max()
    j = 0
    tau = _EPS**(1/3)
    taut = tau*tau
    mu = 0.1

    # Phase one, A potentially positive-definite
    L = A
    E = np.zeros(n, dtype=float)
    while j < n and phase_one:
        a_max = A.diagonal()[j:].max()
        a_min = A.diagonal()[j:].min()
        if (a_max < taut*gamma) or (a_min < -mu*a_max):
            phase_one = False
        else:
            # Pivot on maximum diagonal of remaining submatrix
            i = j + np.argmax(A.diagonal()[j:])
            if i != j:
                A[[i,j],:] = A[[j,i],:]
                A[:, [i,j]] = A[:,[j,i]]
            if ((A.diagonal()[j+1:] - A[j+1:,j]**2/A.diagonal()[j]).min()
                < -mu*gamma):
                phase_one = False # go to phase two
            else:
                # perform jth iteration of factorization
                L[j,j] = np.sqrt(A[j,j])     # L[j j] overwrites A[j j]
                for i in range(j+1, n):
                    L[j,j] = A[i,j]/L[j,j] # L[i j] overwrites A[i j]
                    A[i,j+1:i+1] -= L[i,j]*L[j+1:i+1,j]
                    A[i,i] -= L[i,j]**2
                j += 1
    # End phase 1

    # Phase two, A not positive-definite
    if not phase_one and j == n - 1:
        E[-1] = delta = -A[-1,-1] + max(-tau*A[-1,-1]/(1-tau),
                                        taut*gamma)
        A[-1,-1] += delta
        L[-1,-1] = np.sqrt(A[-1,-1])

    delta_p = 0
    g = np.zeros(n, dtype=float)
    if not phase_one and j < n - 1:
        k = j - 1  # k = number of iterations performed in phase one;
        # Calculate lower Gerschgorin bounds of A[k+1]
        for i in range(k + 1, n):
            g[i] = A[i,i] - abs(A[i,k+1:i]).sum() - abs(A[i+1:,i]).sum()
        # Modified Cholesky Decomposition
        for j in range(k + 1, n - 2):
            # Pivot on maximum lower Gerschgorin bound estimate
            i = j + np.argmax(g[j:])
            if i != j:
                A[[i,j],:] = A[[j,i],:]
                A[:, [i,j]] = A[:,[j,i]]
            # Calculate E[j,j] and add to diagonal
            norm_j = abs(A[j+1:n,j]).sum()
            E[j] = delta = max(0,
                               -A[j,j] + max(norm_j, taut*gamma),
                               delta_p)
            if delta > 0:
                A[j,j] += delta
                delta_p = delta         # delta_p will contain E_inf
            # Update Gerschgorin bound estimates
            if A[j,j] != norm_j:
                temp = 1 - norm_j/A[j,j]
                g[j+1:] += abs(A.diagonal()[j+1:])*temp
             # Perform j th iteration of factorization
            L[j,j] = np.sqrt(A[j,j])     # L[j j] overwrites A[j j]
            for i in range(j+1, n):
                L[j,j] = A[i,j]/L[j,j] # L[i j] overwrites A[i j]
                A[i,j+1:i+1] -= L[i,j]*L[j+1:i+1,j]

        # Final 2 by 2 submatrix
        e = np.linalg.eigvalsh(A[-2:,-2:])
        e.sort()
        delta = max(0,
                    -e[0] + max(tau*(e[1] - e[0])/(1 - tau),
                                   taut*gamma),
                    delta_p)
        if delta > 0:
            A[-2,-2] += delta
            A[-1,-1] += delta
            delta_p = delta
        L[-2,-2] = np.sqrt(A[-2,-2])    # overwrites A[-2,-2]
        L[-1,-2] = A[-1, -2]/L[-2,-2]   # overwrites A[-1,-2]
        L[-1,-1] = np.sqrt(A[-1,-1] - L[-1,-2]**2) # overwrites A[-1,-1]

        return np.tril(L), E
