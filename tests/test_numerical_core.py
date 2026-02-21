"""Tests for py_tools.numerical.core"""

import numpy as np
import pytest

from py_tools.numerical.core import (
    quad_form,
    rsolve,
    gradient,
    hessian,
    svd_inv,
    ghquad_norm,
    gauss_legendre_norm,
    logit,
    logistic,
    bound_transform,
    robust_cholesky,
    my_chol,
)


# --- quad_form ---

class TestQuadForm:
    def test_identity_weight(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(quad_form(A, np.eye(2)), A.T @ A)

    def test_general(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        X = np.array([[3.0, 1.0], [1.0, 2.0]])
        assert np.allclose(quad_form(A, X), A.T @ X @ A)


# --- rsolve ---

class TestRsolve:
    def test_recovers_b(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([[5.0, 7.0], [1.0, 2.0]])
        assert np.allclose(rsolve(b, A) @ A, b)

    def test_diagonal_system(self):
        A = np.diag([2.0, 4.0])
        b = np.array([[6.0, 8.0]])
        assert np.allclose(rsolve(b, A), np.array([[3.0, 2.0]]))


# --- gradient ---

class TestGradient:
    def test_two_sided(self):
        f = lambda x: np.sum(x ** 2)
        x = np.array([1.0, 2.0, 3.0])
        grad = gradient(f, x)
        assert np.allclose(grad.ravel(), 2 * x, atol=1e-8)

    def test_one_sided(self):
        f = lambda x: np.sum(x ** 2)
        x = np.array([1.0, 2.0, 3.0])
        grad = gradient(f, x, two_sided=False)
        assert np.allclose(grad.ravel(), 2 * x, atol=1e-4)

    def test_vector_valued(self):
        # f(x) = x^2 elementwise, Jacobian = diag(2x)
        f = lambda x: x ** 2
        x = np.array([1.0, 3.0])
        grad = gradient(f, x)
        assert np.allclose(grad, np.diag(2 * x), atol=1e-8)

    def test_does_not_mutate_x(self):
        f = lambda x: np.sum(x ** 2)
        x = np.array([1.0, 2.0, 3.0])
        x_orig = x.copy()
        gradient(f, x)
        assert np.array_equal(x, x_orig)


# --- hessian ---

class TestHessian:
    def test_quadratic(self):
        Q = np.array([[2.0, 1.0], [1.0, 3.0]])
        f = lambda x: x @ Q @ x
        H = hessian(f, np.array([1.0, 2.0]))
        assert np.allclose(H, 2 * Q, atol=1e-6)

    def test_diagonal(self):
        f = lambda x: np.sum(x ** 2)
        H = hessian(f, np.zeros(3))
        assert np.allclose(H, 2 * np.eye(3), atol=1e-6)


# --- svd_inv ---

class TestSvdInv:
    def test_full_rank(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        assert np.allclose(A @ svd_inv(A), np.eye(2), atol=1e-10)

    def test_rank_deficient(self):
        # Moore-Penrose pseudoinverse satisfies A @ A+ @ A = A
        v = np.array([1.0, 2.0, 3.0])
        A = np.outer(v, v)
        A_inv = svd_inv(A)
        assert np.allclose(A @ A_inv @ A, A, atol=1e-10)


# --- ghquad_norm ---

class TestGhquadNorm:
    def test_weights_sum_to_one(self):
        _, w = ghquad_norm(5)
        assert np.isclose(np.sum(w), 1.0)

    def test_mean(self):
        mu = 2.0
        x, w = ghquad_norm(10, mu=mu)
        assert np.isclose(w @ x, mu, atol=1e-10)

    def test_variance(self):
        mu, sig = 1.0, 2.0
        x, w = ghquad_norm(10, mu=mu, sig=sig)
        assert np.isclose(w @ (x - mu) ** 2, sig ** 2, atol=1e-10)


# --- gauss_legendre_norm ---

class TestGaussLegendreNorm:
    def test_weights_integrate_constant(self):
        a, b = 1.0, 4.0
        _, w = gauss_legendre_norm(5, a=a, b=b)
        assert np.isclose(np.sum(w), b - a)

    def test_exact_cubic(self):
        # n=2 Gauss-Legendre is exact for polynomials of degree <= 3
        a, b = 0.0, 2.0
        x, w = gauss_legendre_norm(2, a=a, b=b)
        assert np.isclose(w @ x ** 3, (b ** 4 - a ** 4) / 4.0, atol=1e-12)

    def test_nodes_in_interval(self):
        a, b = -2.0, 3.0
        x, _ = gauss_legendre_norm(5, a=a, b=b)
        assert np.all(x >= a) and np.all(x <= b)


# --- logit / logistic ---

class TestLogitLogistic:
    def test_logistic_inverts_logit(self):
        x = np.array([0.2, 0.5, 0.8])
        assert np.allclose(logistic(logit(x)), x)

    def test_logit_inverts_logistic(self):
        y = np.array([-1.0, 0.0, 1.5])
        assert np.allclose(logit(logistic(y)), y)

    def test_custom_bounds(self):
        lb, ub = 2.0, 5.0
        x = np.array([2.5, 3.0, 4.5])
        assert np.allclose(logistic(logit(x, lb=lb, ub=ub), lb=lb, ub=ub), x)


# --- bound_transform ---

class TestBoundTransform:
    def test_both_bounds_roundtrip(self):
        lb = np.array([0.0,  1.0, -2.0])
        ub = np.array([1.0,  3.0,  0.0])
        x  = np.array([0.3,  2.0, -1.0])
        u = bound_transform(x, lb, ub, to_bdd=False)
        assert np.allclose(bound_transform(u, lb, ub, to_bdd=True), x, atol=1e-12)

    def test_lb_only_roundtrip(self):
        lb = np.array([0.0, 1.0])
        ub = np.array([np.inf, np.inf])
        x  = np.array([0.5, 2.5])
        u = bound_transform(x, lb, ub, to_bdd=False)
        assert np.allclose(bound_transform(u, lb, ub, to_bdd=True), x, atol=1e-12)

    def test_ub_only_roundtrip(self):
        lb = np.array([-np.inf, -np.inf])
        ub = np.array([1.0, 3.0])
        x  = np.array([0.5, 2.0])
        u = bound_transform(x, lb, ub, to_bdd=False)
        assert np.allclose(bound_transform(u, lb, ub, to_bdd=True), x, atol=1e-12)


# --- robust_cholesky ---

class TestRobustCholesky:
    def test_pd_matrix(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = robust_cholesky(A)
        assert np.allclose(L @ L.T, A, atol=1e-10)

    def test_output_is_psd(self):
        # One eigenvalue very close to zero — clamping should make output PSD
        A = np.diag([1e-20, 1.0])
        L = robust_cholesky(A)
        assert np.all(np.linalg.eigvalsh(L @ L.T) >= 0)


# --- my_chol ---

class TestMyChol:
    def test_lower_triangular(self):
        A = np.array([[4.0, 2.0, 1.0],
                      [2.0, 5.0, 3.0],
                      [1.0, 3.0, 6.0]])
        assert np.allclose(np.triu(my_chol(A), 1), 0)

    def test_reconstruction(self):
        A = np.array([[4.0, 2.0, 1.0],
                      [2.0, 5.0, 3.0],
                      [1.0, 3.0, 6.0]])
        L = my_chol(A)
        assert np.allclose(L @ L.T, A, atol=1e-10)

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        M = rng.standard_normal((4, 4))
        A = M @ M.T + np.eye(4)
        assert np.allclose(my_chol(A), np.linalg.cholesky(A), atol=1e-10)

    def test_does_not_mutate(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        A_orig = A.copy()
        my_chol(A)
        assert np.array_equal(A, A_orig)
