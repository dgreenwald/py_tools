"""Tests for py_tools.numerical.chebyshev"""

import numpy as np

from py_tools.numerical.chebyshev import (
    grid,
    poly,
    basis_and_gradient,
    gradient,
    ChebFcn,
    TensorChebFcn,
)


# --- grid ---


class TestGrid:
    def test_size(self):
        assert len(grid(7)) == 7

    def test_values_in_range(self):
        x = grid(10)
        assert np.all(x >= -1.0) and np.all(x <= 1.0)

    def test_endpoints(self):
        x = grid(5)
        assert np.isclose(x[0], -1.0)
        assert np.isclose(x[-1], 1.0)

    def test_symmetric(self):
        x = grid(6)
        assert np.allclose(x, -x[::-1])


# --- poly ---


class TestPoly:
    def test_shape(self):
        x = np.linspace(-1, 1, 5)
        assert poly(x, 4).shape == (5, 4)

    def test_T0_is_ones(self):
        x = np.array([-0.5, 0.0, 0.5])
        assert np.allclose(poly(x, 1)[:, 0], 1.0)

    def test_T1_is_x(self):
        x = np.array([-0.5, 0.0, 0.5])
        assert np.allclose(poly(x, 2)[:, 1], x)

    def test_T2(self):
        x = np.array([-0.5, 0.0, 0.5])
        assert np.allclose(poly(x, 3)[:, 2], 2 * x**2 - 1)

    def test_T3(self):
        x = np.array([-0.5, 0.0, 0.5])
        assert np.allclose(poly(x, 4)[:, 3], 4 * x**3 - 3 * x)

    def test_n1_is_all_ones(self):
        x = np.array([0.3, -0.7])
        assert np.allclose(poly(x, 1), np.ones((2, 1)))


# --- basis_and_gradient ---


class TestBasisAndGradient:
    def test_basis_matches_poly(self):
        x = np.linspace(-1, 1, 8)
        T, _ = basis_and_gradient(x, 5)
        assert np.allclose(T, poly(x, 5))

    def test_dT0_is_zero(self):
        x = np.array([0.3, -0.5])
        _, dT = basis_and_gradient(x, 3)
        assert np.allclose(dT[:, 0], 0.0)

    def test_dT1_is_one(self):
        x = np.array([0.3, -0.5])
        _, dT = basis_and_gradient(x, 3)
        assert np.allclose(dT[:, 1], 1.0)

    def test_dT2(self):
        # dT_2/dx = 4x
        x = np.array([0.3, -0.5])
        _, dT = basis_and_gradient(x, 3)
        assert np.allclose(dT[:, 2], 4 * x)

    def test_gradient_matches_finite_difference(self):
        x = np.array([0.1, 0.4, -0.3])
        n, h = 5, 1e-6
        _, dT = basis_and_gradient(x, n)
        dT_fd = (poly(x + h, n) - poly(x - h, n)) / (2 * h)
        assert np.allclose(dT, dT_fd, atol=1e-6)


# --- gradient ---


class TestGradient:
    def test_matches_basis_and_gradient(self):
        x = np.linspace(-0.8, 0.8, 6)
        _, dT_ref = basis_and_gradient(x, 4)
        assert np.allclose(gradient(x, 4), dT_ref)


# --- ChebFcn ---


class TestChebFcn:
    def test_grid_in_range(self):
        cf = ChebFcn(6)
        assert np.all(cf.grid >= -1.0) and np.all(cf.grid <= 1.0)

    def test_scaled_grid_in_domain(self):
        cf = ChebFcn(6, lb=0.0, ub=3.0)
        assert np.all(cf.scaled_grid >= 0.0) and np.all(cf.scaled_grid <= 3.0)

    def test_make_grid_returns_copy(self):
        cf = ChebFcn(5)
        g = cf.make_grid()
        g[0] = 999.0
        assert not np.isclose(cf.grid[0], 999.0)

    def test_fit_and_evaluate_polynomial_exact(self):
        # n=5 basis can represent degree-4 polynomials exactly
        cf = ChebFcn(5)

        def f(x):
            return x**4 - 2 * x**2 + 1

        cf.fit_fcn(f)
        x_test = np.linspace(-1, 1, 20)
        assert np.allclose(cf.evaluate(x_test), f(x_test), atol=1e-10)

    def test_fit_and_evaluate_custom_domain(self):
        cf = ChebFcn(6, lb=1.0, ub=4.0)

        def f(x):
            return (x - 2.5) ** 2

        cf.fit_fcn(f)
        x_test = np.linspace(1.0, 4.0, 15)
        assert np.allclose(cf.evaluate(x_test), f(x_test), atol=1e-10)

    def test_gradient_matches_finite_difference(self):
        cf = ChebFcn(6)
        cf.fit_fcn(lambda x: x**3 - x)
        x_test = np.linspace(-0.9, 0.9, 10)
        h = 1e-6
        df_fd = (cf.evaluate(x_test + h) - cf.evaluate(x_test - h)) / (2 * h)
        assert np.allclose(cf.gradient(x_test), df_fd, atol=1e-6)

    def test_evaluate_with_gradient_consistent(self):
        cf = ChebFcn(7)
        cf.fit_fcn(lambda x: np.sin(np.pi * x))
        x_test = np.linspace(-0.8, 0.8, 8)
        fx, dfx = cf.evaluate_with_gradient(x_test)
        assert np.allclose(fx, cf.evaluate(x_test))
        assert np.allclose(dfx, cf.gradient(x_test))

    def test_scale_to_from_grid_inverse(self):
        cf = ChebFcn(5, lb=2.0, ub=7.0)
        x = np.linspace(2.0, 7.0, 10)
        assert np.allclose(cf.scale_from_grid(cf.scale_to_grid(x)), x)

    def test_scale_x_clamps_to_bounds(self):
        cf = ChebFcn(5, lb=0.0, ub=1.0)
        x = np.array([-0.5, 0.5, 1.5])
        x_clamped = np.array([0.0, 0.5, 1.0])
        assert np.allclose(cf.scale_x(x), cf.scale_to_grid(x_clamped))


# --- TensorChebFcn ---


class TestTensorChebFcn:
    def test_fit_and_evaluate_linear(self):
        # f(x, y) = x + y is degree 1; exact with 2 nodes per dimension
        cf = TensorChebFcn([2, 2])
        cf.fit_fcn(lambda X: X[:, 0] + X[:, 1])
        rng = np.random.default_rng(0)
        x_test = rng.uniform(-1, 1, (20, 2))
        assert np.allclose(cf.evaluate(x_test), x_test[:, 0] + x_test[:, 1], atol=1e-10)

    def test_fit_and_evaluate_custom_domain(self):
        # f(x, y) = x * y over [0, 2]^2; degree 1 in each, exact with 2x2 basis
        cf = TensorChebFcn([2, 2], lb=0.0, ub=2.0)
        cf.fit_fcn(lambda X: X[:, 0] * X[:, 1])
        rng = np.random.default_rng(1)
        x_test = rng.uniform(0, 2, (20, 2))
        assert np.allclose(cf.evaluate(x_test), x_test[:, 0] * x_test[:, 1], atol=1e-10)
