"""Tests for py_tools.numerical.newton"""

import numpy as np
import pytest

from py_tools.numerical.newton import secant, root


# --- secant ---

class TestSecant:
    def test_simple_root(self):
        # f(x) = x^2 - 4, root at x=2
        f = lambda x, : x ** 2 - 4.0
        result = secant(f, (), x0=1.0, x1=3.0, verbose=False)
        assert result is not None
        assert np.isclose(result, 2.0, atol=1e-6)

    def test_residual_at_solution(self):
        f = lambda x, : x ** 3 - 8.0
        result = secant(f, (), x0=1.5, x1=2.5, verbose=False)
        assert result is not None
        assert np.isclose(f(result), 0.0, atol=1e-6)

    def test_with_args(self):
        # f(x; a) = x^2 - a, root at sqrt(a)
        f = lambda x, a: x ** 2 - a
        result = secant(f, (9.0,), x0=2.0, x1=4.0, verbose=False)
        assert result is not None
        assert np.isclose(result, 3.0, atol=1e-6)

    def test_returns_none_when_no_convergence(self):
        # Function that never decreases sufficiently from these starting points
        f = lambda x, : np.sin(x) * 0.0 + 1.0  # constant f=1, no root
        result = secant(f, (), x0=0.0, x1=1.0, max_it_outer=5, verbose=False)
        assert result is None

    def test_verbose_false_suppresses_output(self, capsys):
        f = lambda x, : x - 1.0
        secant(f, (), x0=0.0, x1=2.0, verbose=False)
        assert capsys.readouterr().out == ""

    def test_verbose_true_prints_output(self, capsys):
        f = lambda x, : x - 1.0
        secant(f, (), x0=0.0, x1=2.0, verbose=True)
        assert "Iteration" in capsys.readouterr().out


# --- root ---

class TestRoot:
    def test_scalar_system(self):
        # f(x) = x^2 - 4, root at x=2
        f = lambda x: np.array([x[0] ** 2 - 4.0])
        res = root(f, x0=[3.0], verbose=False)
        assert res['success']
        assert np.isclose(res['x'][0], 2.0, atol=1e-8)

    def test_linear_system(self):
        # f(x) = Ax - b, root at x = A^{-1} b
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 10.0])
        f = lambda x: A @ x - b
        x_true = np.linalg.solve(A, b)
        res = root(f, x0=[0.0, 0.0], verbose=False)
        assert res['success']
        assert np.allclose(res['x'], x_true, atol=1e-8)

    def test_nonlinear_system(self):
        # f(x) = [x0^2 + x1^2 - 1, x0 - x1], root at (1/sqrt(2), 1/sqrt(2))
        f = lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[0] - x[1]])
        res = root(f, x0=[1.0, 0.5], verbose=False)
        assert res['success']
        assert np.allclose(res['f_val'], 0.0, atol=1e-8)

    def test_with_analytic_jacobian(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([4.0, 3.0])
        f = lambda x: A @ x - b
        jac = lambda x: A
        x_true = np.linalg.solve(A, b)
        res = root(f, x0=[0.0, 0.0], grad=jac, verbose=False)
        assert res['success']
        assert np.allclose(res['x'], x_true, atol=1e-8)

    def test_result_fields_on_success(self):
        f = lambda x: np.array([x[0] - 3.0])
        res = root(f, x0=[0.0], verbose=False)
        assert res['success'] is True
        assert 'x' in res
        assert 'f_val' in res
        assert 'dist' in res
        assert res['dist'] < 1e-8

    def test_failure_max_iterations(self):
        # Highly nonlinear function that won't converge from far away in 2 steps
        f = lambda x: np.array([np.sin(x[0]) - 2.0])  # no real root
        res = root(f, x0=[0.0], max_iterations=2, verbose=False)
        assert res['success'] is False

    def test_verbose_false_suppresses_output(self, capsys):
        f = lambda x: np.array([x[0] - 1.0])
        root(f, x0=[0.0], verbose=False)
        assert capsys.readouterr().out == ""

    def test_verbose_true_prints_output(self, capsys):
        f = lambda x: np.array([x[0] - 1.0])
        root(f, x0=[0.0], verbose=True)
        assert "Iteration" in capsys.readouterr().out
