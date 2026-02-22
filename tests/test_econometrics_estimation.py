import numpy as np
import pytest
from py_tools.econometrics.estimation import objfcn, se_nls, nls


# Simple linear model: y = a*x + b, err = y - (a*x + b)
# data: (x, y) pairs
# params: [a, b]
def _err_linear(params, x, y):
    return y - (params[0] * x + params[1])


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, size=100)
    a_true, b_true = 2.5, -1.0
    y = a_true * x + b_true + rng.normal(scale=0.1, size=100)
    return x, y, a_true, b_true


class TestObjfcn:
    def test_zero_at_perfect_fit(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        params = np.array([2.0, 0.0])
        val = objfcn(params, _err_linear, x, y)
        assert abs(val) < 1e-12

    def test_positive_with_error(self, linear_data):
        x, y, a_true, b_true = linear_data
        params = np.array([0.0, 0.0])
        val = objfcn(params, _err_linear, x, y)
        assert val > 0

    def test_returns_scalar(self, linear_data):
        x, y, a_true, b_true = linear_data
        params = np.array([a_true, b_true])
        val = objfcn(params, _err_linear, x, y)
        assert np.isscalar(val) or val.ndim == 0


class TestSeNls:
    def test_output_shapes(self, linear_data):
        x, y, a_true, b_true = linear_data
        params = np.array([a_true, b_true])
        se, V, e = se_nls(_err_linear, params, args=(x, y))
        assert se.shape == (2,)
        assert V.shape == (2, 2)
        assert e.shape == (100,)

    def test_V_symmetric(self, linear_data):
        x, y, a_true, b_true = linear_data
        params = np.array([a_true, b_true])
        se, V, e = se_nls(_err_linear, params, args=(x, y))
        np.testing.assert_allclose(V, V.T, atol=1e-10)

    def test_se_positive(self, linear_data):
        x, y, a_true, b_true = linear_data
        params = np.array([a_true, b_true])
        se, V, e = se_nls(_err_linear, params, args=(x, y))
        assert np.all(se > 0)

    def test_residuals_correct(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 5.0, 8.0])
        params = np.array([2.0, 0.0])
        se, V, e = se_nls(_err_linear, params, args=(x, y))
        expected_e = y - (params[0] * x + params[1])
        np.testing.assert_allclose(e, expected_e, atol=1e-12)


class TestNls:
    def test_recovers_true_params(self, linear_data):
        x, y, a_true, b_true = linear_data
        b0 = np.array([1.0, 0.0])
        out = nls(_err_linear, b0, args=(x, y))
        np.testing.assert_allclose(out["b_hat"], [a_true, b_true], atol=0.05)

    def test_output_keys(self, linear_data):
        x, y, a_true, b_true = linear_data
        b0 = np.array([a_true, b_true])
        out = nls(_err_linear, b0, args=(x, y))
        assert set(out.keys()) == {"b_hat", "e_hat", "V", "se", "res"}

    def test_se_shape(self, linear_data):
        x, y, a_true, b_true = linear_data
        b0 = np.array([a_true, b_true])
        out = nls(_err_linear, b0, args=(x, y))
        assert out["se"].shape == (2,)
        assert out["V"].shape == (2, 2)
        assert out["e_hat"].shape == (100,)

    def test_optimization_successful(self, linear_data):
        x, y, a_true, b_true = linear_data
        b0 = np.array([1.0, 0.0])
        out = nls(_err_linear, b0, args=(x, y))
        assert out["res"].success
