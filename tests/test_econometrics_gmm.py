import numpy as np
import pytest
from py_tools.econometrics.gmm import (
    compute_g, compute_dg, compute_d2g,
    obj_gmm, jac_gmm, hess_gmm, solve_gmm,
)


# Simple moment condition: h(x_t, params) = x_t - params[0]
# At params = mean(x), g = 0; dg = -1; d2g = 0
def _h(x_t, params):
    return x_t - params

def _dh(x_t, params):
    return -np.ones((1, 1))

_d2h = [lambda x_t, params: np.zeros((1, 1))]


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    Nt = 20
    x = rng.normal(loc=3.0, size=Nt)
    data = x[np.newaxis, :]   # shape (1, Nt)
    return data, x


class TestComputeG:

    def test_at_true_mean(self, simple_data):
        data, x = simple_data
        params = np.array([np.mean(x)])
        g = compute_g(params, _h, data)
        np.testing.assert_allclose(g, [0.0], atol=1e-12)

    def test_away_from_mean(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        g = compute_g(params, _h, data)
        np.testing.assert_allclose(g, [np.mean(x)], atol=1e-12)

    def test_single_obs(self):
        data = np.array([[5.0]])
        params = np.array([2.0])
        g = compute_g(params, _h, data)
        np.testing.assert_allclose(g, [3.0], atol=1e-12)

    def test_shape(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        g = compute_g(params, _h, data)
        assert g.shape == (1,)


class TestComputeDg:

    def test_value(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        dg = compute_dg(params, _dh, data)
        np.testing.assert_allclose(dg, [[-1.0]], atol=1e-12)

    def test_shape(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        dg = compute_dg(params, _dh, data)
        assert dg.shape == (1, 1)

    def test_single_obs(self):
        data = np.array([[5.0]])
        params = np.array([2.0])
        dg = compute_dg(params, _dh, data)
        np.testing.assert_allclose(dg, [[-1.0]], atol=1e-12)


class TestComputeD2g:

    def test_zero_second_deriv(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        d2g = compute_d2g(params, _d2h, data)
        assert len(d2g) == 1
        np.testing.assert_allclose(d2g[0], [[0.0]], atol=1e-12)

    def test_single_obs(self):
        data = np.array([[5.0]])
        params = np.array([2.0])
        d2g = compute_d2g(params, _d2h, data)
        assert len(d2g) == 1
        np.testing.assert_allclose(d2g[0], [[0.0]], atol=1e-12)


class TestObjGmm:

    def test_zero_at_true_mean(self, simple_data):
        data, x = simple_data
        params = np.array([np.mean(x)])
        W = np.eye(1)
        val = obj_gmm(params, _h, data, W)
        assert abs(val) < 1e-24

    def test_positive_away_from_mean(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        W = np.eye(1)
        val = obj_gmm(params, _h, data, W)
        assert val > 0


class TestJacGmm:

    def test_shape(self, simple_data):
        data, x = simple_data
        params = np.array([0.0])
        W = np.eye(1)
        jac = jac_gmm(params, _h, data, W, dh=_dh)
        assert jac.shape == (1,)

    def test_sign(self, simple_data):
        # Moment > 0 when params < mean, gradient w.r.t. params should be negative
        data, x = simple_data
        params = np.array([0.0])
        W = np.eye(1)
        jac = jac_gmm(params, _h, data, W, dh=_dh)
        # dg = -1, g > 0 => jac = dg^T W g < 0
        assert jac[0] < 0


class TestSolveGmm:

    def test_recovers_mean(self, simple_data):
        # dogleg requires hess to share the same args tuple as the objective,
        # which precludes passing d2h separately; use BFGS with analytic jac
        data, x = simple_data
        params_guess = np.array([0.0])
        W = np.eye(1)
        res = solve_gmm(params_guess, _h, data, W=W, dh=_dh, algorithm='BFGS')
        np.testing.assert_allclose(res.x, [np.mean(x)], atol=1e-6)

    def test_recovers_mean_bfgs(self, simple_data):
        data, x = simple_data
        params_guess = np.array([0.0])
        res = solve_gmm(params_guess, _h, data, algorithm='BFGS')
        np.testing.assert_allclose(res.x, [np.mean(x)], atol=1e-6)

    def test_default_W_identity(self, simple_data):
        data, x = simple_data
        params_guess = np.array([0.0])
        # Should work without providing W
        res = solve_gmm(params_guess, _h, data)
        np.testing.assert_allclose(res.x, [np.mean(x)], atol=1e-6)
