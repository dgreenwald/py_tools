import numpy as np
import pytest
from py_tools.econometrics.bootstrap import ar1_bootstrap_inner, objfcn_ar1_bootstrap


@pytest.fixture
def ar1_setup():
    rng = np.random.default_rng(7)
    rho = 0.8
    mu = 2.0
    Nboot = 500
    Nt_eps = 50
    eps_boot = rng.normal(scale=0.5, size=(Nboot, Nt_eps))
    x_init = mu
    return rho, mu, eps_boot, x_init, Nboot, Nt_eps


class TestAr1BootstrapInner:
    def test_returns_scalar(self, ar1_setup):
        rho, mu, eps_boot, x_init, Nboot, Nt_eps = ar1_setup
        result = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        assert np.isscalar(result) or np.array(result).ndim == 0

    def test_near_true_rho(self, ar1_setup):
        # With enough bootstrap draws, simulated rho should be near true rho
        rho, mu, eps_boot, x_init, Nboot, Nt_eps = ar1_setup
        result = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        assert abs(result - rho) < 0.15

    def test_rho_zero_near_zero(self):
        # If rho=0, AR(1) is white noise; OLS rho should average near 0
        rng = np.random.default_rng(99)
        rho = 0.0
        mu = 0.0
        Nboot = 1000
        Nt_eps = 100
        eps_boot = rng.normal(size=(Nboot, Nt_eps))
        x_init = 0.0
        result = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        assert abs(result) < 0.1

    def test_output_shape_single_boot(self):
        rng = np.random.default_rng(1)
        eps_boot = rng.normal(size=(1, 20))
        result = ar1_bootstrap_inner(0.5, 1.0, eps_boot, 1.0)
        assert np.isscalar(result) or np.array(result).ndim == 0

    def test_high_rho(self):
        rng = np.random.default_rng(3)
        rho = 0.95
        mu = 0.0
        Nboot = 1000
        Nt_eps = 100
        eps_boot = rng.normal(scale=0.1, size=(Nboot, Nt_eps))
        x_init = 0.0
        result = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        # Should be biased downward from OLS, but still > 0.8
        assert result > 0.8


class TestObjfcnAr1Bootstrap:
    def test_zero_at_solution(self, ar1_setup):
        rho, mu, eps_boot, x_init, Nboot, Nt_eps = ar1_setup
        rho_sim = ar1_bootstrap_inner(rho, mu, eps_boot, x_init)
        # If rho_ols == rho_sim_avg, objective is 0
        val = objfcn_ar1_bootstrap(rho, rho_sim, mu, eps_boot, x_init)
        assert abs(val) < 1e-10

    def test_sign_above(self, ar1_setup):
        rho, mu, eps_boot, x_init, Nboot, Nt_eps = ar1_setup
        # rho_ols > rho_sim => positive objective
        rho_ols = 0.99
        val = objfcn_ar1_bootstrap(rho, rho_ols, mu, eps_boot, x_init)
        assert val > 0

    def test_sign_below(self, ar1_setup):
        rho, mu, eps_boot, x_init, Nboot, Nt_eps = ar1_setup
        # rho_ols < rho_sim => negative objective
        rho_ols = 0.0
        val = objfcn_ar1_bootstrap(rho, rho_ols, mu, eps_boot, x_init)
        assert val < 0
