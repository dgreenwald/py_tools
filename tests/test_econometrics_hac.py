import numpy as np
import pytest

from py_tools.econometrics.hac import (
    gmm_jt_inefficient,
    long_run_cov,
    no_overlap_cov,
    sandwich,
    simplified_hh,
    wald,
)


def test_long_run_cov_weights_and_normalization():
    u = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 3.0], [4.0, 1.0]])
    base = u.T @ u / 4
    lag1 = (u[1:].T @ u[:-1] + u[:-1].T @ u[1:]) / 4
    lag2 = (u[2:].T @ u[:-2] + u[:-2].T @ u[2:]) / 4
    np.testing.assert_allclose(long_run_cov(u, 2, "hh"), base + lag1 + lag2)
    np.testing.assert_allclose(long_run_cov(u, 2, "truncated"), base + lag1 + lag2)
    np.testing.assert_allclose(long_run_cov(u, 2, "nw"), base + 0.5 * lag1)
    np.testing.assert_allclose(
        long_run_cov(u, 2, "bartlett"), base + 2 / 3 * lag1 + lag2 / 3
    )


def test_cp_nw18_equals_bartlett_maxlags17():
    u = np.random.default_rng(16).normal(size=(50, 3))
    np.testing.assert_allclose(long_run_cov(u, 18, "nw"),
                               long_run_cov(u, 17, "bartlett"), atol=1e-12)


def test_simplified_hh_and_no_overlap_small_arrays():
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    e = np.array([1.0, -2.0, 3.0, -4.0])
    expected_simple = np.mean(e**2) * long_run_cov(x, 2, "nw")
    np.testing.assert_allclose(simplified_hh(x, e, 2), expected_simple)

    g = x * e[:, None]
    expected_no_overlap = g[::2].T @ g[::2] / 2 + g[1::2].T @ g[1::2] / 2
    np.testing.assert_allclose(no_overlap_cov(x, e, step=2), expected_no_overlap)


def test_sandwich_wald_and_inefficient_jt():
    d = np.diag([2.0, 3.0])
    S = np.diag([4.0, 9.0])
    np.testing.assert_allclose(sandwich(d, S, 10), np.eye(2) / 10)
    assert wald([1.0, 2.0], np.eye(2), R=[[1.0, -1.0]], r=[0]) == pytest.approx(0.5)

    # One estimated parameter and one overidentifying restriction.
    g = np.array([0.0, 0.2])
    S = np.eye(2)
    d = np.array([[1.0], [0.0]])
    a = np.array([[1.0, 0.0]])
    assert gmm_jt_inefficient(g, S, d, a, T=100, df=1) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="rank"):
        gmm_jt_inefficient(g, S, d, a, T=100, df=2)
