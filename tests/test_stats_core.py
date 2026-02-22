"""Tests for py_tools.stats.core"""

import numpy as np
import pytest
from scipy.stats import norm

from py_tools.stats.core import (
    weighted_quantile,
    wq_by_col,
    weighted_mean,
    weighted_var,
    weighted_std,
    std_norm_z_star,
    std_norm_bands,
    draw_norm,
    draw_norm_multi,
    my_lognorm,
)


# --- weighted_quantile ---

class TestWeightedQuantile:
    def test_uniform_weights_matches_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        w = np.ones(100)
        q = [0.25, 0.5, 0.75]
        assert np.allclose(weighted_quantile(x, w, q), np.quantile(x, q), atol=0.05)

    def test_single_finite_value(self):
        values = np.array([5.0, np.nan, np.nan])
        weights = np.array([1.0, 0.0, 0.0])
        result = weighted_quantile(values, weights, [0.25, 0.75])
        assert np.allclose(result, 5.0)

    def test_all_nonfinite_returns_nan(self):
        values = np.array([np.nan, np.nan])
        weights = np.array([1.0, 1.0])
        result = weighted_quantile(values, weights, [0.5])
        assert np.all(np.isnan(result))

    def test_zero_weight_excluded(self):
        values = np.array([1.0, 999.0])
        weights = np.array([1.0, 0.0])
        result = weighted_quantile(values, weights, [0.5])
        assert np.isclose(result[0], 1.0)

    def test_sort_false_matches_sort_true(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # already sorted
        w = np.ones(5)
        q = [0.25, 0.5, 0.75]
        assert np.allclose(
            weighted_quantile(x, w, q, sort=True),
            weighted_quantile(x, w, q, sort=False),
        )

    def test_higher_weight_shifts_median(self):
        values = np.array([1.0, 10.0])
        weights = np.array([0.01, 0.99])
        result = weighted_quantile(values, weights, [0.5])
        assert result[0] > 5.0


# --- wq_by_col ---

class TestWqByCol:
    def test_shape(self):
        values = np.random.default_rng(1).standard_normal((20, 3))
        weights = np.ones(20)
        result = wq_by_col(values, weights, [0.25, 0.5, 0.75])
        assert result.shape == (3, 3)

    def test_matches_weighted_quantile_per_column(self):
        rng = np.random.default_rng(2)
        values = rng.standard_normal((30, 2))
        weights = np.ones(30)
        q = [0.25, 0.75]
        result = wq_by_col(values, weights, q)
        for icol in range(2):
            expected = weighted_quantile(values[:, icol], weights, q)
            assert np.allclose(result[:, icol], expected)


# --- weighted_mean ---

class TestWeightedMean:
    def test_equal_weights_matches_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(weighted_mean(x, np.ones(4)), np.mean(x))

    def test_custom_weights(self):
        # mean of [0, 10] with weights [3, 1] = 2.5
        x = np.array([0.0, 10.0])
        w = np.array([3.0, 1.0])
        assert np.isclose(weighted_mean(x, w), 2.5)


# --- weighted_var ---

class TestWeightedVar:
    def test_equal_weights_matches_variance(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(weighted_var(x, np.ones(5)), np.var(x))

    def test_constant_values_zero_variance(self):
        x = np.full(5, 3.0)
        assert np.isclose(weighted_var(x, np.ones(5)), 0.0)


# --- weighted_std ---

class TestWeightedStd:
    def test_matches_sqrt_of_var(self):
        x = np.array([1.0, 3.0, 5.0, 7.0])
        w = np.array([1.0, 2.0, 2.0, 1.0])
        assert np.isclose(weighted_std(x, w), np.sqrt(weighted_var(x, w)))


# --- std_norm_z_star ---

class TestStdNormZStar:
    def test_two_sided(self):
        assert np.isclose(std_norm_z_star(0.95, two_sided=True), norm.ppf(0.975))

    def test_one_sided(self):
        assert np.isclose(std_norm_z_star(0.95, two_sided=False), norm.ppf(0.95))

    def test_positive_for_p_gt_half(self):
        assert std_norm_z_star(0.9) > 0


# --- std_norm_bands ---

class TestStdNormBands:
    def test_symmetric(self):
        z_lb, z_ub = std_norm_bands(0.95)
        assert np.isclose(z_lb, -z_ub)

    def test_95_percent(self):
        _, z_ub = std_norm_bands(0.95)
        assert np.isclose(z_ub, norm.ppf(0.975))

    def test_bounds_ordered(self):
        z_lb, z_ub = std_norm_bands(0.90)
        assert z_lb < z_ub


# --- draw_norm ---

class TestDrawNorm:
    def test_shape(self):
        assert draw_norm(np.eye(3)).shape == (3,)

    def test_sample_covariance(self):
        np.random.seed(42)
        Sig = np.array([[2.0, 1.0], [1.0, 1.0]])
        samples = np.array([draw_norm(Sig) for _ in range(5000)])
        np.random.seed(None)
        assert np.allclose(np.cov(samples.T), Sig, atol=0.1)


# --- draw_norm_multi ---

class TestDrawNormMulti:
    def test_shape(self):
        assert draw_norm_multi(np.eye(4), 10).shape == (10, 4)


# --- my_lognorm ---

class TestMyLognorm:
    def test_mean(self):
        mu, sig = 1.0, 0.5
        dist = my_lognorm(mu, sig)
        assert np.isclose(dist.mean(), np.exp(mu + 0.5 * sig ** 2), rtol=1e-6)
