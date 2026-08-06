import numpy as np
import pytest

from py_tools.bayesian.prior import Prior, get_prior


def test_get_prior_none_returns_none():
    assert get_prior(None) is None


def test_get_prior_invalid_type_raises():
    with pytest.raises(ValueError):
        get_prior("not_a_prior", mean=0.0, sd=1.0)

    with pytest.raises(TypeError):
        get_prior(3.14, mean=0.0, sd=1.0)


def test_get_prior_requires_mean_and_sd():
    with pytest.raises(ValueError, match="mean and sd"):
        get_prior("norm", mean=0.0, sd=None)


def test_prior_logpdf_and_sample_shapes():
    p = Prior()
    p.add("norm", mean=0.0, sd=1.0, name="a")
    p.add("gamma", mean=2.0, sd=1.0, name="b")

    vals = np.array([0.2, 1.5])
    lp = p.logpdf(vals)
    assert np.isfinite(lp)

    draws = p.sample(7)
    assert draws.shape == (2, 7)


def test_prior_sample_raises_with_flat_component():
    p = Prior()
    p.add("norm", mean=0.0, sd=1.0)
    p.add(None)

    with pytest.raises(ValueError):
        p.sample(5)


def test_prior_logpdf_ignores_flat_component():
    p = Prior()
    p.add("norm", mean=0.0, sd=1.0)
    p.add(None)
    vals = np.array([0.1, 123.0])
    lp = p.logpdf(vals)
    assert np.isfinite(lp)
