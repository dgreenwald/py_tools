"""Tests for py_tools.econ.financial"""

import numpy as np
import pytest

from py_tools.econ.financial import get_coupon


class TestGetCoupon:
    def test_near_zero_rate_approaches_one_over_years(self):
        # As rate -> 0, coupon rate -> 1 / (years * freq) (pure principal repayment)
        freq = 1
        years = 30
        c = get_coupon(1e-8, freq=freq, years=years)
        assert np.isclose(c, 1.0 / (years * freq), rtol=1e-3)

    def test_positive_rate_positive_coupon(self):
        c = get_coupon(0.05, freq=1, years=30)
        assert c > 0

    def test_higher_rate_higher_coupon(self):
        c1 = get_coupon(0.03, freq=1, years=30)
        c2 = get_coupon(0.06, freq=1, years=30)
        assert c2 > c1

    def test_shorter_maturity_higher_coupon(self):
        # For same rate, shorter maturity => larger periodic coupon rate
        c_short = get_coupon(0.05, freq=1, years=5)
        c_long = get_coupon(0.05, freq=1, years=30)
        assert c_short > c_long

    def test_semiannual_vs_annual(self):
        # Semi-annual coupon (freq=2) on a 30-year bond at 5%
        c = get_coupon(0.05, freq=2, years=30)
        assert c > 0

    def test_known_value_annual_perpetuity_limit(self):
        # As years -> infinity, coupon rate -> rm_t (the periodic rate)
        rm = 0.05
        freq = 1
        rm_t = (1.0 + rm) ** (1.0 / float(freq)) - 1.0
        c_large = get_coupon(rm, freq=freq, years=10000)
        assert np.isclose(c_large, rm_t, atol=1e-4)

    def test_coupon_rate_is_scalar(self):
        c = get_coupon(0.04, freq=4, years=10)
        assert np.isscalar(c) or (isinstance(c, np.ndarray) and c.ndim == 0)
