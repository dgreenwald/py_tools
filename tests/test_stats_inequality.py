"""Tests for py_tools.stats.inequality"""

import numpy as np
import pytest

pandas = pytest.importorskip("pandas")

from py_tools.stats.inequality import compute_gini, get_top_shares  # noqa: E402


# --- compute_gini ---


class TestComputeGini:
    def test_known_gini_two_groups(self):
        # incomes [1, 2] equal weight → Gini = 1/3 (analytically)
        df = pandas.DataFrame({"income": [1.0, 2.0], "weight": [1.0, 1.0]})
        gini, _, _ = compute_gini(df, "income", wvar="weight")
        assert np.isclose(gini, 1.0 / 3.0, atol=1e-6)

    def test_no_wvar_uses_counts(self):
        # Equal counts per group → same Gini as equal-weight case
        df = pandas.DataFrame({"income": [1.0, 1.0, 2.0, 2.0]})
        gini, _, _ = compute_gini(df, "income")
        assert np.isclose(gini, 1.0 / 3.0, atol=1e-6)

    def test_returns_three_values(self):
        df = pandas.DataFrame({"income": [1.0, 2.0, 3.0]})
        result = compute_gini(df, "income")
        assert len(result) == 3

    def test_cumulative_weight_ends_at_one(self):
        df = pandas.DataFrame({"income": [1.0, 2.0, 3.0]})
        _, c_weight, _ = compute_gini(df, "income")
        assert np.isclose(c_weight[-1], 1.0)

    def test_cumulative_shares_ends_at_one(self):
        df = pandas.DataFrame({"income": [1.0, 2.0, 3.0]})
        _, _, c_shares = compute_gini(df, "income")
        assert np.isclose(c_shares[-1], 1.0)

    def test_gini_in_unit_interval(self):
        df = pandas.DataFrame({"income": [1.0, 2.0, 5.0, 10.0]})
        gini, _, _ = compute_gini(df, "income")
        assert 0.0 <= gini <= 1.0


# --- get_top_shares ---


class TestGetTopShares:
    def test_returns_correct_count(self):
        df = pandas.DataFrame({"income": np.arange(1.0, 11.0)})
        shares = get_top_shares(df, "income", shares=[10, 20, 50])
        assert len(shares) == 3

    def test_top_share_in_unit_interval(self):
        df = pandas.DataFrame({"income": np.arange(1.0, 11.0)})
        shares = get_top_shares(df, "income", shares=[10, 50])
        assert all(0.0 < s <= 1.0 for s in shares)

    def test_smaller_percentile_not_larger_share(self):
        # Top 10% share <= top 50% share
        df = pandas.DataFrame({"income": np.arange(1.0, 21.0)})
        shares = get_top_shares(df, "income", shares=[10, 50])
        assert shares[0] <= shares[1]

    def test_with_weights(self):
        df = pandas.DataFrame(
            {
                "income": np.arange(1.0, 11.0),
                "weight": np.ones(10),
            }
        )
        shares_w = get_top_shares(df, "income", shares=[10], wvar="weight")
        shares_no_w = get_top_shares(df, "income", shares=[10])
        assert np.isclose(shares_w[0], shares_no_w[0], rtol=1e-6)
