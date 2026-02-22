"""Tests for py_tools.data.core"""

import numpy as np
import pandas as pd
import pytest

from py_tools.data.core import (
    lowercase,
    absorb,
    bin_data,
    winsorize,
    match_sample,
    match_xy,
    clean,
    dropna_ix,
    compute_histogram,
    get_cluster_groups,
    least_sq,
    standard_errors,
    to_pickle,
    read_pickle,
    demean,
    demean2,
    weight_regression_params,
    collapse,
    safe_sum,
)


# --- lowercase ---


class TestLowercase:
    def test_renames_to_lowercase(self):
        df = pd.DataFrame({"A": [1], "B_C": [2]})
        result = lowercase(df)
        assert list(result.columns) == ["a", "b_c"]

    def test_already_lowercase_unchanged(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert list(lowercase(df).columns) == ["a", "b"]


# --- bin_data ---


class TestBinData:
    def test_length_preserved(self):
        s = pd.Series(np.arange(50.0))
        assert len(bin_data(s, 5)) == 50

    def test_correct_number_of_bins(self):
        s = pd.Series(np.arange(100.0))
        assert bin_data(s, 5).nunique() == 5

    def test_weighted_matches_unweighted_for_uniform_weights(self):
        s = pd.Series(np.arange(20.0))
        w = pd.Series(np.ones(20))
        b_unweighted = bin_data(s, 4)
        b_weighted = bin_data(s, 4, weights=w)
        assert np.array_equal(b_unweighted, b_weighted)


# --- match_sample ---


class TestMatchSample:
    def test_inner_excludes_nan_rows(self):
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        ix, Xs = match_sample(X, how="inner")
        assert np.sum(ix) == 2
        assert Xs.shape == (2, 2)

    def test_outer_keeps_partial_rows(self):
        X = np.array([[1.0, np.nan], [np.nan, np.nan], [4.0, 5.0]])
        ix, Xs = match_sample(X, how="outer")
        assert np.sum(ix) == 2  # rows 0 and 2

    def test_nan_filled_with_zero(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        _, Xs = match_sample(X, how="outer")
        assert np.all(np.isfinite(Xs))

    def test_custom_ix(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ix_in = np.array([True, False, True])
        ix_out, Xs = match_sample(X, how="custom", ix=ix_in)
        assert np.array_equal(ix_out, ix_in)
        assert Xs.shape == (2, 2)

    def test_invalid_how_raises(self):
        with pytest.raises(Exception):
            match_sample(np.ones((3, 2)), how="invalid")


# --- match_xy ---


class TestMatchXy:
    def test_filters_nan_rows(self):
        X = np.array([[1.0], [2.0], [np.nan]])
        z = np.array([4.0, 5.0, 6.0])
        ix, Xs, zs = match_xy(X, z)
        assert np.sum(ix) == 2
        assert Xs.shape == (2, 1)
        assert zs.shape == (2, 1)

    def test_1d_inputs_promoted(self):
        X = np.array([1.0, 2.0, 3.0])
        z = np.array([4.0, 5.0, 6.0])
        _, Xs, zs = match_xy(X, z)
        assert Xs.ndim == 2
        assert zs.ndim == 2


# --- clean ---


class TestClean:
    def test_drops_nan(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
        result = clean(df, ["x", "y"])
        assert len(result) == 2

    def test_replaces_inf_with_nan_then_drops(self):
        df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
        result = clean(df, ["x"])
        assert len(result) == 2

    def test_ignores_missing_vars(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = clean(df, ["x", "nonexistent"])
        assert len(result) == 2


# --- dropna_ix ---


class TestDropnaIx:
    def test_returns_clean_df_and_mask(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
        df_out, ix = dropna_ix(df)
        assert len(df_out) == 2
        assert np.sum(ix) == 2

    def test_mask_aligns_with_dataframe(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        df_out, ix = dropna_ix(df)
        assert np.array_equal(df.loc[ix].values, df_out.values)


# --- winsorize ---


class TestWinsorize:
    def test_clips_extremes(self):
        df = pd.DataFrame({"x": np.arange(100.0)})
        result = winsorize(df, ["x"], p_val=0.9)
        lb = df["x"].quantile(0.05)
        ub = df["x"].quantile(0.95)
        assert result["x"].min() >= lb - 1e-10
        assert result["x"].max() <= ub + 1e-10

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"x": np.arange(10.0)})
        x_orig = df["x"].copy()
        winsorize(df, ["x"], p_val=0.8)
        assert np.array_equal(df["x"].values, x_orig.values)


# --- compute_histogram ---


class TestComputeHistogram:
    def test_length_matches_bins(self):
        result = compute_histogram(pd.Series([1, 2, 3, 4, 5]), bins=5)
        assert len(result) == 5

    def test_counts_sum_to_n(self):
        s = pd.Series(np.arange(20.0))
        result = compute_histogram(s, bins=4)
        assert result.sum() == 20

    def test_index_name(self):
        result = compute_histogram(pd.Series([1, 2, 3]), name="bucket", bins=3)
        assert result.index[0] == "bucket0"


# --- get_cluster_groups ---


class TestGetClusterGroups:
    def test_returns_integer_codes(self):
        df = pd.DataFrame({"g": ["a", "b", "a", "c"]})
        groups = get_cluster_groups(df, "g")
        assert len(groups) == 4
        assert len(np.unique(groups)) == 3

    def test_same_value_same_group(self):
        df = pd.DataFrame({"g": ["x", "y", "x"]})
        groups = get_cluster_groups(df, "g")
        assert groups[0] == groups[2]
        assert groups[0] != groups[1]


# --- least_sq ---


class TestLeastSq:
    def test_identity_design(self):
        X = np.eye(3)
        z = np.array([[1.0], [2.0], [3.0]])
        assert np.allclose(least_sq(X, z), z)

    def test_ols_known_solution(self):
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        z = np.array([[1.0], [3.0], [5.0], [7.0], [9.0]])  # y = 1 + 2x
        params = least_sq(X, z)
        assert np.allclose(params, [[1.0], [2.0]], atol=1e-10)


# --- standard_errors ---


class TestStandardErrors:
    def test_diagonal_covariance(self):
        V = np.diag([4.0, 9.0])
        se = standard_errors(V, T=1)
        assert np.allclose(se, [2.0, 3.0])

    def test_scales_with_T(self):
        V = np.diag([4.0, 4.0])
        assert np.allclose(standard_errors(V, T=4), standard_errors(V, T=1) / 2.0)


# --- to_pickle / read_pickle ---


class TestPickle:
    def test_round_trip(self, tmp_path):
        data = {"key": [1, 2, 3], "val": "hello"}
        path = tmp_path / "test.pkl"
        to_pickle(data, path)
        assert read_pickle(path) == data


# --- demean ---


class TestDemean:
    def test_demeaned_values(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 7.0, 9.0], "g": ["a", "a", "b", "b"]})
        result, names = demean(df, ["x"], "g")
        assert np.allclose(result["x_demeaned"].values, [-1.0, 1.0, -1.0, 1.0])

    def test_group_means(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 7.0, 9.0], "g": ["a", "a", "b", "b"]})
        result, _ = demean(df, ["x"], "g")
        assert np.allclose(result["x_mean"].values, [2.0, 2.0, 8.0, 8.0])

    def test_returns_name_lists(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "g": ["a", "b"]})
        _, names = demean(df, ["x", "y"], "g")
        assert "x_mean" in names and "x_demeaned" in names
        assert "y_mean" in names and "y_demeaned" in names


# --- demean2 ---


class TestDemean2:
    def test_demeaned_in_place(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 7.0, 9.0], "g": ["a", "a", "b", "b"]})
        result = demean2("g", ["x"], df.copy())
        assert np.allclose(result["x"].values, [-1.0, 1.0, -1.0, 1.0])

    def test_with_prefix(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 7.0, 9.0], "g": ["a", "a", "b", "b"]})
        result = demean2("g", ["x"], df.copy(), prefix="dm")
        assert "dm_x" in result.columns
        assert np.allclose(result["dm_x"].values, [-1.0, 1.0, -1.0, 1.0])


# --- weight_regression_params ---


class TestWeightRegParams:
    def test_weighted_sum(self):
        params = np.array([1.0, 2.0, 3.0])
        cov = np.diag([1.0, 1.0, 1.0])
        weights = np.array([1.0, 1.0, 1.0])
        x, se = weight_regression_params(weights, params=params, cov=cov)
        assert np.isclose(x, 6.0)
        assert np.isclose(se, np.sqrt(3.0))

    def test_single_coefficient(self):
        params = np.array([5.0, 0.0])
        cov = np.diag([4.0, 1.0])
        weights = np.array([1.0, 0.0])
        x, se = weight_regression_params(weights, params=params, cov=cov)
        assert np.isclose(x, 5.0)
        assert np.isclose(se, 2.0)


# --- collapse (core.py) ---


class TestCollapse:
    def test_weighted_mean(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "x": [1.0, 3.0, 7.0, 9.0],
                "w": [1.0, 1.0, 1.0, 1.0],
            }
        )
        result = collapse(df, by=["g"], var_list=["x"], wvar="w")
        assert np.isclose(result.loc["a", "x"], 2.0)
        assert np.isclose(result.loc["b", "x"], 8.0)

    def test_weighted_sum(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a"],
                "x": [2.0, 3.0],
                "w": [1.0, 1.0],
            }
        )
        result = collapse(df, method="sum", by=["g"], var_list=["x"], wvar="w")
        assert np.isclose(result.loc["a", "x"], 5.0)


# --- absorb ---


class TestAbsorb:
    def test_demeaned_within_groups(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 7.0, 9.0], "g": ["a", "a", "b", "b"]})
        result = absorb(df, ["g"], "x", restore_mean=False)
        assert np.allclose(result.values, [-1.0, 1.0, -1.0, 1.0], atol=1e-10)

    def test_restore_mean(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 7.0, 9.0], "g": ["a", "a", "b", "b"]})
        result = absorb(df, ["g"], "x", restore_mean=True)
        # global mean = 5.0, demeaned = [-1, 1, -1, 1], restored = [4, 6, 4, 6]
        assert np.allclose(result.values, [4.0, 6.0, 4.0, 6.0], atol=1e-10)


# --- safe_sum ---


class TestSafeSum:
    def test_sum_without_nan(self):
        assert safe_sum(pd.Series([1.0, 2.0, 3.0])) == 6.0

    def test_nan_propagates(self):
        assert np.isnan(safe_sum(pd.Series([1.0, np.nan, 3.0])))
