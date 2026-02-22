"""Tests for py_tools.data.collapser"""

import numpy as np
import pandas as pd
import pytest

from py_tools.data.collapser import Collapser, collapse, collapse_quantile


def make_df():
    return pd.DataFrame({
        'x': [1.0, 3.0, 7.0, 9.0],
        'g': ['a', 'a', 'b', 'b'],
        'w': [1.0, 1.0, 1.0, 1.0],
    })


# --- Collapser ---

class TestCollapser:
    def test_weighted_mean_by_group(self):
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=['g'])
        result = col.get_data()
        assert np.isclose(result.loc['a', 'x'], 2.0)
        assert np.isclose(result.loc['b', 'x'], 8.0)

    def test_singleton_global_mean(self):
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=[])
        result = col.get_data()
        assert np.isclose(result['x'].iloc[0], 5.0)

    def test_singleton_by_list_stays_empty(self):
        # Regression test: bug caused by_list to be set to ['TEMP'] after
        # singleton collapse
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=[])
        assert col.by_list == []

    def test_var_list_stored(self):
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=['g'])
        assert col.var_list == ['x']

    def test_get_weight(self):
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=['g'])
        # Each group has 2 observations with weight 1 → denominator = 2
        w = col.get_weight('x')
        assert np.all(w == 2.0)

    def test_include_denom(self):
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=['g'])
        result = col.get_data(include_denom=True)
        assert 'x_denom' in result.columns


# --- Collapser.collapse (method) ---

class TestCollapserCollapseMethod:
    def test_further_collapse_across_groups(self):
        # Start with by_list=['g'], then collapse to singleton
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=['g'])
        col2 = col.collapse(by_list=[], inplace=False)
        result = col2.get_data()
        # Numerators sum: 1+3+7+9=20; denominators sum: 4
        assert np.isclose(result['x'].iloc[0], 5.0)

    def test_inplace_does_not_return(self):
        col = Collapser(make_df(), var_list=['x'], weight_var='w', by_list=['g'])
        ret = col.collapse(by_list=['g'], inplace=True)
        assert ret is None


# --- collapse (module-level function) ---

class TestCollapseFunction:
    def test_basic_collapse(self):
        result = collapse(make_df(), by_list=['g'], var_list=['x'], weight_var='w')
        assert np.isclose(result.loc['a', 'x'], 2.0)
        assert np.isclose(result.loc['b', 'x'], 8.0)


# --- collapse_quantile ---

class TestCollapseQuantile:
    def test_no_weights_returns_median(self):
        df = pd.DataFrame({'x': [1.0, 3.0, 7.0, 9.0], 'g': ['a', 'a', 'b', 'b']})
        result = collapse_quantile(df, by_list=['g'])
        assert np.isclose(result.loc['a', 'x'], 2.0)
        assert np.isclose(result.loc['b', 'x'], 8.0)
