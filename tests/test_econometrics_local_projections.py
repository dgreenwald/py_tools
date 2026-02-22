import numpy as np
import pandas as pd
import pytest
from py_tools.econometrics.local_projections import (
    lag_var, add_lags, add_leads, formula_lags, get_formula, estimate, LocalProjection,
)


@pytest.fixture
def simple_df():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        'y': rng.normal(size=n),
        'x': rng.normal(size=n),
    })


class TestLagVar:

    def test_lag_positive(self, simple_df):
        df = simple_df.copy()
        df = lag_var(df, 'y', 1)
        assert 'L1_y' in df.columns
        pd.testing.assert_series_equal(
            df['L1_y'].iloc[1:].reset_index(drop=True),
            df['y'].iloc[:-1].reset_index(drop=True),
            check_names=False,
        )

    def test_lead_negative(self, simple_df):
        df = simple_df.copy()
        df = lag_var(df, 'y', -2)
        assert 'F2_y' in df.columns
        pd.testing.assert_series_equal(
            df['F2_y'].iloc[:-2].reset_index(drop=True),
            df['y'].iloc[2:].reset_index(drop=True),
            check_names=False,
        )

    def test_zero_raises(self, simple_df):
        with pytest.raises(Exception):
            lag_var(simple_df.copy(), 'y', 0)

    def test_returns_df(self, simple_df):
        result = lag_var(simple_df.copy(), 'y', 1)
        assert isinstance(result, pd.DataFrame)


class TestAddLags:

    def test_adds_correct_columns(self, simple_df):
        df = add_lags(simple_df.copy(), 'y', 3)
        for lag in range(1, 4):
            assert f'L{lag}_y' in df.columns

    def test_no_columns_added_for_zero(self, simple_df):
        df = add_lags(simple_df.copy(), 'y', 0)
        assert 'L1_y' not in df.columns


class TestAddLeads:

    def test_adds_correct_columns(self, simple_df):
        df = add_leads(simple_df.copy(), 'y', 4)
        for lead in range(1, 5):
            assert f'F{lead}_y' in df.columns

    def test_no_columns_for_zero(self, simple_df):
        df = add_leads(simple_df.copy(), 'y', 0)
        assert 'F1_y' not in df.columns


class TestFormulaLags:

    def test_empty_for_zero(self):
        assert formula_lags('x', 0) == ''

    def test_one_lag(self):
        f = formula_lags('x', 1)
        assert 'L1_x' in f

    def test_three_lags(self):
        f = formula_lags('x', 3)
        for lag in range(1, 4):
            assert f'L{lag}_x' in f


class TestGetFormula:

    def test_horizon_zero(self):
        f = get_formula(0, 'y', 'x', [], [], 0, 0, {})
        assert f.startswith('y ~')
        assert 'x' in f

    def test_horizon_positive(self):
        f = get_formula(3, 'y', 'x', [], [], 0, 0, {})
        assert f.startswith('F3_y ~')

    def test_shock_lags_included(self):
        f = get_formula(1, 'y', 'x', [], [], 2, 0, {})
        assert 'L1_x' in f
        assert 'L2_x' in f

    def test_y_lags_included(self):
        f = get_formula(1, 'y', 'x', [], [], 0, 2, {})
        assert 'L1_y' in f
        assert 'L2_y' in f

    def test_fe_vars_included(self):
        f = get_formula(0, 'y', 'x', [], ['group'], 0, 0, {})
        assert 'C(group)' in f


class TestLocalProjection:

    def test_mutable_default_fixed(self):
        lp1 = LocalProjection()
        lp2 = LocalProjection()
        lp1.labels['foo'] = 'bar'
        assert 'foo' not in lp2.labels, "mutable default argument bug"

    def test_labels_passed(self):
        lp = LocalProjection(labels={'y': 'Output'})
        assert lp.labels['y'] == 'Output'

    def test_empty_labels_default(self):
        lp = LocalProjection()
        assert lp.labels == {}

    def test_df_stored(self, simple_df):
        lp = LocalProjection(df=simple_df)
        assert lp.df is simple_df


class TestEstimate:
    def test_se_matches_attached_covariance_estimator(self, simple_df):
        fr_list, x, se = estimate(
            simple_df,
            y_var="y",
            shock_var="x",
            periods=3,
            shock_lags=0,
            y_lags=0,
        )
        assert len(fr_list) == 3
        assert np.isclose(se[1], fr_list[1].results.bse[1])
