import numpy as np
import pandas as pd

from py_tools.time_series import core as tsc


def test_date_from_q_string_parses_quarters():
    s = pd.Series(["2020Q1", "2020Q4", "2021Q2"])
    out = tsc.date_from_q_string(s)
    expected = pd.DatetimeIndex(["2020-01-01", "2020-10-01", "2021-04-01"])
    assert out.equals(expected)


def test_transform_creates_expected_name_and_values():
    df = pd.DataFrame({"x": [1.0, 2.0, 4.0, 7.0]})
    new_vars = tsc.transform(df, ["x"], lag=1, diff=1)
    assert new_vars == ["L_D_x"]
    expected = df["x"].diff(1).shift(1)
    assert np.allclose(df["L_D_x"].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_merge_date_many_outer_join():
    i1 = pd.DatetimeIndex(["2020-01-01", "2020-04-01"])
    i2 = pd.DatetimeIndex(["2020-04-01", "2020-07-01"])
    df1 = pd.DataFrame({"a": [1.0, 2.0]}, index=i1)
    df2 = pd.DataFrame({"b": [3.0, 4.0]}, index=i2)
    out = tsc.merge_date_many([df1, df2], how="outer")
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 3
    assert np.isnan(out.loc["2020-01-01", "b"])


def test_rolling_forecast_internal_exact_linear_relation():
    x = np.arange(1.0, 16.0)
    X = np.column_stack([np.ones_like(x), x])
    y = (2.0 + 3.0 * x)[:, np.newaxis]
    fcast = tsc.rolling_forecast_internal(y, X, t_min=5)
    assert fcast.shape == y.shape
    assert np.allclose(fcast[5:, 0], y[5:, 0], atol=1e-10)


def test_bandpass_filter_and_interpolate_shapes():
    series = pd.Series(np.linspace(0.0, 1.0, 40))
    out = tsc.bandpass_filter(series, period_lb=2, period_ub=8, nlags=3)
    assert len(out) == len(series)
    assert np.isfinite(out.dropna()).all()

    z = np.array([1.0, 2.0, 4.0])
    x_star = tsc.interpolate_to_high_frequency(z, freq=4)
    assert x_star.shape == (12,)
    # Default aggregation matrix imposes that each block mean matches coarse series.
    block_means = x_star.reshape(3, 4).mean(axis=1)
    assert np.allclose(block_means, z, atol=1e-10)
