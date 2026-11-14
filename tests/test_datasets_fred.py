"""Tests for dated FRED caches and legacy pickle compatibility."""

from datetime import date

import pandas as pd
import pytest
from pandas_datareader import data as web

from py_tools.datasets import fred


@pytest.fixture
def fake_fred(monkeypatch):
    calls = []

    def data_reader(codes, source, start, end):
        calls.append((codes, source, start, end))
        index = pd.to_datetime(["2020-01-01", "2020-02-01"])
        return pd.DataFrame(
            {code: [len(calls), len(calls) + 1] for code in codes},
            index=index,
        )

    monkeypatch.setattr(web, "DataReader", data_reader)
    return calls


def test_download_keeps_older_dates_and_replaces_same_date(
    tmp_path, monkeypatch, fake_fred
):
    monkeypatch.setattr(fred, "_download_date", lambda: date(2026, 1, 2))
    first = fred.load(codes="GDP", data_dir=tmp_path)
    first_path = tmp_path / "GDP" / "2026-01-02.parquet"
    assert first_path.exists()

    second = fred.load(codes="GDP", data_dir=tmp_path, reimport=True)
    assert second.iloc[0, 0] == 2
    assert pd.read_parquet(first_path).iloc[0, 0] == 2

    monkeypatch.setattr(fred, "_download_date", lambda: date(2026, 1, 3))
    third = fred.load(codes="GDP", data_dir=tmp_path, reimport=True)
    assert third.iloc[0, 0] == 3
    assert first_path.exists()
    assert (tmp_path / "GDP" / "2026-01-03.parquet").exists()
    assert len(fake_fred) == 3
    assert fake_fred[0][0:2] == (["GDP"], "fred")
    assert first.iloc[0, 0] == 1
    assert not (tmp_path / "series.pkl").exists()
    assert not (tmp_path / "GDP.pkl").exists()


def test_latest_cache_is_selected_per_series_without_download(tmp_path, fake_fred):
    index = pd.to_datetime(["2020-01-01", "2020-02-01"])
    for series, vintage, values in (
        ("GDP", "2026-01-02", [1, 2]),
        ("GDP", "2026-01-04", [4, 5]),
        ("CPI", "2026-01-03", [10, 11]),
    ):
        directory = tmp_path / series
        directory.mkdir(exist_ok=True)
        pd.DataFrame({series: values}, index=index).to_parquet(
            directory / f"{vintage}.parquet"
        )

    result = fred.load(
        code_names={"GDP": "gdp", "CPI": "cpi"}, data_dir=tmp_path
    )
    assert result.columns.tolist() == ["gdp", "cpi"]
    assert result.iloc[0].to_dict() == {"gdp": 4, "cpi": 10}
    assert fake_fred == []


def test_explicit_vintage_requires_every_series(tmp_path, fake_fred):
    directory = tmp_path / "GDP"
    directory.mkdir()
    pd.DataFrame({"GDP": [1]}).to_parquet(directory / "2026-01-02.parquet")

    result = fred.load(codes="GDP", data_dir=tmp_path, vintage=date(2026, 1, 2))
    assert result.iloc[0, 0] == 1
    with pytest.raises(FileNotFoundError, match="CPI"):
        fred.load(codes=["GDP", "CPI"], data_dir=tmp_path, vintage="2026-01-02")
    with pytest.raises(ValueError, match="reimport"):
        fred.load(codes="GDP", data_dir=tmp_path, vintage="2026-01-02", reimport=True)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        fred.load(codes="GDP", data_dir=tmp_path, vintage="20260102")
    assert fake_fred == []


def test_pickle_fallback_and_mixed_cache_formats(tmp_path, fake_fred):
    index = pd.to_datetime(["2020-01-01", "2020-02-01"])
    legacy = pd.Series([1, 2], index=index, name="GDP")
    legacy.to_pickle(tmp_path / "GDP.pkl")
    pd.Series([7], index=index[:1], name="CPI").to_pickle(tmp_path / "CPI.pkl")
    (tmp_path / "CPI").mkdir()
    pd.DataFrame({"CPI": [10]}, index=index[:1]).to_parquet(
        tmp_path / "CPI" / "2026-01-02.parquet"
    )

    result = fred.load(codes=["GDP", "CPI"], data_dir=tmp_path)
    assert result.loc[index[0]].to_dict() == {"GDP": 1, "CPI": 10}
    assert pd.isna(result.loc[index[1], "CPI"])
    pd.testing.assert_series_equal(pd.read_pickle(tmp_path / "GDP.pkl"), legacy)
    assert fake_fred == []
    with pytest.raises(FileNotFoundError, match="GDP"):
        fred.load(codes="GDP", data_dir=tmp_path, vintage="2026-01-02")


def test_download_only_missing_series(tmp_path, monkeypatch, fake_fred):
    monkeypatch.setattr(fred, "_download_date", lambda: date(2026, 1, 2))
    pd.Series([9], index=pd.to_datetime(["2020-01-01"]), name="GDP").to_pickle(
        tmp_path / "GDP.pkl"
    )

    result = fred.load(codes=["GDP", "CPI"], data_dir=tmp_path)
    assert fake_fred[0][0] == ["CPI"]
    assert result.loc[pd.Timestamp("2020-01-01")].to_dict() == {"GDP": 9, "CPI": 1}
    assert (tmp_path / "CPI" / "2026-01-02.parquet").exists()
