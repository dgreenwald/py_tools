"""Tests for the vintage-aware Zillow loader and downloader."""

import io
from datetime import date

import pandas as pd
import pytest

from py_tools.datasets import zillow


def _write_csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def _legacy_county(path):
    _write_csv(
        path / "201908" / "County" / "County_Zhvi_AllHomes.csv",
        {
            "RegionID": [1, 2],
            "RegionName": ["Autauga", "Aleutians East"],
            "State": ["AL", "AK"],
            "Metro": ["Montgomery", None],
            "StateCodeFIPS": ["01", "02"],
            "MunicipalCodeFIPS": ["001", "013"],
            "SizeRank": [1, 2],
            "2019-07": [100.0, 200.0],
            "2019-08": [101.0, 202.0],
        },
    )


def _current_county(path, vintage="202608"):
    filename = zillow._RAW_DATA_CONFIG[("zhvi", "county")]["filename"]
    destination = path / vintage / "County" / filename
    _write_csv(
        destination,
        {
            "RegionID": [1, 2],
            "SizeRank": [1, 2],
            "RegionName": ["Autauga County", "Aleutians East Borough"],
            "RegionType": ["county", "county"],
            "StateName": ["AL", "AK"],
            "State": ["AL", "AK"],
            "Metro": ["Montgomery, AL", None],
            "StateCodeFIPS": ["01", "02"],
            "MunicipalCodeFIPS": ["001", "013"],
            "2026-05-31": [150.5, 250.5],
            "2026-06-30": [151.5, 251.5],
        },
    )
    return destination


@pytest.mark.parametrize(
    ("vintage", "expected_date", "expected_value"),
    [
        ("201908", pd.Timestamp("2019-08-31"), 101.0),
        ("202608", pd.Timestamp("2026-06-30"), 151.5),
    ],
)
def test_county_vintages_share_canonical_contract(
    tmp_path, vintage, expected_date, expected_value
):
    _legacy_county(tmp_path)
    _current_county(tmp_path)

    result = zillow.load(data_dir=tmp_path, vintage=vintage)

    assert result.index.names == ["fips", "date"]
    assert result.index.is_monotonic_increasing
    fips = result.index.get_level_values("fips")
    assert all(isinstance(value, str) and len(value) == 5 for value in fips)
    assert set(fips) == {"01001", "02013"}
    assert result.loc[("01001", expected_date), "zhvi"] == expected_value
    assert set(result.columns) >= {
        "zhvi",
        "region_id",
        "region_name",
        "state",
        "metro",
        "size_rank",
    }


def test_county_parquet_cache_matches_fresh_import(tmp_path):
    _current_county(tmp_path)
    fresh = zillow.load(data_dir=tmp_path, reimport=True)
    parquet = tmp_path / "202608" / "parquet" / "county_zhvi.parquet"

    assert parquet.exists()
    cached = zillow.load(data_dir=tmp_path)
    pd.testing.assert_frame_equal(cached, fresh)


def test_load_missing_raw_has_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"download_raw.*202608"):
        zillow.load(data_dir=tmp_path)


def test_legacy_geography_paths_are_vintage_specific(tmp_path):
    for geo in ("State", "Metro", "Zip"):
        path = tmp_path / "201908" / geo / f"{geo}_Zhvi_AllHomes.csv"
        _write_csv(path, {"RegionID": [1], "RegionName": ["example"]})
        result = zillow.load_csv(
            data_dir=tmp_path,
            geo=geo,
            dataset="Zhvi_AllHomes",
            vintage="201908",
        )
        assert result.loc[0, "RegionName"] == "example"


def test_legacy_state_metro_and_zip_loaders(tmp_path, monkeypatch):
    _write_csv(
        tmp_path / "201908" / "State" / "State_Zhvi_AllHomes.csv",
        {
            "RegionID": [9],
            "RegionName": ["California"],
            "SizeRank": [1],
            "2019-08": [500.0],
        },
    )
    _write_csv(
        tmp_path / "201908" / "Metro" / "Metro_Zhvi_AllHomes.csv",
        {
            "RegionID": [102001],
            "RegionName": ["United States"],
            "SizeRank": [0],
            "2019-08": [200.0],
        },
    )
    _write_csv(
        tmp_path / "201908" / "Zip" / "Zip_Zhvi_AllHomes.csv",
        {
            "RegionID": [123],
            "RegionName": ["01234"],
            "State": ["MA"],
            "2019-08": [300.0],
        },
    )
    monkeypatch.setattr(
        zillow.misc,
        "load",
        lambda name: pd.DataFrame({"state_name": ["California"], "state_abbr": ["CA"]}),
    )

    state = zillow.load("state", data_dir=tmp_path, vintage="201908")
    metro = zillow.load("metro", data_dir=tmp_path, vintage="201908")
    zip_frame = zillow.load("zip", data_dir=tmp_path, vintage="201908")

    month_end = pd.Timestamp("2019-08-31")
    assert state.loc[("CA", month_end), "zhvi"] == 500.0
    assert metro.loc[("United States", month_end), "zhvi"] == 200.0
    assert zip_frame.loc[("01234", month_end), "zhvi"] == 300.0


def test_download_raw_uses_explicit_vintage_and_canonical_path(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source = _current_county(source_root, vintage="202701")
    payload = source.read_bytes()
    requested = []

    def fake_urlopen(url):
        requested.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(zillow.urllib.request, "urlopen", fake_urlopen)
    destination = zillow.download_raw(data_dir=tmp_path, vintage="202701")

    config = zillow._RAW_DATA_CONFIG[("zhvi", "county")]
    assert destination == tmp_path / "202701" / "County" / config["filename"]
    assert requested == [config["url"]]
    assert zillow._validate_raw_file(destination, config)


def test_download_raw_defaults_to_current_month(tmp_path, monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2027, 3, 14)

    source = _current_county(tmp_path / "source")
    monkeypatch.setattr(zillow, "date", FixedDate)
    monkeypatch.setattr(
        zillow.urllib.request,
        "urlopen",
        lambda url: io.BytesIO(source.read_bytes()),
    )

    destination = zillow.download_raw(data_dir=tmp_path)
    assert destination.parts[-3] == "202703"


def test_download_raw_skips_valid_file_and_can_overwrite(tmp_path, monkeypatch):
    destination = _current_county(tmp_path, vintage="202701")
    original = destination.read_bytes()
    replacement_source = _current_county(tmp_path / "replacement")
    replacement = replacement_source.read_bytes().replace(b"150.5", b"999.5")
    calls = []

    def fake_urlopen(url):
        calls.append(url)
        return io.BytesIO(replacement)

    monkeypatch.setattr(zillow.urllib.request, "urlopen", fake_urlopen)

    zillow.download_raw(data_dir=tmp_path, vintage="202701")
    assert calls == []
    assert destination.read_bytes() == original

    zillow.download_raw(data_dir=tmp_path, vintage="202701", overwrite=True)
    assert len(calls) == 1
    assert destination.read_bytes() == replacement


def test_download_raw_rejects_invalid_content_and_cleans_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        zillow.urllib.request, "urlopen", lambda url: io.BytesIO(b"invalid")
    )

    with pytest.raises(RuntimeError, match="Zillow zhvi/county"):
        zillow.download_raw(data_dir=tmp_path, vintage="202701")

    destination_dir = tmp_path / "202701" / "County"
    assert list(destination_dir.glob("*.part")) == []
    assert list(destination_dir.glob("*.csv")) == []


@pytest.mark.parametrize("vintage", ["2026-08", "2608", "current"])
def test_invalid_vintages_raise(vintage):
    with pytest.raises(ValueError, match="YYYYMM"):
        zillow.load(vintage=vintage)


def test_unsupported_download_combination_raises(tmp_path):
    with pytest.raises(ValueError, match="No Zillow download"):
        zillow.download_raw(geo="state", data_dir=tmp_path, vintage="202701")
