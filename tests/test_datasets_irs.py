"""Tests for IRS raw ZIP-data downloads and archive-backed imports."""

import io
import zipfile

import pandas as pd
import pytest

from py_tools.datasets import irs


def _archive_bytes(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    return output.getvalue()


def test_zip_download_urls_cover_naming_boundary():
    assert irs._raw_download_url("zip", 2012).endswith("/2012zipcode.zip")
    assert irs._raw_download_url("zip", 2013).endswith("/zipcode2013.zip")


def test_general_download_raw_dispatches_by_geography(tmp_path, monkeypatch):
    payload = _archive_bytes({"source.txt": b"raw"})
    monkeypatch.setattr(
        irs.urllib.request, "urlopen", lambda url: io.BytesIO(payload)
    )

    zip_paths = irs.download_raw("zip", 2016, data_dir=tmp_path)
    county_paths = irs.download_raw("county", 2016, data_dir=tmp_path)

    assert zip_paths == [tmp_path / "zip" / "raw" / "2016.zip"]
    assert county_paths == [tmp_path / "county" / "raw" / "2016.zip"]
    with pytest.raises(ValueError, match="Unsupported IRS geography"):
        irs.download_raw("state", 2016, data_dir=tmp_path)


def test_download_zip_raw_accepts_scalar_and_selected_years(tmp_path, monkeypatch):
    payload = _archive_bytes({"source.txt": b"raw"})
    requested_urls = []

    def fake_urlopen(url):
        requested_urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(irs.urllib.request, "urlopen", fake_urlopen)

    scalar_paths = irs.download_zip_raw(2016, data_dir=tmp_path)
    selected_paths = irs.download_zip_raw(
        [2010, 2022, 2010], data_dir=tmp_path
    )

    assert scalar_paths == [tmp_path / "zip" / "raw" / "2016.zip"]
    assert selected_paths == [
        tmp_path / "zip" / "raw" / "2010.zip",
        tmp_path / "zip" / "raw" / "2022.zip",
    ]
    assert requested_urls == [
        irs._raw_download_url("zip", 2016),
        irs._raw_download_url("zip", 2010),
        irs._raw_download_url("zip", 2022),
    ]
    assert all(zipfile.is_zipfile(path) for path in scalar_paths + selected_paths)


def test_download_zip_raw_skips_valid_file_unless_overwriting(tmp_path, monkeypatch):
    payload = _archive_bytes({"source.txt": b"first"})
    replacement = _archive_bytes({"source.txt": b"replacement"})
    destination = tmp_path / "zip" / "raw" / "2016.zip"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    calls = []

    def fake_urlopen(url):
        calls.append(url)
        return io.BytesIO(replacement)

    monkeypatch.setattr(irs.urllib.request, "urlopen", fake_urlopen)

    irs.download_zip_raw(2016, data_dir=tmp_path)
    assert destination.read_bytes() == payload
    assert calls == []

    irs.download_zip_raw(2016, data_dir=tmp_path, overwrite=True)
    assert destination.read_bytes() == replacement
    assert calls == [irs._raw_download_url("zip", 2016)]


def test_download_zip_raw_rejects_bad_year_and_cleans_partial(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="Unsupported"):
        irs.download_zip_raw(2003, data_dir=tmp_path)

    monkeypatch.setattr(
        irs.urllib.request, "urlopen", lambda url: io.BytesIO(b"not a zip")
    )
    with pytest.raises(RuntimeError, match="2016"):
        irs.download_zip_raw(2016, data_dir=tmp_path)

    raw_dir = tmp_path / "zip" / "raw"
    assert not (raw_dir / "2016.zip").exists()
    assert list(raw_dir.glob("*.part")) == []


def test_modern_zip_year_reads_csv_directly_from_archive(tmp_path):
    csv = "ZIPCODE,AGI_STUB,N1,A00100\n10001,1,2,50\n10001,2,3,75\n"
    archive_path = irs._raw_archive_path("zip", 2011, data_dir=tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(
        _archive_bytes({"nested/path/11ZPALLAGI.CSV": csv.encode("latin1")})
    )

    result = irs.import_geo_year_from_2011(2011, "zip", data_dir=tmp_path)

    assert result.loc[0, "zip"] == 10001
    assert result.loc[0, "n_returns"] == 5
    assert result.loc[0, "agi"] == 125


def test_historical_zip_year_reads_xls_members_without_extracting(
    tmp_path, monkeypatch
):
    archive_path = irs._raw_archive_path("zip", 1998, data_dir=tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(
        _archive_bytes(
            {
                "1998ZIPCode/98zp01al.xls": b"first workbook",
                "1998ZIPCode/98zp02ak.XLS": b"second workbook",
            }
        )
    )
    sources = []

    def fake_load_state_zip_year(source, year):
        sources.append((source, year))
        row = [10001] + list(range(1, 19))
        return pd.DataFrame([row])

    monkeypatch.setattr(irs, "load_state_zip_year", fake_load_state_zip_year)

    result = irs.import_zip_year_to_2010(1998, data_dir=str(tmp_path) + "/")

    assert len(result) == 2
    assert all(isinstance(source, io.BytesIO) for source, _ in sources)
    assert [year for _, year in sources] == [1998, 1998]


def test_modern_zip_year_retains_legacy_extracted_csv_fallback(tmp_path):
    zip_dir = tmp_path / "zip"
    zip_dir.mkdir()
    (zip_dir / "11zpallagi.csv").write_text(
        "ZIPCODE,AGI_STUB,N1,A00100\n10001,1,2,50\n",
        encoding="latin1",
    )

    result = irs.import_geo_year_from_2011(2011, "zip", data_dir=tmp_path)

    assert result.loc[0, "zip"] == 10001
    assert result.loc[0, "n_returns"] == 2


def test_missing_zip_raw_data_has_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"download_zip_raw\(years=2011"):
        irs.import_geo_year_from_2011(2011, "zip", data_dir=tmp_path)


def test_county_download_urls_cover_all_naming_eras():
    assert irs._raw_download_url("county", 2010).endswith(
        "/2010countyincome.zip"
    )
    assert irs._raw_download_url("county", 2011).endswith(
        "/2011countydata.zip"
    )
    assert irs._raw_download_url("county", 2012).endswith(
        "/2012countydata.zip"
    )
    assert irs._raw_download_url("county", 2013).endswith("/county2013.zip")


def test_download_county_raw_accepts_scalar_and_selected_years(
    tmp_path, monkeypatch
):
    payload = _archive_bytes({"source.txt": b"raw"})
    requested_urls = []

    def fake_urlopen(url):
        requested_urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(irs.urllib.request, "urlopen", fake_urlopen)

    scalar_paths = irs.download_county_raw(1989, data_dir=tmp_path)
    selected_paths = irs.download_county_raw(
        [2010, 2022, 2010], data_dir=tmp_path
    )

    assert scalar_paths == [tmp_path / "county" / "raw" / "1989.zip"]
    assert selected_paths == [
        tmp_path / "county" / "raw" / "2010.zip",
        tmp_path / "county" / "raw" / "2022.zip",
    ]
    assert requested_urls == [
        irs._raw_download_url("county", 1989),
        irs._raw_download_url("county", 2010),
        irs._raw_download_url("county", 2022),
    ]
    assert all(zipfile.is_zipfile(path) for path in scalar_paths + selected_paths)


def test_download_county_raw_rejects_bad_year_and_cleans_partial(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="Unsupported"):
        irs.download_county_raw(1988, data_dir=tmp_path)

    monkeypatch.setattr(
        irs.urllib.request, "urlopen", lambda url: io.BytesIO(b"not a zip")
    )
    with pytest.raises(RuntimeError, match="2016"):
        irs.download_county_raw(2016, data_dir=tmp_path)

    raw_dir = tmp_path / "county" / "raw"
    assert not (raw_dir / "2016.zip").exists()
    assert list(raw_dir.glob("*.part")) == []


def test_modern_county_year_reads_csv_directly_from_archive(tmp_path):
    csv = (
        "STATEFIPS,COUNTYFIPS,AGI_STUB,N1,A00100\n"
        "1,1,1,2,50\n"
        "1,1,2,3,75\n"
    )
    archive_path = irs._raw_archive_path("county", 2011, data_dir=tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(
        _archive_bytes({"nested/path/11INCYALLAGI.CSV": csv.encode("latin1")})
    )

    result = irs.import_geo_year_from_2011(2011, "county", data_dir=tmp_path)

    assert result.loc[0, "fips"] == 1001
    assert result.loc[0, "n_returns"] == 5
    assert result.loc[0, "agi"] == 125


def test_historical_county_year_reads_xls_members_without_extracting(
    tmp_path, monkeypatch
):
    archive_path = irs._raw_archive_path("county", 2009, data_dir=tmp_path)
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(
        _archive_bytes(
            {
                "2009CountyIncome/09in01al.xls": b"first workbook",
                "2009CountyIncome/09in02ak.XLS": b"second workbook",
            }
        )
    )
    sources = []

    def fake_load_state_county_year(source, skiprows, target_cols):
        sources.append((source, skiprows, target_cols))
        return pd.DataFrame([[1, 1, "Example", 2, 3, 4, 5, 6, 7]])

    monkeypatch.setattr(
        irs, "load_state_county_year", fake_load_state_county_year
    )

    result = irs.import_county_year_to_2009(2009, data_dir=tmp_path)

    assert len(result) == 2
    assert set(result["fips"]) == {1001}
    assert all(isinstance(source, io.BytesIO) for source, _, _ in sources)
    assert all(skiprows == 8 for _, skiprows, _ in sources)


def test_county_year_retains_legacy_extracted_csv_fallback(tmp_path):
    county_dir = tmp_path / "county"
    county_dir.mkdir()
    (county_dir / "11incyallagi.csv").write_text(
        "STATEFIPS,COUNTYFIPS,AGI_STUB,N1,A00100\n1,1,1,2,50\n",
        encoding="latin1",
    )

    result = irs.import_geo_year_from_2011(2011, "county", data_dir=tmp_path)

    assert result.loc[0, "fips"] == 1001
    assert result.loc[0, "n_returns"] == 2


def test_missing_county_raw_data_has_download_instructions(tmp_path):
    with pytest.raises(
        FileNotFoundError, match=r"download_county_raw\(years=2011"
    ):
        irs.import_geo_year_from_2011(2011, "county", data_dir=tmp_path)
