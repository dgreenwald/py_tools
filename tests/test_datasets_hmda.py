"""Tests for HMDA LAR downloads."""

import io
import zipfile

import pytest

from py_tools.datasets import hmda


def _archive_bytes(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    return output.getvalue()


class _ResponseWithLength(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


@pytest.mark.parametrize("year", range(2017, 2026))
def test_snapshot_lar_urls_and_paths(year, tmp_path):
    config = hmda._LAR_SOURCE_CONFIG["ffiec_snapshot"][year]

    assert config["url"] == (
        f"https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/"
        f"{year}_public_lar_csv.zip"
    )
    assert hmda._lar_file_path(year, tmp_path, source="ffiec_snapshot") == (
        tmp_path / "raw" / "ffiec_snapshot" / str(year) / f"{year}_public_lar_csv.zip"
    )


@pytest.mark.parametrize("year", range(2007, 2018))
def test_cfpb_lar_urls_and_paths(year, tmp_path):
    config = hmda._LAR_SOURCE_CONFIG["cfpb"][year]
    filename = f"hmda_{year}_nationwide_all-records_codes.zip"

    assert config["url"] == (
        "https://files.consumerfinance.gov/hmda-historic-loan-data/" + filename
    )
    assert hmda._lar_file_path(year, tmp_path, source="cfpb") == (
        tmp_path / "raw" / "cfpb" / str(year) / filename
    )


@pytest.mark.parametrize("year", range(2017, 2023))
def test_three_year_lar_urls_and_paths(year, tmp_path):
    config = hmda._LAR_SOURCE_CONFIG["ffiec_three_year"][year]
    filename = f"{year}_public_lar_three_year_csv.zip"

    assert config["url"] == (
        f"https://files.ffiec.cfpb.gov/static-data/three-year/{year}/{filename}"
    )
    assert hmda._lar_file_path(year, tmp_path, source="ffiec_three_year") == (
        tmp_path / "raw" / "ffiec_three_year" / str(year) / filename
    )


@pytest.mark.parametrize("year", range(1981, 1990))
def test_national_archives_lar_urls_and_paths(year, tmp_path):
    config = hmda._LAR_SOURCE_CONFIG["nara"][year]
    filename = f"HMD_FACDSB{year % 100:02d}.txt"

    assert config["url"] == (
        "https://catalog.archives.gov/medialz/electronic-records/"
        f"rg-082/hmda/{filename}"
    )
    assert hmda._lar_file_path(year, tmp_path, source="nara") == (
        tmp_path / "raw" / "nara" / str(year) / filename
    )


@pytest.mark.parametrize(
    "year,relative_path",
    sorted(hmda._NATIONAL_ARCHIVES_ZIP_PATHS.items()),
)
def test_national_archives_zip_urls_and_paths(year, relative_path, tmp_path):
    config = hmda._LAR_SOURCE_CONFIG["nara"][year]

    assert config["url"] == (
        "https://catalog.archives.gov/medialz/electronic-records/"
        f"rg-082/hmda/{relative_path}"
    )
    assert hmda._lar_file_path(year, tmp_path, source="nara") == (
        tmp_path / "raw" / "nara" / str(year) / relative_path.rsplit("/", 1)[-1]
    )


def test_download_dispatches_across_sources_and_deduplicates(tmp_path, monkeypatch):
    zip_payload = _archive_bytes({"hmda_lar.csv": b"year,action_type\n2016,1\n"})
    text_payload = b"A" * 100 + b"\r\n" + b"B" * 100 + b"\r\n" + b"C" * 100
    requested_urls = []

    def fake_urlopen(url):
        requested_urls.append(url)
        return io.BytesIO(text_payload if url.endswith(".txt") else zip_payload)

    monkeypatch.setattr(hmda.urllib.request, "urlopen", fake_urlopen)

    paths = hmda.download_lar([1981, 2016, 2017, 2025, 1981], data_dir=tmp_path)

    assert paths == [
        hmda._lar_file_path(1981, tmp_path),
        hmda._lar_file_path(2016, tmp_path),
        hmda._lar_file_path(2017, tmp_path),
        hmda._lar_file_path(2025, tmp_path),
    ]
    assert requested_urls == [
        hmda._LAR_SOURCE_CONFIG["nara"][1981]["url"],
        hmda._LAR_SOURCE_CONFIG["cfpb"][2016]["url"],
        hmda._LAR_SOURCE_CONFIG["ffiec_three_year"][2017]["url"],
        hmda._LAR_SOURCE_CONFIG["ffiec_snapshot"][2025]["url"],
    ]


def test_download_defaults_to_all_configured_years(tmp_path, monkeypatch):
    zip_payload = _archive_bytes({"hmda_lar.csv": b"year\n2025\n"})
    text_payload = b"A" * 100 + b"\n" + b"B" * 100 + b"\n" + b"C" * 100

    def fake_urlopen(url):
        return io.BytesIO(text_payload if url.endswith(".txt") else zip_payload)

    monkeypatch.setattr(hmda.urllib.request, "urlopen", fake_urlopen)

    paths = hmda.download_lar(data_dir=tmp_path)

    assert paths == [hmda._lar_file_path(year, tmp_path) for year in hmda.LAR_YEARS]


def test_download_rejects_unsupported_year(tmp_path):
    with pytest.raises(ValueError, match=r"Supported years are: 2025, .*1981"):
        hmda.download_lar(1980, data_dir=tmp_path)


def test_download_skips_valid_archive_unless_overwriting(tmp_path, monkeypatch):
    original = _archive_bytes({"hmda_lar.csv": b"original"})
    replacement = _archive_bytes({"hmda_lar.csv": b"replacement"})
    destination = hmda._lar_file_path(2016, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(original)
    calls = []

    def fake_urlopen(url):
        calls.append(url)
        return io.BytesIO(replacement)

    monkeypatch.setattr(hmda.urllib.request, "urlopen", fake_urlopen)

    hmda.download_lar(2016, data_dir=tmp_path)
    assert destination.read_bytes() == original
    assert calls == []

    hmda.download_lar(2016, data_dir=tmp_path, overwrite=True)
    assert destination.read_bytes() == replacement
    assert calls == [hmda._LAR_SOURCE_CONFIG["cfpb"][2016]["url"]]


def test_download_progress_bar_is_optional(tmp_path, monkeypatch, capsys):
    payload = _archive_bytes({"hmda_lar.csv": b"year\n2016\n"})
    monkeypatch.setattr(
        hmda.urllib.request,
        "urlopen",
        lambda url: _ResponseWithLength(payload),
    )

    hmda.download_lar(2016, data_dir=tmp_path / "quiet")
    assert capsys.readouterr().err == ""

    hmda.download_lar(2016, data_dir=tmp_path / "visible", progress=True)
    progress_output = capsys.readouterr().err
    assert "HMDA 2016 (cfpb)" in progress_output
    assert "100.0%" in progress_output
    assert "MiB" in progress_output


@pytest.mark.parametrize(
    "payload",
    [
        b"not a zip",
        _archive_bytes({"readme.txt": b"not CSV data"}),
        _archive_bytes({"hmda_lar.csv": b""}),
    ],
)
def test_download_rejects_invalid_content_and_cleans_partial(
    tmp_path, monkeypatch, payload
):
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    with pytest.raises(RuntimeError, match=r"2016.*valid zip_csv"):
        hmda.download_lar(2016, data_dir=tmp_path)

    destination = hmda._lar_file_path(2016, tmp_path)
    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []


def test_failed_overwrite_preserves_existing_archive(tmp_path, monkeypatch):
    original = _archive_bytes({"hmda_lar.csv": b"original"})
    destination = hmda._lar_file_path(2017, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(original)
    monkeypatch.setattr(
        hmda.urllib.request, "urlopen", lambda url: io.BytesIO(b"not a zip")
    )

    with pytest.raises(RuntimeError, match="2017"):
        hmda.download_lar(2017, data_dir=tmp_path, overwrite=True)

    assert destination.read_bytes() == original
    assert list(destination.parent.glob("*.part")) == []


@pytest.mark.parametrize(
    "year,expected_source",
    [
        (1981, "nara"),
        (2006, "nara"),
        (2007, "cfpb"),
        (2016, "cfpb"),
        (2017, "ffiec_three_year"),
        (2022, "ffiec_three_year"),
        (2023, "ffiec_snapshot"),
        (2025, "ffiec_snapshot"),
    ],
)
def test_auto_source_precedence(year, expected_source, tmp_path):
    assert hmda._resolve_lar_source(year) == expected_source
    assert hmda._lar_file_path(year, tmp_path).parts[-3] == expected_source


def test_explicit_overlapping_sources_have_separate_paths(tmp_path):
    paths = {
        source: hmda._lar_file_path(2014, tmp_path, source=source)
        for source in ("cfpb", "nara")
    }

    assert paths["cfpb"] != paths["nara"]
    assert paths["cfpb"].parts[-3] == "cfpb"
    assert paths["nara"].parts[-3] == "nara"


def test_explicit_source_defaults_and_validation(tmp_path, monkeypatch):
    payload = _archive_bytes({"hmda_lar.csv": b"year\n2022\n"})
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    paths = hmda.download_lar(source="ffiec_three_year", data_dir=tmp_path)

    assert paths == [
        hmda._lar_file_path(year, tmp_path, source="ffiec_three_year")
        for year in hmda.THREE_YEAR_LAR_YEARS
    ]
    with pytest.raises(ValueError, match="unavailable.*ffiec_three_year"):
        hmda.download_lar(2023, source="ffiec_three_year", data_dir=tmp_path)
    with pytest.raises(ValueError, match="Supported sources"):
        hmda.download_lar(2020, source="unknown", data_dir=tmp_path)


def test_nara_zip_accepts_non_csv_data_member(tmp_path, monkeypatch):
    payload = _archive_bytes({"Lars.ultimate.2010.dat": b"fixed width data"})
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    paths = hmda.download_lar(2010, source="nara", data_dir=tmp_path)

    assert paths == [hmda._lar_file_path(2010, tmp_path, source="nara")]


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"<html>not fixed width</html>\n",
        b"A" * 100 + b"\n" + b"short\n" + b"B" * 100,
        b"A" * 100 + b"\n" + b"B" * 100 + b"\n\xff",
    ],
)
def test_national_archives_download_rejects_invalid_text(
    tmp_path, monkeypatch, payload
):
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    with pytest.raises(RuntimeError, match=r"1989.*valid txt"):
        hmda.download_lar(1989, data_dir=tmp_path)

    destination = hmda._lar_file_path(1989, tmp_path)
    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []
