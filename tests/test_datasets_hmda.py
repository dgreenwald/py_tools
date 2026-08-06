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


@pytest.mark.parametrize("year", range(2017, 2026))
def test_snapshot_lar_urls_and_paths(year, tmp_path):
    config = hmda._LAR_CONFIG[year]

    assert config["url"] == (
        f"https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/"
        f"{year}_public_lar_csv.zip"
    )
    assert hmda._lar_archive_path(year, tmp_path) == (
        tmp_path / "snapshot" / str(year) / f"{year}_public_lar_csv.zip"
    )


@pytest.mark.parametrize("year", range(2007, 2017))
def test_earlier_lar_urls_and_paths(year, tmp_path):
    config = hmda._LAR_CONFIG[year]
    filename = f"hmda_{year}_nationwide_all-records_codes.zip"

    assert config["url"] == (
        "https://files.consumerfinance.gov/hmda-historic-loan-data/" + filename
    )
    assert hmda._lar_archive_path(year, tmp_path) == (
        tmp_path / "historic" / str(year) / filename
    )


def test_download_dispatches_across_sources_and_deduplicates(tmp_path, monkeypatch):
    payload = _archive_bytes({"hmda_lar.csv": b"year,action_type\n2016,1\n"})
    requested_urls = []

    def fake_urlopen(url):
        requested_urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(hmda.urllib.request, "urlopen", fake_urlopen)

    paths = hmda.download_lar([2016, 2017, 2025, 2016], data_dir=tmp_path)

    assert paths == [
        hmda._lar_archive_path(2016, tmp_path),
        hmda._lar_archive_path(2017, tmp_path),
        hmda._lar_archive_path(2025, tmp_path),
    ]
    assert requested_urls == [
        hmda._LAR_CONFIG[2016]["url"],
        hmda._LAR_CONFIG[2017]["url"],
        hmda._LAR_CONFIG[2025]["url"],
    ]


def test_download_defaults_to_all_configured_years(tmp_path, monkeypatch):
    payload = _archive_bytes({"hmda_lar.csv": b"year\n2025\n"})
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    paths = hmda.download_lar(data_dir=tmp_path)

    assert paths == [hmda._lar_archive_path(year, tmp_path) for year in hmda.LAR_YEARS]


def test_download_rejects_unsupported_year(tmp_path):
    with pytest.raises(ValueError, match=r"Supported years are: 2025, .*2007"):
        hmda.download_lar(2006, data_dir=tmp_path)


def test_download_skips_valid_archive_unless_overwriting(tmp_path, monkeypatch):
    original = _archive_bytes({"hmda_lar.csv": b"original"})
    replacement = _archive_bytes({"hmda_lar.csv": b"replacement"})
    destination = hmda._lar_archive_path(2016, tmp_path)
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
    assert calls == [hmda._LAR_CONFIG[2016]["url"]]


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

    with pytest.raises(RuntimeError, match=r"2016.*nonempty CSV"):
        hmda.download_lar(2016, data_dir=tmp_path)

    destination = hmda._lar_archive_path(2016, tmp_path)
    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []


def test_failed_overwrite_preserves_existing_archive(tmp_path, monkeypatch):
    original = _archive_bytes({"hmda_lar.csv": b"original"})
    destination = hmda._lar_archive_path(2017, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(original)
    monkeypatch.setattr(
        hmda.urllib.request, "urlopen", lambda url: io.BytesIO(b"not a zip")
    )

    with pytest.raises(RuntimeError, match="2017"):
        hmda.download_lar(2017, data_dir=tmp_path, overwrite=True)

    assert destination.read_bytes() == original
    assert list(destination.parent.glob("*.part")) == []
