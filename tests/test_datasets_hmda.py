"""Tests for HMDA snapshot LAR downloads."""

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
    config = hmda._SNAPSHOT_LAR_CONFIG[year]

    assert config["url"] == (
        f"https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/"
        f"{year}_public_lar_csv.zip"
    )
    assert hmda._snapshot_lar_archive_path(year, tmp_path) == (
        tmp_path / "snapshot" / str(year) / f"{year}_public_lar_csv.zip"
    )


def test_download_accepts_default_scalar_and_deduplicated_years(tmp_path, monkeypatch):
    payload = _archive_bytes({"2025_public_lar.csv": b"lei,loan_type\n1,2\n"})
    requested_urls = []

    def fake_urlopen(url):
        requested_urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(hmda.urllib.request, "urlopen", fake_urlopen)

    default_paths = hmda.download_snapshot_lar(data_dir=tmp_path / "default")
    scalar_paths = hmda.download_snapshot_lar(2025, data_dir=tmp_path / "scalar")
    iterable_paths = hmda.download_snapshot_lar(
        [2017, 2025, 2017], data_dir=tmp_path / "iterable"
    )

    assert default_paths == [
        tmp_path / "default" / "snapshot" / str(year) / f"{year}_public_lar_csv.zip"
        for year in hmda.SNAPSHOT_LAR_YEARS
    ]
    assert scalar_paths == [
        tmp_path / "scalar" / "snapshot" / "2025" / "2025_public_lar_csv.zip"
    ]
    assert iterable_paths == [
        tmp_path / "iterable" / "snapshot" / "2017" / "2017_public_lar_csv.zip",
        tmp_path / "iterable" / "snapshot" / "2025" / "2025_public_lar_csv.zip",
    ]
    assert requested_urls == [
        *(hmda._SNAPSHOT_LAR_CONFIG[year]["url"] for year in hmda.SNAPSHOT_LAR_YEARS),
        hmda._SNAPSHOT_LAR_CONFIG[2025]["url"],
        hmda._SNAPSHOT_LAR_CONFIG[2017]["url"],
        hmda._SNAPSHOT_LAR_CONFIG[2025]["url"],
    ]
    assert all(
        hmda._validate_snapshot_lar_archive(path)
        for path in default_paths + scalar_paths + iterable_paths
    )


def test_download_rejects_unsupported_year(tmp_path):
    with pytest.raises(ValueError, match=r"Supported years are: 2025, .*2017"):
        hmda.download_snapshot_lar(2016, data_dir=tmp_path)


def test_download_skips_valid_archive_unless_overwriting(tmp_path, monkeypatch):
    original = _archive_bytes({"2025_public_lar.csv": b"original"})
    replacement = _archive_bytes({"2025_public_lar.csv": b"replacement"})
    destination = hmda._snapshot_lar_archive_path(2025, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(original)
    calls = []

    def fake_urlopen(url):
        calls.append(url)
        return io.BytesIO(replacement)

    monkeypatch.setattr(hmda.urllib.request, "urlopen", fake_urlopen)

    hmda.download_snapshot_lar(2025, data_dir=tmp_path)
    assert destination.read_bytes() == original
    assert calls == []

    hmda.download_snapshot_lar(2025, data_dir=tmp_path, overwrite=True)
    assert destination.read_bytes() == replacement
    assert calls == [hmda._SNAPSHOT_LAR_CONFIG[2025]["url"]]


@pytest.mark.parametrize(
    "payload",
    [
        b"not a zip",
        _archive_bytes({"readme.txt": b"not CSV data"}),
        _archive_bytes({"2025_public_lar.csv": b""}),
    ],
)
def test_download_rejects_invalid_content_and_cleans_partial(
    tmp_path, monkeypatch, payload
):
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    with pytest.raises(RuntimeError, match=r"2025.*nonempty CSV"):
        hmda.download_snapshot_lar(2025, data_dir=tmp_path)

    destination = hmda._snapshot_lar_archive_path(2025, tmp_path)
    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []


def test_failed_overwrite_preserves_existing_archive(tmp_path, monkeypatch):
    original = _archive_bytes({"2025_public_lar.csv": b"original"})
    destination = hmda._snapshot_lar_archive_path(2025, tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(original)
    monkeypatch.setattr(
        hmda.urllib.request, "urlopen", lambda url: io.BytesIO(b"not a zip")
    )

    with pytest.raises(RuntimeError, match="2025"):
        hmda.download_snapshot_lar(2025, data_dir=tmp_path, overwrite=True)

    assert destination.read_bytes() == original
    assert list(destination.parent.glob("*.part")) == []
