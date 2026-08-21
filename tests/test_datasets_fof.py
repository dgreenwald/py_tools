"""Tests for datasets.fof downloads and vintage-aware table loading."""

import io
import zipfile

import pandas as pd
import pytest

from py_tools.datasets import fof


MAPPING = (
    "new,old,title\n"
    "S11.1.b,B.103,Nonfinancial corporate business\n"
).encode()


def _archive_bytes(csv_name="S11_1_b", value=100, date_time=(2026, 6, 10, 0, 0, 0)):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        members = {
            f"csv/{csv_name}.csv": (
                f"date,FL102000005.Q\n2025:Q4,{value}\n"
            ),
            f"data_dictionary/{csv_name}.txt": (
                "FL102000005.Q\tNonfinancial corporate business; total assets"
                f"\tLine 1\t{csv_name}\tMillions of dollars\n"
            ),
        }
        for name, contents in members.items():
            info = zipfile.ZipInfo(name, date_time=date_time)
            archive.writestr(info, contents)
    return output.getvalue()


def _mock_downloads(monkeypatch, archive_payload, mapping_payload=MAPPING):
    requested = []

    def fake_urlopen(request):
        requested.append(request)
        url = request.full_url
        if url == fof.FOF_CSV_URL:
            return io.BytesIO(archive_payload)
        if url == fof.FOF_TABLE_MAPPING_URL:
            return io.BytesIO(mapping_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(fof.urllib.request, "urlopen", fake_urlopen)
    return requested


def test_download_current_installs_expected_vintage_layout(tmp_path, monkeypatch):
    requested = _mock_downloads(monkeypatch, _archive_bytes())

    destination = fof.download_current(vintage="2606", data_dir=tmp_path)

    assert destination == tmp_path / "all_csv" / "2606"
    assert (destination / "csv" / "S11_1_b.csv").exists()
    assert (destination / "data_dictionary" / "S11_1_b.txt").exists()
    assert (destination / "z1_table_mapping.csv").read_bytes() == MAPPING
    assert [request.full_url for request in requested] == [
        fof.FOF_CSV_URL,
        fof.FOF_TABLE_MAPPING_URL,
    ]
    assert all(
        request.get_header("User-agent") == fof.FOF_REQUEST_HEADERS["User-Agent"]
        for request in requested
    )
    assert all(request.get_header("Accept") for request in requested)
    assert list((tmp_path / "all_csv").glob("*.part")) == []


def test_download_current_infers_vintage_from_archive(tmp_path, monkeypatch):
    _mock_downloads(
        monkeypatch,
        _archive_bytes(date_time=(2027, 9, 8, 0, 0, 0)),
    )

    destination = fof.download_current(data_dir=tmp_path)

    assert destination == tmp_path / "all_csv" / "2709"


def test_download_current_skips_existing_unless_overwriting(tmp_path, monkeypatch):
    requested = _mock_downloads(monkeypatch, _archive_bytes(value=100))
    destination = fof.download_current(vintage="2606", data_dir=tmp_path)

    requested.clear()
    assert fof.download_current(vintage="2606", data_dir=tmp_path) == destination
    assert requested == []

    _mock_downloads(monkeypatch, _archive_bytes(value=200))
    fof.download_current(vintage="2606", data_dir=tmp_path, overwrite=True)
    result = pd.read_csv(destination / "csv" / "S11_1_b.csv")
    assert result.loc[0, "FL102000005.Q"] == 200
    assert list((tmp_path / "all_csv").glob(".*.backup")) == []


@pytest.mark.parametrize("vintage", ["2026", "260", "2600", "2613", "abcd"])
def test_download_current_rejects_invalid_vintage(tmp_path, vintage):
    with pytest.raises(ValueError, match="vintage"):
        fof.download_current(vintage=vintage, data_dir=tmp_path)


def test_download_current_rejects_unsafe_archive_and_cleans_up(
    tmp_path, monkeypatch
):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("csv/table.csv", "date,value\n2025:Q4,1\n")
        archive.writestr("data_dictionary/table.txt", "code\tname\n")
        archive.writestr("../escape.csv", "unsafe")
    _mock_downloads(monkeypatch, output.getvalue())

    with pytest.raises(RuntimeError, match="Unsafe path"):
        fof.download_current(vintage="2606", data_dir=tmp_path)

    all_csv = tmp_path / "all_csv"
    assert not (all_csv / "2606").exists()
    assert list(all_csv.iterdir()) == []
    assert not (tmp_path / "escape.csv").exists()


def test_download_current_rejects_invalid_mapping_without_partial_vintage(
    tmp_path, monkeypatch
):
    _mock_downloads(monkeypatch, _archive_bytes(), b"wrong,columns\n1,2\n")

    with pytest.raises(RuntimeError, match="new and old"):
        fof.download_current(vintage="2606", data_dir=tmp_path)

    assert not (tmp_path / "all_csv" / "2606").exists()
    assert list((tmp_path / "all_csv").iterdir()) == []


def test_load_table_resolves_legacy_name_with_current_mapping(
    tmp_path, monkeypatch
):
    _mock_downloads(monkeypatch, _archive_bytes(value=321))
    fof.download_current(vintage="2606", data_dir=tmp_path)

    result = fof.load_table(
        "b103", data_dir=tmp_path, vintage="2606", update_names=True
    )

    assert result.index.tolist() == [pd.Timestamp("2025-10-01")]
    assert result.loc[pd.Timestamp("2025-10-01"), "nfc_total_assets"] == 321


def test_load_table_accepts_new_dotted_table_name(tmp_path, monkeypatch):
    _mock_downloads(monkeypatch, _archive_bytes(value=654))
    fof.download_current(vintage="2606", data_dir=tmp_path)

    result = fof.load_table("S11.1.b", data_dir=tmp_path, vintage="2606")

    assert result.loc[pd.Timestamp("2025-10-01"), "FL102000005.Q"] == 654


def test_predefined_dataset_loads_from_renumbered_release(tmp_path, monkeypatch):
    _mock_downloads(monkeypatch, _archive_bytes(value=456))
    fof.download_current(vintage="2606", data_dir=tmp_path)

    result = fof.load(
        "corporate",
        usecols=["assets"],
        data_dir=tmp_path,
        vintage="2606",
    )

    assert result.loc[pd.Timestamp("2025-10-01"), "assets"] == 456


def test_missing_series_are_filled_with_nan_without_fred(
    tmp_path, monkeypatch
):
    _mock_downloads(monkeypatch, _archive_bytes(value=456))
    fof.download_current(vintage="2606", data_dir=tmp_path)

    def unexpected_fred_load(**kwargs):
        raise AssertionError("FRED should not be called by default")

    monkeypatch.setattr(fof.fred, "load", unexpected_fred_load)
    result = fof.load(
        "corporate",
        usecols=["assets", "loans_asset", "net_dividends_fin"],
        data_dir=tmp_path,
        vintage="2606",
    )

    assert result.loc[pd.Timestamp("2025-10-01"), "assets"] == 456
    assert result["loans_asset"].isna().all()
    assert result["net_dividends_fin"].isna().all()


def test_discontinued_f3_table_has_actionable_error(tmp_path, monkeypatch):
    _mock_downloads(monkeypatch, _archive_bytes())
    fof.download_current(vintage="2606", data_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="discontinued in the June 2026"):
        fof.load_table("f3", data_dir=tmp_path, vintage="2606")


def test_legacy_vintage_still_loads_without_mapping(tmp_path):
    vintage_dir = tmp_path / "all_csv" / "2510"
    (vintage_dir / "csv").mkdir(parents=True)
    (vintage_dir / "csv" / "b103.csv").write_text(
        "date,FL102000005.Q\n2025:Q3,789\n", encoding="utf-8"
    )

    result = fof.load_table("b103", data_dir=tmp_path, vintage="2510")

    assert result.loc[pd.Timestamp("2025-07-01"), "FL102000005.Q"] == 789
