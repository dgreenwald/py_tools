"""Tests for HMDA LAR downloads."""

import io
import zipfile

import pandas as pd
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


def test_all_source_pairs_preserve_year_and_source_order():
    assert hmda._lar_source_year_pairs([2014, 2017, 2014], source="all") == [
        (2014, "cfpb"),
        (2014, "nara"),
        (2017, "ffiec_three_year"),
        (2017, "ffiec_snapshot"),
        (2017, "cfpb"),
    ]

    pairs = hmda._lar_source_year_pairs(source="all")
    assert len(pairs) == sum(len(config) for config in hmda._LAR_SOURCE_CONFIG.values())
    assert pairs[0] == (2025, "ffiec_snapshot")
    assert pairs[-1] == (1981, "nara")


def test_download_all_sources_for_overlapping_years(tmp_path, monkeypatch):
    payload = _archive_bytes({"hmda_lar.csv": b"year\n2017\n"})
    monkeypatch.setattr(hmda.urllib.request, "urlopen", lambda url: io.BytesIO(payload))

    paths = hmda.download_lar([2014, 2017], source="all", data_dir=tmp_path)

    assert paths == [
        hmda._lar_file_path(2014, tmp_path, source="cfpb"),
        hmda._lar_file_path(2014, tmp_path, source="nara"),
        hmda._lar_file_path(2017, tmp_path, source="ffiec_three_year"),
        hmda._lar_file_path(2017, tmp_path, source="ffiec_snapshot"),
        hmda._lar_file_path(2017, tmp_path, source="cfpb"),
    ]


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


def _install_raw_zip(tmp_path, year, source, member, contents):
    path = hmda._lar_file_path(year, tmp_path, source=source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_archive_bytes({member: contents}))
    return path


def _fixed_width_record(widths, values=None):
    values = values or {}
    fields = []
    for index, width in enumerate(widths):
        value = str(values.get(index, ""))
        assert len(value) <= width
        fields.append(value.ljust(width))
    return "".join(fields)


def test_convert_csv_to_source_specific_parquet(tmp_path):
    csv = (
        b"activity_year,lei,state_code,county_code,action_taken,loan_amount,"
        b"interest_rate,unknown_field\n"
        b"2023,00ABC,06,001,1,250000,Exempt,alpha\n"
        b"2023,00DEF,12,003,,300000,4.25,beta\n"
    )
    _install_raw_zip(
        tmp_path,
        2023,
        "ffiec_snapshot",
        "2023_public_lar.csv",
        csv,
    )

    paths = hmda.convert_lar(
        2023, source="ffiec_snapshot", data_dir=tmp_path, chunksize=1
    )

    expected = tmp_path / "parquet" / "ffiec_snapshot" / "2023" / "lar.parquet"
    assert paths == [expected]
    frame = pd.read_parquet(expected)
    assert frame["activity_year"].dtype == pd.Int64Dtype()
    assert frame["action_taken"].dtype == pd.Int64Dtype()
    assert frame["loan_amount"].dtype == pd.Int64Dtype()
    assert frame["lei"].tolist() == ["00ABC", "00DEF"]
    assert frame["county_code"].tolist() == ["001", "003"]
    assert frame["interest_rate"].tolist() == ["Exempt", "4.25"]
    assert frame["unknown_field"].tolist() == ["alpha", "beta"]

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(expected)
    assert parquet.num_row_groups == 2
    metadata = parquet.schema_arrow.metadata
    assert metadata[b"hmda.source"] == b"ffiec_snapshot"
    assert metadata[b"hmda.year"] == b"2023"
    assert metadata[b"hmda.schema_cohort"] == b"csv"


def test_convert_ignores_macos_zip_metadata(tmp_path):
    path = hmda._lar_file_path(2022, tmp_path, source="ffiec_three_year")
    path.parent.mkdir(parents=True)
    path.write_bytes(
        _archive_bytes(
            {
                "2022_public_lar_three_year_csv.csv": (
                    b"activity_year,action_taken\n2022,1\n"
                ),
                "__MACOSX/._2022_public_lar_three_year_csv.csv": b"metadata",
            }
        )
    )

    [output] = hmda.convert_lar(2022, source="ffiec_three_year", data_dir=tmp_path)

    assert pd.read_parquet(output).to_dict("records") == [
        {"activity_year": 2022, "action_taken": 1}
    ]


def test_convert_accepts_nara_lars_archive_member(tmp_path):
    values = {index: "1" for index in range(len(hmda._NARA_1990_2003_WIDTHS))}
    values[1] = "0000123456"
    grouped_amount = values.copy()
    grouped_amount[6] = "1 250"
    grouped_amount[16] = "2 05"
    records = "\r\n".join(
        [
            _fixed_width_record(hmda._NARA_1990_2003_WIDTHS, values),
            _fixed_width_record(hmda._NARA_1990_2003_WIDTHS, grouped_amount),
        ]
    ).encode("ascii")
    _install_raw_zip(tmp_path, 2003, "nara", "HMS.U2003.LARS", records)

    [output] = hmda.convert_lar(2003, source="nara", data_dir=tmp_path)

    frame = pd.read_parquet(output)
    assert len(frame) == 2
    assert frame["respondent_id"].tolist() == ["0000123456", "0000123456"]
    assert frame["loan_amt"].tolist() == [1, 1250]
    assert frame["app_income"].tolist() == [1, 205]


@pytest.mark.parametrize(
    "year,widths,names,cohort",
    [
        (1981, hmda._NARA_PRE_1990_WIDTHS, hmda._NARA_PRE_1990_NAMES, "1981-1989"),
        (
            1990,
            hmda._NARA_1990_2003_WIDTHS,
            hmda._NARA_1990_2003_NAMES,
            "1990-2003",
        ),
        (
            2004,
            hmda._NARA_2004_2014_WIDTHS,
            hmda._NARA_2004_2014_NAMES,
            "2004-2014",
        ),
    ],
)
def test_convert_nara_fixed_width_cohorts(tmp_path, year, widths, names, cohort):
    values = {index: "1" for index in range(len(widths))}
    values[1] = "00001234"
    records = "\r\n".join(
        [_fixed_width_record(widths, values), _fixed_width_record(widths, values)]
    ).encode("latin-1")
    if year < 1990:
        raw_path = hmda._lar_file_path(year, tmp_path, source="nara")
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(records + b"\r\n")
    else:
        _install_raw_zip(tmp_path, year, "nara", f"lar_{year}.dat", records)

    [output] = hmda.convert_lar(year, source="nara", data_dir=tmp_path)
    frame = pd.read_parquet(output)

    assert frame.columns.tolist() == names
    assert len(frame) == 2
    assert frame["respondent_id"].iloc[0] == "00001234"
    assert frame["state_code"].iloc[0] == "1"
    if year == 2004:
        for column in [
            "app_race_1",
            "app_race_2",
            "app_race_3",
            "app_race_4",
            "app_race_5",
            "co_app_race_1",
            "co_app_race_2",
            "co_app_race_3",
            "co_app_race_4",
            "co_app_race_5",
        ]:
            assert frame[column].dtype == pd.Int64Dtype()
    import pyarrow.parquet as pq

    assert pq.ParquetFile(output).schema_arrow.metadata[b"hmda.schema_cohort"] == (
        cohort.encode()
    )


def test_convert_uses_documented_numeric_types_for_cfpb(tmp_path):
    csv = (
        b"as_of_year,respondent_id,state_code,msamd,census_tract_number,"
        b"loan_amount_000s,applicant_income_000s,action_taken,property_type,"
        b"owner_occupancy,applicant_ethnicity,co_applicant_ethnicity,"
        b"applicant_sex,co_applicant_sex,population,"
        b"minority_population,hud_median_family_income,tract_to_msamd_income,"
        b"number_of_owner_occupied_units,number_of_1_to_4_family_units,"
        b"application_date_indicator\n"
        b"2016,0000123456,06,12345,0012.34,250,75.5,1,2,1,2,5,1,2,3177,"
        b"28.77,54000,61.65,160,773,0\n"
    )
    _install_raw_zip(tmp_path, 2016, "cfpb", "hmda_2016.csv", csv)

    [output] = hmda.convert_lar(2016, source="cfpb", data_dir=tmp_path)
    frame = pd.read_parquet(output)

    assert frame["as_of_year"].dtype == pd.Int64Dtype()
    assert frame["loan_amount_000s"].dtype == pd.Int64Dtype()
    assert frame["applicant_income_000s"].dtype == pd.Float64Dtype()
    for column in [
        "property_type",
        "owner_occupancy",
        "applicant_ethnicity",
        "co_applicant_ethnicity",
        "applicant_sex",
        "co_applicant_sex",
        "population",
        "hud_median_family_income",
        "number_of_owner_occupied_units",
        "number_of_1_to_4_family_units",
        "application_date_indicator",
    ]:
        assert frame[column].dtype == pd.Int64Dtype()
    assert frame["minority_population"].dtype == pd.Float64Dtype()
    assert frame["tract_to_msamd_income"].dtype == pd.Float64Dtype()
    assert frame["respondent_id"].iloc[0] == "0000123456"
    assert frame["state_code"].iloc[0] == "06"
    assert frame["msamd"].iloc[0] == "12345"
    assert frame["census_tract_number"].iloc[0] == "0012.34"


def test_convert_uses_numeric_types_for_2017_ffiec_fields(tmp_path):
    csv = (
        b"activity_year,applicant_ethnicity,co_applicant_ethnicity,income,"
        b"tract_one_to_four_family_units\n"
        b"2017,1,5,75.5,1200\n"
        b"2017,2,2,NA,800\n"
    )
    _install_raw_zip(
        tmp_path,
        2017,
        "ffiec_three_year",
        "2017_public_lar_three_year_csv.csv",
        csv,
    )

    [output] = hmda.convert_lar(
        2017, source="ffiec_three_year", data_dir=tmp_path, chunksize=1
    )
    frame = pd.read_parquet(output)

    assert frame["applicant_ethnicity"].dtype == pd.Int64Dtype()
    assert frame["co_applicant_ethnicity"].dtype == pd.Int64Dtype()
    assert frame["income"].dtype == pd.Float64Dtype()
    assert frame["tract_one_to_four_family_units"].dtype == pd.Int64Dtype()
    assert frame["income"].tolist() == [75.5, pd.NA]


def test_load_lar_projects_and_filters_parquet(tmp_path):
    csv = b"activity_year,action_taken,lei\n2023,1,A\n2023,3,B\n"
    _install_raw_zip(tmp_path, 2023, "ffiec_snapshot", "2023_public_lar.csv", csv)
    hmda.convert_lar(2023, source="ffiec_snapshot", data_dir=tmp_path)

    frame = hmda.load_lar(
        2023,
        source="ffiec_snapshot",
        data_dir=tmp_path,
        columns=["lei", "action_taken"],
        filters=[("action_taken", "==", 1)],
    )
    registry_frame = hmda.load(
        yr=2023, source="ffiec_snapshot", data_dir=tmp_path, columns=["lei"]
    )

    assert frame.to_dict("records") == [{"lei": "A", "action_taken": 1}]
    assert registry_frame["lei"].tolist() == ["A", "B"]


def test_load_lar_converts_downloaded_raw_file_if_missing(tmp_path):
    csv = b"activity_year,action_taken,lei\n2023,1,A\n"
    _install_raw_zip(tmp_path, 2023, "ffiec_snapshot", "2023_public_lar.csv", csv)

    with pytest.raises(FileNotFoundError, match="convert_lar"):
        hmda.load_lar(2023, source="ffiec_snapshot", data_dir=tmp_path)

    frame = hmda.load_lar(
        2023,
        source="ffiec_snapshot",
        data_dir=tmp_path,
        convert_if_missing=True,
    )

    assert frame.to_dict("records") == [
        {"activity_year": 2023, "action_taken": 1, "lei": "A"}
    ]
    assert hmda._lar_parquet_path(2023, tmp_path, source="ffiec_snapshot").exists()


def test_conversion_keeps_overlapping_sources_separate(tmp_path):
    _install_raw_zip(
        tmp_path,
        2017,
        "cfpb",
        "hmda_2017.csv",
        b"as_of_year,respondent_id\n2017,OLD\n",
    )
    _install_raw_zip(
        tmp_path,
        2017,
        "ffiec_snapshot",
        "2017_public_lar.csv",
        b"activity_year,lei\n2017,NEW\n",
    )

    [cfpb] = hmda.convert_lar(2017, source="cfpb", data_dir=tmp_path)
    [snapshot] = hmda.convert_lar(2017, source="ffiec_snapshot", data_dir=tmp_path)

    assert cfpb != snapshot
    assert cfpb.parts[-3] == "cfpb"
    assert snapshot.parts[-3] == "ffiec_snapshot"
    assert pd.read_parquet(cfpb)["respondent_id"].tolist() == ["OLD"]
    assert pd.read_parquet(snapshot)["lei"].tolist() == ["NEW"]


def test_convert_all_sources_for_requested_year(tmp_path):
    expected = []
    for source in ("ffiec_three_year", "ffiec_snapshot", "cfpb"):
        path = hmda._lar_parquet_path(2017, tmp_path, source=source)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"source": [source]}).to_parquet(path)
        expected.append(path)

    assert hmda.convert_lar(2017, source="all", data_dir=tmp_path) == expected


def test_conversion_failure_preserves_existing_parquet(tmp_path):
    good = b"activity_year,action_taken\n2023,1\n"
    raw_path = _install_raw_zip(
        tmp_path, 2023, "ffiec_snapshot", "2023_public_lar.csv", good
    )
    [output] = hmda.convert_lar(2023, data_dir=tmp_path)
    original = output.read_bytes()
    raw_path.write_bytes(
        _archive_bytes(
            {"2023_public_lar.csv": b"activity_year,action_taken\n2023,bad\n"}
        )
    )

    with pytest.raises(RuntimeError, match=r"2023.*action_taken"):
        hmda.convert_lar(2023, data_dir=tmp_path, overwrite=True)

    assert output.read_bytes() == original
    assert list(output.parent.glob("*.part")) == []


def test_convert_requires_downloaded_raw_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_lar"):
        hmda.convert_lar(2023, data_dir=tmp_path)


def test_download_cli_dispatches_options_and_prints_paths(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return [tmp_path / "first.zip", tmp_path / "second.zip"]

    monkeypatch.setattr(hmda, "download_lar", fake_download)

    result = hmda.main(
        [
            "download",
            "2013",
            "2014",
            "--source",
            "nara",
            "--data-dir",
            str(tmp_path),
            "--overwrite",
            "--progress",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "years": [2013, 2014],
            "source": "nara",
            "data_dir": str(tmp_path),
            "overwrite": True,
            "progress": True,
        }
    ]
    assert capsys.readouterr().out.splitlines() == [
        str(tmp_path / "first.zip"),
        str(tmp_path / "second.zip"),
    ]


def test_convert_cli_uses_all_source_years_when_omitted(tmp_path, monkeypatch, capsys):
    calls = []
    output = tmp_path / "parquet" / "nara" / "2014" / "lar.parquet"

    def fake_convert(**kwargs):
        calls.append(kwargs)
        return [output]

    monkeypatch.setattr(hmda, "convert_lar", fake_convert)

    result = hmda.main(
        [
            "convert",
            "--source",
            "nara",
            "--data-dir",
            str(tmp_path),
            "--chunksize",
            "25000",
            "--compression",
            "snappy",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "years": None,
            "source": "nara",
            "data_dir": str(tmp_path),
            "overwrite": False,
            "chunksize": 25000,
            "compression": "snappy",
        }
    ]
    assert capsys.readouterr().out == f"{output}\n"


def test_cli_accepts_all_sources(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hmda,
        "download_lar",
        lambda **kwargs: calls.append(kwargs) or [],
    )

    assert hmda.main(["download", "2017", "--source", "all"]) == 0
    assert calls[0]["years"] == [2017]
    assert calls[0]["source"] == "all"
