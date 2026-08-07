import argparse
import os
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

default_dir = config.base_dir() + "hmda/"
DATASET_NAME = "hmda"
DESCRIPTION = "Home Mortgage Disclosure Act (HMDA) dataset loader."
SNAPSHOT_LAR_YEARS = tuple(range(2025, 2016, -1))
_SNAPSHOT_LAR_CONFIG = {
    year: {
        "url": (
            f"https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/"
            f"{year}_public_lar_csv.zip"
        ),
        "filename": f"{year}_public_lar_csv.zip",
        "format": "zip_csv",
    }
    for year in SNAPSHOT_LAR_YEARS
}
THREE_YEAR_LAR_YEARS = tuple(range(2022, 2016, -1))
_THREE_YEAR_LAR_CONFIG = {
    year: {
        "url": (
            f"https://files.ffiec.cfpb.gov/static-data/three-year/{year}/"
            f"{year}_public_lar_three_year_csv.zip"
        ),
        "filename": f"{year}_public_lar_three_year_csv.zip",
        "format": "zip_csv",
    }
    for year in THREE_YEAR_LAR_YEARS
}
CFPB_LAR_YEARS = tuple(range(2017, 2006, -1))
_CFPB_LAR_CONFIG = {
    year: {
        "url": (
            "https://files.consumerfinance.gov/hmda-historic-loan-data/"
            f"hmda_{year}_nationwide_all-records_codes.zip"
        ),
        "filename": f"hmda_{year}_nationwide_all-records_codes.zip",
        "format": "zip_csv",
    }
    for year in CFPB_LAR_YEARS
}
NATIONAL_ARCHIVES_LAR_YEARS = tuple(range(1989, 1980, -1))
_NATIONAL_ARCHIVES_LAR_CONFIG = {
    year: {
        "url": (
            "https://catalog.archives.gov/medialz/electronic-records/"
            f"rg-082/hmda/HMD_FACDSB{year % 100:02d}.txt"
        ),
        "filename": f"HMD_FACDSB{year % 100:02d}.txt",
        "format": "txt",
    }
    for year in NATIONAL_ARCHIVES_LAR_YEARS
}
_NATIONAL_ARCHIVES_ZIP_PATHS = {
    **{year: f"HMS.U{year}.LARS.zip" for year in range(1990, 2001)},
    2001: "ULAR01/HMS.U2001.LARS.PUBLIC.DATA.zip",
    2002: "HMS.U2002.LARS.zip",
    2003: "HMS.U2003.LARS.zip",
    2004: "ULAR04/u2004lar.public.dat.zip",
    2005: "ULAR0506/LARS.ULTIMATE.2005.DAT.zip",
    2006: "ULAR0506/LARS.ULTIMATE.2006.DAT.zip",
    2007: "ULAR0708/lars.ultimate.2007.dat.zip",
    2008: "ULAR0708/lars.ultimate.2008.dat.zip",
    2009: "ULAR09/2009_Ultimate_PUBLIC_LAR.dat.zip",
    2010: "UTL10/Lars.ultimate.2010.dat.zip",
    2011: "UTL11/Lars.ultimate.2011.dat.zip",
    2012: "2012/Lars.ultimate.2012.dat.zip",
    2013: "2013/Lars.ultimate.2013.dat.zip",
    2014: "2014/ULAR_2014.zip",
}
_NATIONAL_ARCHIVES_LAR_CONFIG.update(
    {
        year: {
            "url": (
                "https://catalog.archives.gov/medialz/electronic-records/"
                f"rg-082/hmda/{relative_path}"
            ),
            "filename": Path(relative_path).name,
            "format": "zip",
        }
        for year, relative_path in _NATIONAL_ARCHIVES_ZIP_PATHS.items()
    }
)
NATIONAL_ARCHIVES_LAR_YEARS = tuple(range(2014, 1980, -1))
LAR_SOURCES = ("ffiec_three_year", "ffiec_snapshot", "cfpb", "nara")
LAR_BULK_SOURCES = ("auto", "all", *LAR_SOURCES)
_LAR_SOURCE_CONFIG = {
    "ffiec_three_year": _THREE_YEAR_LAR_CONFIG,
    "ffiec_snapshot": _SNAPSHOT_LAR_CONFIG,
    "cfpb": _CFPB_LAR_CONFIG,
    "nara": _NATIONAL_ARCHIVES_LAR_CONFIG,
}
LAR_YEARS = tuple(range(2025, 1980, -1))
_AUTO_LAR_SOURCE = {
    **{year: "nara" for year in range(1981, 2007)},
    **{year: "cfpb" for year in range(2007, 2017)},
    **{year: "ffiec_three_year" for year in range(2017, 2023)},
    **{year: "ffiec_snapshot" for year in range(2023, 2026)},
}

_NARA_PRE_1990_WIDTHS = [
    28,
    8,
    4,
    6,
    2,
    3,
    1,
    2,
    1,
    4,
    9,
    1,
    4,
    9,
    1,
    4,
    9,
    1,
    4,
    9,
    1,
    4,
    9,
    1,
    1,
]
_NARA_PRE_1990_NAMES = [
    "respondent_name",
    "respondent_id",
    "msa_of_report",
    "census_tract",
    "state_code",
    "county_code",
    "agency_code",
    "census_validity_flag",
    "government_loan_validity_flag",
    "government_loan_count",
    "government_loan_amount_000s",
    "conventional_loan_validity_flag",
    "conventional_loan_count",
    "conventional_loan_amount_000s",
    "home_improvement_validity_flag",
    "home_improvement_loan_count",
    "home_improvement_loan_amount_000s",
    "multifamily_validity_flag",
    "multifamily_loan_count",
    "multifamily_loan_amount_000s",
    "nonoccupant_validity_flag",
    "nonoccupant_loan_count",
    "nonoccupant_loan_amount_000s",
    "record_quality_flag",
    "filler",
]
_NARA_1990_2003_WIDTHS = [
    4,
    10,
    1,
    1,
    1,
    1,
    5,
    1,
    4,
    2,
    3,
    7,
    1,
    1,
    1,
    1,
    4,
    1,
    1,
    1,
    1,
    1,
    7,
]
_NARA_1990_2003_NAMES = [
    "asof_date",
    "respondent_id",
    "agency_code",
    "loan_type",
    "loan_purp",
    "occupancy",
    "loan_amt",
    "action_taken",
    "prop_msa",
    "state_code",
    "county_code",
    "census_tract",
    "app_race",
    "co_app_race",
    "app_sex",
    "co_app_sex",
    "app_income",
    "purchaser_type",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "edit_status",
    "seq_num",
]
_NARA_2004_2014_WIDTHS = [
    4,
    10,
    1,
    1,
    1,
    1,
    5,
    1,
    5,
    2,
    3,
    7,
    1,
    1,
    4,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    5,
    1,
    1,
    7,
]
_NARA_2004_2014_NAMES = [
    "asof_date",
    "respondent_id",
    "agency_code",
    "loan_type",
    "loan_purp",
    "occupancy",
    "loan_amt",
    "action_taken",
    "prop_msa",
    "state_code",
    "county_code",
    "census_tract",
    "app_sex",
    "co_app_sex",
    "app_income",
    "purchaser_type",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "edit_status",
    "prop_type",
    "preapprovals",
    "app_ethnicity",
    "co_app_ethnicity",
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
    "rate_spread",
    "hoepa_status",
    "lien_status",
    "seq_num",
]

_IDENTIFIER_COLUMNS = {
    "lei",
    "respondent_id",
    "respondent_id_ts",
    "sequence_number",
    "seq_num",
    "uli",
}
_GEOGRAPHY_COLUMNS = {
    "census_tract",
    "census_tract_number",
    "county_code",
    "derived_msa_md",
    "msamd",
    "msa_md",
    "msa_of_property",
    "msa_of_report",
    "prop_msa",
    "state_code",
    "zip_code",
}
_INTEGER_COLUMNS = {
    "activity_year",
    "action_taken",
    "app_ethnicity",
    "app_income",
    "app_race",
    "app_sex",
    "applicant_ethnicity",
    "applicant_sex",
    "applicant_credit_score_type",
    "applicant_ethnicity_observed",
    "applicant_sex_observed",
    "as_of_year",
    "asof_date",
    "aus_1",
    "aus_2",
    "aus_3",
    "aus_4",
    "aus_5",
    "balloon_payment",
    "business_or_commercial_purpose",
    "census_validity_flag",
    "co_app_ethnicity",
    "co_app_income",
    "co_app_race",
    "co_app_sex",
    "co_applicant_ethnicity",
    "co_applicant_sex",
    "co_applicant_credit_score_type",
    "co_applicant_ethnicity_observed",
    "co_applicant_sex_observed",
    "construction_method",
    "conventional_loan_amount_000s",
    "conventional_loan_count",
    "conventional_loan_validity_flag",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "denial_reason_4",
    "edit_status",
    "ffiec_msa_md_median_family_income",
    "government_loan_amount_000s",
    "government_loan_count",
    "government_loan_validity_flag",
    "hoepa_status",
    "home_improvement_loan_amount_000s",
    "home_improvement_loan_count",
    "home_improvement_validity_flag",
    "hud_median_family_income",
    "initially_payable_to_institution",
    "interest_only_payment",
    "lien_status",
    "loan_amt",
    "loan_amount",
    "loan_amount_000s",
    "loan_purp",
    "loan_purpose",
    "loan_term",
    "loan_type",
    "manufactured_home_land_property_interest",
    "manufactured_home_secured_property_type",
    "multifamily_affordable_units",
    "multifamily_loan_amount_000s",
    "multifamily_loan_count",
    "multifamily_validity_flag",
    "negative_amortization",
    "nonoccupant_loan_amount_000s",
    "nonoccupant_loan_count",
    "nonoccupant_validity_flag",
    "occupancy",
    "occupancy_type",
    "owner_occupancy",
    "open_end_line_of_credit",
    "other_nonamortizing_features",
    "preapprovals",
    "preapproval",
    "prop_type",
    "property_type",
    "purchaser_type",
    "record_quality_flag",
    "reverse_mortgage",
    "submission_of_application",
    "total_units",
    "application_date_indicator",
    "number_of_1_to_4_family_units",
    "number_of_owner_occupied_units",
    "population",
    "tract_median_age_of_housing_units",
    "tract_one_to_four_family_units",
    "tract_one_to_four_family_homes",
    "tract_owner_occupied_units",
    "tract_population",
}
_FLOAT_COLUMNS = {
    "applicant_income_000s",
    "income",
    "loan_to_value_ratio",
    "minority_population",
    "minority_population_percent",
    "rate_spread",
    "tract_minority_population_percent",
    "tract_to_msamd_income",
    "tract_to_msa_income_percentage",
}
_MIXED_VALUE_COLUMNS = {
    "combined_loan_to_value_ratio",
    "debt_to_income_ratio",
    "discount_points",
    "interest_rate",
    "intro_rate_period",
    "lender_credits",
    "loan_term",
    "multifamily_affordable_units",
    "origination_charges",
    "prepayment_penalty_term",
    "property_value",
    "rate_spread",
    "total_loan_costs",
    "total_points_and_fees",
    "total_units",
}
_INTEGER_PREFIXES = (
    "app_race_",
    "applicant_ethnicity_",
    "applicant_race_",
    "co_app_race_",
    "co_applicant_ethnicity_",
    "co_applicant_race_",
)
_NULL_TOKENS = {"", "NA", "N/A", "NULL", "null"}


def _normalize_lar_years(years, source="auto"):
    """Return source-supported LAR years in caller-specified order."""
    if source == "auto":
        supported_years = LAR_YEARS
    elif source in _LAR_SOURCE_CONFIG:
        supported_years = tuple(sorted(_LAR_SOURCE_CONFIG[source], reverse=True))
    else:
        supported = ", ".join(("auto", *LAR_SOURCES))
        raise ValueError(
            f"Unsupported HMDA LAR source {source!r}. "
            f"Supported sources are: {supported}."
        )
    if years is None:
        requested = list(supported_years)
    elif isinstance(years, (int, np.integer)):
        requested = [int(years)]
    else:
        try:
            requested = [int(year) for year in years]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "years must be an integer or an iterable of integers"
            ) from exc

    unsupported = [year for year in requested if year not in supported_years]
    if unsupported:
        supported = ", ".join(str(year) for year in supported_years)
        raise ValueError(
            f"HMDA LAR year(s) {unsupported} are unavailable from "
            f"source {source!r}. Supported years are: {supported}."
        )

    return list(dict.fromkeys(requested))


def _resolve_lar_source(year, source="auto"):
    """Resolve an explicit source name for a requested year."""
    if source == "auto":
        try:
            return _AUTO_LAR_SOURCE[year]
        except KeyError as exc:
            raise ValueError(f"Unsupported HMDA LAR year: {year}") from exc
    if source not in _LAR_SOURCE_CONFIG:
        supported = ", ".join(("auto", *LAR_SOURCES))
        raise ValueError(
            f"Unsupported HMDA LAR source {source!r}. "
            f"Supported sources are: {supported}."
        )
    if year not in _LAR_SOURCE_CONFIG[source]:
        raise ValueError(f"HMDA LAR year {year} is unavailable from source {source!r}.")
    return source


def _lar_source_year_pairs(years=None, source="auto"):
    """Return ordered ``(year, source)`` pairs for a bulk operation."""
    if source not in LAR_BULK_SOURCES:
        supported = ", ".join(LAR_BULK_SOURCES)
        raise ValueError(
            f"Unsupported HMDA LAR source {source!r}. "
            f"Supported sources are: {supported}."
        )
    if source != "all":
        requested = _normalize_lar_years(years, source=source)
        return [(year, _resolve_lar_source(year, source)) for year in requested]

    requested = _normalize_lar_years(years, source="auto")
    return [
        (year, candidate)
        for year in requested
        for candidate in LAR_SOURCES
        if year in _LAR_SOURCE_CONFIG[candidate]
    ]


def _lar_file_path(year, data_dir=default_dir, source="auto"):
    """Return the canonical local path for an annual LAR source file."""
    resolved_source = _resolve_lar_source(year, source)
    year_config = _LAR_SOURCE_CONFIG[resolved_source][year]
    return (
        Path(data_dir) / "raw" / resolved_source / str(year) / year_config["filename"]
    )


def _is_zip_metadata(member):
    """Return whether a ZIP member is macOS filesystem metadata."""
    parts = Path(member.filename).parts
    return "__MACOSX" in parts or Path(member.filename).name.startswith("._")


def _validate_lar_zip(path, require_csv=False):
    """Return whether a ZIP has a suitable nonempty data member."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(
                not member.is_dir()
                and not _is_zip_metadata(member)
                and member.file_size > 0
                and (not require_csv or Path(member.filename).suffix.lower() == ".csv")
                for member in archive.infolist()
            )
    except (OSError, zipfile.BadZipFile):
        return False


def _validate_lar_text(path):
    """Return whether a file begins with uniform ASCII fixed-width records."""
    try:
        with Path(path).open("rb") as source:
            sample = source.read(8192)
        text = sample.decode("ascii")
    except (OSError, UnicodeDecodeError):
        return False

    complete_lines = text.splitlines()
    if sample and not sample.endswith((b"\n", b"\r")):
        complete_lines = complete_lines[:-1]
    return (
        len(complete_lines) >= 2
        and len({len(line) for line in complete_lines}) == 1
        and len(complete_lines[0]) >= 50
        and all(line.isprintable() and line.strip() for line in complete_lines)
    )


def _validate_lar_file(path, file_format):
    """Return whether a downloaded LAR file has its configured format."""
    if file_format == "zip":
        return _validate_lar_zip(path)
    if file_format == "zip_csv":
        return _validate_lar_zip(path, require_csv=True)
    if file_format == "txt":
        return _validate_lar_text(path)
    raise ValueError(f"Unsupported HMDA LAR file format: {file_format}")


def _response_content_length(response):
    """Return a response's byte length when supplied by the server."""
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    if value is None and hasattr(response, "getheader"):
        value = response.getheader("Content-Length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_progress(label, downloaded, total, finished=False):
    """Write one in-place terminal progress update to stderr."""
    downloaded_mib = downloaded / (1024 * 1024)
    if total:
        fraction = min(downloaded / total, 1.0)
        completed = round(30 * fraction)
        bar = "=" * completed + "-" * (30 - completed)
        total_mib = total / (1024 * 1024)
        status = (
            f"{label} [{bar}] {fraction:6.1%} "
            f"{downloaded_mib:,.1f}/{total_mib:,.1f} MiB"
        )
    else:
        status = f"{label} {downloaded_mib:,.1f} MiB"
    print(status, end="\n" if finished else "\r", file=sys.stderr, flush=True)


def _stream_response(response, output, label=None):
    """Copy a response to an output stream, optionally reporting progress."""
    total = _response_content_length(response)
    downloaded = 0
    try:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if label is not None:
                _write_progress(label, downloaded, total)
    finally:
        if label is not None:
            _write_progress(label, downloaded, total, finished=True)


def download_lar(
    years=None,
    source="auto",
    data_dir=default_dir,
    overwrite=False,
    progress=False,
):
    """Download annual HMDA LAR files from the source for each year.

    Parameters
    ----------
    years : int or iterable of int, optional
        Year or years to download. By default, download every configured year.
    source : {"auto", "all", "ffiec_three_year", "ffiec_snapshot", "cfpb", "nara"}, optional
        Data release to download. ``"auto"`` prefers FFIEC three-year files
        for 2017--2022, FFIEC snapshots for 2023 onward, CFPB files for
        2007--2016, and National Archives files for earlier years. With an
        explicit source and no ``years``, download every year from that source.
        ``"all"`` downloads every available source for each requested year.
    data_dir : str or path-like, optional
        Root directory for HMDA data files.
    overwrite : bool, optional
        Replace valid archives that are already present.
    progress : bool, optional
        Display download progress on standard error. Defaults to ``False``.

    Returns
    -------
    list of pathlib.Path
        Local source paths in requested year and source order.

    Raises
    ------
    ValueError
        If any requested year is unsupported.
    RuntimeError
        If a download fails or has the wrong file format.
    """
    requested = _lar_source_year_pairs(years, source=source)
    paths = []

    for year, resolved_source in requested:
        destination = _lar_file_path(year, data_dir=data_dir, source=resolved_source)
        year_config = _LAR_SOURCE_CONFIG[resolved_source][year]
        paths.append(destination)
        if (
            destination.exists()
            and _validate_lar_file(destination, year_config["format"])
            and not overwrite
        ):
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        url = year_config["url"]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".hmda_{year}_lar.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url) as response:
                    label = f"HMDA {year} ({resolved_source})" if progress else None
                    _stream_response(response, output, label=label)

            if not _validate_lar_file(temporary, year_config["format"]):
                raise RuntimeError(
                    f"downloaded content is not valid {year_config['format']} data"
                )
            os.replace(temporary, destination)
        except Exception as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download HMDA LAR data for {year} from {url}: {exc}"
            ) from exc

    return paths


def _lar_parquet_path(year, data_dir=default_dir, source="auto"):
    """Return the canonical parquet path for one source and year."""
    resolved_source = _resolve_lar_source(year, source)
    return Path(data_dir) / "parquet" / resolved_source / str(year) / "lar.parquet"


def _normalized_column_name(column):
    """Return a source-column name in the form used by dtype maps."""
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _column_kind(column):
    """Return the storage kind for a source column."""
    name = _normalized_column_name(column)
    if name in _IDENTIFIER_COLUMNS or name in _GEOGRAPHY_COLUMNS:
        return "string"
    if name in _MIXED_VALUE_COLUMNS:
        return "string"
    if name in _INTEGER_COLUMNS or name.startswith(_INTEGER_PREFIXES):
        return "integer"
    if name in _FLOAT_COLUMNS:
        return "float"
    return "string"


def _null_numeric_tokens(series):
    """Normalize numeric whitespace/grouping and documented null tokens."""
    values = series.astype("string").str.strip()
    is_null = values.str.upper().isin({token.upper() for token in _NULL_TOKENS})
    values = values.mask(is_null)
    values = values.str.replace(r"\s+", "", regex=True)
    comma_grouped = values.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", na=False)
    values.loc[comma_grouped] = values.loc[comma_grouped].str.replace(
        ",", "", regex=False
    )
    return values


def _coerce_lar_chunk(frame, year, source):
    """Apply stable, analysis-oriented dtypes to a raw LAR chunk."""
    converted = frame.copy()
    for column in converted.columns:
        kind = _column_kind(column)
        values = converted[column].astype("string")
        if kind == "string":
            converted[column] = values.mask(values == "")
            continue

        values = _null_numeric_tokens(values)
        try:
            numeric = pd.to_numeric(values, errors="raise")
            if kind == "integer":
                nonnull = numeric.dropna()
                if not np.equal(nonnull, np.floor(nonnull)).all():
                    raise ValueError("contains a non-integral value")
                converted[column] = numeric.astype("Int64")
            else:
                converted[column] = numeric.astype("Float64")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {kind} data in HMDA {year} ({source}) "
                f"column {column!r}: {exc}"
            ) from exc
    return converted


def _nara_layout(year):
    """Return names, widths, and cohort label for a NARA LAR year."""
    if year < 1990:
        return _NARA_PRE_1990_NAMES, _NARA_PRE_1990_WIDTHS, "1981-1989"
    if year < 2004:
        return _NARA_1990_2003_NAMES, _NARA_1990_2003_WIDTHS, "1990-2003"
    return _NARA_2004_2014_NAMES, _NARA_2004_2014_WIDTHS, "2004-2014"


def _zip_data_member(archive, source):
    """Return the single data member from an HMDA source ZIP."""
    members = [
        member
        for member in archive.infolist()
        if not member.is_dir() and not _is_zip_metadata(member)
    ]
    if source in {"cfpb", "ffiec_snapshot", "ffiec_three_year"}:
        candidates = [
            member
            for member in members
            if member.file_size > 0 and Path(member.filename).suffix.lower() == ".csv"
        ]
    else:
        data_suffixes = {"", ".dat", ".data", ".lars", ".txt"}
        candidates = [
            member
            for member in members
            if member.file_size > 0
            and Path(member.filename).suffix.lower() in data_suffixes
            and not Path(member.filename).name.lower().startswith("readme")
        ]
    if len(candidates) != 1:
        names = ", ".join(member.filename for member in candidates) or "none"
        raise ValueError(
            f"Expected exactly one HMDA data member for source {source!r}; "
            f"found {len(candidates)}: {names}."
        )
    return candidates[0]


@contextmanager
def _open_lar_data(path, source):
    """Yield the binary data stream inside a raw HMDA file."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            member = _zip_data_member(archive, source)
            with archive.open(member) as data:
                yield data
    else:
        with path.open("rb") as data:
            yield data


def _validate_nara_record_width(data, expected_width, year):
    """Check an initial sample of fixed-width NARA records."""
    position = data.tell()
    lengths = []
    for _ in range(20):
        line = data.readline()
        if not line:
            break
        record = line.rstrip(b"\r\n")
        if record:
            lengths.append(len(record))
    data.seek(position)
    invalid = sorted({length for length in lengths if length != expected_width})
    if not lengths or invalid:
        observed = ", ".join(str(length) for length in invalid) or "no records"
        raise ValueError(
            f"Invalid NARA HMDA {year} fixed-width data: expected "
            f"{expected_width}-byte records, observed {observed}."
        )


def _lar_chunk_reader(path, year, source, chunksize):
    """Yield raw DataFrame chunks for one annual HMDA file."""
    with _open_lar_data(path, source) as data:
        if source == "nara":
            names, widths, _ = _nara_layout(year)
            _validate_nara_record_width(data, sum(widths), year)
            yield from pd.read_fwf(
                data,
                widths=widths,
                names=names,
                dtype=str,
                keep_default_na=False,
                chunksize=chunksize,
                encoding="latin-1",
            )
        else:
            yield from pd.read_csv(
                data,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                chunksize=chunksize,
                encoding="utf-8-sig",
            )


def _valid_parquet(path):
    """Return whether a path contains readable parquet metadata."""
    try:
        import pyarrow.parquet as pq

        pq.ParquetFile(path)
        return True
    except Exception:
        return False


def convert_lar(
    years=None,
    source="auto",
    data_dir=default_dir,
    overwrite=False,
    chunksize=100000,
    compression="zstd",
):
    """Convert downloaded HMDA LAR files to source-specific parquet files.

    Raw source columns are retained, while documented measures and numeric
    codes are converted to stable nullable numeric dtypes. Identifiers and
    geographic codes remain strings. Conversion is chunked and each chunk is
    written as a parquet row group.

    Parameters
    ----------
    years : int or iterable of int, optional
        Year or years to convert. By default, convert every configured year.
    source : {"auto", "all", "ffiec_three_year", "ffiec_snapshot", "cfpb", "nara"}
        Source release to convert. ``"auto"`` uses the download precedence.
        ``"all"`` converts every available source for each requested year.
    data_dir : str or path-like, optional
        Root containing the ``raw`` and ``parquet`` HMDA directories.
    overwrite : bool, optional
        Replace valid parquet files that already exist.
    chunksize : int, optional
        Number of source rows per parquet row group.
    compression : str, optional
        Parquet compression codec. Defaults to ``"zstd"``.

    Returns
    -------
    list of pathlib.Path
        Parquet paths in requested year and source order.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "HMDA parquet conversion requires pyarrow; install py_tools[datasets]."
        ) from exc

    if not isinstance(chunksize, (int, np.integer)) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")

    requested = _lar_source_year_pairs(years, source=source)
    outputs = []
    for year, resolved_source in requested:
        raw_path = _lar_file_path(year, data_dir=data_dir, source=resolved_source)
        output_path = _lar_parquet_path(year, data_dir=data_dir, source=resolved_source)
        outputs.append(output_path)

        if output_path.exists() and _valid_parquet(output_path) and not overwrite:
            continue
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw HMDA {year} ({resolved_source}) file not found at "
                f"{raw_path}. Run download_lar first."
            )
        year_config = _LAR_SOURCE_CONFIG[resolved_source][year]
        if not _validate_lar_file(raw_path, year_config["format"]):
            raise ValueError(
                f"Raw HMDA {year} ({resolved_source}) file is not valid "
                f"{year_config['format']} data: {raw_path}."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        writer = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.stem}.",
                suffix=".parquet.part",
                dir=output_path.parent,
                delete=False,
            ) as temp_file:
                temporary = Path(temp_file.name)

            for chunk_number, raw_chunk in enumerate(
                _lar_chunk_reader(raw_path, year, resolved_source, int(chunksize))
            ):
                chunk = _coerce_lar_chunk(raw_chunk, year, resolved_source)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    cohort = (
                        _nara_layout(year)[2] if resolved_source == "nara" else "csv"
                    )
                    metadata = dict(table.schema.metadata or {})
                    metadata.update(
                        {
                            b"hmda.source": resolved_source.encode(),
                            b"hmda.year": str(year).encode(),
                            b"hmda.raw_filename": raw_path.name.encode(),
                            b"hmda.schema_cohort": cohort.encode(),
                        }
                    )
                    table = table.replace_schema_metadata(metadata)
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression=compression,
                        use_dictionary=True,
                    )
                elif table.schema.remove_metadata() != writer.schema.remove_metadata():
                    raise ValueError(
                        f"Schema changed in HMDA {year} ({resolved_source}) "
                        f"at chunk {chunk_number}."
                    )
                writer.write_table(table)

            if writer is None:
                raise ValueError(
                    f"Raw HMDA {year} ({resolved_source}) file contains no records."
                )
            writer.close()
            writer = None
            if not _valid_parquet(temporary):
                raise RuntimeError("converted output does not contain valid parquet")
            os.replace(temporary, output_path)
            temporary = None
        except Exception as exc:
            if writer is not None:
                writer.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to convert HMDA {year} ({resolved_source}) from "
                f"{raw_path}: {exc}"
            ) from exc

    return outputs


def load_lar(
    year,
    source="auto",
    data_dir=default_dir,
    columns=None,
    filters=None,
    convert_if_missing=False,
):
    """Load one source-specific annual HMDA parquet file.

    Parameters
    ----------
    year : int
        HMDA survey year to load.
    source : {"auto", "ffiec_three_year", "ffiec_snapshot", "cfpb", "nara"}
        Source release to load. ``"auto"`` uses the download precedence.
    data_dir : str or path-like, optional
        Root containing the ``raw`` and ``parquet`` HMDA directories.
    columns : list of str, optional
        Columns to read from parquet.
    filters : list, optional
        PyArrow-compatible parquet filters.
    convert_if_missing : bool, optional
        Convert the downloaded raw file with :func:`convert_lar` when the
        parquet file is absent. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        Source-specific HMDA records.
    """
    year = int(year)
    resolved_source = _resolve_lar_source(year, source)
    parquet_path = _lar_parquet_path(year, data_dir, resolved_source)
    if not parquet_path.exists() and convert_if_missing:
        convert_lar(year, source=resolved_source, data_dir=data_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"HMDA {year} ({resolved_source}) parquet file not found at "
            f"{parquet_path}. Run convert_lar first."
        )
    return pd.read_parquet(parquet_path, columns=columns, filters=filters)


def load(data_dir=None, **kwargs):
    """Load HMDA data from a source-specific annual parquet file.

    This registry-compatible wrapper accepts ``year`` or the legacy spelling
    ``yr`` and delegates to :func:`load_lar`.

    Parameters
    ----------
    data_dir : str, optional
        Root directory containing source-specific HMDA parquet files.
    **kwargs
        Additional keyword arguments passed directly to ``load_lar``.

    Returns
    -------
    pandas.DataFrame
        HMDA loan-application records for the requested year.
    """
    if "year" in kwargs and "yr" in kwargs:
        raise TypeError("Specify only one of year or yr")
    if "yr" in kwargs:
        kwargs["year"] = kwargs.pop("yr")
    if data_dir is not None:
        kwargs.setdefault("data_dir", data_dir)
    return load_lar(**kwargs)


def cat(num):
    """Return a list of integers from 1 to num inclusive.

    Parameters
    ----------
    num : int
        Upper bound of the integer range (inclusive).

    Returns
    -------
    list
        ``[1, 2, ..., num]``.
    """
    return list(range(1, num + 1))


def to_float(df, var):
    """Convert a DataFrame column to float64 in-place.

    Non-numeric values are coerced to ``NaN``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose column will be converted.
    var : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        The same ``df`` with ``df[var]`` cast to ``numpy.float64``.
    """
    df[var] = pd.to_numeric(df[var], errors="coerce").astype(np.float64)
    return df


def load_chunk(df, drop_columns=None, obj_columns=None, categories=None):
    """Process one chunk of HMDA data by dropping, casting, or categorising columns.

    For each column in ``df``:

    * columns in ``drop_columns`` are removed;
    * columns in ``obj_columns`` are cast to ``object`` dtype;
    * columns present as keys in ``categories`` are converted to
      ``pandas.Categorical`` with the supplied category list;
    * all remaining columns are converted to ``float64`` via :func:`to_float`.

    Parameters
    ----------
    df : pandas.DataFrame
        A single chunk of raw HMDA data.
    drop_columns : list, optional
        Column names to drop entirely.  Defaults to an empty list.
    obj_columns : list, optional
        Column names to cast to ``object`` dtype.  Defaults to an empty list.
    categories : dict, optional
        Mapping of column name to list of valid category values.  Columns
        present in this mapping are converted to ``pandas.Categorical``.
        Defaults to an empty dict.

    Returns
    -------
    pandas.DataFrame
        The processed chunk with columns dropped, recast, or categorised as
        specified.
    """
    if drop_columns is None:
        drop_columns = []
    if obj_columns is None:
        obj_columns = []
    if categories is None:
        categories = {}

    for col in df.columns:
        if col in drop_columns:
            df.drop(col, axis=1, inplace=True)
        elif col in obj_columns:
            df[col] = df[col].astype("object")
        elif col in categories:
            df[col] = pd.Categorical(df[col], categories=categories[col])
        else:
            to_float(df, col)

    return df


def store(
    yr,
    data_dir=default_dir,
    save_dir=default_dir,
    nrows=None,
    usecols=None,
    reimport=False,
    chunksize=500000,
):
    """Read a raw HMDA fixed-width file in chunks and persist it to HDF5.

    The compressed fixed-width file for ``yr`` is read with
    ``pandas.read_fwf`` in chunks of ``chunksize`` rows.  Each chunk is
    processed by :func:`load_chunk` (columns dropped, cast, or categorised)
    and appended to the HDF5 store under the key ``hmda_{yr}``.

    Parameters
    ----------
    yr : int
        HMDA survey year.  Controls both the source filename and the HDF5
        key used for storage.
    data_dir : str, optional
        Directory that contains the raw compressed fixed-width source files.
        Defaults to the configured HMDA data directory.
    save_dir : str, optional
        Directory where the HDF5 store (``hmda.hd5``) will be written.
        Defaults to the configured HMDA data directory.
    nrows : int, optional
        Maximum number of rows to read from the source file.  ``None`` reads
        all rows.
    usecols : list, optional
        Subset of column names to read from the source file.  ``None`` reads
        all columns.
    reimport : bool, optional
        Currently unused; reserved for future use to force re-ingestion even
        when the store key already exists.
    chunksize : int, optional
        Number of rows per chunk passed to ``pandas.read_fwf``.  Defaults to
        500 000.

    Returns
    -------
    None
    """
    warnings.warn(
        "hmda.store() is deprecated; use convert_lar() for parquet output.",
        DeprecationWarning,
        stacklevel=2,
    )
    store_file = save_dir + "hmda.hd5"
    key = "hmda_{}".format(yr)
    store = pd.HDFStore(store_file)

    if yr == 2001:
        filename = "HMS.U2001.LARS.PUBLIC.DATA"
    elif yr == 2004:
        filename = "u2004lar.public.dat"
    elif yr in [2005, 2006]:
        filename = "LARS.ULTIMATE.{}.DAT".format(yr)
    elif yr in [2007, 2008]:
        filename = "lars.ultimate.{}.dat".format(yr)
    elif yr == 2009:
        filename = "2009_Ultimate_PUBLIC_LAR.dat"
    elif yr > 2009:
        filename = "Lars.ultimate.{0}.dat".format(yr)
    else:
        filename = "HMS.U{}.LARS".format(yr)

    if yr < 2004:
        widths = [
            4,
            10,
            1,
            1,
            1,
            1,
            5,
            1,
            4,
            2,
            3,
            7,
            1,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            7,
        ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_race",
            "co_app_race",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "edit_status",
            "seq_num",
        ]
    else:
        widths = [
            4,
            10,
            1,
            1,
            1,
            1,
            5,
            1,
            5,
            2,
            3,
            7,
            1,
            1,
            4,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            1,
            7,
        ]
        names = [
            "asof_date",
            "resp_id",
            "agency_code",
            "loan_type",
            "loan_purp",
            "occupancy",
            "loan_amt",
            "action_taken",
            "prop_msa",
            "state_code",
            "county_code",
            "census_tract",
            "app_sex",
            "co_app_sex",
            "app_income",
            "purchaser_type",
            "denial_reason_1",
            "denial_reason_2",
            "denial_reason_3",
            "edit_status",
            "prop_type",
            "preapprovals",
            "app_ethnicity",
            "co_app_ethnicity",
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
            "rate_spread",
            "hoepa_status",
            "lien_status",
            "seq_num",
        ]

    # filepath = data_dir + filename + '.zip?download=true'
    filepath = data_dir + filename + ".zip"
    reader = pd.read_fwf(
        filepath,
        widths=widths,
        names=names,
        usecols=usecols,
        nrows=nrows,
        compression="zip",
        chunksize=chunksize,
    )

    data_columns = [
        "loan_type",
        "loan_purp",
        "occupancy",
        "action_taken",
        "lien_status",
        "purchaser_type",
    ]

    obj_columns = []
    drop_columns = [
        "resp_id",
        "agency_code",
        "app_sex",
        "co_app_sex",
        "app_ethnicity",
        "co_app_ethnicity",
        "app_race",
        "co_app_race",
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
        "hoepa_status",
        "seq_num",
    ]

    # cat_vars = ['']
    categories = {
        "loan_type": cat(4),
        "prop_type": cat(3),
        "loan_purp": cat(3),
        "occupancy": cat(3),
        "preapprovals": cat(3),
        "action_taken": cat(8),
        "denial_reason_1": cat(9),
        "denial_reason_2": cat(9),
        "denial_reason_3": cat(9),
        "edit_status": list(range(5, 8)),
        "state_code": list(range(1, 100)),
        "purchaser_type": list(range(10)),
        "lien_status": list(range(5)),
    }

    for ii, df in enumerate(reader):
        print("reading chunk {}".format(ii))

        for col in df.columns:
            if col in drop_columns:
                df.drop(col, axis=1, inplace=True)
            elif col in obj_columns:
                df[col] = df[col].astype("object")
            elif col in categories:
                df[col] = pd.Categorical(df[col], categories=categories[col])
            else:
                to_float(df, col)

        if ii == 0:
            store.append(key, df, append=False, data_columns=data_columns)
        else:
            store.append(key, df, data_columns=data_columns)

    store.close()

    return None


def load_hmda(yr, data_dir=default_dir, save_dir=default_dir, query=None, columns=None):
    """Load HMDA data for a given year from the local HDF5 store.

    Opens the HDF5 store at ``save_dir/hmda.hd5``, selects the table stored
    under the key ``hmda_{yr}``, and returns the result as a DataFrame,
    optionally filtered by a query expression and/or a subset of columns.

    Parameters
    ----------
    yr : int
        HMDA survey year to load (e.g. ``2010``).
    data_dir : str, optional
        Directory containing the raw source files (not used during loading,
        but kept for API consistency with :func:`store`).  Defaults to the
        configured HMDA data directory.
    save_dir : str, optional
        Directory containing the HDF5 store (``hmda.hd5``).  Defaults to the
        configured HMDA data directory.
    query : str, optional
        PyTables query string passed to ``HDFStore.select`` for row filtering
        (e.g. ``"state_code == 6"``).  ``None`` returns all rows.
    columns : list, optional
        List of column names to return.  ``None`` returns all columns.

    Returns
    -------
    pandas.DataFrame
        HMDA loan-application records for the requested year, filtered
        according to ``query`` and ``columns``.
    """
    warnings.warn(
        "hmda.load_hmda() is deprecated; use load_lar() for parquet data.",
        DeprecationWarning,
        stacklevel=2,
    )
    store_file = save_dir + "hmda.hd5"
    key = "hmda_{}".format(yr)
    store = pd.HDFStore(store_file)

    store.open()
    df = store.select(key, query, columns=columns)
    store.close()

    return df


def _build_cli_parser():
    """Build the command-line parser for HMDA download and conversion."""
    parser = argparse.ArgumentParser(
        prog="py-tools-hmda",
        description="Download and convert annual HMDA LAR data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download raw annual LAR files."
    )
    download_parser.add_argument(
        "years",
        metavar="YEAR",
        type=int,
        nargs="*",
        help="Years to download; omit to download all years for the source.",
    )
    download_parser.add_argument("--source", choices=LAR_BULK_SOURCES, default="auto")
    download_parser.add_argument("--data-dir", default=default_dir)
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument(
        "--progress", action="store_true", help="Show download progress."
    )

    convert_parser = subparsers.add_parser(
        "convert", help="Convert downloaded LAR files to parquet."
    )
    convert_parser.add_argument(
        "years",
        metavar="YEAR",
        type=int,
        nargs="*",
        help="Years to convert; omit to convert all years for the source.",
    )
    convert_parser.add_argument("--source", choices=LAR_BULK_SOURCES, default="auto")
    convert_parser.add_argument("--data-dir", default=default_dir)
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.add_argument("--chunksize", type=int, default=100000)
    convert_parser.add_argument("--compression", default="zstd")
    return parser


def main(argv=None):
    """Run the HMDA command-line interface."""
    args = _build_cli_parser().parse_args(argv)
    years = args.years or None
    if args.command == "download":
        paths = download_lar(
            years=years,
            source=args.source,
            data_dir=args.data_dir,
            overwrite=args.overwrite,
            progress=args.progress,
        )
    else:
        paths = convert_lar(
            years=years,
            source=args.source,
            data_dir=args.data_dir,
            overwrite=args.overwrite,
            chunksize=args.chunksize,
            compression=args.compression,
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
