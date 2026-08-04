"""Vintage-aware loaders and downloads for Zillow Research data."""

import os
import re
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from . import config, misc

default_dir = config.base_dir() + "zillow/"
DATASET_NAME = "zillow"
DESCRIPTION = "Zillow housing dataset loader."
DEFAULT_VINTAGE = "202608"

_DATE_COLUMN = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")
_GEOGRAPHIES = {
    "county": "County",
    "state": "State",
    "zip": "Zip",
    "metro": "Metro",
}
_DATASET_ALIASES = {
    "zhvi": "zhvi",
    "Zhvi_AllHomes": "zhvi",
}
_RAW_DATA_CONFIG = {
    ("zhvi", "county"): {
        "url": (
            "https://files.zillowstatic.com/research/public_csvs/zhvi/"
            "County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        ),
        "filename": ("County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"),
        "required_columns": {
            "RegionID",
            "RegionName",
            "StateCodeFIPS",
            "MunicipalCodeFIPS",
        },
    }
}


def _normalize_vintage(vintage, *, use_current=False):
    """Validate a vintage, optionally using the current month by default."""
    if vintage is None:
        vintage = date.today().strftime("%Y%m") if use_current else DEFAULT_VINTAGE
    vintage = str(vintage)
    if not re.fullmatch(r"\d{6}", vintage):
        raise ValueError("vintage must be a six-digit YYYYMM string")
    return vintage


def _normalize_geo(geo):
    """Return the canonical lower-case geography key."""
    key = str(geo).lower()
    if key not in _GEOGRAPHIES:
        supported = ", ".join(_GEOGRAPHIES)
        raise ValueError(
            f"Unsupported Zillow geography {geo!r}. Supported: {supported}."
        )
    return key


def _normalize_dataset(dataset):
    """Return the canonical dataset key, accepting the legacy ZHVI name."""
    try:
        return _DATASET_ALIASES[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Zillow dataset {dataset!r}. Supported: zhvi."
        ) from exc


def _raw_path(data_dir, vintage, dataset, geo):
    """Resolve the existing raw path, preferring the current filename."""
    root = Path(data_dir) / vintage
    geo_dir = root / _GEOGRAPHIES[geo]
    config_entry = _RAW_DATA_CONFIG.get((dataset, geo))
    candidates = []
    if config_entry is not None:
        candidates.append(geo_dir / config_entry["filename"])
    candidates.append(geo_dir / f"{_GEOGRAPHIES[geo]}_Zhvi_AllHomes.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _cache_path(data_dir, vintage, dataset, geo):
    """Return the vintage-specific parquet cache path."""
    return Path(data_dir) / vintage / "parquet" / f"{geo}_{dataset}.parquet"


def _date_columns(columns):
    """Return source columns that are strict Zillow monthly date labels."""
    return [column for column in columns if _DATE_COLUMN.fullmatch(str(column))]


def _fips_component(values, width):
    """Normalize a numeric or textual FIPS component without losing zeros."""
    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(width)
        .astype(object)
    )


def _normalize_frame(df_wide, geo, dataset, data_dir, vintage):
    """Convert a Zillow wide source table to the canonical long form."""
    value_vars = _date_columns(df_wide.columns)
    if not value_vars:
        raise ValueError("Zillow source has no YYYY-MM or YYYY-MM-DD columns")
    id_vars = [column for column in df_wide.columns if column not in value_vars]
    frame = df_wide.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="date",
        value_name=dataset,
    )
    frame["date"] = pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0)

    rename = {
        "RegionID": "region_id",
        "RegionName": "region_name",
        "RegionType": "region_type",
        "StateName": "state_name",
        "State": "state",
        "Metro": "metro",
        "SizeRank": "size_rank",
    }
    frame = frame.rename(columns=rename)

    if geo == "county":
        if {"StateCodeFIPS", "MunicipalCodeFIPS"}.issubset(frame.columns):
            frame["fips"] = _fips_component(
                frame["StateCodeFIPS"], 2
            ) + _fips_component(frame["MunicipalCodeFIPS"], 3)
        else:
            crosswalk = load_crosswalk(data_dir=data_dir, vintage=vintage)
            crosswalk = crosswalk[["CountyRegionID_Zillow", "FIPS"]].copy()
            crosswalk["FIPS"] = _fips_component(crosswalk["FIPS"], 5)
            frame = frame.merge(
                crosswalk,
                left_on="region_id",
                right_on="CountyRegionID_Zillow",
                validate="many_to_one",
            ).rename(columns={"FIPS": "fips"})
        index = ["fips", "date"]
    elif geo == "state":
        if "state" not in frame.columns:
            state_codes = misc.load("state_codes").copy()
            state_codes["state_name"] = state_codes["state_name"].str.title()
            frame = frame.merge(
                state_codes[["state_name", "state_abbr"]],
                left_on="region_name",
                right_on="state_name",
                validate="many_to_one",
            ).rename(columns={"state_abbr": "state"})
        index = ["state", "date"]
    elif geo == "zip":
        frame["zip"] = frame["region_name"].astype("string").str.zfill(5).astype(object)
        index = ["zip", "date"]
    else:
        if "metro" not in frame.columns:
            frame["metro"] = frame["region_name"]
        index = ["metro", "date"]

    drop_columns = [
        "StateCodeFIPS",
        "MunicipalCodeFIPS",
        "CountyRegionID_Zillow",
    ]
    frame = frame.drop(columns=drop_columns, errors="ignore")
    text_columns = [
        "region_name",
        "region_type",
        "state_name",
        "state",
        "metro",
    ]
    for column in text_columns:
        if column in frame.columns and column not in index:
            frame[column] = frame[column].astype("string")
    return frame.set_index(index).sort_index()


def load(
    geo="county",
    data_dir=default_dir,
    dataset="zhvi",
    vintage=DEFAULT_VINTAGE,
    reimport=False,
):
    """Load a Zillow dataset as a normalized, vintage-specific parquet.

    The county result is indexed by five-character ``fips`` and month-end
    ``date``. Other legacy geographies use ``state``, ``zip``, or ``metro``
    as the first index level. Loading never performs an implicit download.
    """
    geo = _normalize_geo(geo)
    dataset = _normalize_dataset(dataset)
    vintage = _normalize_vintage(vintage)
    parquet_file = _cache_path(data_dir, vintage, dataset, geo)

    if parquet_file.exists() and not reimport:
        return pd.read_parquet(parquet_file)

    raw_file = _raw_path(data_dir, vintage, dataset, geo)
    if not raw_file.exists():
        download_hint = (
            f"zillow.download_raw(dataset={dataset!r}, geo={geo!r}, "
            f"vintage={vintage!r}, data_dir=...)"
        )
        raise FileNotFoundError(
            f"Missing Zillow raw data at {raw_file}. Fetch it with {download_hint}."
        )

    df_wide = load_csv(data_dir=data_dir, dataset=dataset, geo=geo, vintage=vintage)
    frame = _normalize_frame(df_wide, geo, dataset, data_dir, vintage)
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_file)
    return frame


def _validate_raw_file(path, config_entry):
    """Return whether a downloaded CSV has the expected Zillow structure."""
    try:
        preview = pd.read_csv(path, nrows=5)
    except (OSError, UnicodeError, pd.errors.ParserError):
        return False
    if not config_entry["required_columns"].issubset(preview.columns):
        return False
    date_columns = _date_columns(preview.columns)
    return bool(date_columns) and preview[date_columns].notna().any().any()


def download_raw(
    dataset="zhvi",
    geo="county",
    vintage=None,
    data_dir=default_dir,
    overwrite=False,
):
    """Download a current Zillow source CSV into a vintage directory."""
    dataset = _normalize_dataset(dataset)
    geo = _normalize_geo(geo)
    vintage = _normalize_vintage(vintage, use_current=True)
    try:
        config_entry = _RAW_DATA_CONFIG[(dataset, geo)]
    except KeyError as exc:
        raise ValueError(
            f"No Zillow download is configured for {dataset}/{geo}."
        ) from exc

    destination_dir = Path(data_dir) / vintage / _GEOGRAPHIES[geo]
    destination = destination_dir / config_entry["filename"]
    if (
        destination.exists()
        and _validate_raw_file(destination, config_entry)
        and not overwrite
    ):
        return destination

    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dataset}_{geo}.",
            suffix=".part",
            dir=destination_dir,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(config_entry["url"]) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)

        if not _validate_raw_file(temporary, config_entry):
            raise RuntimeError("downloaded content is not a valid Zillow CSV")
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download Zillow {dataset}/{geo} data from "
            f"{config_entry['url']}: {exc}"
        ) from exc
    return destination


def load_county(
    data_dir=default_dir,
    dataset="zhvi",
    vintage=DEFAULT_VINTAGE,
    reimport=False,
):
    """Load county Zillow data through :func:`load`."""
    return load(
        "county",
        data_dir=data_dir,
        dataset=dataset,
        vintage=vintage,
        reimport=reimport,
    )


def load_state(
    data_dir=default_dir,
    dataset="zhvi",
    vintage=DEFAULT_VINTAGE,
    reimport=False,
):
    """Load state Zillow data through :func:`load`."""
    return load(
        "state",
        data_dir=data_dir,
        dataset=dataset,
        vintage=vintage,
        reimport=reimport,
    )


def load_crosswalk(data_dir=default_dir, vintage="201908"):
    """Load the vintage-specific legacy Zillow county-FIPS crosswalk."""
    vintage = _normalize_vintage(vintage)
    path = Path(data_dir) / vintage / "CountyCrossWalk_Zillow2.csv"
    return pd.read_csv(path, dtype={"FIPS": "string"})


def load_csv(
    data_dir=default_dir,
    dataset="zhvi",
    geo="county",
    vintage=DEFAULT_VINTAGE,
):
    """Read a raw Zillow CSV selected by dataset, geography, and vintage."""
    geo = _normalize_geo(geo)
    dataset = _normalize_dataset(dataset)
    vintage = _normalize_vintage(vintage)
    path = _raw_path(data_dir, vintage, dataset, geo)
    dtypes = {
        "StateCodeFIPS": "string",
        "MunicipalCodeFIPS": "string",
    }
    if geo == "zip":
        dtypes["RegionName"] = "string"
    return pd.read_csv(
        path,
        encoding="latin1",
        dtype=dtypes,
    )
