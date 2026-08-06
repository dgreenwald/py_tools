import os
import sys
import tempfile
import urllib.request
import zipfile
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


def _lar_file_path(year, data_dir=default_dir, source="auto"):
    """Return the canonical local path for an annual LAR source file."""
    resolved_source = _resolve_lar_source(year, source)
    year_config = _LAR_SOURCE_CONFIG[resolved_source][year]
    return (
        Path(data_dir) / "raw" / resolved_source / str(year) / year_config["filename"]
    )


def _validate_lar_zip(path, require_csv=False):
    """Return whether a ZIP has a suitable nonempty data member."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(
                not member.is_dir()
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

    complete_lines = text.splitlines()[:-1]
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
    source : {"auto", "ffiec_three_year", "ffiec_snapshot", "cfpb", "nara"}, optional
        Data release to download. ``"auto"`` prefers FFIEC three-year files
        for 2017--2022, FFIEC snapshots for 2023 onward, CFPB files for
        2007--2016, and National Archives files for earlier years. With an
        explicit source and no ``years``, download every year from that source.
    data_dir : str or path-like, optional
        Root directory for HMDA data files.
    overwrite : bool, optional
        Replace valid archives that are already present.
    progress : bool, optional
        Display download progress on standard error. Defaults to ``False``.

    Returns
    -------
    list of pathlib.Path
        Local source paths in requested order. Duplicate years appear once.

    Raises
    ------
    ValueError
        If any requested year is unsupported.
    RuntimeError
        If a download fails or has the wrong file format.
    """
    requested = _normalize_lar_years(years, source=source)
    paths = []

    for year in requested:
        resolved_source = _resolve_lar_source(year, source)
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


def load(data_dir=None, **kwargs):
    """Load HMDA data from the local HDF store.

    Backward-compatible wrapper around ``load_hmda``; expects ``yr`` in
    ``kwargs``.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing the HDF store.  When provided it is
        forwarded as the ``data_dir`` keyword argument to ``load_hmda``.
    **kwargs
        Additional keyword arguments passed directly to ``load_hmda``.
        Must include ``yr`` (int) – the HMDA survey year to load.

    Returns
    -------
    pandas.DataFrame
        HMDA loan-application records for the requested year.
    """
    if data_dir is not None:
        kwargs.setdefault("data_dir", data_dir)
    if "save_dir" not in kwargs or kwargs.get("save_dir") is None:
        kwargs["save_dir"] = kwargs.get("data_dir", default_dir)
    return load_hmda(**kwargs)


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
    store_file = save_dir + "hmda.hd5"
    key = "hmda_{}".format(yr)
    store = pd.HDFStore(store_file)

    store.open()
    df = store.select(key, query, columns=columns)
    store.close()

    return df
