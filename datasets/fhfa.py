import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from py_tools import time_series as ts

from . import config

default_dir = config.base_dir() + "fhfa/"
DATASET_NAME = "fhfa"
DESCRIPTION = "FHFA house price index dataset loader."
FHFA_BASE_URL = "https://www.fhfa.gov"
FHFA_RAW_DATASETS = ("county", "zip3", "state", "zip5")
_RAW_DATA_CONFIG = {
    "county": {
        "url": f"{FHFA_BASE_URL}/hpi/download/annual/hpi_at_county.xlsx",
        "filename": "HPI_AT_BDL_county.xlsx",
        "format": "xlsx",
    },
    "zip3": {
        "url": (
            f"{FHFA_BASE_URL}/hpi/download/quarterly_datasets/"
            "hpi_at_3zip.xlsx"
        ),
        "filename": "HPI_AT_3zip.xlsx",
        "format": "xlsx",
    },
    "state": {
        "url": (
            f"{FHFA_BASE_URL}/hpi/download/quarterly_datasets/"
            "hpi_po_state.txt"
        ),
        "filename": "HPI_PO_state.txt",
        "format": "txt",
    },
    "zip5": {
        "url": f"{FHFA_BASE_URL}/hpi/download/annual/hpi_at_zip5.xlsx",
        "filename": "hpi_at_zip5.xlsx",
        "format": "xlsx",
    },
}


def _normalize_raw_datasets(datasets):
    """Return supported raw datasets in caller-specified order."""
    if datasets is None:
        requested = list(FHFA_RAW_DATASETS)
    elif isinstance(datasets, str):
        requested = [datasets]
    else:
        try:
            requested = list(datasets)
        except TypeError as exc:
            raise ValueError(
                "datasets must be a string or an iterable of strings"
            ) from exc

    unsupported = [
        dataset for dataset in requested if dataset not in _RAW_DATA_CONFIG
    ]
    if unsupported:
        supported = ", ".join(FHFA_RAW_DATASETS)
        raise ValueError(
            f"Unsupported FHFA raw dataset(s): {unsupported}. "
            f"Supported datasets are: {supported}."
        )

    return list(dict.fromkeys(requested))


def _validate_raw_file(path, file_format):
    """Return whether a downloaded FHFA file matches its expected format."""
    if file_format == "xlsx":
        return zipfile.is_zipfile(path)
    if file_format == "txt":
        try:
            with Path(path).open(encoding="utf-8-sig") as source:
                columns = set(source.readline().strip().split("\t"))
        except (OSError, UnicodeError):
            return False
        expected = {"state", "yr", "qtr", "index_nsa", "index_sa"}
        return expected.issubset(columns)
    raise ValueError(f"Unsupported FHFA raw file format: {file_format}")


def download_raw(datasets=None, data_dir=default_dir, overwrite=False):
    """Download current FHFA source files for selected datasets.

    Parameters
    ----------
    datasets : str or iterable of str, optional
        One or more of ``'county'``, ``'zip3'``, ``'state'``, and ``'zip5'``.
        By default, download all four datasets.
    data_dir : str or path-like, optional
        Directory where source files are stored.
    overwrite : bool, optional
        Replace valid source files that are already present.

    Returns
    -------
    list of pathlib.Path
        Local source paths in requested order. Duplicate datasets appear once.

    Raises
    ------
    ValueError
        If an unsupported dataset is requested.
    RuntimeError
        If a download fails or its content has the wrong format.
    """
    requested = _normalize_raw_datasets(datasets)
    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for dataset in requested:
        raw_config = _RAW_DATA_CONFIG[dataset]
        destination = destination_dir / raw_config["filename"]
        paths.append(destination)
        if (
            destination.exists()
            and _validate_raw_file(destination, raw_config["format"])
            and not overwrite
        ):
            continue

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{dataset}.",
                suffix=".part",
                dir=destination_dir,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(raw_config["url"]) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

            if not _validate_raw_file(temporary, raw_config["format"]):
                raise RuntimeError(
                    f"downloaded content is not valid {raw_config['format']}"
                )
            os.replace(temporary, destination)
        except Exception as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download FHFA {dataset} data from "
                f"{raw_config['url']}: {exc}"
            ) from exc

    return paths


def _read_excel_table(path, first_column):
    """Read an FHFA workbook by locating its table-header row."""
    preview = pd.read_excel(path, header=None, nrows=20)
    first_values = preview.iloc[:, 0].astype(str).str.strip()
    matches = preview.index[first_values == first_column].tolist()
    if len(matches) != 1:
        raise ValueError(
            f"Expected one FHFA header row beginning with {first_column!r} "
            f"in {path}, found {len(matches)}"
        )
    return pd.read_excel(path, skiprows=matches[0])


def load(
    dataset,
    all_transactions=True,
    reimport=False,
    data_dir=default_dir,
    annual=None,
):
    """Load FHFA house price index data.

    Parameters
    ----------
    dataset : str
        Geographic level of the index. One of ``'metro'``/``'msa'``,
        ``'state'``, ``'county'``, ``'zip3'``, or ``'zip5'``.
    all_transactions : bool, optional
        If ``True``, load the all-transactions index; if ``False``, load
        the purchase-only index.  Not all dataset/index combinations are
        supported (e.g. ``'metro'`` requires ``all_transactions=True``).
    reimport : bool, optional
        If ``True``, re-read the raw source file and overwrite any cached
        parquet.  If ``False``, use the cached parquet when available.
    data_dir : str, optional
        Path to the directory containing the FHFA source files.
    annual : bool or None, optional
        Frequency selector.  If ``None``, use the historical default for the
        requested geographic level: annual for ``'county'`` and ``'zip5'``,
        quarterly for ``'metro'``/``'msa'``, ``'state'``, and ``'zip3'``.
        Set ``annual=True`` to load the annual 3-digit ZIP file.

    Returns
    -------
    pandas.DataFrame
        House price index data with a multi-level index (geographic unit,
        date) and ``hpi`` (and possibly additional) columns.

    Raises
    ------
    ValueError
        When an unsupported combination of ``dataset`` and
        ``all_transactions`` or ``annual`` is requested.
    """
    annual_defaults = {
        "metro": False,
        "msa": False,
        "state": False,
        "county": True,
        "zip3": False,
        "zip5": True,
    }
    if dataset not in annual_defaults:
        raise ValueError(f"Unsupported FHFA dataset: {dataset}")
    if annual is None:
        annual = annual_defaults[dataset]

    if annual and dataset not in ["county", "zip3", "zip5"]:
        raise ValueError(f"Annual FHFA data are not supported for {dataset}")
    if not annual and dataset in ["county", "zip5"]:
        raise ValueError(f"Quarterly FHFA data are not supported for {dataset}")

    suffix = dataset
    if all_transactions:
        suffix += "_at"
    else:
        suffix += "_purch"
    if annual and dataset == "zip3":
        suffix += "_annual"

    basepath = data_dir + "fhfa" + suffix
    parquet_file = basepath + ".parquet"

    if reimport or not os.path.exists(parquet_file):
        if dataset in ["metro", "msa"]:
            if not all_transactions:
                raise ValueError("Metro FHFA data require all_transactions=True")

            # df = pd.read_csv(data_dir + 'HPI_AT_metro.csv')
            df = pd.read_csv(
                data_dir + "HPI_AT_metro.csv",
                header=None,
                names=["MSA", "code", "year", "qtr", "hpi", "unknown"],
            )

            df["date"] = ts.date_from_qtr(df["year"], df["qtr"])

            # df['date'] = ts.date_from_qtr(df['year'], df['qtr'])

            df["hpi"] = pd.to_numeric(df["hpi"], errors="coerce")

            df = df.set_index(["date", "MSA"])
            df = df.drop(columns=["year", "qtr", "unknown"])
            df = df.apply(pd.to_numeric, errors="coerce")
            # df['date'] = df['yr'].astype('str') + '-' + (3*df['qtr'] - 2).astype('str') + '-01'
            # df['date'] = pd.to_datetime(df['date'])
            # df = df.drop(columns=['yr', 'qtr', 'Warning'])

        elif dataset == "state":
            if all_transactions:
                df = pd.read_csv(
                    data_dir + "HPI_AT_state.txt",
                    names=["state", "year", "qtr", "hpi"],
                    sep="\t",
                )
            else:
                df = pd.read_csv(data_dir + "HPI_PO_state.txt", sep="\t")
                df = df.drop(columns=["Warning"])

            df["date"] = ts.date_from_qtr(df["yr"], df["qtr"])
            df = df.set_index(["state", "date"])
            df = df.apply(pd.to_numeric, errors="coerce")

        elif dataset == "county":
            if all_transactions:
                df = _read_excel_table(
                    data_dir + "HPI_AT_BDL_county.xlsx", "State"
                )
                df = df.rename({var: var.lower() for var in df.columns}, axis=1)

                for var in df.columns:
                    if var not in ["state", "county"]:
                        df[var] = pd.to_numeric(df[var], errors="coerce")

                df = df.rename(
                    {
                        "county": "county_name",
                        "fips code": "fips",
                        "hpi": "hpi",
                        "hpi with 1990 base": "hpi_1990_base",
                        "hpi with 2000 base": "hpi_2000_base",
                        "annual change (%)": "annual_change_pct",
                    },
                    axis=1,
                )

                df["date"] = ts.date_from_year(df["year"])
                df = df.set_index(["fips", "date"])
            else:
                raise ValueError("County FHFA data require all_transactions=True")

        elif dataset == "zip3":
            if not all_transactions:
                raise ValueError("ZIP3 FHFA data require all_transactions=True")

            if annual:
                df = _read_excel_table(
                    data_dir + "hpi_at_zip3_annual.xlsx",
                    "Three-Digit ZIP Code",
                )
                df = df.rename({var: var.lower() for var in df.columns}, axis=1)

                for var in df.columns:
                    if var != "three-digit zip code":
                        df[var] = pd.to_numeric(df[var], errors="coerce")

                df = df.rename(
                    {
                        "three-digit zip code": "zip3",
                        "hpi": "hpi",
                        "hpi with 1990 base": "hpi_1990_base",
                        "hpi with 2000 base": "hpi_2000_base",
                        "annual change (%)": "annual_change_pct",
                    },
                    axis=1,
                )

                df["date"] = ts.date_from_year(df["year"])
                df = df.set_index(["zip3", "date"])
            else:
                filename = "HPI_AT_3zip"
                excel_file = data_dir + filename + ".xlsx"
                df = _read_excel_table(
                    excel_file, "Three-Digit ZIP Code"
                )
                df["date"] = ts.date_from_qtr(df["Year"], df["Quarter"])
                df = df.drop(columns=["Index Type"]).rename(columns={"Index (NSA)": "hpi"})

                df = df.rename({"Three-Digit ZIP Code": "zip3"}, axis=1)
                df = df.rename({var: var.lower() for var in df.columns}, axis=1)

                df = df.set_index(["zip3", "date"])

        elif dataset == "zip5":
            if not all_transactions:
                raise ValueError("ZIP5 FHFA data require all_transactions=True")

            df = _read_excel_table(
                data_dir + "hpi_at_zip5.xlsx", "Five-Digit ZIP Code"
            )
            df = df.rename({var: var.lower() for var in df.columns}, axis=1)

            for var in df.columns:
                if var not in ["five-digit zip code", "warning"]:
                    df[var] = pd.to_numeric(df[var], errors="coerce")

            df = df.rename(
                {
                    "five-digit zip code": "zip5",
                    "hpi": "hpi",
                    "hpi with 1990 base": "hpi_1990_base",
                    "hpi with 2000 base": "hpi_2000_base",
                    "annual change (%)": "annual_change_pct",
                },
                axis=1,
            )

            df["date"] = ts.date_from_year(df["year"])
            df = df.set_index(["zip5", "date"])
            df = df.drop(columns=[c for c in ["warning"] if c in df.columns])

        df.to_parquet(parquet_file)

    else:
        df = pd.read_parquet(parquet_file)

    if reimport:
        stata_file = basepath + ".dta"
        df.to_stata(stata_file)

    return df
