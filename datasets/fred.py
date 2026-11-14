import datetime
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

from . import config

default_dir = config.base_dir() + "fred/"
DATASET_NAME = "fred"
DESCRIPTION = "FRED macroeconomic time series dataset loader."


def _normalize_vintage(vintage):
    if isinstance(vintage, datetime.datetime):
        vintage = vintage.date()
    if isinstance(vintage, datetime.date):
        return vintage
    if isinstance(vintage, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", vintage):
        try:
            return datetime.date.fromisoformat(vintage)
        except ValueError:
            pass
    raise ValueError("vintage must be a date or a YYYY-MM-DD string")


def _latest_cache_path(series_dir):
    dated_paths = []
    for path in series_dir.glob("????-??-??.parquet"):
        try:
            vintage = _normalize_vintage(path.stem)
        except ValueError:
            continue
        dated_paths.append((vintage, path))
    return max(dated_paths)[1] if dated_paths else None


def _download_date():
    return datetime.date.today()


def _write_cache(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(fd)
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(
    codes=None,
    code_names=None,
    start=datetime.datetime(1900, 1, 1),
    end=datetime.datetime.today(),
    reimport=False,
    data_dir=default_dir,
    vintage=None,
):
    """Load FRED series, caching each download by local calendar date.

    ``vintage`` selects an exact cached download date and never downloads.
    Without a vintage, each series uses its latest Parquet cache, then its
    legacy pickle cache, or is downloaded if neither exists. ``reimport=True``
    downloads every requested series and replaces any cache from today.

    Parameters
    ----------
    codes : list of str or str or None, optional
        FRED series codes. If omitted, use the keys of ``code_names``.
    code_names : dict, optional
        Mapping from series codes to output column names.
    start, end : datetime.datetime, optional
        Date range passed to FRED when a download is needed.
    reimport : bool, optional
        Download all requested series, even if cached.
    data_dir : str or pathlib.Path, optional
        Directory containing the FRED cache.
    vintage : datetime.date or str or None, optional
        Exact local download date, formatted ``YYYY-MM-DD`` when a string.
        Missing dated caches raise ``FileNotFoundError``. Cannot be combined
        with ``reimport=True``.

    Returns
    -------
    pandas.DataFrame
        Requested series joined on their date index and renamed as requested.
    """
    if vintage is not None:
        vintage = _normalize_vintage(vintage)
        if reimport:
            raise ValueError("vintage cannot be combined with reimport=True")

    if code_names is None:
        code_names = {}
    if codes is None:
        codes = list(code_names)
    if isinstance(codes, str):
        codes = [codes]
    codes = list(dict.fromkeys(codes))
    if not codes:
        return pd.DataFrame()

    cache_root = Path(data_dir)
    cached_paths = {}
    download_series = []
    missing_series = []

    for series in codes:
        series_dir = cache_root / series
        if vintage is not None:
            path = series_dir / f"{vintage.isoformat()}.parquet"
            if path.is_file():
                cached_paths[series] = path
            else:
                missing_series.append(series)
        elif reimport:
            download_series.append(series)
        else:
            path = _latest_cache_path(series_dir)
            if path is not None:
                cached_paths[series] = path
            elif (cache_root / f"{series}.pkl").is_file():
                cached_paths[series] = cache_root / f"{series}.pkl"
            else:
                download_series.append(series)

    if missing_series:
        raise FileNotFoundError(
            f"No FRED cache for vintage {vintage.isoformat()} and series: "
            + ", ".join(missing_series)
        )

    downloaded = None
    if download_series:
        try:
            from pandas_datareader import data as web
        except ImportError as e:
            raise ImportError(
                "pandas-datareader is required for datasets.fred. "
                "Install it with: pip install py_tools[datasets]"
            ) from e

        downloaded = web.DataReader(download_series, "fred", start, end)
        download_date = _download_date().isoformat()
        for series in download_series:
            _write_cache(
                downloaded[[series]],
                cache_root / series / f"{download_date}.parquet",
            )

    frames = []
    for series in codes:
        if series in cached_paths:
            path = cached_paths[series]
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_pickle(path).rename(series).to_frame()
        else:
            frame = downloaded[[series]]
        frames.append(frame)

    return pd.concat(frames, axis=1).rename(columns=code_names)
