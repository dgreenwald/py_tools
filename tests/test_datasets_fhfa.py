"""Tests for datasets.fhfa loader."""

import os
import numpy as np
import pandas as pd
import pytest

from datasets import fhfa


def write_zip5_excel(path, rows=None):
    """Write a minimal zip5-format Excel file for testing."""
    if rows is None:
        rows = [
            (10001, 2000, np.nan, 100.0, 52.0, 51.0),
            (10001, 2001, 5.0, 105.0, 54.6, 53.6),
            (10002, 2000, np.nan, 100.0, 50.0, 49.0),
        ]
    header_rows = [[""] * 6] * 5  # five blank rows before the data header
    col_names = [
        "Five-Digit ZIP Code",
        "Year",
        "Annual Change (%)",
        "HPI",
        "HPI with 1990 base",
        "HPI with 2000 base",
    ]
    data = pd.DataFrame(rows, columns=col_names)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(header_rows).to_excel(writer, index=False, header=False)
        data.to_excel(
            writer,
            index=False,
            header=True,
            startrow=5,
        )


@pytest.fixture()
def zip5_data_dir(tmp_path):
    write_zip5_excel(tmp_path / "hpi_at_zip5.xlsx")
    return str(tmp_path) + "/"


def test_zip5_columns_and_index(zip5_data_dir):
    df = fhfa.load("zip5", data_dir=zip5_data_dir)
    assert df.index.names == ["zip5", "date"]
    assert set(df.columns) == {"annual_change_pct", "hpi", "hpi_1990_base", "hpi_2000_base"}


def test_zip5_shape(zip5_data_dir):
    df = fhfa.load("zip5", data_dir=zip5_data_dir)
    assert df.shape == (3, 4)


def test_zip5_values(zip5_data_dir):
    df = fhfa.load("zip5", data_dir=zip5_data_dir)
    assert df.loc[(10001, pd.Timestamp("2001-01-01")), "hpi"] == pytest.approx(105.0)
    assert np.isnan(df.loc[(10001, pd.Timestamp("2000-01-01")), "annual_change_pct"])


def test_zip5_parquet_cache(zip5_data_dir):
    fhfa.load("zip5", data_dir=zip5_data_dir)
    parquet_path = zip5_data_dir + "fhfazip5_at.parquet"
    assert os.path.exists(parquet_path)

    # Second call reads from cache and returns identical data
    df_cached = fhfa.load("zip5", data_dir=zip5_data_dir)
    df_fresh = fhfa.load("zip5", reimport=True, data_dir=zip5_data_dir)
    pd.testing.assert_frame_equal(df_cached, df_fresh)


def test_zip5_reimport_overwrites_cache(zip5_data_dir):
    fhfa.load("zip5", data_dir=zip5_data_dir)
    parquet_path = zip5_data_dir + "fhfazip5_at.parquet"
    mtime_first = os.path.getmtime(parquet_path)

    fhfa.load("zip5", reimport=True, data_dir=zip5_data_dir)
    assert os.path.getmtime(parquet_path) >= mtime_first
