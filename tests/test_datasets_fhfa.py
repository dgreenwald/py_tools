"""Tests for datasets.fhfa loader."""

import os
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from py_tools.datasets import fhfa


def _write_zip5_xlsx(path, rows):
    """Write a minimal zip5 xlsx fixture matching FHFA's file layout."""
    wb = Workbook()
    ws = wb.active
    ws.append(["HPI for Five-Digit ZIP Codes (All-Transactions Index)"])
    ws.append([None])
    ws.append(["Disclaimer text placeholder."])
    ws.append(["Last updated: January 1, 2025.", None, None, None, None, None])
    ws.append(["Not Seasonally Adjusted (NSA) ", None, None, None, None, None])
    ws.append(["Five-Digit ZIP Code", "Year", "Annual Change (%)", "HPI",
               "HPI with 1990 base", "HPI with 2000 base"])
    for row in rows:
        ws.append(row)
    wb.save(path)


@pytest.fixture()
def zip5_dir(tmp_path):
    rows = [
        [10001, 2010, None,  100.0, 80.0, 75.0],
        [10001, 2011,  5.0,  105.0, 84.0, 78.75],
        [10001, 2012,  3.5,  108.68, 86.94, 81.51],
        [10002, 2010, None,  100.0, 90.0, 85.0],
        [10002, 2011,  3.0,  103.0, 92.7, 87.55],
    ]
    _write_zip5_xlsx(tmp_path / "hpi_at_zip5.xlsx", rows)
    return str(tmp_path) + "/"


def test_zip5_index(zip5_dir):
    df = fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    assert df.index.names == ["zip5", "date"]


def test_zip5_columns(zip5_dir):
    df = fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    assert set(df.columns) >= {"hpi", "hpi_1990_base", "hpi_2000_base", "annual_change_pct"}


def test_zip5_dtypes(zip5_dir):
    df = fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    assert df["hpi"].dtype == np.float64
    assert df["annual_change_pct"].dtype == np.float64
    assert isinstance(df.index.get_level_values("date"), pd.DatetimeIndex)


def test_zip5_values(zip5_dir):
    df = fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    assert df.loc[(10001, pd.Timestamp("2011-01-01")), "hpi"] == pytest.approx(105.0)
    assert df.loc[(10002, pd.Timestamp("2010-01-01")), "hpi_1990_base"] == pytest.approx(90.0)
    assert np.isnan(df.loc[(10001, pd.Timestamp("2010-01-01")), "annual_change_pct"])


def test_zip5_shape(zip5_dir):
    df = fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    assert df.shape[0] == 5
    assert df.index.is_unique


def test_zip5_parquet_cache(zip5_dir):
    fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    parquet_path = zip5_dir + "fhfazip5_at.parquet"
    assert os.path.exists(parquet_path)

    df_cached = fhfa.load("zip5", data_dir=zip5_dir)
    df_fresh = fhfa.load("zip5", data_dir=zip5_dir, reimport=True)
    pd.testing.assert_frame_equal(df_cached, df_fresh)
