"""Tests for py_tools.in_out.core"""

import io
import json
import os
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from py_tools.in_out.core import (
    load_eigen,
    reshape_eigen,
    save_eigen,
    save_pickle,
    load_pickle,
    make_dir,
    write_text,
    write_numeric,
    read_numeric,
    write_json,
    read_json,
    read_zipped,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_eigen_file(path, array):
    """Write a 2-D ndarray in the Eigen binary format used by save_eigen."""
    n_rows, n_cols = array.shape
    with open(path, 'wb') as fid:
        fid.write(struct.pack('i', n_rows))
        fid.write(struct.pack('i', n_cols))
        array.transpose().astype('float64').tofile(fid)


# ---------------------------------------------------------------------------
# save_eigen / load_eigen  (round-trip)
# ---------------------------------------------------------------------------

class TestSaveLoadEigen:
    def test_round_trip_square(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = str(tmp_path / "mat.bin")
        save_eigen(arr, path)
        result = load_eigen(path)
        assert np.allclose(result, arr)

    def test_round_trip_non_square(self, tmp_path):
        arr = np.arange(6.0).reshape(2, 3)
        path = str(tmp_path / "mat.bin")
        save_eigen(arr, path)
        result = load_eigen(path)
        assert result.shape == (2, 3)
        assert np.allclose(result, arr)

    def test_dtype_preserved(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = str(tmp_path / "mat.bin")
        save_eigen(arr, path, dtype='float32')
        result = load_eigen(path, dtype='float32')
        assert np.allclose(result, arr, atol=1e-6)


# ---------------------------------------------------------------------------
# reshape_eigen
# ---------------------------------------------------------------------------

class TestReshapeEigen:
    def test_round_trip_3d(self, tmp_path):
        arr = np.arange(24.0).reshape(4, 6)  # saved as 4×6
        path = str(tmp_path / "mat.bin")
        _write_eigen_file(path, arr)
        # reshape the 24 values into (2, 3, 4)
        result = reshape_eigen(path, (2, 3, 4))
        assert result.shape == (2, 3, 4)
        assert result.size == 24

    def test_wrong_shape_raises_value_error(self, tmp_path):
        arr = np.arange(6.0).reshape(2, 3)
        path = str(tmp_path / "mat.bin")
        _write_eigen_file(path, arr)
        with pytest.raises(ValueError, match="reshape_eigen"):
            reshape_eigen(path, (3, 3))  # 9 ≠ 6


# ---------------------------------------------------------------------------
# save_pickle / load_pickle  (round-trip)
# ---------------------------------------------------------------------------

class TestPickle:
    def test_round_trip_dict(self, tmp_path):
        obj = {"key": [1, 2, 3], "val": "hello"}
        path = str(tmp_path / "data.pkl")
        save_pickle(obj, path)
        assert load_pickle(path) == obj

    def test_round_trip_array(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        path = str(tmp_path / "arr.pkl")
        save_pickle(arr, path)
        assert np.allclose(load_pickle(path), arr)


# ---------------------------------------------------------------------------
# make_dir
# ---------------------------------------------------------------------------

class TestMakeDir:
    def test_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "sub" / "dir")
        make_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_does_not_raise_if_exists(self, tmp_path):
        existing = str(tmp_path)
        make_dir(existing)  # should not raise


# ---------------------------------------------------------------------------
# write_text
# ---------------------------------------------------------------------------

class TestWriteText:
    def test_writes_string(self, tmp_path):
        path = str(tmp_path / "out.txt")
        write_text("hello world", path)
        with open(path, 'r') as f:
            assert f.read() == "hello world"

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "out.txt")
        write_text("first", path)
        write_text("second", path)
        with open(path, 'r') as f:
            assert f.read() == "second"


# ---------------------------------------------------------------------------
# write_numeric / read_numeric  (round-trip)
# ---------------------------------------------------------------------------

class TestNumericRoundTrip:
    def test_default_precision(self, tmp_path):
        path = str(tmp_path / "val.txt")
        write_numeric(3.14159, path)
        result = read_numeric(path)
        assert abs(result - 3.142) < 1e-3

    def test_custom_precision(self, tmp_path):
        path = str(tmp_path / "val.txt")
        write_numeric(1.23456789, path, precision='8.6f')
        result = read_numeric(path)
        assert abs(result - 1.234568) < 1e-6

    def test_integer_value(self, tmp_path):
        path = str(tmp_path / "val.txt")
        write_numeric(42, path, precision='d')
        result = read_numeric(path)
        assert result == 42.0


# ---------------------------------------------------------------------------
# write_json / read_json  (round-trip)
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_dict(self, tmp_path):
        path = str(tmp_path / "data.json")
        obj = {"a": 1, "b": [1, 2, 3]}
        write_json(obj, path)
        assert read_json(path) == obj

    def test_list(self, tmp_path):
        path = str(tmp_path / "data.json")
        obj = [1, "two", 3.0]
        write_json(obj, path)
        assert read_json(path) == obj

    def test_kwargs_forwarded(self, tmp_path):
        path = str(tmp_path / "data.json")
        obj = {"z": 1, "a": 2}
        write_json(obj, path, sort_keys=True)
        with open(path, 'r') as f:
            raw = f.read()
        assert raw.index('"a"') < raw.index('"z"')


# ---------------------------------------------------------------------------
# read_zipped
# ---------------------------------------------------------------------------

class TestReadZipped:
    def _make_zip(self, tmp_path, csv_content, csv_name="data.csv"):
        zip_path = str(tmp_path / "archive.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr(csv_name, csv_content)
        return zip_path

    def test_reads_csv_from_zip(self, tmp_path):
        csv = "a,b\n1,2\n3,4\n"
        zip_path = self._make_zip(tmp_path, csv)
        df = read_zipped(zip_path, "data.csv")
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_values_correct(self, tmp_path):
        csv = "x,y\n10,20\n30,40\n"
        zip_path = self._make_zip(tmp_path, csv)
        df = read_zipped(zip_path, "data.csv")
        assert df["x"].tolist() == [10, 30]
        assert df["y"].tolist() == [20, 40]
