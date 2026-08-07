"""Tests for py_tools.data.validation."""

import pandas as pd
import pytest

from py_tools.data.validation import (
    require_columns,
    require_datetime_dtype,
    require_unique_key,
    validate_dataframe,
)


def test_require_columns_uses_requested_exception_type():
    class SourceError(Exception):
        pass

    with pytest.raises(SourceError, match="source is missing required columns: b"):
        require_columns(pd.DataFrame({"a": [1]}), ["b"], "source", SourceError)


def test_require_unique_key_accepts_iterables_and_returns_mask():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": [2, 2, 3]})

    mask = require_unique_key(frame, (column for column in ["a", "b"]), error_type=None)

    assert mask.tolist() == [True, True, False]


def test_require_datetime_dtype_checks_column_presence_and_dtype():
    frame = pd.DataFrame({"date": ["2024-01-01"]})

    with pytest.raises(ValueError, match="datetime dtype"):
        require_datetime_dtype(frame, "date")
    with pytest.raises(ValueError, match="missing required columns"):
        require_datetime_dtype(frame, "missing")


def test_validate_dataframe_checks_canonical_dtypes_and_returns_copy():
    frame = pd.DataFrame(
        {
            "s": pd.Series(["a"], dtype="string"),
            "i": pd.Series([1], dtype="Int64"),
            "n": pd.Series([1.5], dtype="Float64"),
            "b": pd.Series([True], dtype="boolean"),
            "d": pd.to_datetime(["2024-01-01"]),
        }
    )

    result = validate_dataframe(
        frame,
        string_columns=["s"],
        integer_columns=["i"],
        numeric_columns=["n"],
        binary_columns=["b"],
        datetime_columns=["d"],
        assert_nonmissing=["s", "d"],
        assert_unique="i",
    )

    assert result.equals(frame)
    assert result is not frame


def test_validate_dataframe_rejects_object_strings():
    frame = pd.DataFrame({"s": pd.Series(["a"], dtype=object)})

    with pytest.raises(ValueError, match="pandas string dtype"):
        validate_dataframe(frame, string_columns=["s"])


def test_validate_dataframe_checks_reserved_columns_before_projection():
    frame = pd.DataFrame({"required": [1], "generated": [2]})

    with pytest.raises(ValueError, match="contains reserved columns: generated"):
        validate_dataframe(
            frame,
            usecols=["required"],
            reserved_columns=["generated"],
        )
