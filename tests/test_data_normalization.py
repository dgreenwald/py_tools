"""Tests for py_tools.data.normalization."""

import warnings

import pandas as pd
import pytest

from py_tools.data.normalization import (
    normalize_binary,
    normalize_columns,
    normalize_dataframe,
    normalize_datetime,
    normalize_integer,
    normalize_numeric,
    select_columns,
)


def test_normalize_columns_strips_lowercases_and_copies():
    frame = pd.DataFrame({" Name ": ["Acme"], "VALUE": [1]})

    result = normalize_columns(frame)

    assert list(result.columns) == ["name", "value"]
    assert list(frame.columns) == [" Name ", "VALUE"]


def test_normalize_columns_rejects_collisions_and_multiindex():
    collision = pd.DataFrame([[1, 2]], columns=[" A ", "a"])
    hierarchical = pd.DataFrame(
        [[1]], columns=pd.MultiIndex.from_tuples([("A", "value")])
    )

    with pytest.raises(ValueError, match="duplicate labels"):
        normalize_columns(collision)
    with pytest.raises(ValueError, match="MultiIndex"):
        normalize_columns(hierarchical)


def test_select_columns_can_ignore_absent_optional_columns():
    frame = pd.DataFrame({"a": [1], "b": [2]})

    result = select_columns(frame, ["b", "missing"], require=False)

    assert list(result.columns) == ["b"]
    result.loc[0, "b"] = 3
    assert frame.loc[0, "b"] == 2


def test_normalize_integer_uses_nullable_dtype_and_rejects_nonintegers():
    result = normalize_integer(pd.Series(["1", 2.0, None]), allow_missing=True)

    assert result.dtype == "Int64"
    assert result.tolist() == [1, 2, pd.NA]

    for value in (True, 1.5, float("inf")):
        with pytest.raises(ValueError, match="integer values"):
            normalize_integer(pd.Series([value]))


def test_normalize_numeric_rejects_infinity_by_default():
    with pytest.raises(ValueError, match="invalid numeric values"):
        normalize_numeric(pd.Series([float("inf")]))

    result = normalize_numeric(pd.Series([float("inf")]), allow_infinite=True)
    assert result.iloc[0] == float("inf")


def test_normalize_binary_accepts_unambiguous_default_representations():
    values = pd.Series(
        [
            True,
            False,
            1,
            0,
            1.0,
            0.0,
            " TRUE ",
            "false",
            "1.00",
            "0.0",
            None,
        ]
    )

    result = normalize_binary(values)

    assert result.dtype == "boolean"
    assert result.tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        pd.NA,
    ]


def test_normalize_binary_supports_custom_aliases_and_rejects_overlap():
    result = normalize_binary(
        pd.Series([" Y ", "n"]), true_values=("y",), false_values=("N",)
    )
    assert result.tolist() == [True, False]

    with pytest.raises(ValueError, match="overlapping aliases"):
        normalize_binary(
            pd.Series([1]), true_values=(1,), false_values=("1.0",)
        )


def test_normalize_binary_preserves_duplicate_index_alignment():
    series = pd.Series([1, 0], index=[5, 5])

    result = normalize_binary(series)

    assert result.index.tolist() == [5, 5]
    assert result.tolist() == [True, False]


def test_normalize_datetime_requires_consistent_timezone_policy():
    values = pd.Series(
        ["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00-05:00"]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(ValueError, match="utc=True"):
            normalize_datetime(values)

    result = normalize_datetime(values, utc=True)
    assert isinstance(result.dtype, pd.DatetimeTZDtype)
    assert str(result.dtype.tz) == "UTC"


def test_normalize_dataframe_normalizes_groups_and_reports_column_errors():
    frame = pd.DataFrame(
        {
            " ID ": ["1"],
            "Name": [" Acme "],
            "Amount": ["bad"],
            "Active": [1.0],
        }
    )

    with pytest.raises(ValueError, match="source column 'amount'"):
        normalize_dataframe(
            frame,
            normalize_columns=True,
            integer_columns=["id"],
            string_columns=["name"],
            numeric_columns=["amount"],
            binary_columns=["active"],
            description="source",
        )

    frame["Amount"] = "4.5"
    result = normalize_dataframe(
        frame,
        normalize_columns=True,
        integer_columns=["id"],
        string_columns=["name"],
        numeric_columns=["amount"],
        binary_columns=["active"],
    )
    assert result["id"].dtype == "Int64"
    assert result["name"].dtype == "string"
    assert result["amount"].dtype == "Float64"
    assert result["active"].dtype == "boolean"
