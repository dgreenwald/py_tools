"""DataFrame and Series normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from numbers import Real

import numpy as np
import pandas as pd

from .validation import require_columns, require_unique_key


__all__ = (
    "normalize_binary",
    "normalize_columns",
    "normalize_dataframe",
    "normalize_datetime",
    "normalize_integer",
    "normalize_numeric",
    "normalize_string",
    "select_columns",
)


def normalize_columns(
    frame: pd.DataFrame,
    *,
    copy: bool = True,
    error_type: type[Exception] = ValueError,
) -> pd.DataFrame:
    """Strip and lowercase column labels, rejecting ambiguous results."""
    if isinstance(frame.columns, pd.MultiIndex):
        raise error_type("Cannot normalize MultiIndex column labels.")

    result = frame.copy() if copy else frame
    columns = pd.Index([str(column).strip().lower() for column in result.columns])
    duplicated = columns[columns.duplicated(keep=False)].unique().tolist()
    if duplicated:
        raise error_type(
            "Column normalization produces duplicate labels: "
            f"{', '.join(repr(column) for column in duplicated)}"
        )
    result.columns = columns
    return result


_normalize_column_labels = normalize_columns


def select_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    description: str = "dataframe",
    *,
    require: bool = True,
    error_type: type[Exception] = ValueError,
) -> pd.DataFrame:
    """Select columns and return an independent copy."""
    requested = list(columns)
    if require:
        require_columns(frame, requested, description, error_type)
        selected = requested
    else:
        selected = [column for column in requested if column in frame.columns]
    return frame.loc[:, selected].copy()


def normalize_string(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    """Strip strings, convert blanks to missing, and optionally uppercase."""
    result = series.astype("string").str.strip()
    result = result.mask(result.eq(""), pd.NA)
    return result.str.upper() if uppercase else result


def normalize_integer(
    series: pd.Series,
    description: str = "series",
    *,
    allow_missing: bool = False,
    error_type: type[Exception] = ValueError,
) -> pd.Series:
    """Convert integer-like values to nullable integers.

    Missing values are rejected by default. Booleans and fractional values
    are never treated as integers.
    """
    boolean = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    numeric = pd.to_numeric(series, errors="coerce")
    failed_conversion = series.notna() & numeric.isna()
    nonfinite = numeric.notna() & ~np.isfinite(numeric)
    fractional = numeric.notna() & numeric.mod(1).ne(0)
    missing = numeric.isna() if not allow_missing else pd.Series(False, index=series.index)
    invalid = boolean | failed_conversion | nonfinite | fractional | missing
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        missing_text = "" if allow_missing else "nonmissing "
        raise error_type(
            f"{description} must contain {missing_text}integer values; "
            f"examples: {examples}"
        )
    return numeric.astype("Int64")


def normalize_numeric(
    series: pd.Series,
    description: str = "series",
    *,
    allow_missing: bool = True,
    allow_infinite: bool = False,
    nonnegative: bool = False,
    error_type: type[Exception] = ValueError,
) -> pd.Series:
    """Convert values to nullable floats and validate numeric constraints."""
    numeric = pd.to_numeric(series, errors="coerce").astype("Float64")
    failed_conversion = series.notna() & numeric.isna()
    missing = numeric.isna() if not allow_missing else pd.Series(False, index=series.index)
    nonfinite = (
        pd.Series(False, index=series.index)
        if allow_infinite
        else numeric.notna() & ~np.isfinite(numeric.astype(float))
    )
    invalid = failed_conversion | missing | nonfinite
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise error_type(f"{description} contains invalid numeric values: {examples}")
    negative = numeric.lt(0).fillna(False)
    if nonnegative and negative.any():
        examples = numeric.loc[negative].head(5).tolist()
        raise error_type(f"{description} cannot be negative; examples: {examples}")
    return numeric


def _canonical_binary_value(value: object) -> tuple[str, object]:
    """Return a stable comparison token for a nonmissing binary alias."""
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "false"}:
            return ("binary", text == "true")
        try:
            numeric = Decimal(text)
        except InvalidOperation:
            return ("string", text)
        if numeric.is_finite() and numeric in {Decimal(0), Decimal(1)}:
            return ("binary", bool(numeric))
        return ("string", text)
    if isinstance(value, (bool, np.bool_)):
        return ("binary", bool(value))
    if isinstance(value, (Real, Decimal)):
        if value == 0 or value == 1:
            return ("binary", bool(value))
        return ("number", value)
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"Binary aliases must be hashable; got {value!r}.") from exc
    return ("object", value)


def normalize_binary(
    series: pd.Series,
    description: str = "series",
    *,
    true_values: Iterable[object] = (True, 1, 1.0, "true", "1", "1.0"),
    false_values: Iterable[object] = (False, 0, 0.0, "false", "0", "0.0"),
    allow_missing: bool = True,
    error_type: type[Exception] = ValueError,
) -> pd.Series:
    """Normalize unambiguous true/false representations to nullable booleans."""
    try:
        true_tokens = {_canonical_binary_value(value) for value in true_values}
        false_tokens = {_canonical_binary_value(value) for value in false_values}
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    overlap = true_tokens.intersection(false_tokens)
    if overlap:
        raise error_type("true_values and false_values contain overlapping aliases.")

    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    invalid = np.zeros(len(series), dtype=bool)
    for position, value in enumerate(series):
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        if is_missing:
            invalid[position] = not allow_missing
            continue
        token = _canonical_binary_value(value)
        if token in true_tokens:
            result.iloc[position] = True
        elif token in false_tokens:
            result.iloc[position] = False
        else:
            invalid[position] = True
    if invalid.any():
        examples = series.iloc[invalid].head(5).tolist()
        raise error_type(f"{description} contains invalid binary values: {examples}")
    return result


def normalize_datetime(
    series: pd.Series,
    description: str = "series",
    *,
    allow_missing: bool = True,
    utc: bool = False,
    format: str | None = None,
    error_type: type[Exception] = ValueError,
) -> pd.Series:
    """Convert date-like values to a consistent pandas datetime dtype."""
    converted = pd.to_datetime(series, errors="coerce", utc=utc, format=format)
    failed_conversion = series.notna() & converted.isna()
    missing = converted.isna() if not allow_missing else pd.Series(False, index=series.index)
    invalid = failed_conversion | missing
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        missing_text = "" if allow_missing else "nonmissing "
        raise error_type(
            f"{description} must contain {missing_text}valid datetimes; "
            f"examples: {examples}"
        )
    if not pd.api.types.is_datetime64_any_dtype(converted):
        raise error_type(
            f"{description} contains datetimes with inconsistent time zones; "
            "pass utc=True to normalize them."
        )
    return converted


def normalize_dataframe(
    frame: pd.DataFrame,
    *,
    usecols: Iterable[str] | None = None,
    normalize_columns: bool = False,
    assert_nonmissing: str | Iterable[str] | None = None,
    assert_unique: str | Iterable[str] | None = None,
    assert_unique_options: Mapping[str, object] | None = None,
    string_columns: Iterable[str] = (),
    integer_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
    binary_columns: Iterable[str] = (),
    datetime_columns: Iterable[str] = (),
    string_options: Mapping[str, object] | None = None,
    integer_options: Mapping[str, object] | None = None,
    numeric_options: Mapping[str, object] | None = None,
    binary_options: Mapping[str, object] | None = None,
    datetime_options: Mapping[str, object] | None = None,
    description: str | None = None,
    error_type: type[Exception] = ValueError,
) -> pd.DataFrame:
    """Select and normalize mutually exclusive groups of dataframe columns."""
    result = frame.copy()
    label = description or "dataframe"
    if normalize_columns:
        result = _normalize_column_labels(result, copy=False, error_type=error_type)
    if usecols is not None:
        result = select_columns(result, usecols, label, error_type=error_type)

    groups = {
        "string": list(string_columns),
        "integer": list(integer_columns),
        "numeric": list(numeric_columns),
        "binary": list(binary_columns),
        "datetime": list(datetime_columns),
    }
    assigned: dict[str, str] = {}
    for group_name, columns in groups.items():
        for column in columns:
            previous_group = assigned.get(column)
            if previous_group is not None:
                raise error_type(
                    f"{label} column {column!r} is assigned to both "
                    f"{previous_group} and {group_name} normalization"
                )
            assigned[column] = group_name
        require_columns(result, columns, label, error_type)

    options = {
        "string": dict(string_options or {}),
        "integer": dict(integer_options or {}),
        "numeric": dict(numeric_options or {}),
        "binary": dict(binary_options or {}),
        "datetime": dict(datetime_options or {}),
    }
    converters = {
        "string": normalize_string,
        "integer": normalize_integer,
        "numeric": normalize_numeric,
        "binary": normalize_binary,
        "datetime": normalize_datetime,
    }
    for group_name, columns in groups.items():
        for column in columns:
            kwargs = options[group_name]
            if group_name == "string":
                result[column] = converters[group_name](result[column], **kwargs)
            else:
                result[column] = converters[group_name](
                    result[column],
                    f"{label} column {column!r}",
                    error_type=error_type,
                    **kwargs,
                )

    if assert_nonmissing is not None:
        required = (
            [assert_nonmissing]
            if isinstance(assert_nonmissing, str)
            else list(assert_nonmissing)
        )
        require_columns(result, required, label, error_type)
        missing_columns = [column for column in required if result[column].isna().any()]
        if missing_columns:
            raise error_type(
                f"{label} contains missing values in required columns: "
                f"{', '.join(missing_columns)}"
            )
    if assert_unique is not None:
        require_unique_key(
            result,
            assert_unique,
            description=label,
            error_type=error_type,
            **dict(assert_unique_options or {}),
        )
    return result
