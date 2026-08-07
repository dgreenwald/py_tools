"""DataFrame validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


__all__ = (
    "require_columns",
    "require_datetime_dtype",
    "require_unique_key",
    "validate_dataframe",
)


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    description: str = "dataframe",
    error_type: type[Exception] = ValueError,
) -> None:
    """Require that *frame* contains every named column."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise error_type(
            f"{description} is missing required columns: {', '.join(missing)}"
        )


def require_unique_key(
    frame: pd.DataFrame,
    key: str | Iterable[str],
    description: str | None = None,
    error_type: type[Exception] | None = ValueError,
    *,
    keep: str | bool = False,
) -> pd.Series:
    """Return a duplicate mask and, by default, reject duplicate keys.

    Pass ``error_type=None`` to obtain the diagnostic mask without raising.
    Missing key values follow :meth:`pandas.DataFrame.duplicated` semantics
    and are therefore duplicates when they occur more than once.
    """
    columns = [key] if isinstance(key, str) else list(key)
    require_columns(frame, columns, description or "dataframe", error_type or ValueError)
    duplicated = frame.duplicated(columns, keep=keep)
    if duplicated.any() and error_type is not None:
        label = description or "dataframe"
        if len(columns) == 1:
            examples = (
                frame.loc[duplicated, columns[0]]
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            key_text = columns[0]
        else:
            examples = (
                frame.loc[duplicated, columns]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            key_text = str(columns)
        raise error_type(
            f"{label} must be unique on {key_text}; duplicate examples: {examples}"
        )
    return duplicated


def require_datetime_dtype(
    frame: pd.DataFrame,
    column: str,
    description: str = "dataframe",
    error_type: type[Exception] = ValueError,
) -> None:
    """Require that a frame column already has a datetime dtype."""
    require_columns(frame, [column], description, error_type)
    if not pd.api.types.is_datetime64_any_dtype(frame[column]):
        raise error_type(f"{description} column {column!r} must have a datetime dtype.")


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    usecols: Iterable[str] | None = None,
    reserved_columns: Iterable[str] = (),
    assert_nonmissing: str | Iterable[str] | None = None,
    assert_unique: str | Iterable[str] | None = None,
    assert_unique_options: Mapping[str, object] | None = None,
    string_columns: Iterable[str] = (),
    integer_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
    binary_columns: Iterable[str] = (),
    datetime_columns: Iterable[str] = (),
    description: str | None = None,
    error_type: type[Exception] = ValueError,
) -> pd.DataFrame:
    """Validate a dataframe contract and return an independent copy."""
    label = description or "dataframe"
    reserved = sorted(set(reserved_columns).intersection(frame.columns))
    if reserved:
        raise error_type(f"{label} contains reserved columns: {', '.join(reserved)}")

    if usecols is None:
        result = frame.copy()
    else:
        selected = list(usecols)
        require_columns(frame, selected, label, error_type)
        result = frame.loc[:, selected].copy()

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
                    f"{previous_group} and {group_name} validation"
                )
            assigned[column] = group_name
        require_columns(result, columns, label, error_type)

    for column in groups["string"]:
        if not isinstance(result[column].dtype, pd.StringDtype):
            raise error_type(
                f"{label} column {column!r} must have a pandas string dtype."
            )
    for column in groups["integer"]:
        if not pd.api.types.is_integer_dtype(result[column]):
            raise error_type(f"{label} column {column!r} must have an integer dtype.")
    for column in groups["numeric"]:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise error_type(f"{label} column {column!r} must have a numeric dtype.")
    for column in groups["binary"]:
        if not pd.api.types.is_bool_dtype(result[column]):
            raise error_type(f"{label} column {column!r} must have a boolean dtype.")
    for column in groups["datetime"]:
        require_datetime_dtype(result, column, label, error_type)

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
