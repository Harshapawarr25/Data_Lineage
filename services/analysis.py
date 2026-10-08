from __future__ import annotations

from collections.abc import Mapping
import re

import pandas as pd


def compare_schema(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, list[str]]:
    reference_columns = list(reference.columns)
    current_columns = list(current.columns)
    reference_set = set(reference_columns)
    current_set = set(current_columns)
    return {
        "shared": [column for column in reference_columns if column in current_set],
        "missing_from_current": [
            column for column in reference_columns if column not in current_set
        ],
        "extra_in_current": [
            column for column in current_columns if column not in reference_set
        ],
    }


def infer_column_types(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, str]:
    schema = compare_schema(reference, current)
    columns = list(reference.columns) + [
        column for column in current.columns if column not in reference.columns
    ]
    inferred: dict[str, str] = {}
    for column in columns:
        series_list = [
            frame[column]
            for frame in (reference, current)
            if column in frame.columns
        ]
        numeric = all(
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
            for series in series_list
        )
        if not numeric:
            inferred[column] = "categorical"
            continue
        values = pd.concat(series_list, ignore_index=True).dropna()
        inferred[column] = (
            "categorical"
            if values.nunique(dropna=True) <= 2
            else "numeric"
        )
    for column in schema["shared"]:
        if column not in inferred:
            raise ValueError(f"Could not infer a type for shared column {column!r}.")
    return inferred


def recommended_nonnegative_columns(
    column_types: Mapping[str, str],
) -> list[str]:
    tokens = (
        "age",
        "amount",
        "charge",
        "charges",
        "cost",
        "costs",
        "count",
        "duration",
        "fee",
        "fees",
        "price",
        "quantity",
        "tenure",
        "qty",
    )
    return [
        column
        for column, kind in column_types.items()
        if kind == "numeric"
        and bool(set(re.split(r"[^a-z0-9]+", column.casefold())) & set(tokens))
    ]
