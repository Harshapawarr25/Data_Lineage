from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _signal(rate: float, warning_rate: float, critical_rate: float) -> str:
    if rate >= critical_rate:
        return "red"
    if rate >= warning_rate:
        return "amber"
    return "green"


def _safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _missingness_tests(
    data: pd.DataFrame, column: str, categorical_columns: list[str]
) -> list[dict[str, Any]]:
    results = []
    missing = data[column].isna()
    for group_column in categorical_columns:
        if group_column == column:
            continue
        grouped = pd.crosstab(data[group_column].fillna("<missing>"), missing)
        if grouped.shape[0] < 2 or grouped.shape[1] < 2:
            continue
        try:
            statistic, p_value, _, expected = chi2_contingency(grouped)
        except ValueError:
            continue
        expected_minimum = float(np.min(expected))
        results.append(
            {
                "group_column": group_column,
                "chi2_statistic": float(statistic),
                "p_value": float(p_value),
                "minimum_expected_count": expected_minimum,
                "reliable": expected_minimum >= 5,
            }
        )
    return results


def _outlier_metrics(series: pd.Series) -> dict[str, float | int]:
    values = _safe_numeric(series)
    if values.empty:
        return {
            "valid_numeric_count": 0,
            "iqr_outlier_count": 0,
            "iqr_outlier_rate": 0.0,
            "zscore_outlier_count": 0,
            "zscore_outlier_rate": 0.0,
        }
    first_quartile = float(values.quantile(0.25))
    third_quartile = float(values.quantile(0.75))
    spread = third_quartile - first_quartile
    iqr_mask = (values < first_quartile - 1.5 * spread) | (
        values > third_quartile + 1.5 * spread
    )
    standard_deviation = float(values.std(ddof=0))
    zscore_mask = (
        (values - float(values.mean())).abs() > 3 * standard_deviation
        if standard_deviation > 0
        else pd.Series(False, index=values.index)
    )
    count = len(values)
    return {
        "valid_numeric_count": count,
        "iqr_outlier_count": int(iqr_mask.sum()),
        "iqr_outlier_rate": float(iqr_mask.mean()),
        "zscore_outlier_count": int(zscore_mask.sum()),
        "zscore_outlier_rate": float(zscore_mask.mean()),
    }


def _suggestions(
    missing_rate: float,
    duplicate_rate: float,
    iqr_rate: float,
    invalid_count: int,
    unexpected_categories: list[str],
    missingness_tests: list[dict[str, Any]],
) -> list[str]:
    suggestions = []
    if missing_rate:
        suggestions.append("Inspect source-system nulls and define an imputation policy.")
    if duplicate_rate:
        suggestions.append("Review the business key and deduplicate only verified repeats.")
    if iqr_rate:
        suggestions.append("Review IQR outliers against domain rules before capping or removing.")
    if invalid_count:
        suggestions.append("Correct values outside the configured domain bounds at ingestion.")
    if unexpected_categories:
        suggestions.append("Validate new category labels and update the approved reference domain.")
    if any(
        test["reliable"] and test["p_value"] < 0.05
        for test in missingness_tests
    ):
        suggestions.append("Investigate group-dependent missingness before imputing.")
    if not suggestions:
        suggestions.append("No action suggested; continue monitoring this feature.")
    return suggestions


def analyze_quality(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_types: Mapping[str, str],
    bounds: Mapping[str, Mapping[str, float | None]],
    warning_rate: float = 0.01,
    critical_rate: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not 0 <= warning_rate < critical_rate <= 1:
        raise ValueError("Quality thresholds must satisfy 0 <= warning < critical <= 1.")

    shared_categorical = [
        column
        for column, kind in column_types.items()
        if kind == "categorical" and column in reference.columns
    ]
    reference_domains = {
        column: set(reference[column].dropna().unique())
        for column in shared_categorical
    }
    results: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for dataset_name, data in (("reference", reference), ("current", current)):
        missing_cells = int(data.isna().sum().sum())
        total_cells = int(data.size)
        duplicate_rows = int(data.duplicated().sum())
        duplicate_rate = duplicate_rows / len(data) if len(data) else 0.0
        dataset_rows = []
        invalid_count_total = 0
        unexpected_count_total = 0
        outlier_rates = []

        for column in data.columns:
            missing_count = int(data[column].isna().sum())
            missing_rate = missing_count / len(data) if len(data) else 0.0
            numeric_metrics = (
                _outlier_metrics(data[column])
                if column_types.get(column) == "numeric"
                else _outlier_metrics(pd.Series(dtype=float))
            )
            lower = bounds.get(column, {}).get("minimum")
            upper = bounds.get(column, {}).get("maximum")
            invalid_mask = pd.Series(False, index=data.index)
            values = pd.to_numeric(data[column], errors="coerce")
            if lower is not None:
                invalid_mask |= values < lower
            if upper is not None:
                invalid_mask |= values > upper
            invalid_mask &= data[column].notna()
            invalid_count = int(invalid_mask.sum())
            invalid_rate = invalid_count / len(data) if len(data) else 0.0
            invalid_count_total += invalid_count

            unexpected_categories: list[str] = []
            if (
                dataset_name == "current"
                and column in reference_domains
                and column_types.get(column) == "categorical"
            ):
                reference_domain = reference_domains[column]
                unexpected_categories = sorted(
                    (
                        value
                        for value in data[column].dropna().unique()
                        if value not in reference_domain
                    ),
                    key=str,
                )
            unexpected_count = int(data[column].isin(unexpected_categories).sum())
            unexpected_rate = (
                unexpected_count / len(data) if len(data) else 0.0
            )
            unexpected_count_total += unexpected_count
            missingness_tests = _missingness_tests(
                data, column, shared_categorical
            )
            feature_rates = [
                missing_rate,
                float(numeric_metrics["iqr_outlier_rate"]),
                float(numeric_metrics["zscore_outlier_rate"]),
                invalid_rate,
                unexpected_rate,
            ]
            feature_signal = max(
                (_signal(rate, warning_rate, critical_rate) for rate in feature_rates),
                key={"green": 0, "amber": 1, "red": 2}.get,
            )
            if (invalid_count or unexpected_count) and feature_signal == "green":
                feature_signal = "amber"
            if any(
                test["reliable"] and test["p_value"] < 0.05
                for test in missingness_tests
            ) and feature_signal == "green":
                feature_signal = "amber"
            if column_types.get(column) == "numeric":
                outlier_rates.append(
                    max(
                        float(numeric_metrics["iqr_outlier_rate"]),
                        float(numeric_metrics["zscore_outlier_rate"]),
                    )
                )

            row = {
                "dataset": dataset_name,
                "feature": column,
                "signal": feature_signal,
                "metrics": {
                    "inferred_type": column_types.get(column, "categorical"),
                    "row_count": int(len(data)),
                    "missing_count": missing_count,
                    "missing_rate": missing_rate,
                    "duplicate_rows": duplicate_rows,
                    "duplicate_rate": duplicate_rate,
                    **numeric_metrics,
                    "invalid_count": invalid_count,
                    "invalid_rate": invalid_rate,
                    "minimum_allowed": lower,
                    "maximum_allowed": upper,
                    "unexpected_categories": [str(value) for value in unexpected_categories],
                    "unexpected_category_count": unexpected_count,
                    "unexpected_category_rate": unexpected_rate,
                    "missingness_group_tests": missingness_tests,
                },
                "suggestions": _suggestions(
                    missing_rate,
                    duplicate_rate,
                    float(numeric_metrics["iqr_outlier_rate"]),
                    invalid_count,
                    [str(value) for value in unexpected_categories],
                    missingness_tests,
                ),
            }
            results.append(row)
            dataset_rows.append(row)

        issue_rate = min(
            1.0,
            (missing_cells / total_cells if total_cells else 0.0)
            + duplicate_rate
            + (invalid_count_total / total_cells if total_cells else 0.0)
            + (unexpected_count_total / total_cells if total_cells else 0.0)
            + (float(np.mean(outlier_rates)) if outlier_rates else 0.0),
        )
        severity_order = {"green": 0, "amber": 1, "red": 2}
        overall_signal = max(
            (row["signal"] for row in dataset_rows),
            key=severity_order.get,
            default="green",
        )
        summaries[dataset_name] = {
            "quality_score": max(0.0, 100.0 * (1.0 - issue_rate)),
            "signal": overall_signal,
            "issue_rate": issue_rate,
            "row_count": int(len(data)),
            "column_count": int(len(data.columns)),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "duplicate_rate": duplicate_rate,
            "warning_rate": warning_rate,
            "critical_rate": critical_rate,
        }
    return results, summaries
