from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, wasserstein_distance

MIN_INFERENCE_SAMPLE_SIZE = 5


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("P-values must be between 0 and 1.")
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_minimum = 1.0
    for reverse_index in range(count - 1, -1, -1):
        original_index = int(order[reverse_index])
        rank = reverse_index + 1
        running_minimum = min(
            running_minimum, p_values[original_index] * count / rank
        )
        adjusted[original_index] = running_minimum
    return [float(value) for value in np.clip(adjusted, 0, 1)]


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _psi_numeric(reference: np.ndarray, current: np.ndarray) -> float:
    if reference.size == 0 or current.size == 0:
        return 0.0
    unique = np.unique(reference)
    if unique.size == 1:
        center = float(unique[0])
        epsilon = max(abs(center) * 1e-6, 1e-6)
        edges = np.array([-np.inf, center - epsilon, center + epsilon, np.inf])
    else:
        quantiles = np.linspace(0, 1, 11)[1:-1]
        inner_edges = np.unique(np.quantile(reference, quantiles))
        edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
        if edges.size < 3:
            center = float(np.median(reference))
            epsilon = max(abs(center) * 1e-6, 1e-6)
            edges = np.array([-np.inf, center - epsilon, center + epsilon, np.inf])
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    reference_share = reference_counts / reference.size
    current_share = current_counts / current.size
    epsilon = 1e-6
    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(
                (current_share + epsilon) / (reference_share + epsilon)
            )
        )
    )


def _psi_categorical(
    reference: pd.Series, current: pd.Series
) -> float:
    categories = sorted(
        set(reference.dropna().unique()) | set(current.dropna().unique()),
        key=str,
    )
    if not categories:
        return 0.0
    reference_counts = reference.value_counts().reindex(categories, fill_value=0)
    current_counts = current.value_counts().reindex(categories, fill_value=0)
    reference_share = reference_counts.to_numpy(dtype=float) / max(1, len(reference.dropna()))
    current_share = current_counts.to_numpy(dtype=float) / max(1, len(current.dropna()))
    epsilon = 1e-6
    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(
                (current_share + epsilon) / (reference_share + epsilon)
            )
        )
    )


def _adjusted_signal(
    adjusted_p_value: float | None,
    psi: float,
    new_category_count: int,
    alpha: float,
    psi_warning: float,
    psi_critical: float,
) -> str:
    if psi >= psi_critical:
        return "red"
    if new_category_count:
        return "red"
    if psi >= psi_warning:
        return "amber"
    if adjusted_p_value is not None and adjusted_p_value < alpha:
        return "amber"
    return "green"


def analyze_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_types: Mapping[str, str],
    alpha: float = 0.05,
    psi_warning: float = 0.1,
    psi_critical: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < alpha < 1:
        raise ValueError("Significance threshold alpha must be between 0 and 1.")
    if not 0 <= psi_warning < psi_critical:
        raise ValueError("PSI thresholds must satisfy 0 <= warning < critical.")

    shared_columns = [
        column for column in reference.columns if column in current.columns
    ]
    results: list[dict[str, Any]] = []
    p_value_locations: list[tuple[int, str]] = []

    for column in shared_columns:
        kind = column_types.get(column, "categorical")
        row: dict[str, Any] = {
            "feature": column,
            "feature_type": kind,
            "signal": "green",
            "metrics": {
                "reference_count": int(reference[column].notna().sum()),
                "current_count": int(current[column].notna().sum()),
                "new_categories": [],
                "new_category_count": 0,
            },
        }
        if kind == "numeric":
            ref_values = _numeric_values(reference[column])
            cur_values = _numeric_values(current[column])
            row["metrics"].update(
                {
                    "ks_statistic": None,
                    "ks_p_value": None,
                    "mann_whitney_u": None,
                    "mann_whitney_p_value": None,
                    "wasserstein_distance": None,
                    "psi": None,
                    "test_note": None,
                }
            )
            if ref_values.size and cur_values.size:
                psi = _psi_numeric(ref_values, cur_values)
                row["metrics"].update(
                    {
                        "wasserstein_distance": float(
                            wasserstein_distance(ref_values, cur_values)
                        ),
                        "psi": psi,
                    }
                )
                if (
                    ref_values.size >= MIN_INFERENCE_SAMPLE_SIZE
                    and cur_values.size >= MIN_INFERENCE_SAMPLE_SIZE
                ):
                    ks_result = ks_2samp(ref_values, cur_values)
                    mwu_result = mannwhitneyu(
                        ref_values, cur_values, alternative="two-sided"
                    )
                    row["metrics"].update(
                        {
                            "ks_statistic": float(ks_result.statistic),
                            "ks_p_value": float(ks_result.pvalue),
                            "mann_whitney_u": float(mwu_result.statistic),
                            "mann_whitney_p_value": float(mwu_result.pvalue),
                        }
                    )
                    p_value_locations.extend(
                        [
                            (len(results), "ks_p_value"),
                            (len(results), "mann_whitney_p_value"),
                        ]
                    )
                else:
                    row["metrics"]["test_note"] = (
                        f"Inferential tests require at least "
                        f"{MIN_INFERENCE_SAMPLE_SIZE} non-missing observations "
                        "in each dataset."
                    )
            else:
                row["metrics"]["test_note"] = (
                    "Distribution metrics need non-missing numeric values in both datasets."
                )
        else:
            ref_values = reference[column].dropna()
            cur_values = current[column].dropna()
            categories = sorted(
                set(ref_values.unique()) | set(cur_values.unique()), key=str
            )
            chi2_statistic = None
            chi2_p_value = None
            js_distance = None
            test_note = None
            if len(categories) >= 2 and len(ref_values) and len(cur_values):
                ref_counts = ref_values.value_counts().reindex(categories, fill_value=0)
                cur_counts = cur_values.value_counts().reindex(categories, fill_value=0)
                contingency = np.vstack(
                    [ref_counts.to_numpy(), cur_counts.to_numpy()]
                )
                try:
                    chi2_statistic, candidate_p_value, _, expected = chi2_contingency(
                        contingency
                    )
                    if float(np.min(expected)) >= 5:
                        chi2_p_value = float(candidate_p_value)
                    else:
                        test_note = (
                            "Chi-square p-value omitted because one or more "
                            "expected category counts are below 5."
                        )
                except ValueError:
                    chi2_statistic = None
                    chi2_p_value = None
                    test_note = "Chi-square test could not be calculated for this table."
                reference_distribution = ref_counts.to_numpy(dtype=float) / len(ref_values)
                current_distribution = cur_counts.to_numpy(dtype=float) / len(cur_values)
                js_distance = float(
                    jensenshannon(reference_distribution, current_distribution)
                )
            elif len(categories) < 2:
                test_note = "Chi-square test needs at least two observed categories."
            reference_categories = set(ref_values.unique())
            new_categories = sorted(
                (value for value in cur_values.unique() if value not in reference_categories),
                key=str,
            )
            psi = _psi_categorical(ref_values, cur_values)
            row["metrics"].update(
                {
                    "chi2_statistic": (
                        float(chi2_statistic)
                        if chi2_statistic is not None
                        else None
                    ),
                    "chi2_p_value": (
                        float(chi2_p_value) if chi2_p_value is not None else None
                    ),
                    "jensen_shannon_distance": js_distance,
                    "psi": psi,
                    "new_categories": [str(value) for value in new_categories],
                    "new_category_count": len(new_categories),
                    "new_category_rate": (
                        float(cur_values.isin(new_categories).mean())
                        if len(cur_values)
                        else 0.0
                    ),
                    "test_note": test_note,
                }
            )
            if chi2_p_value is not None:
                p_value_locations.append((len(results), "chi2_p_value"))
        results.append(row)

    all_p_values = [
        float(results[row_index]["metrics"][metric_name])
        for row_index, metric_name in p_value_locations
        if results[row_index]["metrics"][metric_name] is not None
    ]
    adjusted_values = benjamini_hochberg(all_p_values)
    for (row_index, metric_name), adjusted in zip(
        p_value_locations, adjusted_values
    ):
        results[row_index]["metrics"][metric_name.replace("_p_value", "_adjusted_p_value")] = adjusted

    severity = {"green": 0, "amber": 1, "red": 2}
    for row in results:
        p_metrics = [
            value
            for name, value in row["metrics"].items()
            if name.endswith("_adjusted_p_value") and value is not None
        ]
        minimum_adjusted_p = min(p_metrics) if p_metrics else None
        row["metrics"]["minimum_adjusted_p_value"] = minimum_adjusted_p
        row["metrics"]["alpha"] = alpha
        row["metrics"]["psi_warning"] = psi_warning
        row["metrics"]["psi_critical"] = psi_critical
        row["signal"] = _adjusted_signal(
            minimum_adjusted_p,
            float(row["metrics"]["psi"] or 0.0),
            int(row["metrics"]["new_category_count"]),
            alpha,
            psi_warning,
            psi_critical,
        )
        if row["metrics"]["test_note"] and row["signal"] == "green":
            row["signal"] = "amber"

    overall_signal = max(
        (row["signal"] for row in results),
        key=severity.get,
        default="green",
    )
    summary = {
        "signal": overall_signal,
        "features_compared": len(results),
        "drifted_features": sum(row["signal"] == "red" for row in results),
        "watch_features": sum(row["signal"] == "amber" for row in results),
        "alpha": alpha,
        "psi_warning": psi_warning,
        "psi_critical": psi_critical,
        "benjamini_hochberg_test_count": len(all_p_values),
    }
    return results, summary
