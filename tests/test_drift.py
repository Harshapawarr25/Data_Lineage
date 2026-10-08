import numpy as np
import pandas as pd
import pytest

from services.drift import analyze_drift, benjamini_hochberg


def test_benjamini_hochberg_adjusts_and_preserves_input_order():
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    assert benjamini_hochberg([]) == []


def test_mixed_drift_reports_new_categories_effect_and_corrected_tests():
    reference = pd.DataFrame(
        {
            "charge": list(range(20)),
            "region": ["North", "South"] * 10,
            "binary": [0, 1] * 10,
        }
    )
    current = pd.DataFrame(
        {
            "charge": list(range(100, 120)),
            "region": ["Central", "Central"] * 10,
            "binary": [1, 1] * 10,
        }
    )
    results, summary = analyze_drift(
        reference,
        current,
        {"charge": "numeric", "region": "categorical", "binary": "categorical"},
    )
    by_feature = {row["feature"]: row for row in results}

    assert by_feature["region"]["metrics"]["new_categories"] == ["Central"]
    assert by_feature["region"]["signal"] == "red"
    assert by_feature["region"]["metrics"]["chi2_adjusted_p_value"] is not None
    assert by_feature["charge"]["metrics"]["ks_adjusted_p_value"] is not None
    assert by_feature["charge"]["metrics"]["mann_whitney_adjusted_p_value"] is not None
    assert by_feature["charge"]["metrics"]["wasserstein_distance"] > 0
    assert by_feature["charge"]["metrics"]["psi"] > 0.25
    assert summary["signal"] == "red"
    assert summary["benjamini_hochberg_test_count"] == 4


def test_numeric_drift_handles_infinite_and_constant_values():
    reference = pd.DataFrame({"x": [1.0, 1.0, np.inf, np.nan]})
    current = pd.DataFrame({"x": [2.0, 2.0, -np.inf, np.nan]})

    results, _ = analyze_drift(reference, current, {"x": "numeric"})

    assert len(results) == 1
    assert results[0]["metrics"]["psi"] > 0
    assert np.isfinite(results[0]["metrics"]["wasserstein_distance"])


def test_drift_thresholds_are_validated():
    data = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="PSI thresholds"):
        analyze_drift(data, data, {"x": "numeric"}, psi_warning=0.3, psi_critical=0.2)


def test_drift_marks_small_samples_and_sparse_categories_as_unavailable():
    reference = pd.DataFrame(
        {"number": [1, 2, 3], "category": ["a", "b", "a"]}
    )
    current = pd.DataFrame(
        {"number": [2, 3, 4], "category": ["b", "c", "b"]}
    )

    results, summary = analyze_drift(
        reference,
        current,
        {"number": "numeric", "category": "categorical"},
    )
    by_feature = {row["feature"]: row for row in results}

    assert by_feature["number"]["metrics"]["ks_p_value"] is None
    assert "at least 5" in by_feature["number"]["metrics"]["test_note"]
    assert by_feature["category"]["metrics"]["chi2_p_value"] is None
    assert "expected category counts" in by_feature["category"]["metrics"]["test_note"]
    assert all(row["signal"] in {"amber", "red"} for row in results)
    assert summary["benjamini_hochberg_test_count"] == 0
