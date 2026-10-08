import pandas as pd

from services.quality import analyze_quality


def test_quality_detects_invalid_values_outliers_and_new_categories():
    reference = pd.DataFrame(
        {
            "monthly_charges": [10, 11, 12, 13, 14, 15],
            "region": ["North", "South", "North", "South", "North", "South"],
            "churned": [0, 1, 0, 1, 0, 1],
        }
    )
    current = pd.DataFrame(
        {
            "monthly_charges": [-2, 11, 12, 13, 14, 100],
            "region": [
                "Central",
                "South",
                "North",
                "South",
                "North",
                "South",
            ],
            "churned": [0, 1, 0, 1, 0, 1],
        }
    )
    types = {
        "monthly_charges": "numeric",
        "region": "categorical",
        "churned": "categorical",
    }

    results, summaries = analyze_quality(
        reference,
        current,
        types,
        {"monthly_charges": {"minimum": 0, "maximum": None}},
    )
    current_results = {
        row["feature"]: row for row in results if row["dataset"] == "current"
    }

    assert current_results["monthly_charges"]["metrics"]["invalid_count"] == 1
    assert current_results["monthly_charges"]["metrics"]["iqr_outlier_count"] == 2
    assert current_results["region"]["metrics"]["unexpected_categories"] == [
        "Central"
    ]
    assert current_results["region"]["signal"] == "red"
    assert summaries["current"]["quality_score"] < 100
    assert summaries["current"]["signal"] in {"amber", "red"}
    assert current_results["monthly_charges"]["suggestions"]


def test_quality_detects_duplicates_missingness_and_group_association():
    reference = pd.DataFrame(
        {
            "value": [
                None,
                1,
                None,
                2,
                None,
                3,
                None,
                4,
                None,
                5,
                None,
                6,
                None,
                7,
                None,
                8,
                None,
                9,
                None,
                10,
                *range(20, 40),
            ],
            "cohort": ["A"] * 20 + ["B"] * 20,
            "row_id": list(range(40)),
        }
    )
    current = pd.concat(
        [reference, reference.iloc[[0]]],
        ignore_index=True,
    )

    results, summaries = analyze_quality(
        reference,
        current,
        {"value": "numeric", "cohort": "categorical", "row_id": "numeric"},
        {},
    )
    current_value = next(
        row
        for row in results
        if row["dataset"] == "current" and row["feature"] == "value"
    )

    assert summaries["current"]["duplicate_rows"] == 1
    assert current_value["metrics"]["missing_count"] > 0
    assert current_value["metrics"]["missingness_group_tests"]
    assert any(
        suggestion.startswith("Investigate group-dependent missingness")
        for suggestion in current_value["suggestions"]
    )


def test_quality_thresholds_must_be_ordered():
    data = pd.DataFrame({"x": [1, 2, 3]})

    try:
        analyze_quality(data, data, {"x": "numeric"}, {}, 0.05, 0.01)
    except ValueError as error:
        assert "warning < critical" in str(error)
    else:
        raise AssertionError("Invalid thresholds should be rejected.")
