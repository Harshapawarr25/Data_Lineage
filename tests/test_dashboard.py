from ui_pages.dashboard import (
    _compact_values,
    _drift_overview,
    _drift_table,
    _quality_issue_overview,
)


def test_compact_values_summarizes_long_lists():
    values = [f"CUST{i:05d}" for i in range(12)]

    assert _compact_values(values, limit=4) == (
        "CUST00000, CUST00001, CUST00002, CUST00003, +8 more"
    )


def test_drift_table_compacts_new_categories():
    records = [
        {
            "feature": "customer_id",
            "feature_type": "categorical",
            "signal": "red",
            "metrics": {
                "minimum_adjusted_p_value": None,
                "psi": 0.5,
                "new_categories": [f"CUST{i:05d}" for i in range(10)],
                "test_note": None,
                "chi2_statistic": None,
                "chi2_p_value": None,
                "chi2_adjusted_p_value": None,
                "jensen_shannon_distance": None,
            },
        }
    ]

    table = _drift_table(records)

    assert table.loc[0, "New categories"] == (
        "CUST00000, CUST00001, CUST00002, CUST00003, "
        "CUST00004, CUST00005, CUST00006, CUST00007, +2 more"
    )


def test_drift_overview_orders_by_psi_and_new_categories():
    records = [
        {
            "feature": "low",
            "signal": "green",
            "metrics": {"psi": 0.1, "new_category_count": 0},
        },
        {
            "feature": "customer_id",
            "signal": "red",
            "metrics": {"psi": 13.823, "new_category_count": 1000},
        },
    ]

    overview = _drift_overview(records)

    assert overview.iloc[0].to_dict() == {
        "Feature": "customer_id",
        "Signal": "RED",
        "PSI": 13.823,
        "New categories": 1000,
    }


def test_drift_overview_handles_empty_records():
    overview = _drift_overview([])

    assert list(overview.columns) == ["Feature", "Signal", "PSI", "New categories"]
    assert overview.empty


def test_quality_issue_overview_orders_by_worst_rate_and_skips_overall():
    records = [
        {
            "dataset": "current",
            "feature": "__overall__",
            "signal": "amber",
            "metrics": {},
            "suggestions": [],
        },
        {
            "dataset": "current",
            "feature": "age",
            "signal": "amber",
            "metrics": {
                "missing_rate": 0.02,
                "invalid_rate": 0.0,
                "iqr_outlier_rate": 0.05,
                "zscore_outlier_rate": 0.01,
            },
            "suggestions": ["Review outliers."],
        },
        {
            "dataset": "reference",
            "feature": "income",
            "signal": "red",
            "metrics": {
                "missing_rate": 0.2,
                "invalid_rate": 0.0,
                "iqr_outlier_rate": 0.0,
                "zscore_outlier_rate": 0.0,
            },
            "suggestions": ["Impute missing values."],
        },
    ]

    overview = _quality_issue_overview(records)

    assert list(overview["Feature"]) == ["income", "age"]
    assert overview.iloc[0]["Issue score"] == 0.2
