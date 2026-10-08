import pandas as pd

from services.analysis import (
    compare_schema,
    infer_column_types,
    recommended_nonnegative_columns,
)


def test_schema_comparison_reports_missing_and_extra_columns():
    reference = pd.DataFrame({"amount": [1, 2], "region": ["N", "S"]})
    current = pd.DataFrame({"amount": [3, 4], "segment": ["a", "b"]})

    assert compare_schema(reference, current) == {
        "shared": ["amount"],
        "missing_from_current": ["region"],
        "extra_in_current": ["segment"],
    }


def test_binary_numeric_fields_are_categorical():
    reference = pd.DataFrame({"churned": [0, 1, 0], "charges": [1.5, 2.5, 3.5]})
    current = pd.DataFrame({"churned": [1, 1, 0], "charges": [4.5, 5.5, 6.5]})

    assert infer_column_types(reference, current) == {
        "churned": "categorical",
        "charges": "numeric",
    }


def test_nonnegative_suggestions_are_conservative_and_name_based():
    inferred = {
        "monthly_charges": "numeric",
        "record_count": "numeric",
        "profit": "numeric",
        "region": "categorical",
    }

    assert recommended_nonnegative_columns(inferred) == [
        "monthly_charges",
        "record_count",
    ]
