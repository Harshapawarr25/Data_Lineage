from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from services.quality import analyze_quality
from services.storage import (
    get_quality_results,
    load_run_frames,
    save_quality_results,
)
from ui_pages.chrome import page_title


def _format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render(run: dict) -> None:
    page_title(
        "Data quality",
        "Tune thresholds, validate field-level quality, and review fixes for both datasets.",
    )
    st.caption(f"Run `{run['run_id']}` - {run['status'].replace('_', ' ').title()}")
    if run["status"] == "created":
        reference, current = load_run_frames(run)
        numeric_columns = [
            column
            for column in run["shared_columns"]
            if run["column_types"].get(column) == "numeric"
        ]
        with st.form(f"quality-form-{run['run_id']}"):
            st.subheader("Quality signal thresholds")
            warning_percent = st.slider(
                "Amber starts at an anomaly rate",
                0.0,
                20.0,
                1.0,
                0.5,
                format="%.1f%%",
            )
            critical_percent = st.slider(
                "Red starts at an anomaly rate",
                warning_percent + 0.5,
                50.0,
                max(5.0, warning_percent + 0.5),
                0.5,
                format="%.1f%%",
            )
            warning_rate = warning_percent / 100
            critical_rate = critical_percent / 100
            if critical_rate <= warning_rate:
                st.warning("The red threshold must be higher than the amber threshold.")

            st.subheader("Allowed numeric ranges")
            st.caption(
                "Likely non-negative fields are prefilled. Edit or clear bounds to "
                "match your domain; these rules apply to both datasets."
            )
            current_bounds = run.get("quality_config", {}).get("bounds", {})
            recommended_columns = run.get("quality_config", {}).get(
                "recommended_nonnegative"
            )
            if recommended_columns is None:
                from services.analysis import recommended_nonnegative_columns

                recommended_columns = recommended_nonnegative_columns(
                    run["column_types"]
                )
            constraints = pd.DataFrame(
                [
                    {
                        "Column": column,
                        "Minimum": current_bounds.get(column, {}).get(
                            "minimum",
                            0.0
                            if column in recommended_columns
                            else np.nan,
                        ),
                        "Maximum": current_bounds.get(column, {}).get(
                            "maximum", np.nan
                        ),
                    }
                    for column in numeric_columns
                ],
                columns=["Column", "Minimum", "Maximum"],
            )
            edited_constraints = st.data_editor(
                constraints,
                width="stretch",
                hide_index=True,
                disabled=["Column"],
                num_rows="fixed",
                column_config={
                    "Minimum": st.column_config.NumberColumn(
                        "Minimum", help="Leave blank for no lower bound."
                    ),
                    "Maximum": st.column_config.NumberColumn(
                        "Maximum", help="Leave blank for no upper bound."
                    ),
                },
                key=f"quality-bounds-{run['run_id']}",
            )
            submitted = st.form_submit_button(
                "Run quality checks", type="primary", width="stretch"
            )
        if submitted:
            if critical_rate <= warning_rate:
                st.error("Set the red threshold higher than the amber threshold.")
                return
            bounds = {}
            for row in edited_constraints.to_dict(orient="records"):
                minimum = row["Minimum"]
                maximum = row["Maximum"]
                minimum = None if pd.isna(minimum) else float(minimum)
                maximum = None if pd.isna(maximum) else float(maximum)
                if minimum is not None and maximum is not None and minimum > maximum:
                    st.error(f"Minimum exceeds maximum for {row['Column']}.")
                    return
                if minimum is not None or maximum is not None:
                    bounds[row["Column"]] = {
                        "minimum": minimum,
                        "maximum": maximum,
                    }
            try:
                results, summaries = analyze_quality(
                    reference,
                    current,
                    run["column_types"],
                    bounds,
                    warning_rate,
                    critical_rate,
                )
                save_quality_results(
                    run["run_id"],
                    results,
                    summaries,
                    {
                        "warning_rate": warning_rate,
                        "critical_rate": critical_rate,
                        "bounds": bounds,
                        "recommended_nonnegative": recommended_columns,
                    },
                )
            except (OSError, sqlite3.Error, ValueError) as error:
                st.error(f"Could not save quality results: {error}")
            else:
                st.success("Quality results saved. Drift Detection is now unlocked.")
                st.rerun()
        return

    records = get_quality_results(run["run_id"])
    if not records:
        st.info("No saved quality results are available for this run.")
        return
    overall = {
        result["dataset"]: result
        for result in records
        if result["feature"] == "__overall__"
    }
    current_summary = overall.get("current", {}).get("metrics", {})
    reference_summary = overall.get("reference", {}).get("metrics", {})
    metrics = st.columns(4)
    metrics[0].metric(
        "Current quality score",
        f"{current_summary.get('quality_score', 0):.1f}%",
        delta=(
            f"{current_summary.get('quality_score', 0) - reference_summary.get('quality_score', 0):+.1f} pts vs reference"
            if reference_summary
            else None
        ),
    )
    metrics[1].metric(
        "Current quality signal", current_summary.get("signal", "-").upper()
    )
    metrics[2].metric(
        "Current duplicate rows",
        f"{current_summary.get('duplicate_rows', 0):,}",
    )
    metrics[3].metric(
        "Current missing cells",
        f"{current_summary.get('missing_cells', 0):,}",
    )
    st.caption(
        "The score combines missing cells, duplicate rows, invalid values, "
        "unexpected categories, and numeric outlier rates."
    )
    view = st.radio(
        "Dataset",
        ["Current", "Reference"],
        horizontal=True,
        key=f"quality-dataset-{run['run_id']}",
    ).lower()
    rows = [
        result
        for result in records
        if result["dataset"] == view and result["feature"] != "__overall__"
    ]
    table = pd.DataFrame(
        [
            {
                "Feature": result["feature"],
                "Type": result["metrics"]["inferred_type"],
                "Signal": result["signal"].upper(),
                "Missing %": _format_percent(result["metrics"]["missing_rate"]),
                "IQR outliers": result["metrics"]["iqr_outlier_count"],
                "Z-score outliers": result["metrics"]["zscore_outlier_count"],
                "Invalid values": result["metrics"]["invalid_count"],
                "Unexpected categories": ", ".join(
                    result["metrics"]["unexpected_categories"]
                )
                or "-",
                "Group missingness p": ", ".join(
                    f"{test['group_column']}: {test['p_value']:.3g}"
                    for test in result["metrics"]["missingness_group_tests"]
                )
                or "-",
            }
            for result in rows
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)
    with st.expander("Suggested fixes by feature"):
        for result in rows:
            st.markdown(f"**{result['feature']} - {result['signal'].upper()}**")
            for suggestion in result["suggestions"]:
                st.markdown(f"- {suggestion}")
    st.download_button(
        "Download quality results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{run['run_id']}-quality-{view}.csv",
        mime="text/csv",
    )
