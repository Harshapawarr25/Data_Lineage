from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from services.drift import analyze_drift
from services.storage import (
    get_drift_results,
    get_run,
    load_run_frames,
    save_drift_results,
)
from ui_pages.chrome import page_title, pill


def _compact_values(values: list, limit: int = 8) -> str:
    if not values:
        return "-"
    visible_values = [str(value) for value in values[:limit]]
    suffix = len(values) - len(visible_values)
    summary = ", ".join(visible_values)
    if suffix > 0:
        summary += f", +{suffix} more"
    return summary


def _result_table(results: list[dict]) -> pd.DataFrame:
    records = []
    for result in results:
        metrics = result["metrics"]
        common = {
            "Feature": result["feature"],
            "Type": result["feature_type"],
            "Signal": result["signal"].upper(),
            "PSI": metrics.get("psi"),
            "Minimum adjusted p": metrics.get("minimum_adjusted_p_value"),
            "New categories": _compact_values(metrics.get("new_categories", [])),
            "Test note": metrics.get("test_note"),
        }
        if result["feature_type"] == "numeric":
            common.update(
                {
                    "KS statistic": metrics.get("ks_statistic"),
                    "KS p": metrics.get("ks_p_value"),
                    "KS adjusted p": metrics.get("ks_adjusted_p_value"),
                    "Mann-Whitney U": metrics.get("mann_whitney_u"),
                    "Mann-Whitney p": metrics.get("mann_whitney_p_value"),
                    "MW adjusted p": metrics.get("mann_whitney_adjusted_p_value"),
                    "Wasserstein": metrics.get("wasserstein_distance"),
                }
            )
        else:
            common.update(
                {
                    "Chi-square statistic": metrics.get("chi2_statistic"),
                    "Chi-square p": metrics.get("chi2_p_value"),
                    "Chi-square adjusted p": metrics.get("chi2_adjusted_p_value"),
                    "Jensen-Shannon": metrics.get("jensen_shannon_distance"),
                }
            )
        records.append(common)
    return pd.DataFrame(records)


def render(run: dict) -> None:
    page_title(
        "Drift detection",
        "Compare shared features with statistical tests, PSI thresholds, and new-category detection.",
    )
    if run["status"] == "quality_complete":
        reference, current = load_run_frames(run)
        with st.form(f"drift-form-{run['run_id']}"):
            alpha = st.slider(
                "Adjusted p-value significance threshold",
                0.001,
                0.20,
                0.05,
                0.001,
                format="%.3f",
                help="Statistical significance alone produces a watch signal; PSI adds an effect-size check.",
            )
            psi_warning = st.slider("Amber PSI threshold", 0.0, 0.5, 0.1, 0.01)
            psi_critical = st.slider("Red PSI threshold", 0.01, 1.0, 0.25, 0.01)
            if psi_critical <= psi_warning:
                st.warning("The red PSI threshold must exceed the amber threshold.")
            submitted = st.form_submit_button(
                "Run drift checks", type="primary", width="stretch"
            )
        if submitted:
            if psi_critical <= psi_warning:
                st.error("Set the red PSI threshold higher than the amber threshold.")
                return
            try:
                results, summary = analyze_drift(
                    reference,
                    current,
                    run["column_types"],
                    alpha,
                    psi_warning,
                    psi_critical,
                )
                save_drift_results(
                    run["run_id"],
                    results,
                    summary,
                    {
                        "alpha": alpha,
                        "psi_warning": psi_warning,
                        "psi_critical": psi_critical,
                    },
                )
            except (OSError, sqlite3.Error, ValueError) as error:
                st.error(f"Could not save drift results: {error}")
            else:
                st.success("Drift results saved. The Dashboard is now unlocked.")
                st.rerun()
        return

    records = get_drift_results(run["run_id"])
    run = get_run(run["run_id"]) or run
    config = run["drift_config"]
    red_count = sum(record["signal"] == "red" for record in records)
    amber_count = sum(record["signal"] == "amber" for record in records)
    columns = st.columns(4)
    columns[0].metric("Overall signal", config.get("signal", "green").upper())
    columns[1].metric("Features compared", len(records))
    columns[2].metric("Alert features", red_count)
    columns[3].metric("Watch features", amber_count)
    st.caption(
        f"Adjusted p-value threshold {config.get('alpha', 0.05):.3f} - "
        f"PSI thresholds {config.get('psi_warning', 0.1):.2f} / "
        f"{config.get('psi_critical', 0.25):.2f} - "
        f"{config.get('benjamini_hochberg_test_count', 0)} tests in correction family"
    )
    st.markdown("**Feature signal strip**")
    if records:
        st.markdown(
            " ".join(
                pill(f"{result['feature']} - {result['signal'].upper()}", result["signal"])
                for result in records
            ),
            unsafe_allow_html=True,
        )
    table = _result_table(records)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download drift results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{run['run_id']}-drift.csv",
        mime="text/csv",
    )
