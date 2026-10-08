from __future__ import annotations

import pandas as pd
import streamlit as st

from services.storage import get_drift_results, get_quality_results
from ui_pages.chrome import page_title


def render(run: dict) -> None:
    page_title(
        "Root-cause investigation",
        "Prioritize features where drift and quality signals overlap. These indicators guide review; they do not prove a cause.",
    )
    quality = get_quality_results(run["run_id"])
    drift = get_drift_results(run["run_id"])
    current_quality = [
        record
        for record in quality
        if record["dataset"] == "current" and record["feature"] != "__overall__"
    ]
    by_feature = {record["feature"]: record for record in current_quality}
    rows = []
    for result in drift:
        metrics = result["metrics"]
        quality_result = by_feature.get(result["feature"])
        quality_metrics = quality_result["metrics"] if quality_result else {}
        rows.append(
            {
                "Feature": result["feature"],
                "Drift signal": result["signal"].upper(),
                "Drift type": result["feature_type"],
                "PSI": metrics.get("psi"),
                "Adjusted p": metrics.get("minimum_adjusted_p_value"),
                "New categories": ", ".join(metrics.get("new_categories", []))
                or "-",
                "Missing %": 100 * quality_metrics.get("missing_rate", 0.0),
                "Invalid values": quality_metrics.get("invalid_count", 0),
                "IQR outliers": quality_metrics.get("iqr_outlier_count", 0),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("There are no shared features to investigate.")
    else:
        signal_rank = {"RED": 2, "AMBER": 1, "GREEN": 0}
        table["_signal_rank"] = table["Drift signal"].map(signal_rank)
        table = table.sort_values(
            ["_signal_rank", "PSI"], ascending=[False, False], na_position="last"
        ).drop(columns=["_signal_rank"])
        st.dataframe(table, width="stretch", hide_index=True)
        st.subheader("Suggested quality follow-up")
        for result in sorted(
            current_quality,
            key=lambda row: {"red": 2, "amber": 1, "green": 0}[row["signal"]],
            reverse=True,
        ):
            if result["signal"] == "green":
                continue
            with st.expander(
                f"{result['feature']} - {result['signal'].upper()}"
            ):
                for suggestion in result["suggestions"]:
                    st.markdown(f"- {suggestion}")
