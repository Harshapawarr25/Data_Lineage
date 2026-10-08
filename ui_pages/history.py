from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_pages.chrome import page_title, style_chart


def render(runs: list[dict]) -> None:
    page_title(
        "Run timeline",
        "Review saved analysis runs, workflow stages, row counts, and retained local artifacts.",
    )
    if not runs:
        st.info("No analysis runs yet. Upload datasets in Intake to get started.")
        return
    rows = [
        {
            "Run ID": run["run_id"],
            "Created": pd.to_datetime(run["created_at"], utc=True),
            "Stage": run["status"].replace("_", " ").title(),
            "Reference": run["reference_name"],
            "Current": run["current_name"],
            "Rows (reference)": run["reference_rows"],
            "Rows (current)": run["current_rows"],
            "Shared columns": len(run["shared_columns"]),
        }
        for run in runs
    ]
    table = pd.DataFrame(rows)
    figure = px.scatter(
        table,
        x="Created",
        y="Run ID",
        color="Stage",
        hover_data=["Reference", "Current", "Rows (reference)", "Rows (current)"],
        color_discrete_map={
            "Created": "#91a6b8",
            "Quality Complete": "#e4ad3c",
            "Drift Complete": "#167d83",
        },
        title="Persisted analysis runs",
    )
    style_chart(figure, height=max(280, min(620, 40 * len(rows))))
    st.plotly_chart(figure, width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Run inputs and results are stored under the local `data/` directory. "
        "Exclude this directory from source control."
    )
