from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from services.storage import get_drift_results, get_quality_results, list_runs
from ui_pages.chrome import page_title, style_chart


def _compact_values(values: list, limit: int = 8) -> str:
    if not values:
        return "-"
    visible_values = [str(value) for value in values[:limit]]
    summary = ", ".join(visible_values)
    remaining_count = len(values) - len(visible_values)
    if remaining_count > 0:
        summary += f", +{remaining_count} more"
    return summary


def _quality_table(records: list[dict], dataset: str | None = None) -> pd.DataFrame:
    selected = [
        record
        for record in records
        if record["feature"] != "__overall__"
        and (dataset is None or record["dataset"] == dataset)
    ]
    return pd.DataFrame(
        [
            {
                "Dataset": record["dataset"].title(),
                "Feature": record["feature"],
                "Type": record["metrics"]["inferred_type"],
                "Signal": record["signal"].upper(),
                "Missing count": record["metrics"]["missing_count"],
                "Missing rate": record["metrics"]["missing_rate"],
                "Duplicate rows": record["metrics"]["duplicate_rows"],
                "IQR outliers": record["metrics"]["iqr_outlier_count"],
                "IQR outlier rate": record["metrics"]["iqr_outlier_rate"],
                "Z-score outliers": record["metrics"]["zscore_outlier_count"],
                "Z-score outlier rate": record["metrics"]["zscore_outlier_rate"],
                "Invalid values": record["metrics"]["invalid_count"],
                "Invalid rate": record["metrics"]["invalid_rate"],
                "Missingness group tests": "; ".join(
                    f"{test['group_column']}: p={test['p_value']:.4g}"
                    f"{'' if test['reliable'] else ' (sparse)'}"
                    for test in record["metrics"]["missingness_group_tests"]
                )
                or "-",
                "Unexpected categories": ", ".join(
                    record["metrics"]["unexpected_categories"]
                )
                or "-",
                "Suggested fixes": " ".join(record["suggestions"]),
            }
            for record in selected
        ]
    )


def _drift_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record["metrics"]
        row = {
            "Feature": record["feature"],
            "Type": record["feature_type"],
            "Signal": record["signal"].upper(),
            "Adjusted p (min)": metrics.get("minimum_adjusted_p_value"),
            "PSI": metrics.get("psi"),
            "New categories": _compact_values(metrics.get("new_categories", [])),
            "Test note": metrics.get("test_note"),
        }
        if record["feature_type"] == "numeric":
            row.update(
                {
                    "KS statistic": metrics.get("ks_statistic"),
                    "KS p": metrics.get("ks_p_value"),
                    "KS adjusted p": metrics.get("ks_adjusted_p_value"),
                    "Mann-Whitney U": metrics.get("mann_whitney_u"),
                    "MW p": metrics.get("mann_whitney_p_value"),
                    "MW adjusted p": metrics.get("mann_whitney_adjusted_p_value"),
                    "Wasserstein": metrics.get("wasserstein_distance"),
                }
            )
        else:
            row.update(
                {
                    "Chi-square": metrics.get("chi2_statistic"),
                    "Chi-square p": metrics.get("chi2_p_value"),
                    "Chi-square adjusted p": metrics.get(
                        "chi2_adjusted_p_value"
                    ),
                    "Jensen-Shannon": metrics.get("jensen_shannon_distance"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _drift_overview(records: list[dict], limit: int = 8) -> pd.DataFrame:
    columns = ["Feature", "Signal", "PSI", "New categories"]
    rows = [
        {
            "Feature": record["feature"],
            "Signal": record["signal"].upper(),
            "PSI": float(record["metrics"].get("psi") or 0.0),
            "New categories": int(record["metrics"].get("new_category_count") or 0),
        }
        for record in records
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["PSI", "New categories"], ascending=False)
        .head(limit)
    )


def _quality_issue_overview(records: list[dict], limit: int = 10) -> pd.DataFrame:
    columns = [
        "Dataset",
        "Feature",
        "Signal",
        "Issue score",
        "Missing rate",
        "Invalid rate",
        "IQR outlier rate",
        "Z-score outlier rate",
        "Suggested fixes",
    ]
    rows = []
    for record in records:
        if record["feature"] == "__overall__":
            continue
        metrics = record["metrics"]
        issue_score = max(
            float(metrics.get("missing_rate") or 0.0),
            float(metrics.get("invalid_rate") or 0.0),
            float(metrics.get("iqr_outlier_rate") or 0.0),
            float(metrics.get("zscore_outlier_rate") or 0.0),
        )
        rows.append(
            {
                "Dataset": record["dataset"].title(),
                "Feature": record["feature"],
                "Signal": record["signal"].upper(),
                "Issue score": issue_score,
                "Missing rate": float(metrics.get("missing_rate") or 0.0),
                "Invalid rate": float(metrics.get("invalid_rate") or 0.0),
                "IQR outlier rate": float(metrics.get("iqr_outlier_rate") or 0.0),
                "Z-score outlier rate": float(
                    metrics.get("zscore_outlier_rate") or 0.0
                ),
                "Suggested fixes": " ".join(record["suggestions"]),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("Issue score", ascending=False)
        .head(limit)
    )


def _signal_html(signal: str, title: str, subtitle: str) -> str:
    background = {
        "green": "#e3f5e9",
        "amber": "#fff2d5",
        "red": "#ffe4df",
    }.get(signal, "#edf1f4")
    color = {"green": "#176239", "amber": "#805900", "red": "#9d2f25"}.get(
        signal, "#4f6171"
    )
    return (
        f"<div style='background:{background};border-radius:16px;padding:1.15rem 1.3rem;"
        f"border:1px solid {color}22;min-height:112px'>"
        f"<div style='color:#52677a;font-size:.88rem'>{html.escape(title)}</div>"
        f"<div style='color:{color};font-size:1.65rem;font-weight:750;margin:.25rem 0'>"
        f"{html.escape(signal.upper())}</div>"
        f"<div style='color:#52677a;font-size:.86rem'>{html.escape(subtitle)}</div></div>"
    )


def _timeline_chart() -> None:
    runs = list_runs()[:12]
    if not runs:
        return
    rows = [
        {
            "Run": run["run_id"][:8],
            "Created": pd.to_datetime(run["created_at"], utc=True),
            "Stage": run["status"].replace("_", " ").title(),
        }
        for run in reversed(runs)
    ]
    figure = px.scatter(
        pd.DataFrame(rows),
        x="Created",
        y="Run",
        color="Stage",
        title="Recent analysis runs",
        color_discrete_map={
            "Created": "#91a6b8",
            "Quality Complete": "#e4ad3c",
            "Drift Complete": "#167d83",
        },
    )
    style_chart(figure, height=260)
    st.plotly_chart(figure, width="stretch")


def _key_changes_charts(
    current_quality: dict,
    reference_quality: dict,
    drift_records: list[dict],
) -> None:
    quality_delta = current_quality["quality_score"] - reference_quality["quality_score"]
    top_drift = _drift_overview(drift_records)
    highest_psi = top_drift.iloc[0] if not top_drift.empty else None

    st.subheader("Key changes")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric(
        "Current quality",
        f"{current_quality['quality_score']:.1f}%",
        f"{quality_delta:+.1f} pts vs reference",
    )
    kpi_columns[1].metric(
        "Top PSI",
        f"{highest_psi['PSI']:.3f}" if highest_psi is not None else "0.000",
        str(highest_psi["Feature"]) if highest_psi is not None else "No shared features",
    )
    kpi_columns[2].metric(
        "New categories",
        f"{int(top_drift['New categories'].sum()):,}" if not top_drift.empty else "0",
        "Across top drift features",
    )
    kpi_columns[3].metric(
        "Red drift features",
        f"{sum(record['signal'] == 'red' for record in drift_records):,}",
        "High-severity signals",
    )

    chart_columns = st.columns(2)
    quality_chart = pd.DataFrame(
        [
            {"Dataset": "Reference", "Quality score": reference_quality["quality_score"]},
            {"Dataset": "Current", "Quality score": current_quality["quality_score"]},
        ]
    )
    quality_figure = px.bar(
        quality_chart,
        x="Dataset",
        y="Quality score",
        color="Dataset",
        title="Quality score comparison",
        color_discrete_sequence=["#4f86c6", "#d95f59"],
        text="Quality score",
    )
    quality_figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    style_chart(quality_figure, height=320)
    quality_figure.update_layout(showlegend=False, margin=dict(l=0, r=16, t=48, b=28))
    quality_figure.update_yaxes(range=[0, 100])
    with chart_columns[0]:
        st.plotly_chart(quality_figure, width="stretch")

    if top_drift.empty:
        with chart_columns[1]:
            st.info("No shared features were available for drift testing.")
    else:
        drift_figure = px.bar(
            top_drift.sort_values("PSI"),
            x="PSI",
            y="Feature",
            color="Signal",
            orientation="h",
            title="Top drift features by PSI",
            color_discrete_map={
                "GREEN": "#2f8f5b",
                "AMBER": "#c8891a",
                "RED": "#d95f59",
            },
            hover_data={"New categories": ":,", "PSI": ":.3f"},
            text="PSI",
        )
        drift_figure.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        style_chart(drift_figure, height=320)
        drift_figure.update_layout(legend_title_text="Signal")
        with chart_columns[1]:
            st.plotly_chart(drift_figure, width="stretch")


def _render_quality_results(
    quality_records: list[dict],
    quality_table: pd.DataFrame,
) -> None:
    st.subheader("Quality results")
    overview = _quality_issue_overview(quality_records)
    current_rows = quality_table[quality_table["Dataset"] == "Current"]
    reference_rows = quality_table[quality_table["Dataset"] == "Reference"]

    highlight_tab, current_tab, reference_tab, raw_tab = st.tabs(
        ["Highlights", "Current data", "Reference data", "All fields"]
    )
    with highlight_tab:
        kpi_columns = st.columns(3)
        kpi_columns[0].metric(
            "Fields reviewed",
            f"{len(quality_table):,}",
        )
        kpi_columns[1].metric(
            "Fields needing attention",
            f"{len(quality_table[quality_table['Signal'] != 'GREEN']):,}",
        )
        kpi_columns[2].metric(
            "Invalid values",
            f"{int(quality_table['Invalid values'].sum()):,}"
            if not quality_table.empty
            else "0",
        )

        if overview.empty:
            st.success("No field-level quality issues were found.")
        else:
            issue_figure = px.bar(
                overview.sort_values("Issue score"),
                x="Issue score",
                y="Feature",
                color="Signal",
                orientation="h",
                title="Top quality issues by rate",
                color_discrete_map={
                    "GREEN": "#2f8f5b",
                    "AMBER": "#c8891a",
                    "RED": "#d95f59",
                },
                hover_data={
                    "Dataset": True,
                    "Missing rate": ":.2%",
                    "Invalid rate": ":.2%",
                    "IQR outlier rate": ":.2%",
                    "Z-score outlier rate": ":.2%",
                    "Issue score": ":.2%",
                },
                text="Issue score",
            )
            issue_figure.update_traces(
                texttemplate="%{text:.1%}", textposition="outside"
            )
            style_chart(issue_figure, height=360)
            issue_figure.update_layout(
                xaxis_tickformat=".0%",
                legend_title_text="Signal",
            )
            st.plotly_chart(issue_figure, width="stretch")
            st.dataframe(
                overview[
                    [
                        "Dataset",
                        "Feature",
                        "Signal",
                        "Missing rate",
                        "Invalid rate",
                        "Suggested fixes",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with current_tab:
        st.dataframe(current_rows, width="stretch", hide_index=True)
    with reference_tab:
        st.dataframe(reference_rows, width="stretch", hide_index=True)
    with raw_tab:
        st.dataframe(quality_table, width="stretch", hide_index=True)


def _render_drift_results(
    drift_records: list[dict],
    drift_table: pd.DataFrame,
) -> None:
    st.subheader("Statistical drift results")
    overview = _drift_overview(drift_records, limit=12)
    signal_counts = (
        drift_table["Signal"].value_counts().rename_axis("Signal").reset_index(name="Count")
        if not drift_table.empty
        else pd.DataFrame(columns=["Signal", "Count"])
    )
    new_category_rows = drift_table[
        drift_table["New categories"].astype(str) != "-"
    ] if not drift_table.empty else drift_table

    signal_tab, category_tab, raw_tab = st.tabs(
        ["Signal map", "New categories", "All tests"]
    )
    with signal_tab:
        kpi_columns = st.columns(3)
        kpi_columns[0].metric("Features tested", f"{len(drift_table):,}")
        kpi_columns[1].metric(
            "Red signals",
            f"{len(drift_table[drift_table['Signal'] == 'RED']):,}"
            if not drift_table.empty
            else "0",
        )
        kpi_columns[2].metric(
            "Watch signals",
            f"{len(drift_table[drift_table['Signal'] == 'AMBER']):,}"
            if not drift_table.empty
            else "0",
        )

        if drift_table.empty:
            st.info("No shared features were available for drift testing.")
        else:
            chart_columns = st.columns(2)
            signal_figure = px.pie(
                signal_counts,
                names="Signal",
                values="Count",
                title="Drift signal mix",
                color="Signal",
                color_discrete_map={
                    "GREEN": "#2f8f5b",
                    "AMBER": "#c8891a",
                    "RED": "#d95f59",
                },
                hole=0.45,
            )
            style_chart(signal_figure, height=330)
            with chart_columns[0]:
                st.plotly_chart(signal_figure, width="stretch")

            drift_figure = px.bar(
                overview.sort_values("PSI"),
                x="PSI",
                y="Feature",
                color="Signal",
                orientation="h",
                title="Highest PSI features",
                color_discrete_map={
                    "GREEN": "#2f8f5b",
                    "AMBER": "#c8891a",
                    "RED": "#d95f59",
                },
                hover_data={"New categories": ":,", "PSI": ":.3f"},
                text="PSI",
            )
            drift_figure.update_traces(
                texttemplate="%{text:.3f}", textposition="outside"
            )
            style_chart(drift_figure, height=330)
            drift_figure.update_layout(legend_title_text="Signal")
            with chart_columns[1]:
                st.plotly_chart(drift_figure, width="stretch")

    with category_tab:
        if new_category_rows.empty:
            st.success("No new categories were detected.")
        else:
            st.dataframe(
                new_category_rows[
                    ["Feature", "Type", "Signal", "PSI", "New categories"]
                ],
                width="stretch",
                hide_index=True,
            )
    with raw_tab:
        st.dataframe(drift_table, width="stretch", hide_index=True)


def _report_html(
    run: dict,
    quality_summary: dict,
    drift_summary: dict,
    quality_table: pd.DataFrame,
    drift_table: pd.DataFrame,
) -> str:
    current_quality = html.escape(quality_summary.get("signal", "unknown").upper())
    current_score = float(quality_summary.get("quality_score", 0))
    drift_signal = html.escape(drift_summary.get("signal", "unknown").upper())
    story = (
        f"Current data quality is {current_quality} ({current_score:.1f}%). "
        f"Overall drift is {drift_signal}; {int(drift_summary.get('drifted_features', 0))} "
        f"feature(s) have high-severity drift and "
        f"{int(drift_summary.get('watch_features', 0))} need review."
    )
    quality_config = run["quality_config"]
    drift_config = run["drift_config"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oberva analysis {html.escape(run['run_id'])}</title>
<style>
body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1250px;
margin:2rem auto;padding:0 1.2rem;color:#162638;background:#f4f7fa}}
header,section{{background:white;border:1px solid #e5ebef;border-radius:14px;
padding:1.2rem 1.5rem;margin:1rem 0;overflow:auto}}
h1,h2{{margin-top:0}}.meta{{color:#617286}}table{{border-collapse:collapse;width:100%;
font-size:.88rem}}th,td{{border:1px solid #e5ebef;padding:.45rem;text-align:left}}
th{{background:#edf3f6}}.story{{font-size:1.05rem}}
</style></head><body>
<header><h1>Oberva analysis report</h1>
<div class="meta">Run {html.escape(run['run_id'])} - {html.escape(run['created_at'])}</div>
<div class="meta">Reference: {html.escape(run['reference_name'])} -
Current: {html.escape(run['current_name'])}</div></header>
<section><h2>Summary</h2><p class="story">{html.escape(story)}</p>
<p>Quality thresholds: amber {_percent(quality_config.get('warning_rate', 0.01))},
red {_percent(quality_config.get('critical_rate', 0.05))}.</p>
<p>Drift thresholds: adjusted p &lt; {drift_config.get('alpha', 0.05):.3f},
PSI amber {drift_config.get('psi_warning', 0.1):.2f},
PSI red {drift_config.get('psi_critical', 0.25):.2f}.</p></section>
<section><h2>Quality results</h2>
{quality_table.to_html(index=False, escape=True, border=0, float_format=lambda value: f'{value:.4f}')}
</section>
<section><h2>Drift results</h2>
{drift_table.to_html(index=False, escape=True, border=0, float_format=lambda value: f'{value:.4f}')}
</section></body></html>"""


def _percent(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def render(run: dict) -> None:
    page_title(
        "Analysis dashboard",
        "Monitor quality, drift, alerts, and investigation-ready details for the selected run.",
    )
    st.caption(
        f"Run `{run['run_id']}` - {run['created_at']} - "
        f"{run['reference_name']} -> {run['current_name']}"
    )
    quality_records = get_quality_results(run["run_id"])
    drift_records = get_drift_results(run["run_id"])
    current_quality_record = next(
        record
        for record in quality_records
        if record["dataset"] == "current" and record["feature"] == "__overall__"
    )
    reference_quality_record = next(
        record
        for record in quality_records
        if record["dataset"] == "reference" and record["feature"] == "__overall__"
    )
    current_quality = current_quality_record["metrics"]
    reference_quality = reference_quality_record["metrics"]
    drift_config = run["drift_config"]
    drift_signal = drift_config.get("signal", "green")
    alerts = []
    if current_quality["signal"] == "red":
        alerts.append(
            f"Current data quality is critical ({current_quality['quality_score']:.1f}% score)."
        )
    elif current_quality["signal"] == "amber":
        alerts.append("Current data quality has a watch-level signal.")
    if drift_signal == "red":
        alerts.append(
            f"{drift_config.get('drifted_features', 0)} feature(s) have high-severity drift."
        )
    elif drift_signal == "amber":
        alerts.append(
            f"{drift_config.get('watch_features', 0)} feature(s) need drift review."
        )
    if any(
        record["metrics"].get("new_category_count", 0) > 0
        for record in drift_records
    ):
        alerts.append("New categories were found in one or more current features.")

    quality_column, drift_column = st.columns(2)
    with quality_column:
        st.markdown(
            _signal_html(
                current_quality["signal"],
                "Current data quality",
                f"{current_quality['quality_score']:.1f}% - "
                f"{current_quality['duplicate_rows']:,} duplicate row(s)",
            ),
            unsafe_allow_html=True,
        )
    with drift_column:
        st.markdown(
            _signal_html(
                drift_signal,
                "Feature drift",
                f"{drift_config.get('drifted_features', 0)} alert(s) - "
                f"{drift_config.get('watch_features', 0)} watch item(s)",
            ),
            unsafe_allow_html=True,
        )
    if current_quality["signal"] == "red" or drift_signal == "red":
        st.error("ALERT - " + " ".join(alerts))
    elif alerts:
        st.warning("WATCH - " + " ".join(alerts))
    else:
        st.success("No high-severity quality or drift alerts in this run.")

    _key_changes_charts(current_quality, reference_quality, drift_records)
    st.caption(
        "Statistical signals identify changes for investigation; they do not prove "
        "a business cause."
    )

    quality_table = _quality_table(quality_records)
    drift_table = _drift_table(drift_records)
    _render_quality_results(quality_records, quality_table)
    _render_drift_results(drift_records, drift_table)
    _timeline_chart()

    report = _report_html(
        run,
        current_quality,
        drift_config,
        quality_table,
        drift_table,
    )
    st.download_button(
        "Download self-contained HTML report",
        data=report.encode("utf-8"),
        file_name=f"oberva-report-{run['run_id']}.html",
        mime="text/html",
        type="primary",
    )
