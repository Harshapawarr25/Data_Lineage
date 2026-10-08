from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

CHART_FONT_COLOR = "#213547"
CHART_GRID_COLOR = "#d8e1e8"


def page_title(title: str, description: str | None = None) -> None:
    description_html = (
        f"<p>{html.escape(description)}</p>" if description else ""
    )
    st.markdown(
        f"""
        <div class="page-title">
          <span>WORKSPACE</span>
          <h2>{html.escape(title)}</h2>
          {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(label: str, tone: str = "neutral") -> str:
    return (
        f"<span class='pill pill-{html.escape(tone)}'>"
        f"{html.escape(label)}</span>"
    )


def style_chart(figure: go.Figure, height: int) -> None:
    figure.update_layout(
        height=height,
        font=dict(color=CHART_FONT_COLOR),
        title_font=dict(color=CHART_FONT_COLOR, size=16),
        legend=dict(
            font=dict(color=CHART_FONT_COLOR),
            title_font=dict(color=CHART_FONT_COLOR),
        ),
        margin=dict(l=0, r=16, t=48, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(
        color=CHART_FONT_COLOR,
        title_font=dict(color=CHART_FONT_COLOR),
        tickfont=dict(color=CHART_FONT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
    )
    figure.update_yaxes(
        color=CHART_FONT_COLOR,
        title_font=dict(color=CHART_FONT_COLOR),
        tickfont=dict(color=CHART_FONT_COLOR),
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
    )
    figure.update_traces(textfont=dict(color=CHART_FONT_COLOR))
