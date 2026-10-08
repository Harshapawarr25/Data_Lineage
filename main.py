import streamlit as st

from services.storage import get_run, initialize_store, list_runs
from ui_pages import (
    dashboard,
    documents,
    drift_page,
    history,
    intake,
    quality_page,
    root_cause,
)


APP_NAME = "Oberva"


st.set_page_config(
    page_title=f"{APP_NAME} | Data & Document Intelligence",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172333;
        --muted: #5d6f82;
        --faint: #8ea0b2;
        --surface: #ffffff;
        --surface-soft: #f7fafc;
        --canvas: #eef3f7;
        --line: #dbe5ec;
        --accent: #167d83;
        --accent-strong: #0f6067;
        --blue: #4f86c6;
        --red: #d95f59;
        --amber: #c8891a;
        --green: #2f8f5b;
    }
    .stApp {
        background:
          linear-gradient(180deg, rgba(255,255,255,.72) 0, rgba(238,243,247,.92) 18rem),
          var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(238, 243, 247, 0.86);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(219,229,236,.65);
    }
    [data-testid="stSidebar"] {
        background: #102333;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #edf5f7; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #edf5f7; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 10px;
        padding: .25rem .35rem;
        margin: .08rem 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.08);
    }
    .brand-lockup {
        padding: .65rem 0 1rem;
        border-bottom: 1px solid rgba(255,255,255,.12);
        margin-bottom: 1rem;
    }
    .brand-mark {
        display: inline-grid;
        place-items: center;
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 12px;
        margin-right: .65rem;
        background: linear-gradient(135deg, #2ca0a7, #4f86c6);
        color: #ffffff;
        font-weight: 800;
    }
    .brand-name {
        display: inline-block;
        vertical-align: middle;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0;
    }
    .brand-subtitle {
        color: #a9bac7;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08rem;
        margin-top: .45rem;
    }
    .hero {
        padding: 1.45rem 1.7rem;
        border: 1px solid rgba(255,255,255,.62);
        border-radius: 16px;
        color: #ffffff;
        background:
          linear-gradient(120deg, rgba(16,35,51,.96), rgba(22,125,131,.92)),
          radial-gradient(circle at 90% 20%, rgba(79,134,198,.45), transparent 30%);
        margin: .25rem 0 1.25rem;
        box-shadow: 0 18px 42px rgba(16, 35, 51, .14);
    }
    .hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: 2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: .5rem 0 0;
        color: #d9eef0;
        max-width: 58rem;
    }
    .page-title {
        margin: .2rem 0 1rem;
        padding-bottom: .45rem;
        border-bottom: 1px solid var(--line);
    }
    .page-title span {
        color: var(--accent-strong);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .08rem;
    }
    .page-title h2 {
        margin: .15rem 0 .2rem;
        color: var(--ink);
        font-size: 1.55rem;
        line-height: 1.2;
    }
    .page-title p {
        margin: 0;
        color: var(--muted);
        max-width: 62rem;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        padding: .95rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(23, 35, 51, .045);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stMetricDelta"] { color: var(--accent-strong); }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 12px;
        background: var(--surface);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--surface-soft);
        border: 1px solid var(--line);
        border-bottom: 0;
        border-radius: 10px 10px 0 0;
        color: var(--muted);
        font-weight: 700;
    }
    .stDataFrame, [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 9px;
        font-weight: 700;
        border: 1px solid var(--line);
        background: #ffffff;
        color: var(--ink);
    }
    .stButton button p, .stDownloadButton button p { color: inherit; }
    .stButton button:hover, .stDownloadButton button:hover {
        border-color: var(--accent);
        color: var(--accent-strong);
    }
    .stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
    }
    .stButton button[kind="primary"]:hover,
    .stDownloadButton button[kind="primary"]:hover {
        background: var(--accent-strong);
        border-color: var(--accent-strong);
        color: #ffffff;
    }
    .pill {
        display: inline-block;
        padding: .28rem .55rem;
        border-radius: 999px;
        margin: .12rem .18rem .12rem 0;
        font-size: .82rem;
        font-weight: 700;
        border: 1px solid transparent;
    }
    .pill-green { background: #e5f5eb; color: #236844; border-color: #bce1ca; }
    .pill-amber { background: #fff1ce; color: #805900; border-color: #f1d28b; }
    .pill-red { background: #ffe4df; color: #9d2f25; border-color: #efb6ae; }
    .pill-neutral { background: #eef3f7; color: #405569; border-color: #d8e3ea; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_store()
if "run_selector_version" not in st.session_state:
    st.session_state["run_selector_version"] = 0

runs = list_runs()
st.sidebar.markdown(
    f"""
    <div class="brand-lockup">
      <span class="brand-mark">O</span>
      <span class="brand-name">{APP_NAME}</span>
      <div class="brand-subtitle">DATA & DOCUMENT INTELLIGENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if runs:
    run_ids = [run["run_id"] for run in runs]
    options = ["__latest__", *run_ids]

    def format_run(option: str) -> str:
        if option == "__latest__":
            latest = runs[0]
            return f"Latest - {latest['run_id'][:8]} - {latest['status'].replace('_', ' ')}"
        selected = next(item for item in runs if item["run_id"] == option)
        return (
            f"{selected['created_at'][:19]} - {selected['run_id'][:8]} - "
            f"{selected['status'].replace('_', ' ')}"
        )

    selected_id = st.sidebar.selectbox(
        "Analysis run",
        options,
        format_func=format_run,
        key=f"active-run-{st.session_state['run_selector_version']}",
    )
    active_run = get_run(runs[0]["run_id"] if selected_id == "__latest__" else selected_id)
else:
    active_run = None
    st.sidebar.info("No saved runs yet.")

status = active_run["status"] if active_run else None
page_options = ["Intake"]
if active_run:
    page_options.append("Data Quality")
if status in ("quality_complete", "drift_complete"):
    page_options.append("Drift Detection")
if status == "drift_complete":
    page_options.extend(["Root Cause", "Dashboard"])
page_options.extend(["Project Library", "Data Lineage"])

default_page = {
    None: "Intake",
    "created": "Data Quality",
    "quality_complete": "Drift Detection",
    "drift_complete": "Dashboard",
}.get(status, "Intake")
if default_page not in page_options:
    default_page = page_options[0]
workspace_key = f"workspace-{active_run['run_id'] if active_run else 'new'}-{status}"
selected_page = st.sidebar.radio(
    "Workspace",
    page_options,
    index=page_options.index(default_page),
    key=workspace_key,
    label_visibility="collapsed",
)
st.sidebar.divider()
if active_run:
    completed = {
        "created": "Run created",
        "quality_complete": "Run created - Quality complete",
        "drift_complete": "Run created - Quality complete - Drift complete",
    }[status]
    st.sidebar.caption(f"WORKFLOW - {completed}")
    if status == "created":
        st.sidebar.caption("Next: complete Data Quality.")
    elif status == "quality_complete":
        st.sidebar.caption("Next: complete Drift Detection.")
    elif status == "drift_complete":
        st.sidebar.caption("Dashboard and investigation unlocked.")
st.sidebar.caption("Run files and results are saved locally under `data/`.")

st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_NAME}</h1>
      <p>Audit-ready data quality, drift detection, and project context in a clean local workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_page == "Intake":
    intake.render()
elif selected_page == "Data Quality":
    if active_run is not None:
        quality_page.render(active_run)
elif selected_page == "Drift Detection":
    if active_run is not None:
        drift_page.render(active_run)
elif selected_page == "Root Cause":
    if active_run is not None:
        root_cause.render(active_run)
elif selected_page == "Dashboard":
    if active_run is not None:
        dashboard.render(active_run)
elif selected_page == "Project Library":
    documents.render()
elif selected_page == "Data Lineage":
    history.render(runs)
