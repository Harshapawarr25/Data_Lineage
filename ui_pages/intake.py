from __future__ import annotations

import io
import sqlite3

import pandas as pd
import streamlit as st

from services.analysis import compare_schema, infer_column_types
from services.storage import create_run
from ui_pages.chrome import page_title


def _read_csv(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> pd.DataFrame | None:
    try:
        data = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        st.error(f"Could not read {uploaded_file.name}: {error}")
        return None
    if data.shape[1] == 0:
        st.error(f"{uploaded_file.name} does not contain any columns.")
        return None
    return data


def render() -> None:
    page_title(
        "Dataset intake",
        "Compare a trusted reference with the latest dataset, review schema changes, and create a persistent analysis run.",
    )
    reference_column, current_column = st.columns(2)
    with reference_column:
        st.subheader("Reference dataset")
        reference_file = st.file_uploader(
            "Choose the baseline CSV",
            type=["csv"],
            key="intake_reference",
        )
        reference = _read_csv(reference_file) if reference_file is not None else None
    with current_column:
        st.subheader("Current dataset")
        current_file = st.file_uploader(
            "Choose the current CSV",
            type=["csv"],
            key="intake_current",
        )
        current = _read_csv(current_file) if current_file is not None else None

    if reference is None or current is None:
        st.info("Upload both CSV files to review the schema and create a run.")
        return

    st.divider()
    schema = compare_schema(reference, current)
    reference_metric, current_metric, common_metric = st.columns(3)
    reference_metric.metric(
        "Reference dataset",
        f"{len(reference):,} rows - {len(reference.columns):,} columns",
    )
    current_metric.metric(
        "Current dataset",
        f"{len(current):,} rows - {len(current.columns):,} columns",
    )
    common_metric.metric("Shared columns", len(schema["shared"]))

    if schema["missing_from_current"] or schema["extra_in_current"]:
        st.warning(
            "The schemas differ. Comparisons will use shared columns; missing and "
            "extra fields are retained in the run record."
        )
        missing_column, extra_column = st.columns(2)
        with missing_column:
            st.markdown("**Missing from current**")
            st.write(schema["missing_from_current"] or "None")
        with extra_column:
            st.markdown("**Extra in current**")
            st.write(schema["extra_in_current"] or "None")
    else:
        st.success("The reference and current datasets have matching columns.")

    if not schema["shared"]:
        st.error("At least one shared column is required to create an analysis run.")
        return

    column_types = infer_column_types(reference, current)
    type_table = pd.DataFrame(
        [
            {
                "Column": column,
                "Type": column_types[column],
                "In Reference": column in reference.columns,
                "In Current": column in current.columns,
            }
            for column in column_types
        ]
    )
    st.subheader("Inferred feature types")
    st.caption(
        "Two-valued numeric columns (for example, churned 0/1) are treated as "
        "categorical; other numeric columns remain numeric."
    )
    st.dataframe(type_table, width="stretch", hide_index=True)

    preview_reference, preview_current = st.columns(2)
    with preview_reference:
        st.caption(f"Reference preview - {reference_file.name}")
        st.dataframe(reference.head(8), width="stretch", hide_index=True)
    with preview_current:
        st.caption(f"Current preview - {current_file.name}")
        st.dataframe(current.head(8), width="stretch", hide_index=True)

    st.divider()
    st.info(
        "Creating a run saves both CSV files and the schema snapshot to the local "
        "`data/` directory. Files are not sent to an external service."
    )
    if st.button("Run Analysis - Create run", type="primary", width="stretch"):
        try:
            run = create_run(
                reference_file.getvalue(),
                current_file.getvalue(),
                reference_file.name,
                current_file.name,
                reference,
                current,
                schema["shared"],
                schema["missing_from_current"],
                schema["extra_in_current"],
                column_types,
            )
        except (OSError, sqlite3.Error, ValueError) as error:
            st.error(f"Could not create the analysis run: {error}")
        else:
            st.session_state["run_selector_version"] += 1
            st.success(f"Created run `{run['run_id']}`. Continue to Data Quality.")
            st.rerun()
