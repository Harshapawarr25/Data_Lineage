import sqlite3

import pandas as pd
import pytest

from services.analysis import compare_schema, infer_column_types
from services.storage import (
    create_run,
    get_drift_results,
    get_quality_results,
    get_run,
    list_runs,
    load_run_frames,
    save_drift_results,
    save_quality_results,
)
import services.storage as storage


def _run(tmp_path):
    reference = pd.DataFrame(
        {"churned": [0, 1, 0], "region": ["N", "S", "N"], "charge": [10, 20, 30]}
    )
    current = pd.DataFrame(
        {"churned": [1, 1, 0], "region": ["C", "S", "N"], "charge": [11, 25, 35]}
    )
    schema = compare_schema(reference, current)
    return create_run(
        reference.to_csv(index=False).encode(),
        current.to_csv(index=False).encode(),
        "reference.csv",
        "current.csv",
        reference,
        current,
        schema["shared"],
        schema["missing_from_current"],
        schema["extra_in_current"],
        infer_column_types(reference, current),
        tmp_path,
    )


def test_run_files_results_and_workflow_status_persist(tmp_path):
    run = _run(tmp_path)
    reference, current = load_run_frames(run)
    assert len(reference) == 3
    assert len(current) == 3
    assert get_run(run["run_id"], tmp_path)["status"] == "created"

    with pytest.raises(ValueError, match="completed quality"):
        save_drift_results(run["run_id"], [], {}, {}, tmp_path)

    quality_rows = [
        {
            "dataset": dataset,
            "feature": "charge",
            "signal": "green",
            "metrics": {"missing_count": 0},
            "suggestions": [],
        }
        for dataset in ("reference", "current")
    ]
    summaries = {
        dataset: {"quality_score": 100, "signal": "green"}
        for dataset in ("reference", "current")
    }
    save_quality_results(run["run_id"], quality_rows, summaries, {}, tmp_path)
    assert get_run(run["run_id"], tmp_path)["status"] == "quality_complete"
    assert len(get_quality_results(run["run_id"], tmp_path)) == 4

    drift_row = {
        "feature": "charge",
        "feature_type": "numeric",
        "signal": "green",
        "metrics": {"psi": 0.0},
    }
    save_drift_results(run["run_id"], [drift_row], {"signal": "green"}, {}, tmp_path)
    persisted = get_run(run["run_id"], tmp_path)
    assert persisted["status"] == "drift_complete"
    assert persisted["drift_config"]["signal"] == "green"
    assert get_drift_results(run["run_id"], tmp_path)[0]["metrics"]["psi"] == 0.0
    assert [item["run_id"] for item in list_runs(tmp_path)] == [run["run_id"]]
    with sqlite3.connect(tmp_path / "oberva.db") as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"runs", "quality_results", "drift_results"} <= table_names


def test_results_cannot_be_written_twice_for_a_completed_stage(tmp_path):
    run = _run(tmp_path)
    quality_rows = []
    summaries = {
        dataset: {"quality_score": 100, "signal": "green"}
        for dataset in ("reference", "current")
    }
    save_quality_results(run["run_id"], quality_rows, summaries, {}, tmp_path)

    with pytest.raises(ValueError, match="new run"):
        save_quality_results(run["run_id"], quality_rows, summaries, {}, tmp_path)


def test_legacy_observa_database_is_still_used(tmp_path):
    with sqlite3.connect(tmp_path / "observa.db") as connection:
        connection.execute("CREATE TABLE marker (id INTEGER PRIMARY KEY)")

    with storage._connect(tmp_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert "marker" in table_names
    assert not (tmp_path / "oberva.db").exists()


def test_legacy_database_is_used_when_new_database_is_empty(tmp_path):
    with sqlite3.connect(tmp_path / "oberva.db") as connection:
        connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
    with sqlite3.connect(tmp_path / "observa.db") as connection:
        connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO runs (run_id) VALUES ('legacy-run')")

    assert storage._database_path(tmp_path) == tmp_path / "observa.db"
