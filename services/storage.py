from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = Path(
    os.environ.get("OBERVA_DATA_DIR")
    or os.environ.get("OBSERVA_DATA_DIR", PROJECT_ROOT / "data")
)


def _data_directory(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else DATA_DIRECTORY


def _run_count(database_path: Path) -> int:
    if not database_path.exists():
        return 0
    try:
        with sqlite3.connect(database_path) as connection:
            table_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'runs'
                """
            ).fetchone()
            if table_exists is None:
                return 0
            return int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    except sqlite3.Error:
        return 0


def _database_path(root: Path) -> Path:
    database_path = root / "oberva.db"
    legacy_path = root / "observa.db"
    if legacy_path.exists() and _run_count(database_path) == 0 < _run_count(legacy_path):
        return legacy_path
    return database_path


def _connect(data_dir: str | Path | None = None) -> sqlite3.Connection:
    root = _data_directory(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_database_path(root), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def _connection(
    data_dir: str | Path | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    connection = _connect(data_dir)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_store(data_dir: str | Path | None = None) -> None:
    with _connection(data_dir) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                reference_name TEXT NOT NULL,
                current_name TEXT NOT NULL,
                reference_path TEXT NOT NULL,
                current_path TEXT NOT NULL,
                reference_rows INTEGER NOT NULL,
                current_rows INTEGER NOT NULL,
                reference_columns_json TEXT NOT NULL,
                current_columns_json TEXT NOT NULL,
                shared_columns_json TEXT NOT NULL,
                missing_from_current_json TEXT NOT NULL,
                extra_in_current_json TEXT NOT NULL,
                column_types_json TEXT NOT NULL,
                quality_config_json TEXT NOT NULL DEFAULT '{}',
                drift_config_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_runs_created_at
                ON runs(created_at DESC);
            CREATE TABLE IF NOT EXISTS quality_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                dataset TEXT NOT NULL,
                feature TEXT NOT NULL,
                signal TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                suggestions_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_quality_results_run
                ON quality_results(run_id, dataset);
            CREATE TABLE IF NOT EXISTS drift_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                feature TEXT NOT NULL,
                feature_type TEXT NOT NULL,
                signal TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_drift_results_run
                ON drift_results(run_id);
            """
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def create_run(
    reference_bytes: bytes,
    current_bytes: bytes,
    reference_name: str,
    current_name: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    shared_columns: list[str],
    missing_from_current: list[str],
    extra_in_current: list[str],
    column_types: dict[str, str],
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    initialize_store(data_dir)
    root = _data_directory(data_dir)
    run_id = uuid.uuid4().hex
    run_directory = root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    reference_path = run_directory / "reference.csv"
    current_path = run_directory / "current.csv"
    try:
        reference_path.write_bytes(reference_bytes)
        current_path.write_bytes(current_bytes)
        created_at = datetime.now(timezone.utc).isoformat()
        with _connection(data_dir) as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, status, reference_name, current_name,
                    reference_path, current_path, reference_rows, current_rows,
                    reference_columns_json, current_columns_json, shared_columns_json,
                    missing_from_current_json, extra_in_current_json, column_types_json
                ) VALUES (?, ?, 'created', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    Path(reference_name).name,
                    Path(current_name).name,
                    str(reference_path.resolve()),
                    str(current_path.resolve()),
                    len(reference),
                    len(current),
                    _json(list(reference.columns)),
                    _json(list(current.columns)),
                    _json(shared_columns),
                    _json(missing_from_current),
                    _json(extra_in_current),
                    _json(column_types),
                ),
            )
    except Exception:
        shutil.rmtree(run_directory, ignore_errors=True)
        raise
    return get_run(run_id, data_dir)


def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in (
        "reference_columns_json",
        "current_columns_json",
        "shared_columns_json",
        "missing_from_current_json",
        "extra_in_current_json",
        "column_types_json",
        "quality_config_json",
        "drift_config_json",
    ):
        result[key.removesuffix("_json")] = json.loads(result.pop(key))
    return result


def get_run(run_id: str, data_dir: str | Path | None = None) -> dict[str, Any] | None:
    initialize_store(data_dir)
    with _connection(data_dir) as connection:
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _decode_run(row) if row is not None else None


def list_runs(data_dir: str | Path | None = None) -> list[dict[str, Any]]:
    initialize_store(data_dir)
    with _connection(data_dir) as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()
    return [_decode_run(row) for row in rows]


def load_run_frames(
    run: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_path = Path(run["reference_path"])
    current_path = Path(run["current_path"])
    if not reference_path.is_file() or not current_path.is_file():
        raise FileNotFoundError(
            f"Stored CSV artifacts for run {run['run_id']} are missing."
        )
    return pd.read_csv(reference_path), pd.read_csv(current_path)


def save_quality_results(
    run_id: str,
    results: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    quality_config: dict[str, Any],
    data_dir: str | Path | None = None,
) -> None:
    initialize_store(data_dir)
    records = list(results)
    for dataset, summary in summaries.items():
        records.append(
            {
                "dataset": dataset,
                "feature": "__overall__",
                "signal": summary["signal"],
                "metrics": summary,
                "suggestions": [],
            }
        )
    with _connection(data_dir) as connection:
        run = connection.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(f"Run {run_id} does not exist.")
        if run["status"] != "created":
            raise ValueError("Quality checks can only be saved for a new run.")
        connection.executemany(
            """
            INSERT INTO quality_results
                (run_id, dataset, feature, signal, metrics_json, suggestions_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    record["dataset"],
                    record["feature"],
                    record["signal"],
                    _json(record["metrics"]),
                    _json(record["suggestions"]),
                )
                for record in records
            ],
        )
        connection.execute(
            """
            UPDATE runs
            SET status = 'quality_complete', quality_config_json = ?
            WHERE run_id = ?
            """,
            (_json(quality_config), run_id),
        )


def get_quality_results(
    run_id: str, data_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    initialize_store(data_dir)
    with _connection(data_dir) as connection:
        rows = connection.execute(
            """
            SELECT dataset, feature, signal, metrics_json, suggestions_json
            FROM quality_results
            WHERE run_id = ?
            ORDER BY dataset, feature
            """,
            (run_id,),
        ).fetchall()
    return [
        {
            "dataset": row["dataset"],
            "feature": row["feature"],
            "signal": row["signal"],
            "metrics": json.loads(row["metrics_json"]),
            "suggestions": json.loads(row["suggestions_json"]),
        }
        for row in rows
    ]


def save_drift_results(
    run_id: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    drift_config: dict[str, Any],
    data_dir: str | Path | None = None,
) -> None:
    initialize_store(data_dir)
    with _connection(data_dir) as connection:
        run = connection.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(f"Run {run_id} does not exist.")
        if run["status"] != "quality_complete":
            raise ValueError("Drift checks require completed quality checks.")
        connection.executemany(
            """
            INSERT INTO drift_results
                (run_id, feature, feature_type, signal, metrics_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    record["feature"],
                    record["feature_type"],
                    record["signal"],
                    _json(record["metrics"]),
                )
                for record in results
            ],
        )
        connection.execute(
            """
            UPDATE runs
            SET status = 'drift_complete', drift_config_json = ?
            WHERE run_id = ?
            """,
            (_json({**drift_config, **summary}), run_id),
        )


def get_drift_results(
    run_id: str, data_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    initialize_store(data_dir)
    with _connection(data_dir) as connection:
        rows = connection.execute(
            """
            SELECT feature, feature_type, signal, metrics_json
            FROM drift_results
            WHERE run_id = ?
            ORDER BY feature
            """,
            (run_id,),
        ).fetchall()
    return [
        {
            "feature": row["feature"],
            "feature_type": row["feature_type"],
            "signal": row["signal"],
            "metrics": json.loads(row["metrics_json"]),
        }
        for row in rows
    ]
