"""Tests for the ``contexts`` table: display columns, migration, accessors and status resets."""

import sqlite3
import time

import pytest

from rapidfireai.evals.utils.constants import ContextStatus, DBConfig


@pytest.fixture
def rf_db(tmp_path, monkeypatch):
    """A real RFDatabase on a fresh temp SQLite file."""
    monkeypatch.setattr(DBConfig, "DB_PATH", str(tmp_path / "evals.db"))
    from rapidfireai.evals.db.rf_db import RFDatabase

    db = RFDatabase()
    yield db
    db.close()


def _column_names(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_fresh_schema_has_display_columns(rf_db):
    cols = _column_names(rf_db.db.conn, "contexts")
    assert "enable_gpu" in cols
    assert "vector_store_json" in cols


def test_migration_adds_columns_to_old_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE contexts (
            context_id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_hash TEXT NOT NULL UNIQUE,
            rag_config_json TEXT,
            prompt_config_json TEXT,
            status TEXT NOT NULL,
            error TEXT DEFAULT '',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            duration_seconds REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO contexts (context_hash, status, completed_at, duration_seconds) VALUES (?, ?, ?, ?)",
        ("legacy", "ongoing", 1700000010.0, 10.0),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(DBConfig, "DB_PATH", str(db_path))
    from rapidfireai.evals.db.rf_db import RFDatabase

    db = RFDatabase()
    try:
        cols = _column_names(db.db.conn, "contexts")
        assert "enable_gpu" in cols and "vector_store_json" in cols

        # Legacy row is readable with defaults, and a second init is a no-op.
        legacy = db.get_context_by_hash("legacy")
        assert legacy["enable_gpu"] is False
        assert legacy["vector_store_json"] is None
        assert legacy["status"] == "ongoing"
        assert legacy["completed_at"] == 1700000010.0
        db._initialize_schema()
        assert _column_names(db.db.conn, "contexts").count("enable_gpu") == 1
    finally:
        db.close()


def test_create_and_get_all_contexts_roundtrip(rf_db):
    a = rf_db.create_context(
        "hash-a", status=ContextStatus.ONGOING, enable_gpu=True, vector_store_json='{"type": "faiss"}'
    )
    b = rf_db.create_context("hash-b", status=ContextStatus.ONGOING)
    assert rf_db.create_context("hash-a") == a  # dedupe on hash, existing row untouched

    rows = rf_db.get_all_contexts()
    assert [r["context_id"] for r in rows] == [a, b]
    assert rows[0]["enable_gpu"] is True
    assert rows[0]["vector_store_json"] == '{"type": "faiss"}'
    assert rows[1]["enable_gpu"] is False
    assert rows[1]["vector_store_json"] is None
    assert set(rows[0]) == {
        "context_id",
        "context_hash",
        "rag_config_json",
        "prompt_config_json",
        "status",
        "error",
        "started_at",
        "completed_at",
        "duration_seconds",
        "enable_gpu",
        "vector_store_json",
    }
    assert rf_db.get_context(a) == rows[0]
    assert rf_db.get_context_by_hash("hash-b") == rows[1]


def test_get_all_contexts_empty(rf_db):
    assert rf_db.get_all_contexts() == []


def test_start_context_build_resets_result_fields(rf_db):
    cid = rf_db.create_context("hash-a", status=ContextStatus.ONGOING)
    rf_db.set_context_end_time(cid, 1700000010.0, 10.0)
    rf_db.set_context_status(cid, ContextStatus.COMPLETED)
    rf_db.set_context_error(cid, "boom")

    t0 = time.time()
    rf_db.start_context_build(cid, t0)
    ctx = rf_db.get_context(cid)
    assert ctx["status"] == "ongoing"
    assert ctx["started_at"] == t0
    assert ctx["completed_at"] is None
    assert ctx["duration_seconds"] is None
    assert ctx["error"] == ""


def test_reset_experiment_states_keeps_completed_contexts(rf_db):
    done = rf_db.create_context("done", status=ContextStatus.COMPLETED)
    building = rf_db.create_context("building", status=ContextStatus.ONGOING)
    fresh = rf_db.create_context("fresh", status=ContextStatus.NEW)
    failed = rf_db.create_context("failed", status=ContextStatus.FAILED)

    rf_db.reset_experiment_states()

    status = {r["context_id"]: r["status"] for r in rf_db.get_all_contexts()}
    assert status[done] == "completed"
    assert status[building] == "failed"
    assert status[fresh] == "failed"
    assert status[failed] == "failed"
