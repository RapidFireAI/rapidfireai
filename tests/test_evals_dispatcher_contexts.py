"""Tests for the evals dispatcher's ``GET /dispatcher/get-all-contexts`` endpoint."""

from unittest.mock import Mock

import pytest

import rapidfireai.evals.dispatcher.dispatcher as dispatcher_module

NOW = 1_700_000_100.0


def _ctx(**overrides):
    row = {
        "context_id": 1,
        "context_hash": "a" * 64,
        "rag_config_json": "{}",
        "prompt_config_json": "{}",
        "status": "ongoing",
        "error": "",
        "started_at": NOW - 30.0,
        "completed_at": None,
        "duration_seconds": None,
        "enable_gpu": False,
        "vector_store_json": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def dispatcher(monkeypatch):
    """A Dispatcher whose RFDatabase is a Mock (no SQLite, no experiment)."""
    mock_db = Mock()
    monkeypatch.setattr(dispatcher_module, "RFDatabase", Mock(return_value=mock_db))
    d = dispatcher_module.Dispatcher()
    d.mock_db = mock_db
    return d


@pytest.fixture
def client(dispatcher):
    dispatcher.app.config["TESTING"] = True
    return dispatcher.app.test_client()


def test_empty_list(client, dispatcher):
    dispatcher.mock_db.get_all_contexts.return_value = []
    resp = client.get("/dispatcher/get-all-contexts")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_options_preflight(client):
    resp = client.options("/dispatcher/get-all-contexts")
    assert resp.status_code == 200


def test_building_row_reports_elapsed_duration(dispatcher):
    row = dispatcher._transform_context_to_row(_ctx(), now=NOW)
    assert row["status"] == "Building"
    assert row["duration_seconds"] == pytest.approx(30.0)
    assert row["device"] == "CPU"
    assert row["vector_store"] is None
    assert row["context_hash"] == "a" * 64
    assert row["error"] == ""


def test_completed_row(dispatcher):
    row = dispatcher._transform_context_to_row(
        _ctx(status="completed", completed_at=NOW - 10.0, duration_seconds=20.0, enable_gpu=True), now=NOW
    )
    assert row["status"] == "Complete"
    assert row["duration_seconds"] == 20.0
    assert row["device"] == "GPU"


def test_legacy_ongoing_with_completed_at_is_complete(dispatcher):
    row = dispatcher._transform_context_to_row(
        _ctx(status="ongoing", completed_at=NOW - 10.0, duration_seconds=20.0), now=NOW
    )
    assert row["status"] == "Complete"
    assert row["duration_seconds"] == 20.0


def test_failed_row_carries_error(dispatcher):
    row = dispatcher._transform_context_to_row(
        _ctx(status="failed", duration_seconds=3.5, error="connection refused"), now=NOW
    )
    assert row["status"] == "Failed"
    assert row["duration_seconds"] == 3.5
    assert row["error"] == "connection refused"


def test_new_row_without_start_time_has_no_duration(dispatcher):
    row = dispatcher._transform_context_to_row(_ctx(status="new", started_at=None), now=NOW)
    assert row["status"] == "Building"
    assert row["duration_seconds"] is None


def test_non_numeric_started_at_is_tolerated(dispatcher):
    row = dispatcher._transform_context_to_row(_ctx(started_at="2026-10-05T10:00:00"), now=NOW)
    assert row["status"] == "Building"
    assert row["duration_seconds"] is None
    assert row["started_at"] is None


def test_pgvector_connection_is_masked(dispatcher):
    vs = (
        '{"type": "pgvector", "pgvector_connection": "postgresql://rf:s3cret@db:5432/rf",'
        ' "pgvector_collection_name": "docs", "pinecone_index_name": "-", "pinecone_namespace": "-"}'
    )
    row = dispatcher._transform_context_to_row(_ctx(vector_store_json=vs), now=NOW)
    assert row["vector_store"] == {
        "type": "pgvector",
        "label": "PGVector",
        "connection": "postgresql://rf:***@db:5432/rf",
        "collection_name": "docs",
        "index_name": None,
        "namespace": None,
    }
    assert "s3cret" not in str(row)


def test_faiss_and_pinecone_vector_store_shapes(dispatcher):
    faiss = dispatcher._transform_context_to_row(_ctx(vector_store_json='{"type": "faiss"}'), now=NOW)
    assert faiss["vector_store"] == {
        "type": "faiss",
        "label": "FAISS",
        "connection": None,
        "collection_name": None,
        "index_name": None,
        "namespace": None,
    }
    pinecone = dispatcher._transform_context_to_row(
        _ctx(vector_store_json='{"type": "pinecone", "pinecone_index_name": "rf-abc", "pinecone_namespace": ""}'),
        now=NOW,
    )
    assert pinecone["vector_store"]["label"] == "Pinecone"
    assert pinecone["vector_store"]["index_name"] == "rf-abc"
    assert pinecone["vector_store"]["namespace"] == "''"


def test_malformed_vector_store_json_is_null(dispatcher):
    row = dispatcher._transform_context_to_row(_ctx(vector_store_json="{not json"), now=NOW)
    assert row["vector_store"] is None


def test_endpoint_serialises_rows_and_skips_bad_ones(client, dispatcher):
    dispatcher.mock_db.get_all_contexts.return_value = [
        _ctx(context_id=1, status="completed", completed_at=NOW, duration_seconds=5.0),
        "not-a-row",  # transform raises -> skipped, not a 500
        _ctx(context_id=3),
    ]
    resp = client.get("/dispatcher/get-all-contexts")
    assert resp.status_code == 200
    body = resp.get_json()
    assert [r["context_id"] for r in body] == [1, 3]
    assert body[0]["status"] == "Complete"
    assert body[1]["status"] == "Building"


def test_db_failure_is_500_with_error(client, dispatcher):
    dispatcher.mock_db.get_all_contexts.side_effect = RuntimeError("db locked")
    resp = client.get("/dispatcher/get-all-contexts")
    assert resp.status_code == 500
    body = resp.get_json()
    assert "db locked" in body["error"]
    assert "traceback" in body
