"""Regression test: get_all_pipelines must not 500 when a single row has a
corrupt dill blob in pipeline_config."""

import json
from unittest.mock import Mock

from rapidfireai.evals.db.rf_db import RFDatabase


def _make_db(rows):
    db_iface = Mock()
    db_iface.execute.return_value = rows
    rf = RFDatabase.__new__(RFDatabase)
    rf.db = db_iface
    return rf


def _row(
    pipeline_id,
    pipeline_config_blob,
    json_config=None,
    flattened_config=None,
    status="Ongoing",
):
    return (
        pipeline_id,
        "ctx-1",
        "sft",
        pipeline_config_blob,
        json.dumps(json_config or {}),
        json.dumps(flattened_config or {}),
        status,
        0,
        0,
        0,
        None,
        None,
        "2026-04-21",
    )


def test_get_all_pipelines_tolerates_corrupt_dill_blob():
    rows = [
        _row(1, None),
        _row(2, "!!not-valid-base64-or-dill!!"),
        _row(3, None),
    ]
    rf = _make_db(rows)

    pipelines = rf.get_all_pipelines()

    assert len(pipelines) == 3
    assert [p["pipeline_id"] for p in pipelines] == [1, 2, 3]
    assert all(p["pipeline_config"] is None for p in pipelines)


def test_get_all_pipelines_returns_empty_list_when_no_rows():
    rf = _make_db([])
    assert rf.get_all_pipelines() == []
