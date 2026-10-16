"""Unit tests for the shared RAG-context display helpers and the notebook index table."""

from unittest.mock import patch

import pandas as pd
import pytest

from rapidfireai.evals.utils.context_info import (
    collect_vector_store_info,
    redact_connection,
    vector_store_columns,
)
from rapidfireai.evals.utils.progress_display import ContextBuildingDisplay

HASH = "a" * 64


class _Spec:
    def __init__(self, vector_store_cfg):
        self.vector_store_cfg = vector_store_cfg

    def get_hash(self):
        return HASH


# ---------------------------------------------------------------------------
# collect_vector_store_info
# ---------------------------------------------------------------------------


def test_collect_returns_none_without_vector_store():
    assert collect_vector_store_info(None) is None
    assert collect_vector_store_info(_Spec(None)) is None
    assert collect_vector_store_info(_Spec({})) is None


def test_collect_faiss():
    info = collect_vector_store_info(_Spec({"type": "faiss"}))
    assert info == {
        "type": "faiss",
        "pgvector_connection": "-",
        "pgvector_collection_name": "-",
        "pinecone_index_name": "-",
        "pinecone_namespace": "-",
    }


def test_collect_pgvector_explicit_and_default_collection():
    explicit = collect_vector_store_info(
        _Spec({"type": "pgvector", "connection": "postgresql://u:p@h/db", "collection_name": "docs"})
    )
    assert explicit["pgvector_connection"] == "postgresql://u:p@h/db"
    assert explicit["pgvector_collection_name"] == "docs"

    default = collect_vector_store_info(_Spec({"type": "pgvector", "connection": "postgresql://u:p@h/db"}))
    assert default["pgvector_collection_name"] == HASH


def test_collect_pinecone_explicit_and_default_index_namespace():
    explicit = collect_vector_store_info(_Spec({"type": "pinecone", "index_namespace": ("my-index", "ns1")}))
    assert explicit["pinecone_index_name"] == "my-index"
    assert explicit["pinecone_namespace"] == "ns1"

    default = collect_vector_store_info(_Spec({"type": "pinecone"}))
    assert default["pinecone_index_name"] == "rf-" + HASH[:42]
    assert default["pinecone_namespace"] == ""


# ---------------------------------------------------------------------------
# vector_store_columns
# ---------------------------------------------------------------------------


def test_columns_none():
    assert vector_store_columns(None) == {}


def test_columns_faiss():
    assert vector_store_columns({"type": "faiss"}) == {"Vector Store": "FAISS"}


def test_columns_pgvector():
    cols = vector_store_columns(
        {"type": "pgvector", "pgvector_connection": "postgresql://h/db", "pgvector_collection_name": "docs"}
    )
    assert cols == {"Vector Store": "PGVector", "Connection": "postgresql://h/db", "Collection Name": "docs"}


def test_columns_pinecone_renders_empty_namespace_as_quotes():
    cols = vector_store_columns({"type": "pinecone", "pinecone_index_name": "rf-abc", "pinecone_namespace": ""})
    assert cols == {"Vector Store": "Pinecone", "Index Name": "rf-abc", "Namespace": "''"}
    cols = vector_store_columns({"type": "pinecone", "pinecone_index_name": "rf-abc", "pinecone_namespace": "ns"})
    assert cols["Namespace"] == "ns"


def test_columns_unknown_type():
    assert vector_store_columns({"type": "chroma"}) == {"Vector Store": "chroma"}
    assert vector_store_columns({"type": None}) == {"Vector Store": "Unknown"}
    assert vector_store_columns({}) == {"Vector Store": "Unknown"}


# ---------------------------------------------------------------------------
# redact_connection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("postgresql://rf:s3cret@db:5432/rf", "postgresql://rf:***@db:5432/rf"),
        ("postgresql+psycopg://rf:p%40ss@db/rf?sslmode=require", "postgresql+psycopg://rf:***@db/rf?sslmode=require"),
        ("postgresql://rf@db/rf", "postgresql://rf@db/rf"),  # no password: unchanged
        ("postgresql://db/rf", "postgresql://db/rf"),
        ("postgresql://host/db?password=secret", "postgresql://host/db?password=***"),
        (
            "postgresql://u:p@host/db?password=secret&sslmode=require",
            "postgresql://u:***@host/db?password=***&sslmode=require",
        ),
        ("host=db user=rf password=s3cret dbname=rf", "host=db user=rf password=*** dbname=rf"),
        ("host=db PASSWORD = s3cret", "host=db PASSWORD = ***"),
        ("-", "-"),
        ("", ""),
        (None, None),
        (42, 42),
    ],
)
def test_redact_connection(raw, expected):
    assert redact_connection(raw) == expected


# ---------------------------------------------------------------------------
# ContextBuildingDisplay keeps rendering the same table
# ---------------------------------------------------------------------------


def _contexts():
    return [
        {
            "context_hash": "f" * 64,
            "context_id": 1,
            "enable_gpu": True,
            "start_time": None,
            "vector_store_info": {"type": "faiss"},
        },
        {
            "context_hash": "p" * 64,
            "context_id": 2,
            "enable_gpu": False,
            "start_time": None,
            "vector_store_info": {
                "type": "pgvector",
                "pgvector_connection": "postgresql://h/db",
                "pgvector_collection_name": "docs",
            },
        },
        {
            "context_hash": "n" * 64,
            "context_id": 3,
            "enable_gpu": False,
            "start_time": None,
            "vector_store_info": {"type": "pinecone", "pinecone_index_name": "rf-abc", "pinecone_namespace": ""},
        },
        {"context_hash": "0" * 64, "context_id": 4, "enable_gpu": False, "start_time": None, "vector_store_info": None},
    ]


def test_context_building_display_dataframe_matches_notebook_columns():
    with patch("rapidfireai.evals.utils.progress_display.display"):
        disp = ContextBuildingDisplay(_contexts())
        disp.update_context("f" * 64, status="complete", duration=12.34)
        disp.update_context("p" * 64, status="failed", duration=1.0)
        df = disp._create_dataframe()

    assert list(df.columns) == [
        "RAG Source ID",
        "Status",
        "Duration",
        "Device",
        "Vector Store",
        "Connection",
        "Collection Name",
        "Index Name",
        "Namespace",
    ]
    rows = df.fillna("<nan>").to_dict("records")
    assert rows[0] == {
        "RAG Source ID": 1,
        "Status": "Complete",
        "Duration": "12.3s",
        "Device": "GPU",
        "Vector Store": "FAISS",
        "Connection": "<nan>",
        "Collection Name": "<nan>",
        "Index Name": "<nan>",
        "Namespace": "<nan>",
    }
    assert rows[1]["Status"] == "Failed"
    assert rows[1]["Vector Store"] == "PGVector"
    assert rows[1]["Connection"] == "postgresql://h/db"
    assert rows[1]["Collection Name"] == "docs"
    assert rows[2]["Status"] == "Building"
    assert rows[2]["Duration"] == "-"
    assert rows[2]["Index Name"] == "rf-abc"
    assert rows[2]["Namespace"] == "''"
    assert rows[3]["Vector Store"] == "<nan>"
    assert isinstance(df, pd.DataFrame)
