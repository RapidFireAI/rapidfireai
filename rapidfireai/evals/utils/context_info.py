"""
Pure helpers describing a RAG context (vector-store index) for display.

Shared by the notebook index table (``ContextBuildingDisplay``), the evals
controller (which persists the fields on the ``contexts`` row) and the
dispatcher's ``get-all-contexts`` endpoint, so the three never drift apart.
This module must stay free of IPython / Ray imports so the dispatcher and the
unit tests can import it cheaply.
"""

from __future__ import annotations

import re
from typing import Any

# ``password=<value>`` tokens in libpq-style key/value connection strings.
_PASSWORD_KV_RE = re.compile(r"(?i)(password\s*=\s*)([^\s;&]+)")


def collect_vector_store_info(rag_spec: Any) -> dict[str, Any] | None:
    """Return the vector-store summary dict for a RAG spec, or None when it has no vector store.

    Mirrors the dict the notebook index table has always rendered: the raw
    ``type`` plus the pgvector / Pinecone identifiers, with the same defaults
    the RAG spec itself applies when the user omits them.
    """
    if not rag_spec or not getattr(rag_spec, "vector_store_cfg", None):
        return None

    cfg = rag_spec.vector_store_cfg
    store_type = cfg.get("type", None)
    connection = cfg.get("connection", "-")
    collection_name = cfg.get("collection_name", rag_spec.get_hash()) if store_type == "pgvector" else "-"
    if store_type == "pinecone":
        index_namespace = cfg.get("index_namespace", ("rf-" + rag_spec.get_hash()[:42], ""))
    else:
        index_namespace = ("-", "-")
    return {
        "type": store_type,
        "pgvector_connection": connection,
        "pgvector_collection_name": collection_name,
        "pinecone_index_name": index_namespace[0],
        "pinecone_namespace": index_namespace[1],
    }


def vector_store_columns(info: dict[str, Any] | None) -> dict[str, str]:
    """Map a ``collect_vector_store_info`` dict to the notebook's display columns.

    Returns an empty dict when there is no vector store. Column names and
    values are exactly what the notebook index table prints.
    """
    if info is None:
        return {}
    raw_type = info.get("type", None)
    if raw_type == "faiss":
        return {"Vector Store": "FAISS"}
    if raw_type == "pgvector":
        return {
            "Vector Store": "PGVector",
            "Connection": info.get("pgvector_connection", "-"),
            "Collection Name": info.get("pgvector_collection_name", "-"),
        }
    if raw_type == "pinecone":
        namespace = info.get("pinecone_namespace", "-")
        return {
            "Vector Store": "Pinecone",
            "Index Name": info.get("pinecone_index_name", "-"),
            "Namespace": "''" if namespace == "" else namespace,
        }
    return {"Vector Store": raw_type or "Unknown"}


def redact_connection(value: Any) -> Any:
    """Mask the password in a database connection string; return non-strings untouched.

    Handles URL-shaped strings (``postgresql://user:secret@host/db`` ->
    ``postgresql://user:***@host/db``) and libpq key/value strings
    (``host=h password=secret`` -> ``host=h password=***``). Never raises.
    """
    if not isinstance(value, str) or not value:
        return value

    if "://" in value:
        scheme, _, rest = value.partition("://")
        netloc, slash, tail = rest.partition("/")
        if "@" in netloc:
            userinfo, _, hostport = netloc.rpartition("@")
            if ":" in userinfo:
                user = userinfo.split(":", 1)[0]
                return f"{scheme}://{user}:***@{hostport}{slash}{tail}"
        return value

    return _PASSWORD_KV_RE.sub(r"\1***", value)
