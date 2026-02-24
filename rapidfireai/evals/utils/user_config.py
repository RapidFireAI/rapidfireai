"""Extract user-facing pipeline config for display in eval tables and final results."""

from typing import Any


def extract_pipeline_user_config(pipeline: Any) -> dict[str, Any]:
    """
    Extract all user-facing config knobs from a pipeline for display in tables and final results.
    """
    out: dict[str, Any] = {}

    # Generator model config (exclude "model" name; that is shown as model_name)
    if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
        model_config_copy = pipeline.model_config.copy()
        model_config_copy.pop("model", None)
        if model_config_copy:
            out["model_config"] = model_config_copy

    if hasattr(pipeline, "rag") and pipeline.rag is not None:
        rag = pipeline.rag

        # Search: search_type and search_kwargs as separate columns (rag_k is inside search_kwargs)
        if hasattr(rag, "search_spec") and rag.search_spec is not None:
            out["search_type"] = getattr(rag.search_spec, "search_type", None)
            out["search_kwargs"] = dict(rag.search_spec.search_kwargs or {})
        else:
            out["search_type"] = getattr(rag, "search_type", None)
            out["search_kwargs"] = dict(getattr(rag, "search_kwargs", None) or {})

        # Reranker: reranker_cls and reranker_kwargs as separate columns (top_n is inside reranker_kwargs)
        if hasattr(rag, "reranker_spec") and rag.reranker_spec is not None:
            out["reranker_cls"] = getattr(rag.reranker_spec.cls, "__name__", None)
            out["reranker_kwargs"] = dict(rag.reranker_spec.kwargs or {})
        elif hasattr(rag, "reranker_kwargs") and rag.reranker_kwargs is not None:
            out["reranker_cls"] = None
            out["reranker_kwargs"] = dict(rag.reranker_kwargs)

        # Text splitter params (chunk_size, chunk_overlap, type name)
        if hasattr(rag, "text_splitter") and rag.text_splitter is not None:
            ts = rag.text_splitter
            out["text_splitter_params"] = {
                "chunk_size": getattr(ts, "_chunk_size", None),
                "chunk_overlap": getattr(ts, "_chunk_overlap", None),
                "text_splitter_type": type(ts).__name__,
            }

        # Embedding: embedding_cls and embedding_kwargs as separate columns
        if hasattr(rag, "embedding_spec") and rag.embedding_spec is not None:
            es = rag.embedding_spec
            out["embedding_cls"] = getattr(es.cls, "__name__", None)
            out["embedding_kwargs"] = dict(es.kwargs or {})

        # Vector store name (e.g. FAISS when default is used)
        if getattr(rag, "vector_store", None) is not None:
            out["vector_store_name"] = type(rag.vector_store).__name__

    # Sampling params (generation)
    if hasattr(pipeline, "sampling_params") and pipeline.sampling_params is not None:
        sp = getattr(pipeline, "_user_params", {}).get("sampling_params", None)
        if sp is not None:
            out["sampling_params"] = sp

    # Prompt manager: k, embedding_spec (cls + kwargs), example_selector_cls (skip instructions, examples, template)
    if hasattr(pipeline, "prompt_manager") and pipeline.prompt_manager is not None:
        pm = pipeline.prompt_manager
        k = getattr(pm, "k", None)
        if k is not None:
            out["prompt_manager_k"] = k
        if getattr(pm, "embedding_spec", None) is not None:
            es = pm.embedding_spec
            out["prompt_manager_embedding_cls"] = getattr(es.cls, "__name__", None)
            out["prompt_manager_embedding_kwargs"] = dict(es.kwargs or {})
        if getattr(pm, "example_selector_cls", None) is not None:
            out["prompt_manager_example_selector_cls"] = getattr(pm.example_selector_cls, "__name__", None)

    # Drop keys whose value is None to avoid clutter
    return {k: v for k, v in out.items() if v is not None}
