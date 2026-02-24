import base64
import json
from typing import Any

import dill
from rapidfireai.automl.model_config import RFvLLMModelConfig, RFOpenAIAPIModelConfig


def extract_pipeline_display_metadata(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract all user-visible knobs from a pipeline config for display in progress tables.
    """
    metadata: dict[str, Any] = {}
    pipeline = pipeline_config.get("pipeline")
    if pipeline is None:
        return metadata

    # Model name
    model_name = "Unknown"
    if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
        if "model" in pipeline.model_config:
            model_name = pipeline.model_config["model"]
        model_config_copy = pipeline.model_config.copy()
        model_config_copy.pop("model", None)
        if model_config_copy:
            metadata["model_config"] = model_config_copy
    metadata["model_name"] = model_name

    # --- RAG-related fields ---
    if hasattr(pipeline, "rag") and pipeline.rag is not None:
        rag = pipeline.rag

        # Text splitter
        if hasattr(rag, "text_splitter") and rag.text_splitter is not None:
            metadata["text_splitter"] = type(rag.text_splitter).__name__
            chunk_size = getattr(rag.text_splitter, "_chunk_size", None)
            chunk_overlap = getattr(rag.text_splitter, "_chunk_overlap", None)
            if chunk_size is not None:
                metadata["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                metadata["chunk_overlap"] = chunk_overlap

        # Embedding
        if hasattr(rag, "embedding_cls") and rag.embedding_cls is not None:
            metadata["embedding_class"] = rag.embedding_cls.__name__
        if hasattr(rag, "embedding_kwargs") and rag.embedding_kwargs:
            emb_model = rag.embedding_kwargs.get("model_name")
            if emb_model is not None:
                metadata["embedding_model"] = emb_model
            remaining = {k: v for k, v in rag.embedding_kwargs.items() if k != "model_name"}
            if remaining:
                metadata["embedding_kwargs"] = remaining

        # Vector store
        if hasattr(rag, "vector_store") and rag.vector_store is not None:
            metadata["vector_store"] = type(rag.vector_store).__name__
        elif hasattr(rag, "retriever") and rag.retriever is not None:
            metadata["vector_store"] = "Custom Retriever"
        else:
            gpu_flag = getattr(rag, "enable_gpu_search", False)
            metadata["vector_store"] = f"FAISS (default, {'GPU' if gpu_flag else 'CPU'})"

        # Search
        search_type = getattr(rag, "search_type", None)
        if search_type is not None:
            metadata["search_type"] = search_type
        if hasattr(rag, "search_kwargs") and rag.search_kwargs:
            metadata["search_kwargs"] = rag.search_kwargs

        # Reranker
        if hasattr(rag, "reranker_cls") and rag.reranker_cls is not None:
            metadata["reranker_class"] = rag.reranker_cls.__name__
        if hasattr(rag, "reranker_kwargs") and rag.reranker_kwargs:
            reranker_model = rag.reranker_kwargs.get("model_name")
            if reranker_model is not None:
                metadata["reranker_model"] = reranker_model
            top_n = rag.reranker_kwargs.get("top_n")
            if top_n is not None:
                metadata["reranker_top_n"] = top_n

        # GPU search
        gpu_search = getattr(rag, "enable_gpu_search", None)
        if gpu_search is not None:
            metadata["gpu_search"] = gpu_search

    # Sampling params (original dict, not the SamplingParams object)
    if hasattr(pipeline, "sampling_params") and pipeline.sampling_params is not None:
        if hasattr(pipeline, "_user_params"):
            sp = pipeline._user_params.get("sampling_params")
            if sp is not None:
                metadata["sampling_params"] = sp

    # Prompt manager
    if hasattr(pipeline, "prompt_manager") and pipeline.prompt_manager is not None:
        pm = pipeline.prompt_manager
        pm_k = getattr(pm, "k", None)
        if pm_k is not None:
            metadata["prompt_manager_k"] = pm_k

        # Prompt manager embedding (used for few-shot example selector)
        if hasattr(pm, "embedding_cls") and pm.embedding_cls is not None:
            metadata["embedding_class"] = pm.embedding_cls.__name__
        if hasattr(pm, "embedding_kwargs") and pm.embedding_kwargs:
            pm_emb_model = pm.embedding_kwargs.get("model_name")
            if pm_emb_model is not None:
                metadata["embedding_model"] = pm_emb_model
            pm_remaining = {k: v for k, v in pm.embedding_kwargs.items() if k != "model_name"}
            if pm_remaining:
                metadata["embedding_kwargs"] = pm_remaining

        # Example selector class
        pm_selector = getattr(pm, "example_selector_cls", None)
        if pm_selector is not None:
            metadata["example_selector"] = pm_selector.__name__

    return metadata


def encode_payload(payload: object) -> str:
    """Encode the payload for the database"""
    return base64.b64encode(dill.dumps(payload)).decode("utf-8")


def decode_db_payload(payload: str) -> object:
    """Decode the payload from the database"""
    return dill.loads(base64.b64decode(payload))


def extract_pipeline_config_json(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract JSON-serializable data from a pipeline config dictionary.

    Extracts only serializable parameters (dicts, strings, ints, etc.) and ignores
    functions, classes, and other non-serializable objects. This is used for storing
    a JSON representation in the database for analytics/display purposes.

    The actual pipeline config (with functions and classes) should be stored using
    encode_payload/decode_db_payload in the pipeline_config column.

    Args:
        pipeline_config: Pipeline config dict with keys:
            - "pipeline": RFvLLMModelConfig or RFOpenAIAPIModelConfig instance
            - "batch_size": int
            - "preprocess_fn": function (skipped)
            - "postprocess_fn": function (skipped)
            - "compute_metrics_fn": function (skipped)
            - "accumulate_metrics_fn": function (skipped)
            - "online_strategy_kwargs": dict (optional)

    Returns:
        Dictionary with only JSON-serializable data from the pipeline config
    """
    json_config = {}

    # Extract batch_size if present
    if "batch_size" in pipeline_config:
        json_config["batch_size"] = pipeline_config["batch_size"]

    # Extract online_strategy_kwargs if present
    if "online_strategy_kwargs" in pipeline_config:
        json_config["online_strategy_kwargs"] = pipeline_config[
            "online_strategy_kwargs"
        ]

    # Extract pipeline type and model-specific params
    if "pipeline" in pipeline_config:
        pipeline = pipeline_config["pipeline"]

        # Helper function to extract RAG params
        def extract_rag_params(rag_spec):
            """Extract RAG parameters from rag_spec for JSON storage."""
            if rag_spec is None:
                return None

            rag_config = {}

            # Text splitter
            if hasattr(rag_spec, "text_splitter") and rag_spec.text_splitter is not None:
                rag_config["text_splitter"] = type(rag_spec.text_splitter).__name__
                rag_config["chunk_size"] = getattr(rag_spec.text_splitter, "_chunk_size", None)
                rag_config["chunk_overlap"] = getattr(rag_spec.text_splitter, "_chunk_overlap", None)

            # Embedding
            if hasattr(rag_spec, "embedding_cls") and rag_spec.embedding_cls is not None:
                rag_config["embedding_class"] = rag_spec.embedding_cls.__name__
            if hasattr(rag_spec, "embedding_kwargs") and rag_spec.embedding_kwargs:
                emb_model = rag_spec.embedding_kwargs.get("model_name")
                if emb_model is not None:
                    rag_config["embedding_model"] = emb_model

            # Vector store
            if hasattr(rag_spec, "vector_store") and rag_spec.vector_store is not None:
                rag_config["vector_store"] = type(rag_spec.vector_store).__name__

            # Search
            rag_config["search_type"] = getattr(rag_spec, "search_type", None)
            if hasattr(rag_spec, "search_kwargs") and rag_spec.search_kwargs is not None:
                rag_config["search_kwargs"] = rag_spec.search_kwargs

            # Reranker
            if hasattr(rag_spec, "reranker_cls") and rag_spec.reranker_cls is not None:
                rag_config["reranker_class"] = rag_spec.reranker_cls.__name__
            if hasattr(rag_spec, "reranker_kwargs") and rag_spec.reranker_kwargs:
                reranker_model = rag_spec.reranker_kwargs.get("model_name")
                if reranker_model is not None:
                    rag_config["reranker_model"] = reranker_model
                top_n = rag_spec.reranker_kwargs.get("top_n")
                if top_n is not None:
                    rag_config["reranker_top_n"] = top_n

            # GPU search
            gpu_search = getattr(rag_spec, "enable_gpu_search", None)
            if gpu_search is not None:
                rag_config["gpu_search"] = gpu_search

            filtered_rag_config = {k: v for k, v in rag_config.items() if v is not None}
            return filtered_rag_config if filtered_rag_config else None

        if isinstance(pipeline, RFvLLMModelConfig):
            json_config["pipeline_type"] = "vllm"

            # Extract model_config (dict)
            if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
                json_config["model_config"] = pipeline.model_config

            # Extract sampling_params from _user_params (original dict, not SamplingParams object)
            if hasattr(pipeline, "_user_params") and "sampling_params" in pipeline._user_params:
                json_config["sampling_params"] = pipeline._user_params["sampling_params"]

            # Extract RAG params if present
            if hasattr(pipeline, "rag") and pipeline.rag is not None:
                rag_config = extract_rag_params(pipeline.rag)
                if rag_config:
                    json_config["rag_config"] = rag_config

        elif isinstance(pipeline, RFOpenAIAPIModelConfig):
            json_config["pipeline_type"] = "openai"

            # Extract client_config (dict) - filter out sensitive keys
            if (
                hasattr(pipeline, "client_config")
                and pipeline.client_config is not None
            ):
                sensitive_keys = {"api_key", "secret", "token", "password", "key"}
                json_config["client_config"] = {
                    k: v for k, v in pipeline.client_config.items()
                    if k.lower() not in sensitive_keys
                }

            # Extract model_config (dict)
            if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
                json_config["model_config"] = pipeline.model_config

            # Extract sampling_params using sampling_params_to_dict (extracts from model_config)
            if (
                hasattr(pipeline, "sampling_params")
                and pipeline.sampling_params is not None
            ):
                json_config["sampling_params"] = pipeline.sampling_params_to_dict()

            # Extract rate limiting params
            if hasattr(pipeline, "rpm_limit") and pipeline.rpm_limit is not None:
                json_config["rpm_limit"] = pipeline.rpm_limit
            if hasattr(pipeline, "tpm_limit") and pipeline.tpm_limit is not None:
                json_config["tpm_limit"] = pipeline.tpm_limit
            if (
                hasattr(pipeline, "max_completion_tokens")
                and pipeline.max_completion_tokens is not None
            ):
                json_config["max_completion_tokens"] = pipeline.max_completion_tokens

            # Extract RAG params if present
            if hasattr(pipeline, "rag") and pipeline.rag is not None:
                rag_config = extract_rag_params(pipeline.rag)
                if rag_config:
                    json_config["rag_config"] = rag_config

    # Validate JSON serializability
    try:
        json.dumps(json_config)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize pipeline config to JSON: {e}") from e

    return json_config