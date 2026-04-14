import base64
import json
from typing import Any

import dill

from rapidfireai.automl import RFvLLMModelConfig, RFAPIModelConfig
from rapidfireai.evals.utils.constants import SEARCH_TYPE_KEYS


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
            - "pipeline": RFvLLMModelConfig or RFAPIModelConfig instance
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

        # Helper function to extract RAG retrieval params for the clone-modify dialog.
        # Only exposes retrieval-stage params (search_cfg, reranker_cfg) in the same
        # format as LangChainRagSpec constructor args. Indexing-stage params
        # (embedding_cfg, vector_store_cfg, text_splitter) are intentionally excluded
        # because cloned pipelines always reuse the parent's pre-built index.
        def extract_rag_params(rag_spec):
            if rag_spec is None:
                return None

            rag_config = {}

            # search_cfg: mirrors the search_cfg constructor arg {"type": ..., <type-specific kwargs>}
            # Only include kwargs relevant to the chosen search type.
            search_type = getattr(rag_spec, "search_type", None)
            search_kwargs = getattr(rag_spec, "search_kwargs", None) or {}
            if search_type is not None:
                allowed_keys = SEARCH_TYPE_KEYS.get(search_type, set(search_kwargs.keys()))
                rag_config["search_cfg"] = {
                    "type": search_type,
                    **{k: v for k, v in search_kwargs.items() if k in allowed_keys and v is not None},
                }

            # reranker_cfg: full constructor-style dict including "class" and all kwargs.
            # The reranker is now instantiated per-pipeline at query time (not at index time),
            # so the user can change both the class and its kwargs via clone-modify.
            reranker_cls = getattr(rag_spec, "reranker_cls", None)
            reranker_kwargs = getattr(rag_spec, "reranker_kwargs", None) or {}
            if reranker_cls is not None:
                rag_config["reranker_cfg"] = {
                    "class": reranker_cls.__qualname__,
                    **{k: v for k, v in reranker_kwargs.items() if v is not None},
                }

            return rag_config if rag_config else None

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

        elif isinstance(pipeline, RFAPIModelConfig):
            json_config["pipeline_type"] = "api"

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

            # Extract endpoint_config - filter out api_key
            if hasattr(pipeline, "endpoint_config") and pipeline.endpoint_config is not None:
                endpoint_cfg = dict(pipeline.endpoint_config)
                endpoint_cfg.pop("api_key", None)
                json_config["endpoint_config"] = endpoint_cfg

            # Extract model_config (sampling parameters)
            if hasattr(pipeline, "model_config") and pipeline.model_config:
                json_config["model_config"] = pipeline.sampling_params_to_dict()

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