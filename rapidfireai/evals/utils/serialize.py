import base64
import json
from typing import Any

import dill
from rapidfireai.evals.automl.model_config import RFvLLMModelConfig, RFOpenAIAPIModelConfig


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

        if isinstance(pipeline, RFvLLMModelConfig):
            json_config["pipeline_type"] = "vllm"

            # Extract model_config (dict)
            if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
                json_config["model_config"] = pipeline.model_config

            # Extract sampling_params (convert SamplingParams to dict)
            if (
                hasattr(pipeline, "sampling_params")
                and pipeline.sampling_params is not None
            ):
                json_config["sampling_params"] = pipeline.sampling_params_to_dict()

        elif isinstance(pipeline, RFOpenAIAPIModelConfig):
            json_config["pipeline_type"] = "openai"

            # Extract client_config (dict)
            if (
                hasattr(pipeline, "client_config")
                and pipeline.client_config is not None
            ):
                json_config["client_config"] = pipeline.client_config

            # Extract model_config (dict)
            if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
                json_config["model_config"] = pipeline.model_config

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

    # Validate JSON serializability
    try:
        json.dumps(json_config)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize pipeline config to JSON: {e}") from e

    return json_config