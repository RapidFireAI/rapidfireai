"""This module contains utility functions for the AutoML module."""

from typing import Any

from rapidfireai.automl.base import AutoMLAlgorithm
from rapidfireai.fit.utils.exceptions import AutoMLException


def _is_valid_reranker_top_n_vs_k(pipeline: Any) -> bool:
    """
    Check if pipeline has valid top_n <= k when reranker with top_n is present.                                                                                         
    """
    if pipeline is None or not hasattr(pipeline, "rag") or pipeline.rag is None:
        return True
    rag = pipeline.rag

    top_n = None
    if hasattr(rag, "reranker_spec") and rag.reranker_spec is not None:
        top_n = rag.reranker_spec.kwargs.get("top_n")
    elif hasattr(rag, "reranker_kwargs") and rag.reranker_kwargs:
        top_n = rag.reranker_kwargs.get("top_n")
    if top_n is None:
        return True
    if not isinstance(top_n, (int, float)):
        return True

    k = None
    if hasattr(rag, "search_spec") and rag.search_spec is not None:
        k = rag.search_spec.search_kwargs.get("k")
    elif hasattr(rag, "search_kwargs") and rag.search_kwargs:
        k = rag.search_kwargs.get("k")
    if k is None:
        return True  # No k to compare; user config may be incomplete
    if not isinstance(k, (int, float)):
        return True
    return top_n <= k


def filter_evals_runs_valid_reranker(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Filter evals runs to only include configs where reranker top_n <= k.
    """
    filtered = [r for r in runs if _is_valid_reranker_top_n_vs_k(r.get("pipeline"))]
    if not filtered:
        raise AutoMLException(
            "No valid configurations: when using a reranker with top_n, "
            "top_n must be <= k (search_kwargs.k). "
            "Only add top_n values that are less than or equal to k."
        )
    return filtered

# TODO: add code to validate param_config


def get_flattened_config_leaf(
    param_config: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Flattens the param_config dictionary into a single hierarchy"""
    items = []
    for k, v in param_config.items():
        # Skip empty keys and specific keys
        if not k or k in [
            "compute_metrics",
            "formatting_func",
            "output_dir",
            "logging_dir",
            "reward_funcs",
            "task_type",
            "torch_dtype",
        ]:
            continue

        # Create the full key name with prefix to avoid collisions
        full_key = f"{prefix}.{k}" if prefix else str(k)

        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(get_flattened_config_leaf(v, full_key).items())
        else:
            # Handle output_dir conversion safely
            if k == "output_dir" and hasattr(v, "as_posix"):
                # Only call as_posix() if it's actually a Path object
                v = v.as_posix()
            elif k == "output_dir" and isinstance(v, str):
                # If it's already a string, leave it as is
                pass

            # add to items
            items.append((full_key, v))
    return dict(items)


def get_runs(
    param_config: AutoMLAlgorithm | dict[str, Any] | list[Any], seed: int
) -> list[dict[str, Any]]:
    """Get the runs for the given param_config."""
    # FIXME: how do we handle seed for dict and list?
    if isinstance(param_config, AutoMLAlgorithm):
        return param_config.get_runs(seed)
    if isinstance(param_config, dict):
        return [param_config]
    if isinstance(param_config, list):
        config_leaves = []
        for config in param_config:
            config_leaves.extend(get_runs(config, seed))
        return config_leaves
    else:
        raise ValueError(f"Invalid param_config type: {type(param_config)}")
