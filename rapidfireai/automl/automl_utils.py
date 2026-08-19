"""This module contains utility functions for the AutoML module."""

import re
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
    if not hasattr(rag, "reranker_kwargs") or not rag.reranker_kwargs:
        return True
    top_n = rag.reranker_kwargs.get("top_n")
    if top_n is None:
        return True
    k = None
    if hasattr(rag, "search_kwargs") and rag.search_kwargs:
        k = rag.search_kwargs.get("k")
    if k is None:
        return True  # No k to compare; user config may be incomplete
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


# ---------------------------------------------------------------------------
# Secret-key stripping for flattened configs logged to tracking backends.
#
# Token-derived knobs are intentionally NOT treated as secrets: ``max_tokens``,
# ``max_completion_tokens`` (generation-length knobs), ``tokenizer`` /
# ``tokenizer_kwargs`` and the HuggingFace ``*_token_id`` leaves
# (``pad_token_id``, ``eos_token_id``, ``bos_token_id``) are kept. Credential
# token fields use the bare ``token`` leaf or the singular ``*_token`` suffix
# (``api_token``, ``access_token``, ``hf_token``), which are still matched.
_SECRET_SUBSTRINGS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "access_key",
    "private_key",
    "credential",
)
_TOKEN_EXEMPT_PREFIXES = ("tokenizer",)  # covers ``tokenizer`` & ``tokenizer_kwargs``
_TOKEN_EXEMPT_SUFFIXES = (
    "_token_id",  # ``pad_token_id``, ``eos_token_id``, ``bos_token_id``, ...
    "_tokens",  # ``max_tokens``, ``max_completion_tokens`` (generation-length knobs)
)


def is_secret_key(dotted_key: str) -> bool:
    """True if a flattened config key is a secret/API-key knob to drop.

    Matched against the last dotted segment (lowercased). Token-derived knobs
    are exempted (kept); credential token fields are matched.
    """
    leaf = dotted_key.rsplit(".", 1)[-1].lower()
    if any(leaf.startswith(p) for p in _TOKEN_EXEMPT_PREFIXES):
        return False
    if any(leaf.endswith(s) for s in _TOKEN_EXEMPT_SUFFIXES):
        return False
    if leaf == "token" or leaf.endswith("_token"):
        return True
    return any(s in leaf for s in _SECRET_SUBSTRINGS)


def strip_secret_keys(flat: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``flat`` with every secret/API-key knob removed."""
    return {k: v for k, v in flat.items() if not is_secret_key(k)}

# Credentials also hide inside otherwise-innocent *values*: a pgvector
# ``connection`` is a DSN of the form ``postgresql+psycopg://user:password@host``.
# ``is_secret_key`` can't catch these because it matches on the key name, and
# ``connection`` is a legitimate knob we want to keep visible (host/port/database
# are worth comparing across runs). Redact only the userinfo component.
_URI_CREDENTIALS_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)([^/@\s]+)@")


def redact_uri_credentials(value: Any) -> Any:
    """Replace the userinfo component of a URI-shaped string with ``***``.

    ``postgresql+psycopg://user:pw@localhost:6024/db`` becomes
    ``postgresql+psycopg://***@localhost:6024/db``. Non-strings and strings that
    aren't URIs with credentials are returned unchanged.
    """
    if not isinstance(value, str):
        return value
    return _URI_CREDENTIALS_RE.sub(r"\1***@", value, count=1)


def redact_secret_values(flat: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``flat`` with credentials embedded in values redacted."""
    return {k: redact_uri_credentials(v) for k, v in flat.items()}


def sanitize_flat_config(flat: dict[str, Any]) -> dict[str, Any]:
    """Apply both secret rules: drop secret-named keys, redact secret-bearing values.

    Every surface that persists or displays a flattened config should use this
    rather than either rule alone, so a new consumer can't silently pick up only
    half the policy.
    """
    return redact_secret_values(strip_secret_keys(flat))


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
