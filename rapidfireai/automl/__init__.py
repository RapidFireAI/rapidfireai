"""AutoML module for hyperparameter optimization (unified for fit and evals)."""

from .base import AutoMLAlgorithm
from .datatypes import List, Range
from .grid_search import RFGridSearch
from .random_search import RFRandomSearch
from .automl_utils import get_flattened_config_leaf, get_runs
from .model_config import _make_unavailable_class

# Optuna integration (conditionally available)
try:
    from .optuna_search import RFOptuna
    _OPTUNA_AVAILABLE = True
except ImportError as _optuna_import_error:

    class RFOptuna:  # type: ignore[misc]
        """Stub so imports succeed; instantiation explains how to enable Optuna."""

        def __new__(cls, *args, **kwargs):  # noqa: ARG004
            raise ImportError(
                "RFOptuna requires Optuna importable from this Python environment. "
                "Install into the **same interpreter as your Jupyter kernel**, then restart the kernel:\n"
                "  python -m pip install optuna\n"
                "Check in a notebook cell: import sys; print(sys.executable)\n"
                "Original error: "
                + str(_optuna_import_error)
            ) from _optuna_import_error

    _OPTUNA_AVAILABLE = False

# Import fit mode configs (conditionally available)
try:
    from .model_config import (
        RFDPOConfig,
        RFGRPOConfig,
        RFLoraConfig,
        RFModelConfig,
        RFSFTConfig,
    )
    _FIT_CONFIGS_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    # Sentinel classes (not None) so isinstance(x, RFXXX) is safely False
    # instead of raising ``TypeError: isinstance() arg 2 must be a type``.
    RFDPOConfig = _make_unavailable_class("RFDPOConfig", "trl")
    RFGRPOConfig = _make_unavailable_class("RFGRPOConfig", "trl")
    RFLoraConfig = _make_unavailable_class("RFLoraConfig", "peft")
    RFModelConfig = _make_unavailable_class("RFModelConfig", "rapidfireai[fit]")
    RFSFTConfig = _make_unavailable_class("RFSFTConfig", "trl")
    _FIT_CONFIGS_AVAILABLE = False

# Import evals mode configs (conditionally available)
try:
    from .model_config import (
        ModelConfig,
        RFvLLMModelConfig,
        RFAPIModelConfig,
    )
    _EVALS_CONFIGS_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    ModelConfig = _make_unavailable_class("ModelConfig", "rapidfireai[evals]")
    RFvLLMModelConfig = _make_unavailable_class("RFvLLMModelConfig", "vllm")
    RFAPIModelConfig = _make_unavailable_class("RFAPIModelConfig", "rapidfireai[evals]")
    _EVALS_CONFIGS_AVAILABLE = False

# Conditionally import evals-specific helper classes
try:
    from .model_config import RFLangChainRagSpec, RFPromptManager
    _EVALS_HELPERS_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    RFLangChainRagSpec = _make_unavailable_class(
        "RFLangChainRagSpec", "rapidfireai[evals]"
    )
    RFPromptManager = _make_unavailable_class(
        "RFPromptManager", "rapidfireai[evals]"
    )
    _EVALS_HELPERS_AVAILABLE = False

__all__ = [
    "List",
    "Range",
    "RFGridSearch",
    "RFRandomSearch",
    "AutoMLAlgorithm",
    # Utility functions
    "get_flattened_config_leaf",
    "get_runs",
]

__all__.append("RFOptuna")

# Conditionally add fit mode configs to __all__
if _FIT_CONFIGS_AVAILABLE:
    __all__.extend([
        "RFModelConfig",
        "RFLoraConfig",
        "RFSFTConfig",
        "RFDPOConfig",
        "RFGRPOConfig",
    ])

# Conditionally add evals mode configs to __all__
if _EVALS_CONFIGS_AVAILABLE:
    __all__.extend([
        "ModelConfig",
        "RFvLLMModelConfig",
        "RFAPIModelConfig",
    ])

# Conditionally add evals helper classes to __all__
if _EVALS_HELPERS_AVAILABLE:
    __all__.extend(["RFLangChainRagSpec", "RFPromptManager"])
