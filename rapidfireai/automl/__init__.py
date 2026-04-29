"""AutoML module for hyperparameter optimization (unified for fit and evals)."""

from .base import AutoMLAlgorithm
from .datatypes import List, Range
from .grid_search import RFGridSearch
from .random_search import RFRandomSearch
from .automl_utils import get_flattened_config_leaf, get_runs

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
    RFDPOConfig = None
    RFGRPOConfig = None
    RFLoraConfig = None
    RFModelConfig = None
    RFSFTConfig = None
    _FIT_CONFIGS_AVAILABLE = False

# Import evals mode configs (conditionally available)
try:
    from .model_config import (
        ModelConfig,
        RFvLLMModelConfig,
        RFOpenAIAPIModelConfig,
        RFGeminiAPIModelConfig,
    )
    _EVALS_CONFIGS_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    ModelConfig = None
    RFvLLMModelConfig = None
    RFOpenAIAPIModelConfig = None
    RFGeminiAPIModelConfig = None
    _EVALS_CONFIGS_AVAILABLE = False

# Conditionally import evals-specific helper classes
try:
    from .model_config import RFLangChainRagSpec, RFPromptManager
    _EVALS_HELPERS_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    RFLangChainRagSpec = None
    RFPromptManager = None
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
        "RFOpenAIAPIModelConfig",
        "RFGeminiAPIModelConfig",
    ])

# Conditionally add evals helper classes to __all__
if _EVALS_HELPERS_AVAILABLE:
    __all__.extend(["RFLangChainRagSpec", "RFPromptManager"])
