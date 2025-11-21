"""AutoML module for hyperparameter optimization (unified for fit and evals)."""

from .base import AutoMLAlgorithm
from .datatypes import List, Range
from .grid_search import RFGridSearch
from .random_search import RFRandomSearch
from .automl_utils import get_flattened_config_leaf, get_runs

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
    )
    _EVALS_CONFIGS_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    ModelConfig = None
    RFvLLMModelConfig = None
    RFOpenAIAPIModelConfig = None
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
    ])

# Conditionally add evals helper classes to __all__
if _EVALS_HELPERS_AVAILABLE:
    __all__.extend(["RFLangChainRagSpec", "RFPromptManager"])
