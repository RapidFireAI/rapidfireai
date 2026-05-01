"""Model configuration for AutoML training and evaluation."""

from __future__ import annotations
import copy
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Type, get_type_hints

from rapidfireai.automl.datatypes import List, Range


# Fit mode dependencies (peft, trl)
try:
    from peft import LoraConfig
    from trl import DPOConfig, GRPOConfig, SFTConfig

    _FIT_DEPS_AVAILABLE = True
except ImportError:
    # Handle case where fit dependencies are not available
    LoraConfig = None
    DPOConfig = None
    GRPOConfig = None
    SFTConfig = None
    _FIT_DEPS_AVAILABLE = False

# Evals mode dependencies (vllm)
try:
    from vllm import SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    # Handle case where vllm is not available
    SamplingParams = None
    _VLLM_AVAILABLE = False

# Evals mode dependencies (evals modules)
try:
    from rapidfireai.evals.rag.rag_pipeline import LangChainRagSpec
    from rapidfireai.evals.rag.prompt_manager import PromptManager
    from rapidfireai.evals.actors.inference_engines import (
        InferenceEngine,
        APIInferenceEngine,
        VLLMInferenceEngine,
    )

    _EVALS_MODULES_AVAILABLE = True
except ImportError:
    # Handle case where evals modules are not available
    LangChainRagSpec = None
    PromptManager = None
    InferenceEngine = None
    APIInferenceEngine = None
    VLLMInferenceEngine = None
    _EVALS_MODULES_AVAILABLE = False


def _make_unavailable_class(class_name: str, missing: str) -> type:
    """Build a sentinel placeholder class for an optional RF* config.

    Used when the optional dependency a config relies on is not installed
    (e.g. ``vllm`` for ``RFvLLMModelConfig``). Returning a real class — rather
    than ``None`` — keeps ``isinstance(x, RFXXX)`` safely ``False`` (no
    instance is ever produced) instead of raising
    ``TypeError: isinstance() arg 2 must be a type``.

    Trying to actually instantiate the sentinel raises an ``ImportError``
    that names the missing dependency.
    """
    msg = (
        f"{class_name} is unavailable because optional dependency "
        f"'{missing}' is not installed in this environment."
    )

    def __init__(self, *args, **kwargs):  # noqa: D401 - sentinel
        raise ImportError(msg)

    return type(class_name, (), {"__init__": __init__, "__doc__": msg})


def _create_rf_class(base_class: type, class_name: str):
    """Creating a RF class that dynamically inherits all constructor parameters and supports singleton, list, and Range values."""
    if not inspect.isclass(base_class):
        raise ValueError(f"base_class must be a class, got {type(base_class)}")

    sig = inspect.signature(base_class.__init__)
    constructor_params = [p for p in sig.parameters if p != "self"]

    type_hints = get_type_hints(base_class)
    new_type_hints = {}

    for param_name, param_type in type_hints.items():
        if param_name in constructor_params:
            new_type_hints[param_name] = param_type | List | Range

    def __init__(self, **kwargs):
        self._user_params = copy.deepcopy(kwargs)
        self._constructor_params = constructor_params
        self._initializing = True

        self._initializing = True

        parent_kwargs = {}
        for key, value in kwargs.items():
            if not isinstance(value, (List | Range)):
                parent_kwargs[key] = value

        base_class.__init__(self, **parent_kwargs)

        self._initializing = False

    def copy_config(self):
        """Create a deep copy of the configuration."""
        copied_params = copy.deepcopy(self._user_params)
        new_instance = self.__class__(**copied_params)

        return new_instance

    def __setattr__(self, name, value):
        """Override setattr to update _user_params when constructor parameters are modified."""

        if (
            hasattr(self, "_constructor_params")
            and name in self._constructor_params
            and hasattr(self, "_user_params")
            and name in self._user_params
            and not getattr(self, "_initializing", True)
        ):  # Don't update during init
            self._user_params[name] = value

        base_class.__setattr__(self, name, value)

    return type(
        class_name,
        (base_class,),
        {
            "__doc__": f"RF version of {base_class.__name__}",
            "__annotations__": new_type_hints,
            "__init__": __init__,
            "copy": copy_config,
            "__setattr__": __setattr__,
        },
    )


# ============================================================================
# Fit mode model configs
# ============================================================================

# Create RF wrapper classes for external libraries (fit mode)
# Only create these if fit dependencies are available
if _FIT_DEPS_AVAILABLE:
    RFLoraConfig = _create_rf_class(LoraConfig, "RFLoraConfig")
    RFSFTConfig = _create_rf_class(SFTConfig, "RFSFTConfig")
    RFDPOConfig = _create_rf_class(DPOConfig, "RFDPOConfig")
    RFGRPOConfig = _create_rf_class(GRPOConfig, "RFGRPOConfig")
else:
    RFLoraConfig = _make_unavailable_class("RFLoraConfig", "peft")
    RFSFTConfig = _make_unavailable_class("RFSFTConfig", "trl")
    RFDPOConfig = _make_unavailable_class("RFDPOConfig", "trl")
    RFGRPOConfig = _make_unavailable_class("RFGRPOConfig", "trl")


@dataclass
class RFModelConfig:
    """Model configuration for AutoML training."""

    model_name: str = None
    tokenizer: str | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    formatting_func: Callable | List | None = None
    compute_metrics: Callable | List | None = None
    peft_config: RFLoraConfig | List | None = None
    training_args: RFSFTConfig | RFDPOConfig | RFGRPOConfig | None = None
    # training_args = None
    model_type: str | None = "causal_lm"
    model_kwargs: dict[str, Any] | None = None
    ref_model_name: str | None = None
    ref_model_type: str | None = None
    ref_model_kwargs: dict[str, Any] | None = None
    reward_funcs: str | List | Callable | Any | None = None
    generation_config: dict[str, Any] | None = None
    num_gpus: int | None = None

    def copy(self):  # FIXME: Handle similar to create_rf_class
        """Create a deep copy of the RFModelConfig."""
        return copy.deepcopy(self)


# ============================================================================
# Evals mode model configs
# ============================================================================

# Conditionally define ModelConfig base class only if evals modules are available
if _EVALS_MODULES_AVAILABLE and InferenceEngine is not None:

    class ModelConfig(ABC):
        """Base configuration for model backends (evals mode)."""

        def __init__(self):
            pass

        @abstractmethod
        def get_engine_class(self) -> type[InferenceEngine]:
            """Return the inference engine class to use."""
            pass

        @abstractmethod
        def get_engine_kwargs(self) -> dict[str, Any]:
            """Return the kwargs needed to instantiate the inference engine."""
            pass

else:
    # Define a placeholder ABC if evals modules are not available
    class ModelConfig(ABC):
        """Base configuration for model backends (evals mode)."""

        def __init__(self):
            pass

        @abstractmethod
        def get_engine_class(self) -> Any:
            """Return the inference engine class to use."""
            pass

        @abstractmethod
        def get_engine_kwargs(self) -> dict[str, Any]:
            """Return the kwargs needed to instantiate the inference engine."""
            pass


def _create_rf_class_evals(base_class: Type, class_name: str):
    """Creating a RF class for evals that dynamically inherits all constructor parameters and supports singleton, list, and Range values."""
    if not inspect.isclass(base_class):
        raise ValueError(f"base_class must be a class, got {type(base_class)}")

    sig = inspect.signature(base_class.__init__)
    constructor_params = [p for p in sig.parameters.keys() if p != "self"]

    type_hints = get_type_hints(base_class)
    new_type_hints = {}

    for param_name, param_type in type_hints.items():
        if param_name in constructor_params:
            new_type_hints[param_name] = param_type | List | Range

    def __init__(self, **kwargs):
        self._user_params = copy.deepcopy(kwargs)
        self._constructor_params = constructor_params
        self._initializing = True

        parent_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, List):
                # Sample a default value from List for parent initialization
                # Keep original List in _user_params for AutoML sampling
                parent_kwargs[key] = value.sample()
            elif isinstance(value, Range):
                # Sample a default value from Range for parent initialization
                # Keep original Range in _user_params for AutoML sampling
                parent_kwargs[key] = value.sample()
            else:
                parent_kwargs[key] = value

        base_class.__init__(self, **parent_kwargs)

        self._initializing = False

    def copy_config(self):
        """Create a deep copy of the configuration."""
        copied_params = copy.deepcopy(self._user_params)
        new_instance = self.__class__(**copied_params)

        return new_instance

    def __setattr__(self, name, value):
        """Override setattr to update _user_params when constructor parameters are modified."""

        if (
            hasattr(self, "_constructor_params")
            and name in self._constructor_params
            and hasattr(self, "_user_params")
            and name in self._user_params
            and not getattr(self, "_initializing", True)
        ):  # Don't update during init
            self._user_params[name] = value

        base_class.__setattr__(self, name, value)

    return type(
        class_name,
        (base_class,),
        {
            "__doc__": f"RF version of {base_class.__name__}",
            "__annotations__": new_type_hints,
            "__init__": __init__,
            "copy": copy_config,
            "__setattr__": __setattr__,
        },
    )


# Conditionally create evals helper classes
if (
    _EVALS_MODULES_AVAILABLE
    and LangChainRagSpec is not None
    and PromptManager is not None
):
    RFLangChainRagSpec = _create_rf_class_evals(LangChainRagSpec, "RFLangChainRagSpec")
    RFPromptManager = _create_rf_class_evals(PromptManager, "RFPromptManager")
else:
    RFLangChainRagSpec = _make_unavailable_class(
        "RFLangChainRagSpec", "rapidfireai[evals]"
    )
    RFPromptManager = _make_unavailable_class(
        "RFPromptManager", "rapidfireai[evals]"
    )


# Conditionally define vLLM evals model config (requires vllm)
if (
    _VLLM_AVAILABLE
    and _EVALS_MODULES_AVAILABLE
    and SamplingParams is not None
    and InferenceEngine is not None
):

    class RFvLLMModelConfig(ModelConfig):
        """VLLM model configuration for evals mode."""

        def __init__(
            self,
            model_config: dict[str, Any],
            sampling_params: dict[str, Any],
            rag: LangChainRagSpec = None,
            prompt_manager: PromptManager = None,
        ):
            """
            Initialize VLLM model configuration.

            Args:
                model_config: VLLM model configuration (model name, dtype, etc.)
                sampling_params: Sampling parameters (temperature, top_p, etc.)
                rag: Optional RAG specification (index will be built automatically by Controller)
                prompt_manager: Optional prompt manager for few-shot examples
            """
            super().__init__()
            self.model_config = model_config
            self.sampling_params = SamplingParams(**sampling_params)
            self.rag = rag
            self.prompt_manager = prompt_manager
            self._user_params = {
                "model_config": model_config,
                "sampling_params": sampling_params,
                "rag": rag,
                "prompt_manager": prompt_manager,
            }

        @property
        def model_name(self) -> str:
            """Return the model name from model_config."""
            return self.model_config.get("model", "Unknown")

        def get_engine_class(self) -> type[InferenceEngine]:
            """Return VLLMInferenceEngine class."""
            return VLLMInferenceEngine

        def get_engine_kwargs(self) -> dict[str, Any]:
            """Return configuration for VLLMInferenceEngine."""
            return {
                "model_config": self.model_config,
                "sampling_params": self.sampling_params,
            }

        def sampling_params_to_dict(self) -> dict[str, Any]:
            """
            Convert vLLM SamplingParams object to dictionary.

            Extracts all sampling parameters from the vLLM SamplingParams object
            into a JSON-serializable dictionary for database storage.

            Returns:
                Dictionary of sampling parameters.
            """
            # Use vars() to get only the attributes actually set on the object
            # This works across different vLLM versions
            return dict(vars(self.sampling_params))

else:
    RFvLLMModelConfig = _make_unavailable_class("RFvLLMModelConfig", "vllm")


# Conditionally define API evals model config (does NOT require vllm)
if (
    _EVALS_MODULES_AVAILABLE
    and InferenceEngine is not None
    and APIInferenceEngine is not None
):

    class RFAPIModelConfig(ModelConfig):
        """General API model configuration for evals mode.

        Works with any provider (OpenAI, Gemini, Anthropic, etc.) via the MLflow
        gateway, which exposes all providers through an OpenAI-compatible API.

        During ``__init__``, the gateway secret, model definition(s), and
        endpoint(s) described by *endpoint_config* are provisioned (or reused
        if they already exist).  This means errors such as an unreachable
        MLflow server or invalid API keys surface immediately.

        ``endpoint`` inside *endpoint_config* can be:

        * A single ``dict``  – one endpoint for this pipeline (leaf config).
        * An ``automl.List`` – grid-search will expand it into individual
          ``RFAPIModelConfig`` instances, one per endpoint dict.

        A plain ``list`` is **not** supported because grid-search would not
        expand it, leading to silent misconfiguration.
        """

        def __init__(
            self,
            client_config: dict[str, Any],
            endpoint_config: dict[str, Any],
            model_config: dict[str, Any] = None,
            rag: LangChainRagSpec = None,
            prompt_manager: PromptManager = None,
            rpm_limit: int = None,
            tpm_limit: int = None,
            itpm_limit: int = None,
            otpm_limit: int = None,
            max_completion_tokens: int = None,
            verbose: bool = True,
        ):
            """
            Initialize API model configuration.

            Provisions the MLflow gateway resources (secret, model definitions,
            endpoints) described by *endpoint_config*.  Raises on failure so
            that misconfiguration is caught early.

            Args:
                client_config: OpenAI-compatible client kwargs (max_retries, timeout, …).
                    ``base_url`` and ``api_key`` are injected automatically by
                    :meth:`get_engine_kwargs` to point at the gateway.
                endpoint_config: MLflow gateway endpoint definition containing:

                    - ``provider`` (str): e.g. ``"openai"``, ``"gemini"``
                    - ``api_key_name`` (str): Name for the gateway secret
                    - ``api_key`` (str): The actual API key value
                    - ``api_base_url`` (str, optional): Provider base URL
                    - ``endpoint``: ``dict`` or ``automl.List`` of dicts.
                      Each dict must have ``"name"``; ``"model"`` is required
                      only when the endpoint does not already exist.

                model_config: Sampling/generation parameters for inference
                    (temperature, max_completion_tokens, etc.)
                rag: Optional RAG specification
                prompt_manager: Optional prompt manager for few-shot examples
                rpm_limit: Requests per minute limit (required)
                tpm_limit: Combined tokens per minute limit. Required for
                    every provider **except** ``"anthropic"``, where it is
                    rejected — Anthropic enforces separate input/output
                    quotas, so re-using a single ``tpm`` value as both
                    ``itpm`` and ``otpm`` would silently double the budget.
                itpm_limit: Input tokens per minute limit. Only accepted for
                    ``provider="anthropic"``; must be paired with *otpm_limit*.
                otpm_limit: Output tokens per minute limit. Only accepted for
                    ``provider="anthropic"``; must be paired with *itpm_limit*.
                max_completion_tokens: Max completion tokens per request.
                    Extracted from *model_config* if not provided.
                verbose: Print provisioning progress to stdout (default
                    ``True``).  Set to ``False`` to suppress prints while
                    still logging.  Used internally during grid/random
                    search expansion to avoid duplicate output.
            """
            super().__init__()
            self.client_config = client_config
            self.model_config = model_config or {}
            self.rag = rag
            self.prompt_manager = prompt_manager

            # --- Rate-limit validation ---
            # Providers fall into two schemes:
            #   * combined-tpm scheme:  tpm_limit only          (e.g. OpenAI, Gemini)
            #   * split scheme:         itpm_limit + otpm_limit (Anthropic)
            #
            # Mixing the two is dangerous: ``FineGrainedBaseRateLimiter`` falls
            # back to ``limits.get("itpm", limits.get("tpm"))`` and the same
            # for ``otpm``, so a single ``tpm`` value would silently be applied
            # as *both* itpm and otpm, effectively doubling the intended
            # token budget and triggering upstream 429s.
            provider = endpoint_config.get("provider", "openai")
            has_tpm = tpm_limit is not None
            has_itpm = itpm_limit is not None
            has_otpm = otpm_limit is not None
            has_itpm_otpm = has_itpm or has_otpm

            if has_itpm_otpm and provider != "anthropic":
                raise ValueError(
                    f"itpm_limit/otpm_limit are only supported for provider='anthropic'. "
                    f"Got provider='{provider}'. Use tpm_limit instead."
                )

            if has_tpm and provider == "anthropic":
                raise ValueError(
                    "tpm_limit is not supported for provider='anthropic'. "
                    "Anthropic enforces separate input/output token rate limits — "
                    "use itpm_limit and otpm_limit instead. (Re-using a single "
                    "tpm value would silently apply it as both itpm and otpm, "
                    "doubling the intended budget.)"
                )

            if has_tpm and has_itpm_otpm:
                raise ValueError(
                    "Specify either tpm_limit OR (itpm_limit and otpm_limit), not both."
                )

            if has_itpm != has_otpm:
                raise ValueError(
                    "Both itpm_limit and otpm_limit must be provided together."
                )

            if provider == "anthropic" and not has_itpm_otpm:
                raise ValueError(
                    "provider='anthropic' requires itpm_limit and otpm_limit "
                    "(Anthropic enforces separate input/output token quotas)."
                )

            if provider != "anthropic" and not has_tpm:
                raise ValueError(
                    f"provider='{provider}' requires tpm_limit (combined tokens per minute)."
                )

            self.rpm_limit = rpm_limit
            self.tpm_limit = tpm_limit
            self.itpm_limit = itpm_limit
            self.otpm_limit = otpm_limit

            if max_completion_tokens is None:
                max_completion_tokens = self.model_config.get("max_completion_tokens", 150)
            self.max_completion_tokens = max_completion_tokens

            # Preserve original user params (keeps automl List objects for grid search)
            self._user_params = {
                "client_config": client_config,
                "endpoint_config": endpoint_config,
                "model_config": model_config,
                "rag": rag,
                "prompt_manager": prompt_manager,
                "rpm_limit": rpm_limit,
                "tpm_limit": tpm_limit,
                "itpm_limit": itpm_limit,
                "otpm_limit": otpm_limit,
                "max_completion_tokens": max_completion_tokens,
            }

            # Validate endpoint: must be a single dict (leaf config) or an
            # automl List (grid-search will expand into individual dicts).
            # Plain lists are rejected — they would silently skip expansion.
            endpoint_raw = endpoint_config.get("endpoint")
            if endpoint_raw is None:
                raise ValueError("endpoint_config must include 'endpoint'")

            if not isinstance(endpoint_raw, (dict, List)):
                raise TypeError(
                    f"endpoint_config['endpoint'] must be a dict or automl List. "
                    f"Got: {type(endpoint_raw).__name__}. "
                    f"Use List([...]) from rapidfireai.automl to specify multiple endpoints."
                )

            # Provision gateway resources (idempotent get-or-create).
            # For an automl List the gateway client normalises to a plain
            # list internally; for a single dict it wraps it in a list.
            from rapidfireai.evals.utils.gateway_utils import MLflowGatewayClient

            gateway = MLflowGatewayClient()
            self._gateway_endpoints = gateway.provision_endpoints(endpoint_config, verbose=verbose)

            self.endpoint_config = endpoint_config

        @property
        def model_name(self) -> str:
            """Return the endpoint name for display/logging."""
            endpoint = self.endpoint_config.get("endpoint")
            if isinstance(endpoint, dict):
                return endpoint.get("name", "unknown")
            if isinstance(endpoint, List):
                return endpoint.values[0].get("name", "unknown") if endpoint.values else "unknown"
            return "unknown"

        def get_rate_limit_dict(self) -> dict[str, int]:
            """Build the rate-limit dict expected by the rate limiter actor.

            Returns a dict with ``"rpm"`` and either ``"tpm"`` (combined) or
            ``"itpm"``/``"otpm"`` (fine-grained), depending on what the user
            provided.
            """
            if self.itpm_limit is not None:
                return {"rpm": self.rpm_limit, "itpm": self.itpm_limit, "otpm": self.otpm_limit}
            return {"rpm": self.rpm_limit, "tpm": self.tpm_limit}

        def get_engine_class(self) -> type[InferenceEngine]:
            """Return APIInferenceEngine class."""
            return APIInferenceEngine

        def get_engine_kwargs(self) -> dict[str, Any]:
            """
            Return configuration for APIInferenceEngine.

            Injects the MLflow gateway ``base_url`` and a placeholder
            ``api_key`` into *client_config* so the engine's OpenAI client
            routes through the gateway (which handles real authentication).
            """
            from rapidfireai.utils.constants import MLflowConfig

            engine_client_config = {
                **self.client_config,
                "base_url": f"{MLflowConfig.URL}/gateway/mlflow/v1",
                "api_key": "dummy",
            }
            return {
                "client_config": engine_client_config,
                "endpoint_config": self.endpoint_config,
                "model_config": self.model_config,
            }

        def sampling_params_to_dict(self) -> dict[str, Any]:
            """Return the sampling configuration as a dictionary.

            Excludes keys that are surfaced separately in the serialized
            output or are passed explicitly per-request by
            :class:`APIInferenceEngine` (rather than via ``**model_config``):

            * ``"max_completion_tokens"`` — lifted to ``self.max_completion_tokens``
              and serialized at the top level; including it here would
              duplicate it in the IC Ops JSON view.
            * ``"model"`` — the endpoint name comes from ``endpoint_config``;
              and the engine passes ``model=self.model_name`` explicitly.
            * ``"messages"`` — never a sampling param; belongs in the
              request body.
            """
            _non_sampling_keys = {"max_completion_tokens", "model", "messages"}
            return {k: v for k, v in self.model_config.items() if k not in _non_sampling_keys}

else:
    RFAPIModelConfig = _make_unavailable_class(
        "RFAPIModelConfig", "rapidfireai[evals]"
    )
