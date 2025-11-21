"""Model configuration for AutoML training."""

import inspect
import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from rapidfireai.evals.rag.rag_pipeline import LangChainRagSpec
from rapidfireai.evals.rag.prompt_manager import PromptManager
from rapidfireai.evals.automl.datatypes import List, Range
from abc import ABC, abstractmethod
from typing import Any
from vllm import SamplingParams
from rapidfireai.evals.actors.inference_engines import InferenceEngine, OpenAIInferenceEngine, VLLMInferenceEngine

class ModelConfig(ABC):
    """Base configuration for model backends."""

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


class RFvLLMModelConfig(ModelConfig):
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
            "prompt_manager": prompt_manager
        }

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


class RFOpenAIAPIModelConfig(ModelConfig):
    def __init__(
        self,
        client_config: dict[str, Any],
        model_config: dict[str, Any],
        rag: LangChainRagSpec = None,
        prompt_manager: PromptManager = None,
        rpm_limit: int = None,
        tpm_limit: int = None,
        max_completion_tokens: int = None,
    ):
        """
        Initialize OpenAI API model configuration.

        Args:
            client_config: OpenAI client configuration (api_key, base_url, etc.)
            model_config: Model configuration (model name, temperature, etc.)
            rag: Optional RAG specification (index will be built automatically by Controller)
            prompt_manager: Optional prompt manager for few-shot examples
            rpm_limit: Requests per minute limit for this model (required for rate limiting)
            tpm_limit: Tokens per minute limit for this model (required for rate limiting)
            max_completion_tokens: Maximum completion tokens per request. If None, will be extracted
                                  from model_config if present, otherwise defaults to 150.
        """
        super().__init__()
        self.client_config = client_config
        self.model_config = model_config
        self.rag = rag
        self.prompt_manager = prompt_manager
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        
        # Extract max_completion_tokens from model_config if not provided
        if max_completion_tokens is None:
            max_completion_tokens = model_config.get("max_completion_tokens", 150)
        self.max_completion_tokens = max_completion_tokens
        
        self._user_params = {
            "client_config": client_config,
            "model_config": model_config,
            "rag": rag,
            "prompt_manager": prompt_manager,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
            "max_completion_tokens": max_completion_tokens,
        }

    def get_engine_class(self) -> type[InferenceEngine]:
        """Return OpenAIInferenceEngine class."""
        return OpenAIInferenceEngine

    def get_engine_kwargs(self) -> dict[str, Any]:
        """
        Return configuration for OpenAIInferenceEngine.

        Note: rate_limiter_actor will be added by Controller when creating actors.
        max_completion_tokens is available via self.max_completion_tokens if needed.
        """
        return {
            "client_config": self.client_config,
            "model_config": self.model_config,
        }

    def sampling_params_to_dict(self) -> dict[str, Any]:
        """
        Extract sampling parameters from OpenAI model_config.

        For OpenAI models, sampling parameters are stored directly in model_config
        (e.g., temperature, top_p, max_completion_tokens).

        Returns:
            Dictionary of sampling parameters.
        """
        # Extract sampling-related parameters from model_config
        sampling_keys = [
            "temperature", "top_p", "max_completion_tokens",
            "frequency_penalty", "presence_penalty", "seed",
            "reasoning_effort"  # For o1 models
        ]
        return {key: self.model_config.get(key) for key in sampling_keys if key in self.model_config}

def _create_rf_class(base_class: Type, class_name: str):
    """Creating a RF class that dynamically inherits all constructor parameters and supports singleton, list, and Range values."""
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
        
        if (hasattr(self, '_constructor_params') and 
            name in self._constructor_params and 
            hasattr(self, '_user_params') and
            name in self._user_params and
            not getattr(self, '_initializing', True)):  # Don't update during init
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
            "__setattr__": __setattr__
        },
    )

RFLangChainRagSpec = _create_rf_class(LangChainRagSpec, "RFLangChainRagSpec")
RFPromptManager = _create_rf_class(PromptManager, "RFPromptManager")
