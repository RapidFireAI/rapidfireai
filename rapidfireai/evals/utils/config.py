from abc import ABC, abstractmethod
from typing import Any

from vllm import SamplingParams

from rapidfireai.evals.actors.inference_engines import InferenceEngine, OpenAIInferenceEngine, VLLMInferenceEngine
from rapidfireai.evals.rag.context_generator import ContextGenerator


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


class VLLMModelConfig(ModelConfig):
    def __init__(
        self,
        model_config: dict[str, Any],
        sampling_params: dict[str, Any],
        context_generator: ContextGenerator = None,
    ):
        """
        Initialize VLLM model configuration.

        Args:
            model_config: VLLM model configuration (model name, dtype, etc.)
            sampling_params: Sampling parameters (temperature, top_p, etc.)
            context_generator: Optional context generator containing rag_spec and/or prompt_manager
                             (index will be built automatically by Controller)
        """
        super().__init__()
        self.model_config = model_config
        self.sampling_params = SamplingParams(**sampling_params)
        self.context_generator = context_generator

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


class OpenAIAPIModelConfig(ModelConfig):
    def __init__(
        self,
        client_config: dict[str, Any],
        model_config: dict[str, Any],
        context_generator: ContextGenerator = None,
    ):
        """
        Initialize OpenAI API model configuration.

        Args:
            client_config: OpenAI client configuration (api_key, base_url, etc.)
            model_config: Model configuration (model name, temperature, etc.)
            context_generator: Optional context generator containing rag_spec and/or prompt_manager
                             (index will be built automatically by Controller)

        Note:
            Rate limiting (rpm_limit, tpm_limit, max_completion_tokens) is now configured at the
            Experiment level, since rate limits are per API key (experiment-wide), not per config.
        """
        super().__init__()
        self.client_config = client_config
        self.model_config = model_config
        self.context_generator = context_generator

    def get_engine_class(self) -> type[InferenceEngine]:
        """Return OpenAIInferenceEngine class."""
        return OpenAIInferenceEngine

    def get_engine_kwargs(self) -> dict[str, Any]:
        """
        Return configuration for OpenAIInferenceEngine.

        Note: rate_limiter_actor and max_completion_tokens will be added by Controller when creating actors.
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
