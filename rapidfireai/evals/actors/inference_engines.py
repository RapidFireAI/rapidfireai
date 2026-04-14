"""
Inference engines for different model backends.

This module provides pluggable inference engines that can be used with
QueryProcessingActor for model inference. Each engine encapsulates the
logic for a specific backend (VLLM for local models, OpenAI for API).
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, override

import mlflow
from mlflow.entities import SpanType
from rapidfireai.evals.utils.mlflow_utils import mlflow_trace, mlflow_start_span, mlflow_get_current_active_span
from rapidfireai.evals.utils.ratelimiter import RequestStatus


class InferenceEngine(ABC):
    """Abstract base class for model inference engines."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the inference engine with configuration."""
        pass

    @abstractmethod
    def generate(self, prompts: list, **kwargs) -> list[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompts (each prompt can be a list of message dicts)
            **kwargs: Additional generation parameters

        Returns:
            List of generated text strings
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (models, connections, etc.)."""
        pass


class VLLMInferenceEngine(InferenceEngine):
    """VLLM-based inference engine for local model inference."""

    def __init__(self, model_config: dict[str, Any], sampling_params: Any):
        """
        Initialize VLLM inference engine.

        Args:
            model_config: Configuration for VLLM LLM (model name, dtype, etc.)
            sampling_params: VLLM SamplingParams object
        """
        # Set environment variables before importing to disable all progress bars
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        from vllm import LLM

        # Disable VLLM's logging for cleaner output
        model_config["disable_log_stats"] = True
        self.model_name = model_config.get("model", "unknown")
        self.llm = LLM(**model_config)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = sampling_params

    @override
    @mlflow_trace(name="vllm_generate", span_type=SpanType.LLM)
    def generate(self, prompts: list, **kwargs) -> list[str]:
        """
        Generate responses using VLLM.

        Args:
            prompts: List of prompts (each is a list of message dicts)

        Returns:
            List of generated text strings
        """
        span = mlflow_get_current_active_span()
        if span is not None:
            span.set_attribute("model", self.model_name)

        # Apply chat template to format prompts
        formatted_prompts = [
            self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts
        ]

        # Generate with VLLM (disable internal progress bar to avoid tqdm issues)
        outputs = self.llm.generate(formatted_prompts, sampling_params=self.sampling_params, use_tqdm=False)

        # Create a child span per prompt-output pair
        results = []
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            generated_text = output.outputs[0].text
            with mlflow_start_span(name=f"prompt_{i}", span_type=SpanType.LLM) as child_span:
                child_span.set_inputs({
                    "messages": prompt,
                    "formatted_prompt": formatted_prompts[i],
                })
                child_span.set_outputs({"generated_text": generated_text})
                child_span.set_attributes({
                    "model": self.model_name,
                    "index": i,
                    "finish_reason": output.outputs[0].finish_reason,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                })
            results.append(generated_text)

        return results

    @override
    def cleanup(self):
        """Clean up VLLM resources."""
        del self.llm

class APIInferenceEngine(InferenceEngine):
    """API-based inference engine with distributed rate limiting."""

    def __init__(
        self,
        client_config: dict[str, Any],
        endpoint_config: dict[str, Any],
        model_config: dict[str, Any],
        rate_limiter_actor: Any,
        max_completion_tokens: int = 150,
    ):
        """
        Initialize shared API engine state.

        After this call the following attributes are set:

        - ``self.endpoint_config``: MLflow gateway endpoint definition
        - ``self.model_config``: copy of *model_config* with request-level
          keys removed (they are passed explicitly per-request)
        - ``self.model_name``: primary endpoint name from *endpoint_config*
        - ``self.rate_limiter_actor``: the shared Ray rate-limiter handle
        - ``self.max_completion_tokens``: maximum output tokens per request
        """
        if rate_limiter_actor is None:
            raise ValueError(
                f"rate_limiter_actor cannot be None for {type(self).__name__}. "
                "API pipelines require rate limiting. Please ensure the Controller "
                "properly injects the rate_limiter_actor."
            )
        self.endpoint_config = endpoint_config
        self.model_config = model_config.copy() if model_config else {}
        endpoint = endpoint_config.get("endpoint", {})
        self.model_name = endpoint["name"] if isinstance(endpoint, dict) else "unknown"
        self.rate_limiter_actor = rate_limiter_actor
        self.max_completion_tokens = max_completion_tokens
        self.model_config.pop("messages", None)
        self.model_config.pop("max_completion_tokens", None)

        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(**client_config)

    async def create_completion(self, messages: list[dict], request_id: int) -> str:
        """
        Create a completion using the API.

        Args:
            messages: List of message dicts for this request

        Returns:
            Generated text string or error message
        """
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_completion_tokens=self.max_completion_tokens,
            **self.model_config,
        )

        # Get the response content
        content = response.choices[0].message.content
        # Check if response is empty
        if content is None or content.strip() == "":
            # Silently handle empty response (avoid flooding notebook output)
            if request_id is not None:
                update_ref = self.rate_limiter_actor.update_actual_usage.remote(
                    request_id,
                    response.usage.total_tokens,
                    RequestStatus.EMPTY_RESPONSE,
                )
                await update_ref
            return ""

        # Successfully completed with content
        if request_id is not None:
            update_ref = self.rate_limiter_actor.update_actual_usage.remote(
                request_id,
                response.usage.total_tokens,
                RequestStatus.COMPLETED,
            )
            await update_ref

        return content
    
    @override
    def generate(self, prompts: list, **kwargs) -> list[str]:
        """
        Generate responses using API with rate limiting.

        Args:
            prompts: List of prompts (each is a list of message dicts)

        Returns:
            List of generated text strings
        """
        # Run async batch completions
        try:
            loop = asyncio.get_running_loop()
            # Already in event loop (Ray actor context)
            return loop.run_until_complete(self._batch_completions(prompts))
        except RuntimeError:
            # No event loop
            return asyncio.run(self._batch_completions(prompts))

    async def _batch_completions(self, prompts: list) -> list[str]:
        """
        Process batch of prompts with concurrent API calls.

        Args:
            prompts: List of prompts (message lists)

        Returns:
            List of generated text strings
        """
        tasks = [self._rate_limited_request(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Ensure all results are strings (convert any exception objects)
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # If gather returned an exception object, convert to error string
                # Silently convert to error string (avoid flooding notebook output)
                processed_results.append(f"ERROR: {str(result)}")
            elif result is None:
                # Handle None case (silently)
                processed_results.append("")
            else:
                processed_results.append(result)

        return processed_results

    async def _rate_limited_request(self, messages):
        """
        Make a single rate-limited API request using centralized Ray actor.

        Args:
            messages: List of message dicts for this request

        Returns:
            Generated text string or error message
        """
        # Estimate tokens using remote call to rate limiter actor
        estimated_tokens_ref = self.rate_limiter_actor.estimate_total_tokens.remote(messages, self.model_name)
        estimated_tokens = await estimated_tokens_ref

        # Wait for rate limit slot using remote call
        while True:
            acquire_ref = self.rate_limiter_actor.acquire_slot.remote(estimated_tokens, self.model_name)
            can_proceed, wait_time, request_id = await acquire_ref

            if can_proceed:
                break
            # Silently wait for rate limit (no print to avoid flooding notebook output)
            await asyncio.sleep(wait_time)

        try:
            content = await self.create_completion(messages, request_id)
            return content

        except Exception as e:
            # Request failed - update status as FAILED
            # Suppress print for rate limit errors (429) - they're expected and handled by the rate limiter
            from rapidfireai.evals.utils.error_utils import is_rate_limit_error

            if not is_rate_limit_error(e):
                # Only print non-rate-limit errors
                print(f"API Error for request {request_id}: {str(e)}")

            if request_id is not None:
                update_ref = self.rate_limiter_actor.update_actual_usage.remote(
                    request_id,
                    0,
                    RequestStatus.FAILED
                )
                await update_ref
            # Return error message as string
            return f"ERROR: {str(e)}"

    @override
    def cleanup(self):
        """Clean up API client resources (no-op for stateless HTTP clients)."""
        pass


# Export classes for external use
__all__ = ["InferenceEngine", "VLLMInferenceEngine", "APIInferenceEngine"]
