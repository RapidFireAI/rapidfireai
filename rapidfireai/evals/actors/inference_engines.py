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

from mlflow.entities import SpanType

from rapidfireai.evals.utils.mlflow_utils import mlflow_get_current_active_span, mlflow_start_span, mlflow_trace
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
        model_config: dict[str, Any],
        rate_limiter_actor: Any,
        max_completion_tokens: int = 150,
    ):
        """
        Initialize shared API engine state.

        Subclasses must call ``super().__init__()`` and then create their
        specific client.  After this call the following attributes are set:

        - ``self.model_config``: copy of *model_config* with ``"messages"``
          removed (it goes in the request, not the config)
        - ``self.model_name``: value of ``model_config["model"]``
        - ``self.rate_limiter_actor``: the shared Ray rate-limiter handle
        - ``self.max_completion_tokens``: maximum output tokens per request
        """
        if rate_limiter_actor is None:
            raise ValueError(
                f"rate_limiter_actor cannot be None for {type(self).__name__}. "
                "API pipelines require rate limiting. Please ensure the Controller "
                "properly injects the rate_limiter_actor."
            )
        self.model_config = model_config.copy()
        self.model_name = self.model_config["model"]
        self.rate_limiter_actor = rate_limiter_actor
        self.max_completion_tokens = max_completion_tokens
        # "messages" belongs in the request body, not the reusable config
        self.model_config.pop("messages", None)

    @abstractmethod
    async def create_completion(self, messages: list[dict], request_id: int) -> str:
        """
        Create a completion using the API.

        Args:
            messages: List of message dicts for this request

        Returns:
            Generated text string or error message
        """
        pass

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


class OpenAIInferenceEngine(APIInferenceEngine):
    """OpenAI API-based inference engine with distributed rate limiting."""

    def __init__(
        self,
        client_config: dict[str, Any],
        model_config: dict[str, Any],
        rate_limiter_actor: Any,
        max_completion_tokens: int = 150,
    ):
        """
        Initialize OpenAI inference engine.

        Args:
            client_config: Configuration for AsyncOpenAI client (api_key, base_url, etc.)
            model_config: Model configuration (model name, temperature, etc.)
            rate_limiter_actor: Ray ActorHandle to RateLimiterActor for distributed rate limiting
            max_completion_tokens: Maximum completion tokens per request
        """
        from openai import AsyncOpenAI

        super().__init__(
            client_config=client_config,
            model_config=model_config,
            rate_limiter_actor=rate_limiter_actor,
            max_completion_tokens=max_completion_tokens,
        )
        self.client = AsyncOpenAI(**client_config)
        # "max_completion_tokens" is passed explicitly in the request, not via model_config
        self.model_config.pop("max_completion_tokens", None)

    @override
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

class AnthropicInferenceEngine(APIInferenceEngine):
    """Anthropic API-based inference engine with distributed rate limiting."""

    def __init__(
        self,
        client_config: dict[str, Any],
        model_config: dict[str, Any],
        rate_limiter_actor: Any,
        max_completion_tokens: int = 150,
    ):
        """
        Initialize Anthropic inference engine.

        Args:
            client_config: Configuration for AsyncAnthropic client (api_key, base_url, etc.)
            model_config: Model configuration (model name, temperature, etc.)
            rate_limiter_actor: Ray ActorHandle to RateLimiterActor for distributed rate limiting
            max_completion_tokens: Maximum completion tokens per request (passed to Anthropic as max_tokens)
        """
        from anthropic import AsyncAnthropic

        super().__init__(
            client_config=client_config,
            model_config=model_config,
            rate_limiter_actor=rate_limiter_actor,
            max_completion_tokens=max_completion_tokens,
        )
        self.client = AsyncAnthropic(**client_config)
        # "max_tokens" is passed explicitly in the request, not via model_config
        self.model_config.pop("max_tokens", None)

    @override
    async def create_completion(self, messages: list[dict], request_id: int) -> str:
        """
        Create a completion using the Anthropic API.

        Args:
            messages: List of message dicts for this request

        Returns:
            Generated text string or error message
        """
        response = await self.client.messages.create(
            messages=messages,
            max_tokens=self.max_completion_tokens,
            **self.model_config,
        )

        # Extract response text from the first content block
        content = response.content[0].text if response.content else None

        # Calculate total tokens from Anthropic's split usage fields
        total_tokens = 0
        if response.usage:
            total_tokens = response.usage.input_tokens + response.usage.output_tokens

        # Check if response is empty
        if content is None or content.strip() == "":
            if request_id is not None:
                update_ref = self.rate_limiter_actor.update_actual_usage.remote(
                    request_id,
                    total_tokens,
                    RequestStatus.EMPTY_RESPONSE,
                )
                await update_ref
            return ""

        # Successfully completed with content
        if request_id is not None:
            update_ref = self.rate_limiter_actor.update_actual_usage.remote(
                request_id,
                total_tokens,
                RequestStatus.COMPLETED,
            )
            await update_ref

        return content


class GoogleGeminiInferenceEngine(APIInferenceEngine):
    """Google Gemini API-based inference engine with distributed rate limiting."""

    def __init__(
        self,
        client_config: dict[str, Any],
        model_config: dict[str, Any],
        rate_limiter_actor: Any,
        max_completion_tokens: int = 150,
    ):
        """
        Initialize Google Gemini inference engine.

        Args:
            client_config: Configuration for genai.Client (api_key, vertexai, project, etc.)
            model_config: Model configuration (model name, temperature, etc.)
            rate_limiter_actor: Ray ActorHandle to RateLimiterActor for distributed rate limiting
            max_completion_tokens: Maximum output tokens per request (passed to Gemini as max_output_tokens)
        """
        from google import genai

        super().__init__(
            client_config=client_config,
            model_config=model_config,
            rate_limiter_actor=rate_limiter_actor,
            max_completion_tokens=max_completion_tokens,
        )
        # Use the async client directly; model is passed separately in each request
        self.client = genai.Client(**client_config).aio
        # Gemini receives model as a positional arg to generate_content, not via config
        self.model_config.pop("model", None)
        self.model_config.pop("max_output_tokens", None)

    @staticmethod
    def _normalize_parts(parts: list) -> list[dict]:
        """Convert plain strings in a parts list to Gemini Part dicts: {"text": "..."}."""
        return [p if isinstance(p, dict) else {"text": p} for p in parts]

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[list[dict], str | None]:
        """
        Partition messages into Gemini-compatible contents and a system instruction string.

        Normalizes ``parts`` entries from plain strings to ``{"text": "..."}`` dicts so
        notebooks can use the readable shorthand ``"parts": ["some text"]``.

        Returns:
            (contents, system_instruction) where system_instruction is None if absent.
        """
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        non_system = [msg for msg in messages if msg["role"] != "system"]

        # Normalize parts in non-system messages to {"text": "..."} dicts
        contents = [
            {**msg, "parts": GoogleGeminiInferenceEngine._normalize_parts(msg.get("parts", []))}
            for msg in non_system
        ]

        # Flatten system parts to a single instruction string
        system_parts = [part for msg in system_msgs for part in msg.get("parts", [])]
        system_instruction = "\n".join(
            p if isinstance(p, str) else p.get("text", "") for p in system_parts
        ) if system_parts else None

        return contents, system_instruction

    @override
    async def create_completion(self, messages: list[dict], request_id: int) -> str:
        """
        Create a completion using the API.

        Args:
            messages: List of message dicts for this request

        Returns:
            Generated text string or error message
        """
        from google.genai.types import GenerateContentConfigDict

        contents, system_instruction = self._convert_messages(messages)

        config = {**self.model_config, "max_output_tokens": self.max_completion_tokens}
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = await self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfigDict(**config),
        )

        content = response.text
        total_tokens = (
            response.usage_metadata.total_token_count
            if response.usage_metadata else 0
        )

        if not content or not content.strip():
            if request_id is not None:
                await self.rate_limiter_actor.update_actual_usage.remote(
                    request_id, total_tokens, RequestStatus.EMPTY_RESPONSE
                )
            return ""

        if request_id is not None:
            await self.rate_limiter_actor.update_actual_usage.remote(
                request_id, total_tokens, RequestStatus.COMPLETED
            )
        return content


# Export classes for external use
__all__ = [
    "InferenceEngine",
    "VLLMInferenceEngine",
    "APIInferenceEngine",
    "OpenAIInferenceEngine",
    "AnthropicInferenceEngine",
    "GoogleGeminiInferenceEngine",
]
