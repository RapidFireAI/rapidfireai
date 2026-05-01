"""
Ray Actor for centralized API rate limiting.

Provides a single point of coordination for rate limiting across all
distributed query processing actors. The ``backend`` parameter maps to the
provider string from ``endpoint_config["provider"]``.  Supported backends
have dedicated tokenizer implementations; unsupported providers fall back
to the OpenAI rate limiter (tiktoken with gpt-5 tokenizer).
"""
import ray

from rapidfireai.utils.constants import RF_EXPERIMENT_PATH
from rapidfireai.evals.utils.ratelimiter import (
    OpenAIRateLimiter,
    GoogleGeminiRateLimiter,
    AnthropicRateLimiter,
    RequestStatus,
)
from rapidfireai.evals.utils.logger import RFLogger

_BACKEND_MAP = {
    "openai": OpenAIRateLimiter,
    "azure": OpenAIRateLimiter,
    "gemini": GoogleGeminiRateLimiter,
    "anthropic": AnthropicRateLimiter,
}


@ray.remote
class RateLimiterActor:
    """
    Centralized rate limiter as a Ray actor.

    All query processing actors make remote calls to this single actor
    to coordinate rate limiting across the entire distributed system.

    The ``backend`` parameter accepts any provider string (e.g. ``"openai"``,
    ``"gemini"``, ``"anthropic"``, ``"cohere"``).  Providers with a dedicated
    implementation in ``_BACKEND_MAP`` get their specialised tokenizer;
    all others fall back to ``OpenAIRateLimiter`` which uses tiktoken.
    """

    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        backend: str = "openai",
        experiment_name: str = "unknown",
        experiment_path: str = RF_EXPERIMENT_PATH,
    ):
        """
        Initialize the centralized rate limiter with per-model rate limits.

        Args:
            model_rate_limits: Dict mapping endpoint name to rate limits, e.g.
                {"my-chat-endpoint": {"rpm": 500, "tpm": 50000}}
            max_completion_tokens: Maximum completion tokens per request
            limit_safety_ratio: Safety margin (default 0.98 = 98% of limit)
            minimum_wait_time: Minimum wait time when rate limited (seconds)
            backend: Provider name (e.g. ``"openai"``, ``"gemini"``,
                ``"anthropic"``).  Falls back to OpenAI if unsupported.
            experiment_name: Name of the experiment for logging
            experiment_path: Path to experiment logs/artifacts
        """
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        logger = logging_manager.get_logger("RateLimiterActor")

        limiter_cls = _BACKEND_MAP.get(backend)
        if limiter_cls is None:
            logger.warning(
                f"No dedicated rate limiter for provider '{backend}', "
                f"falling back to OpenAI rate limiter (tiktoken)"
            )
            limiter_cls = OpenAIRateLimiter

        self.limiter = limiter_cls(
            model_rate_limits=model_rate_limits,
            max_completion_tokens=max_completion_tokens,
            limit_safety_ratio=limit_safety_ratio,
            minimum_wait_time=minimum_wait_time,
            logger=logger,
        )

    async def acquire_slot(self, estimated_input_tokens: int, model_name: str):
        """
        Try to acquire a slot for a new request for a specific model.

        Args:
            estimated_input_tokens: Projected input/prompt token usage.
                Output tokens are projected internally by the limiter.
            model_name: Name of the model making the request

        Returns:
            Tuple of (can_proceed: bool, wait_time: float, request_id: Optional[int])
        """
        return await self.limiter.acquire_slot(estimated_input_tokens, model_name)

    async def update_actual_usage(
        self,
        request_id: int,
        actual_input_tokens: int,
        actual_output_tokens: int,
        status: RequestStatus,
    ):
        """
        Update actual token usage after request completion.

        Args:
            request_id: ID of the request
            actual_input_tokens: Actual input/prompt tokens used
            actual_output_tokens: Actual output/completion tokens used
            status: Request status (COMPLETED, FAILED, EMPTY_RESPONSE)
        """
        await self.limiter.update_actual_usage(
            request_id, actual_input_tokens, actual_output_tokens, status,
        )

    def count_prompt_tokens(self, messages: list[dict], model_name: str) -> int:
        """
        Count input/prompt tokens for a request.

        Args:
            messages: List of message dicts
            model_name: Model name (must match a key in model_rate_limits)

        Returns:
            Estimated input token count
        """
        return self.limiter.count_prompt_tokens(messages, model_name)

    async def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current RPM, TPM, and limits per model
        """
        return await self.limiter.get_current_usage()


# Export for use in other modules
__all__ = ["RateLimiterActor"]

