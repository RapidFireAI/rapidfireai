"""
Ray Actor for centralized OpenAI API rate limiting.

Provides a single point of coordination for rate limiting across all
distributed query processing actors.
"""
import ray

from rapidfireai.utils.constants import RF_EXPERIMENT_PATH
from rapidfireai.evals.utils.ratelimiter import OpenAIRateLimiter, RequestStatus
from rapidfireai.evals.utils.logger import RFLogger


@ray.remote
class RateLimiterActor:
    """
    Centralized rate limiter as a Ray actor.

    All query processing actors make remote calls to this single actor
    to coordinate rate limiting across the entire distributed system.
    """

    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        experiment_name: str = "unknown",
        experiment_path: str = RF_EXPERIMENT_PATH,
    ):
        """
        Initialize the centralized rate limiter with per-model rate limits.

        Args:
            model_rate_limits: Dict mapping model name to rate limits, e.g.
                {"gpt-4": {"rpm": 500, "tpm": 50000}, "gpt-3.5-turbo": {"rpm": 1000, "tpm": 100000}}
            max_completion_tokens: Maximum completion tokens per request
            limit_safety_ratio: Safety margin (default 0.98 = 98% of limit)
            minimum_wait_time: Minimum wait time when rate limited (seconds)
            experiment_name: Name of the experiment for logging
            experiment_path: Path to experiment logs/artifacts
        """
        # Initialize logger
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        logger = logging_manager.get_logger("RateLimiterActor")

        self.limiter = OpenAIRateLimiter(
            model_rate_limits=model_rate_limits,
            max_completion_tokens=max_completion_tokens,
            limit_safety_ratio=limit_safety_ratio,
            minimum_wait_time=minimum_wait_time,
            logger=logger,
        )

    async def acquire_slot(self, estimated_tokens: int, model_name: str):
        """
        Try to acquire a slot for a new request for a specific model.

        Args:
            estimated_tokens: Projected token usage for this request
            model_name: Name of the model making the request

        Returns:
            Tuple of (can_proceed: bool, wait_time: float, request_id: Optional[int])
        """
        return await self.limiter.acquire_slot(estimated_tokens, model_name)

    async def update_actual_usage(self, request_id: int, actual_tokens: int, status: RequestStatus):
        """
        Update actual token usage after request completion.

        Args:
            request_id: ID of the request
            actual_tokens: Actual tokens used
            status: Request status (COMPLETED, FAILED, EMPTY_RESPONSE)
        """
        await self.limiter.update_actual_usage(request_id, actual_tokens, status)

    def estimate_total_tokens(self, messages: list[dict], model_name: str) -> int:
        """
        Estimate total tokens for a request.

        Args:
            messages: List of message dicts
            model_name: OpenAI model name

        Returns:
            Estimated total tokens (prompt + completion)
        """
        return self.limiter.estimate_total_tokens(messages, model_name)

    async def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current RPM, TPM, and limits per model
        """
        return await self.limiter.get_current_usage()


# Export for use in other modules
__all__ = ["RateLimiterActor"]

