"""
Ray Actor for centralized OpenAI API rate limiting.

Provides a single point of coordination for rate limiting across all
distributed query processing actors.
"""

import ray

from rapidfireai.infer.utils.ratelimiter import OpenAIRateLimiter, RequestStatus


@ray.remote
class RateLimiterActor:
    """
    Centralized rate limiter as a Ray actor.

    All query processing actors make remote calls to this single actor
    to coordinate rate limiting across the entire distributed system.
    """

    def __init__(
        self,
        model_names: list[str] | str,
        rpm_limit: int = 500,
        tpm_limit: int = 500000,
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
    ):
        """
        Initialize the centralized rate limiter.

        Args:
            model_names: OpenAI model name(s) for token counting
            rpm_limit: Requests per minute limit
            tpm_limit: Tokens per minute limit
            max_completion_tokens: Maximum completion tokens per request
            limit_safety_ratio: Safety margin (default 0.98 = 98% of limit)
            minimum_wait_time: Minimum wait time when rate limited (seconds)
        """
        self.limiter = OpenAIRateLimiter(
            model_names=model_names,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            max_completion_tokens=max_completion_tokens,
            limit_safety_ratio=limit_safety_ratio,
            minimum_wait_time=minimum_wait_time,
        )

    async def acquire_slot(self, estimated_tokens: int):
        """
        Try to acquire a slot for a new request.

        Args:
            estimated_tokens: Projected token usage for this request

        Returns:
            Tuple of (can_proceed: bool, wait_time: float, request_id: Optional[int])
        """
        return await self.limiter.acquire_slot(estimated_tokens)

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

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current RPM, TPM, and limits
        """
        return {
            "rpm_limit": self.limiter.enforced_rpm_limit,
            "tpm_limit": self.limiter.enforced_tpm_limit,
            "total_requests": len(self.limiter._all_requests),
        }


# Export for use in other modules
__all__ = ["RateLimiterActor"]

