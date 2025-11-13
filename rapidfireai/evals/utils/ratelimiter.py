import asyncio
import time
from dataclasses import dataclass
from enum import Enum

import tiktoken


class RequestStatus(Enum):
    """Status of a rate-limited request"""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    EMPTY_RESPONSE = "empty_response"


@dataclass
class RequestRecord:
    """Record for tracking an individual request"""

    request_id: int
    timestamp: float  # When the request acquired a slot
    projected_tokens: int  # Estimated max tokens
    status: RequestStatus
    model_name: str  # Model name for this request
    actual_tokens: int | None = None  # Actual usage, known only after completion


class OpenAIRateLimiter:
    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        logger=None,
    ):
        """
        Initialize the rate limiter with sliding window tracking and per-model rate limits.

        Args:
            model_rate_limits: Dict mapping model name to rate limits, e.g.
                {"gpt-4": {"rpm": 500, "tpm": 50000}, "gpt-3.5-turbo": {"rpm": 1000, "tpm": 100000}}
            max_completion_tokens: Maximum completion tokens per request
            limit_safety_ratio: Safety margin as percentage of limits (default 0.98 = 98%)
            minimum_wait_time: Minimum wait time when rate limited (default 3.0 seconds)
            logger: Optional logger instance for logging rate limit messages
        """
        # Configuration
        self.limit_safety_ratio = limit_safety_ratio
        self.minimum_wait_time = minimum_wait_time
        self.max_completion_tokens = max_completion_tokens
        self.logger = logger

        # Throttling for rate limit messages (log 1 in 500)
        self._rate_limit_message_counter = 0
        self._log_throttle_ratio = 500

        # Per-model rate limits
        self.model_rate_limits = model_rate_limits
        self.model_names = list(model_rate_limits.keys())

        # Actual API limits per model (as specified by the API provider)
        self.actual_rpm_limits: dict[str, int] = {
            model: limits["rpm"] for model, limits in model_rate_limits.items()
        }
        self.actual_tpm_limits: dict[str, int] = {
            model: limits["tpm"] for model, limits in model_rate_limits.items()
        }

        # Enforced limits per model (with safety margin applied)
        self.enforced_rpm_limits: dict[str, int] = {
            model: int(limit_safety_ratio * limits["rpm"])
            for model, limits in model_rate_limits.items()
        }
        self.enforced_tpm_limits: dict[str, int] = {
            model: int(limit_safety_ratio * limits["tpm"])
            for model, limits in model_rate_limits.items()
        }

        # Request tracking
        self._request_counter = 0  # Unique ID generator

        # Current requests (sliding 60-second window) - used for rate limiting
        self._current_requests: dict[int, RequestRecord] = {}  # request_id -> RequestRecord

        # Historical requests (all requests since start) - never deleted
        self._all_requests: dict[int, RequestRecord] = {}  # request_id -> RequestRecord

        # For token counting - support multiple models
        self.encoders: dict[str, tiktoken.Encoding] = {}

        # Initialize encoders for all models
        for model_name in self.model_names:
            try:
                self.encoders[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base encoding if model not recognized
                print(f"Warning: Model '{model_name}' not recognized, using cl100k_base encoding")
                self.encoders[model_name] = tiktoken.get_encoding("cl100k_base")

        # Thread safety
        self._lock = asyncio.Lock()

        # Last cleanup time
        self._last_cleanup = time.time()

        # Start time for tracking session duration
        self._start_time = time.time()

    def count_prompt_tokens(self, messages, model_name: str):
        """
        Count tokens in input messages for a specific model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Name of the model to use for token counting

        Returns:
            int: Number of tokens in the messages
        """
        # Get the encoder for this model
        if model_name not in self.encoders:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.encoders.keys())}")

        encoding = self.encoders[model_name]

        total_tokens = 0
        for message in messages:
            # Message formatting overhead
            total_tokens += 4  # role + content formatting
            total_tokens += len(encoding.encode(message.get("content", "")))
        total_tokens += 2  # conversation start/end tokens
        return total_tokens

    def estimate_total_tokens(self, messages, model_name: str):
        """
        Estimate total tokens: prompt + completion for a specific model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Name of the model to use for token counting

        Returns:
            int: Estimated total tokens (prompt + max completion)
        """
        prompt_tokens = self.count_prompt_tokens(messages, model_name)
        return prompt_tokens + self.max_completion_tokens

    def _cleanup_old_requests(self):
        """Remove requests older than 1 minute from the sliding window (not from history)"""
        current_time = time.time()

        # Only cleanup every minimum_wait_time seconds to reduce overhead
        if current_time - self._last_cleanup < self.minimum_wait_time:
            return

        self._last_cleanup = current_time
        minute_ago = current_time - 60

        # Remove old requests from current window (but keep in history)
        expired_ids = [req_id for req_id, record in self._current_requests.items() if record.timestamp < minute_ago]

        for req_id in expired_ids:
            del self._current_requests[req_id]

    def _calculate_current_usage(self, model_name: str):
        """
        Calculate current RPM and TPM usage in the sliding window for a specific model.

        For pending requests: use projected_tokens
        For completed requests: use actual_tokens

        Args:
            model_name: Model name to calculate usage for

        Returns:
            Tuple of (current_rpm, current_tpm) for this model
        """
        current_rpm = 0
        current_tpm = 0

        for record in self._current_requests.values():
            if record.model_name != model_name:
                continue

            current_rpm += 1

            if record.status == RequestStatus.COMPLETED and record.actual_tokens is not None:
                current_tpm += record.actual_tokens
            else:
                # Use projected tokens for pending requests
                current_tpm += record.projected_tokens

        return current_rpm, current_tpm

    async def acquire_slot(self, estimated_tokens: int, model_name: str):
        """
        Try to acquire a slot for a new request for a specific model.

        Args:
            estimated_tokens: Projected token usage for this request
            model_name: Name of the model making the request

        Returns:
            Tuple of (can_proceed: bool, wait_time: float, request_id: Optional[int])
        """
        if model_name not in self.enforced_rpm_limits:
            raise ValueError(
                f"Model '{model_name}' not found in rate limits. Available models: {list(self.enforced_rpm_limits.keys())}"
            )

        async with self._lock:
            self._cleanup_old_requests()

            current_rpm, current_tpm = self._calculate_current_usage(model_name)
            enforced_rpm_limit = self.enforced_rpm_limits[model_name]
            enforced_tpm_limit = self.enforced_tpm_limits[model_name]

            # Check if this request would exceed enforced RPM limit for this model
            if current_rpm >= enforced_rpm_limit:
                # Find the oldest request for this model to determine wait time
                model_requests = [r for r in self._current_requests.values() if r.model_name == model_name]
                if model_requests:
                    oldest_timestamp = min(record.timestamp for record in model_requests)
                    wait_time = max(self.minimum_wait_time, 60 - (time.time() - oldest_timestamp))
                else:
                    wait_time = self.minimum_wait_time

                # Throttled logging: only log 1 in 500 messages
                self._rate_limit_message_counter += 1
                if self.logger and (self._rate_limit_message_counter % self._log_throttle_ratio == 0):
                    self.logger.info(
                        f"RPM limit hit for {model_name} - waiting {wait_time:.1f}s "
                        f"(RPM: {current_rpm}/{enforced_rpm_limit}, TPM: {current_tpm}/{enforced_tpm_limit})"
                    )
                return False, wait_time, None

            # Check if this request would exceed enforced TPM limit for this model
            if current_tpm + estimated_tokens >= enforced_tpm_limit:
                # Find the oldest request for this model to determine wait time
                model_requests = [r for r in self._current_requests.values() if r.model_name == model_name]
                if model_requests:
                    oldest_timestamp = min(record.timestamp for record in model_requests)
                    wait_time = max(self.minimum_wait_time, 60 - (time.time() - oldest_timestamp))
                else:
                    wait_time = self.minimum_wait_time

                # Throttled logging: only log 1 in 500 messages
                self._rate_limit_message_counter += 1
                if self.logger and (self._rate_limit_message_counter % self._log_throttle_ratio == 0):
                    self.logger.info(
                        f"TPM limit hit for {model_name} - waiting {wait_time:.1f}s "
                        f"(RPM: {current_rpm}/{enforced_rpm_limit}, TPM: {current_tpm}/{enforced_tpm_limit})"
                    )
                return False, wait_time, None

            # Reserve the slot
            request_id = self._request_counter
            self._request_counter += 1

            record = RequestRecord(
                request_id=request_id,
                timestamp=time.time(),
                projected_tokens=estimated_tokens,
                status=RequestStatus.PENDING,
                model_name=model_name,
                actual_tokens=None,
            )

            # Add to both current window and full history
            self._current_requests[request_id] = record
            self._all_requests[request_id] = record

            return True, 0, request_id

    async def update_actual_usage(
        self,
        request_id: int | None,
        actual_tokens: int = 0,
        status: RequestStatus = RequestStatus.COMPLETED,
    ):
        """
        Update request with actual token usage and status after completion.

        Args:
            request_id: The ID of the request to update
            actual_tokens: Actual token usage from the API response (default 0 for failed requests)
            status: Status of the request (COMPLETED, FAILED, or EMPTY_RESPONSE)
        """
        if request_id is None:
            return

        async with self._lock:
            # Update in current window (if still there)
            if request_id in self._current_requests:
                record = self._current_requests[request_id]
                record.status = status
                record.actual_tokens = actual_tokens

            # Always update in historical records
            if request_id in self._all_requests:
                record = self._all_requests[request_id]
                record.status = status
                record.actual_tokens = actual_tokens
            # If request_id not found in either, something went wrong (shouldn't happen)

    async def get_current_usage(self):
        """
        Get current rate limit usage statistics per model.

        Returns:
            Dict with current usage information per model (both sliding window and historical)
        """
        async with self._lock:
            self._cleanup_old_requests()

            # Aggregate stats per model
            per_model_stats = {}
            for model_name in self.model_names:
                current_rpm, current_tpm = self._calculate_current_usage(model_name)

                # Count current requests by status for this model (sliding window)
                model_current = [r for r in self._current_requests.values() if r.model_name == model_name]
                current_pending = sum(1 for r in model_current if r.status == RequestStatus.PENDING)
                current_completed = sum(1 for r in model_current if r.status == RequestStatus.COMPLETED)
                current_failed = sum(1 for r in model_current if r.status == RequestStatus.FAILED)
                current_empty = sum(1 for r in model_current if r.status == RequestStatus.EMPTY_RESPONSE)

                # Count all historical requests by status for this model
                model_all = [r for r in self._all_requests.values() if r.model_name == model_name]
                total_pending = sum(1 for r in model_all if r.status == RequestStatus.PENDING)
                total_completed = sum(1 for r in model_all if r.status == RequestStatus.COMPLETED)
                total_failed = sum(1 for r in model_all if r.status == RequestStatus.FAILED)
                total_empty = sum(1 for r in model_all if r.status == RequestStatus.EMPTY_RESPONSE)

                # Calculate total historical tokens for this model
                total_tokens = sum(
                    r.actual_tokens if r.actual_tokens is not None else r.projected_tokens
                    for r in model_all
                )

                per_model_stats[model_name] = {
                    # Current sliding window (60 seconds)
                    "current_requests": current_rpm,
                    "current_tokens": current_tpm,
                    "current_pending_requests": current_pending,
                    "current_completed_requests": current_completed,
                    "current_failed_requests": current_failed,
                    "current_empty_response_requests": current_empty,
                    # Historical (all time since start)
                    "total_requests": len(model_all),
                    "total_tokens": total_tokens,
                    "total_pending_requests": total_pending,
                    "total_completed_requests": total_completed,
                    "total_failed_requests": total_failed,
                    "total_empty_response_requests": total_empty,
                    # Actual API limits
                    "actual_rpm_limit": self.actual_rpm_limits[model_name],
                    "actual_tpm_limit": self.actual_tpm_limits[model_name],
                    # Enforced limits (with safety margin)
                    "enforced_rpm_limit": self.enforced_rpm_limits[model_name],
                    "enforced_tpm_limit": self.enforced_tpm_limits[model_name],
                    # Utilization against enforced limits (current window)
                    "rpm_utilization": current_rpm / self.enforced_rpm_limits[model_name]
                    if self.enforced_rpm_limits[model_name] > 0
                    else 0,
                    "tpm_utilization": current_tpm / self.enforced_tpm_limits[model_name]
                    if self.enforced_tpm_limits[model_name] > 0
                    else 0,
                    # Utilization against actual limits (shows safety buffer)
                    "actual_rpm_utilization": current_rpm / self.actual_rpm_limits[model_name]
                    if self.actual_rpm_limits[model_name] > 0
                    else 0,
                    "actual_tpm_utilization": current_tpm / self.actual_tpm_limits[model_name]
                    if self.actual_tpm_limits[model_name] > 0
                    else 0,
                }

            # Session duration
            session_duration = time.time() - self._start_time

            # Overall totals (across all models)
            total_all_requests = len(self._all_requests)
            total_all_tokens = sum(
                r.actual_tokens if r.actual_tokens is not None else r.projected_tokens
                for r in self._all_requests.values()
            )

            return {
                "per_model": per_model_stats,
                # Overall session info
                "session_duration_seconds": session_duration,
                "average_requests_per_minute": (total_all_requests / session_duration * 60)
                if session_duration > 0
                else 0,
                "average_tokens_per_minute": (total_all_tokens / session_duration * 60)
                if session_duration > 0
                else 0,
                # Configuration
                "limit_safety_ratio": self.limit_safety_ratio,
                "minimum_wait_time": self.minimum_wait_time,
                "supported_models": self.model_names,
            }
