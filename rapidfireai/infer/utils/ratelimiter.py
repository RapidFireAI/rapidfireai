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
    actual_tokens: int | None = None  # Actual usage, known only after completion


class OpenAIRateLimiter:
    def __init__(
        self,
        model_names: list[str],
        rpm_limit: int = 500,
        tpm_limit: int = 500000,
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
    ):
        """
        Initialize the rate limiter with sliding window tracking and support for multiple models.

        Args:
            model_names: List of OpenAI model names for token counting (or single model name string)
            rpm_limit: Requests per minute limit (actual API limit)
            tpm_limit: Tokens per minute limit (actual API limit)
            max_completion_tokens: Maximum completion tokens per request
            limit_safety_ratio: Safety margin as percentage of limits (default 0.98 = 98%)
            minimum_wait_time: Minimum wait time when rate limited (default 3.0 seconds)
        """
        # Configuration
        self.limit_safety_ratio = limit_safety_ratio
        self.minimum_wait_time = minimum_wait_time
        self.max_completion_tokens = max_completion_tokens

        # Actual API limits (as specified by the API provider)
        self.actual_rpm_limit = rpm_limit
        self.actual_tpm_limit = tpm_limit

        # Enforced limits (with safety margin applied)
        self.enforced_rpm_limit = int(limit_safety_ratio * rpm_limit)
        self.enforced_tpm_limit = int(limit_safety_ratio * tpm_limit)

        # Request tracking
        self._request_counter = 0  # Unique ID generator

        # Current requests (sliding 60-second window) - used for rate limiting
        self._current_requests: dict[int, RequestRecord] = {}  # request_id -> RequestRecord

        # Historical requests (all requests since start) - never deleted
        self._all_requests: dict[int, RequestRecord] = {}  # request_id -> RequestRecord

        # For token counting - support multiple models
        self.model_names = model_names
        self.encoders: dict[str, tiktoken.Encoding] = {}

        # Initialize encoders for all models
        for model_name in model_names:
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

    def _calculate_current_usage(self):
        """
        Calculate current RPM and TPM usage in the sliding window.

        For pending requests: use projected_tokens
        For completed requests: use actual_tokens
        """
        current_rpm = 0
        current_tpm = 0

        for record in self._current_requests.values():
            current_rpm += 1

            if record.status == RequestStatus.COMPLETED and record.actual_tokens is not None:
                current_tpm += record.actual_tokens
            else:
                # Use projected tokens for pending requests
                current_tpm += record.projected_tokens

        return current_rpm, current_tpm

    async def acquire_slot(self, estimated_tokens: int):
        """
        Try to acquire a slot for a new request.

        Args:
            estimated_tokens: Projected token usage for this request

        Returns:
            Tuple of (can_proceed: bool, wait_time: float, request_id: Optional[int])
        """
        async with self._lock:
            self._cleanup_old_requests()

            current_rpm, current_tpm = self._calculate_current_usage()

            # Check if this request would exceed enforced RPM limit
            if current_rpm >= self.enforced_rpm_limit:
                # Find the oldest request to determine wait time
                oldest_timestamp = min(record.timestamp for record in self._current_requests.values())
                wait_time = max(self.minimum_wait_time, 60 - (time.time() - oldest_timestamp))
                print(
                    f"RPM limit hit - waiting {wait_time:.1f}s (RPM: {current_rpm}/{self.enforced_rpm_limit}, TPM: {current_tpm}/{self.enforced_tpm_limit})"
                )
                return False, wait_time, None

            # Check if this request would exceed enforced TPM limit
            if current_tpm + estimated_tokens >= self.enforced_tpm_limit:
                # Find the oldest request to determine wait time
                oldest_timestamp = min(record.timestamp for record in self._current_requests.values())
                wait_time = max(self.minimum_wait_time, 60 - (time.time() - oldest_timestamp))
                print(
                    f"TPM limit hit - waiting {wait_time:.1f}s (RPM: {current_rpm}/{self.enforced_rpm_limit}, TPM: {current_tpm}/{self.enforced_tpm_limit})"
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
        Get current rate limit usage statistics.

        Returns:
            Dict with current usage information (both sliding window and historical)
        """
        async with self._lock:
            self._cleanup_old_requests()
            current_rpm, current_tpm = self._calculate_current_usage()

            # Count current requests by status (sliding window)
            current_pending = sum(1 for r in self._current_requests.values() if r.status == RequestStatus.PENDING)
            current_completed = sum(1 for r in self._current_requests.values() if r.status == RequestStatus.COMPLETED)
            current_failed = sum(1 for r in self._current_requests.values() if r.status == RequestStatus.FAILED)
            current_empty = sum(1 for r in self._current_requests.values() if r.status == RequestStatus.EMPTY_RESPONSE)

            # Count all historical requests by status
            total_pending = sum(1 for r in self._all_requests.values() if r.status == RequestStatus.PENDING)
            total_completed = sum(1 for r in self._all_requests.values() if r.status == RequestStatus.COMPLETED)
            total_failed = sum(1 for r in self._all_requests.values() if r.status == RequestStatus.FAILED)
            total_empty = sum(1 for r in self._all_requests.values() if r.status == RequestStatus.EMPTY_RESPONSE)

            # Calculate total historical tokens
            total_tokens = sum(
                r.actual_tokens if r.actual_tokens is not None else r.projected_tokens
                for r in self._all_requests.values()
            )

            # Session duration
            session_duration = time.time() - self._start_time

            return {
                # Current sliding window (60 seconds)
                "current_requests": current_rpm,
                "current_tokens": current_tpm,
                "current_pending_requests": current_pending,
                "current_completed_requests": current_completed,
                "current_failed_requests": current_failed,
                "current_empty_response_requests": current_empty,
                # Historical (all time since start)
                "total_requests": len(self._all_requests),
                "total_tokens": total_tokens,
                "total_pending_requests": total_pending,
                "total_completed_requests": total_completed,
                "total_failed_requests": total_failed,
                "total_empty_response_requests": total_empty,
                # Session info
                "session_duration_seconds": session_duration,
                "average_requests_per_minute": (len(self._all_requests) / session_duration * 60)
                if session_duration > 0
                else 0,
                "average_tokens_per_minute": (total_tokens / session_duration * 60) if session_duration > 0 else 0,
                # Actual API limits
                "actual_rpm_limit": self.actual_rpm_limit,
                "actual_tpm_limit": self.actual_tpm_limit,
                # Enforced limits (with safety margin)
                "enforced_rpm_limit": self.enforced_rpm_limit,
                "enforced_tpm_limit": self.enforced_tpm_limit,
                # Utilization against enforced limits (current window)
                "rpm_utilization": current_rpm / self.enforced_rpm_limit if self.enforced_rpm_limit > 0 else 0,
                "tpm_utilization": current_tpm / self.enforced_tpm_limit if self.enforced_tpm_limit > 0 else 0,
                # Utilization against actual limits (shows safety buffer)
                "actual_rpm_utilization": current_rpm / self.actual_rpm_limit if self.actual_rpm_limit > 0 else 0,
                "actual_tpm_utilization": current_tpm / self.actual_tpm_limit if self.actual_tpm_limit > 0 else 0,
                # Configuration
                "limit_safety_ratio": self.limit_safety_ratio,
                "minimum_wait_time": self.minimum_wait_time,
                "supported_models": self.model_names,
            }
