import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


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


# ---------------------------------------------------------------------------
# Base class — all sliding-window logic lives here
# ---------------------------------------------------------------------------

class BaseRateLimiter(ABC):
    """
    Sliding-window rate limiter shared by all API backends.

    Subclasses must implement:
    - ``_init_encoders()``: set up whatever is needed for token counting
    - ``count_prompt_tokens(messages, model_name) -> int``: count input tokens
    """

    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        logger=None,
    ):
        """
        Args:
            model_rate_limits: Dict mapping model name to rate limits, e.g.
                {"gpt-4": {"rpm": 500, "tpm": 50000}}
            max_completion_tokens: Maximum output/completion tokens per request
            limit_safety_ratio: Safety margin as fraction of limits (default 0.98)
            minimum_wait_time: Minimum wait time when rate limited (seconds)
            logger: Optional logger instance
        """
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

        self._request_counter = 0
        self._current_requests: dict[int, RequestRecord] = {}
        self._all_requests: dict[int, RequestRecord] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
        self._start_time = time.time()

        self._init_encoders()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_encoders(self) -> None:
        """Set up token-counting resources (encoders, clients, etc.)."""

    @abstractmethod
    def count_prompt_tokens(self, messages: list[dict], model_name: str) -> int:
        """
        Count the number of tokens in *messages* for *model_name*.

        Args:
            messages: List of message dicts with at least a ``"content"`` key.
            model_name: Model to count tokens for.

        Returns:
            Estimated token count of the prompt.
        """

    # ------------------------------------------------------------------
    # Common token estimation
    # ------------------------------------------------------------------

    def estimate_total_tokens(self, messages: list[dict], model_name: str) -> int:
        """
        Estimate total tokens: prompt tokens + max output tokens.

        Args:
            messages: List of message dicts
            model_name: Model name for token counting

        Returns:
            Estimated total tokens (prompt + max output)
        """
        return self.count_prompt_tokens(messages, model_name) + self.max_completion_tokens

    # ------------------------------------------------------------------
    # Sliding-window bookkeeping (identical for all backends)
    # ------------------------------------------------------------------

    def _cleanup_old_requests(self) -> None:
        """Remove requests older than 60 s from the sliding window."""
        current_time = time.time()
        if current_time - self._last_cleanup < self.minimum_wait_time:
            return
        self._last_cleanup = current_time
        minute_ago = current_time - 60
        expired = [
            req_id
            for req_id, record in self._current_requests.items()
            if record.timestamp < minute_ago
        ]
        for req_id in expired:
            del self._current_requests[req_id]

    def _calculate_current_usage(self, model_name: str) -> tuple[int, int]:
        """
        Calculate RPM and TPM usage in the current sliding window for *model_name*.

        Pending requests use ``projected_tokens``; completed ones use ``actual_tokens``.
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
                current_tpm += record.projected_tokens
        return current_rpm, current_tpm

    async def acquire_slot(
        self, estimated_tokens: int, model_name: str
    ) -> tuple[bool, float, int | None]:
        """
        Try to acquire a slot for a new request.

        Args:
            estimated_tokens: Projected token usage for this request
            model_name: Model name for the request

        Returns:
            ``(can_proceed, wait_time, request_id)``
        """
        if model_name not in self.enforced_rpm_limits:
            raise ValueError(
                f"Model '{model_name}' not found in rate limits. "
                f"Available models: {list(self.enforced_rpm_limits.keys())}"
            )

        async with self._lock:
            self._cleanup_old_requests()

            current_rpm, current_tpm = self._calculate_current_usage(model_name)
            enforced_rpm_limit = self.enforced_rpm_limits[model_name]
            enforced_tpm_limit = self.enforced_tpm_limits[model_name]

            def _wait_time_for_model() -> float:
                model_requests = [
                    r for r in self._current_requests.values() if r.model_name == model_name
                ]
                if model_requests:
                    oldest = min(r.timestamp for r in model_requests)
                    return max(self.minimum_wait_time, 60 - (time.time() - oldest))
                return self.minimum_wait_time

            if current_rpm >= enforced_rpm_limit:
                wait_time = _wait_time_for_model()
                self._rate_limit_message_counter += 1
                if self.logger and self._rate_limit_message_counter % self._log_throttle_ratio == 0:
                    self.logger.info(
                        f"RPM limit hit for {model_name} — waiting {wait_time:.1f}s "
                        f"(RPM: {current_rpm}/{enforced_rpm_limit}, TPM: {current_tpm}/{enforced_tpm_limit})"
                    )
                return False, wait_time, None

            if current_tpm + estimated_tokens >= enforced_tpm_limit:
                wait_time = _wait_time_for_model()
                self._rate_limit_message_counter += 1
                if self.logger and self._rate_limit_message_counter % self._log_throttle_ratio == 0:
                    self.logger.info(
                        f"TPM limit hit for {model_name} — waiting {wait_time:.1f}s "
                        f"(RPM: {current_rpm}/{enforced_rpm_limit}, TPM: {current_tpm}/{enforced_tpm_limit})"
                    )
                return False, wait_time, None

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
            self._current_requests[request_id] = record
            self._all_requests[request_id] = record # Potentially could become a long list resulting in heavy memory usage
            return True, 0, request_id

    async def update_actual_usage(
        self,
        request_id: int | None,
        actual_tokens: int = 0,
        status: RequestStatus = RequestStatus.COMPLETED,
    ) -> None:
        """
        Update request with actual token usage and status after completion.

        Args:
            request_id: ID of the request to update (no-op if ``None``)
            actual_tokens: Actual token usage from the API response
            status: Final status of the request
        """
        if request_id is None:
            return
        async with self._lock:
            for store in (self._current_requests, self._all_requests):
                if request_id in store:
                    store[request_id].status = status
                    store[request_id].actual_tokens = actual_tokens

    async def get_current_usage(self) -> dict:
        """
        Return current and historical usage statistics per model.

        Returns:
            Dict with per-model stats and session-level aggregates.
        """
        async with self._lock:
            self._cleanup_old_requests()

            per_model_stats = {}
            for model_name in self.model_names:
                current_rpm, current_tpm = self._calculate_current_usage(model_name)

                model_current = [r for r in self._current_requests.values() if r.model_name == model_name]
                model_all = [r for r in self._all_requests.values() if r.model_name == model_name]

                def _count_by_status(records, s):
                    return sum(1 for r in records if r.status == s)

                total_tokens = sum(
                    r.actual_tokens if r.actual_tokens is not None else r.projected_tokens
                    for r in model_all
                )
                enf_rpm = self.enforced_rpm_limits[model_name]
                enf_tpm = self.enforced_tpm_limits[model_name]
                act_rpm = self.actual_rpm_limits[model_name]
                act_tpm = self.actual_tpm_limits[model_name]

                per_model_stats[model_name] = {
                    "current_requests": current_rpm,
                    "current_tokens": current_tpm,
                    "current_pending_requests": _count_by_status(model_current, RequestStatus.PENDING),
                    "current_completed_requests": _count_by_status(model_current, RequestStatus.COMPLETED),
                    "current_failed_requests": _count_by_status(model_current, RequestStatus.FAILED),
                    "current_empty_response_requests": _count_by_status(model_current, RequestStatus.EMPTY_RESPONSE),
                    "total_requests": len(model_all),
                    "total_tokens": total_tokens,
                    "total_pending_requests": _count_by_status(model_all, RequestStatus.PENDING),
                    "total_completed_requests": _count_by_status(model_all, RequestStatus.COMPLETED),
                    "total_failed_requests": _count_by_status(model_all, RequestStatus.FAILED),
                    "total_empty_response_requests": _count_by_status(model_all, RequestStatus.EMPTY_RESPONSE),
                    "actual_rpm_limit": act_rpm,
                    "actual_tpm_limit": act_tpm,
                    "enforced_rpm_limit": enf_rpm,
                    "enforced_tpm_limit": enf_tpm,
                    "rpm_utilization": current_rpm / enf_rpm if enf_rpm > 0 else 0,
                    "tpm_utilization": current_tpm / enf_tpm if enf_tpm > 0 else 0,
                    "actual_rpm_utilization": current_rpm / act_rpm if act_rpm > 0 else 0,
                    "actual_tpm_utilization": current_tpm / act_tpm if act_tpm > 0 else 0,
                }

            session_duration = time.time() - self._start_time
            total_all_requests = len(self._all_requests)
            total_all_tokens = sum(
                r.actual_tokens if r.actual_tokens is not None else r.projected_tokens
                for r in self._all_requests.values()
            )
            return {
                "per_model": per_model_stats,
                "session_duration_seconds": session_duration,
                "average_requests_per_minute": (
                    total_all_requests / session_duration * 60 if session_duration > 0 else 0
                ),
                "average_tokens_per_minute": (
                    total_all_tokens / session_duration * 60 if session_duration > 0 else 0
                ),
                "limit_safety_ratio": self.limit_safety_ratio,
                "minimum_wait_time": self.minimum_wait_time,
                "supported_models": self.model_names,
            }


# ---------------------------------------------------------------------------
# OpenAI — tiktoken-based token counting
# ---------------------------------------------------------------------------

class OpenAIRateLimiter(BaseRateLimiter):
    """Rate limiter for OpenAI API endpoints. Uses tiktoken for token counting."""

    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        logger=None,
    ):
        super().__init__(
            model_rate_limits=model_rate_limits,
            max_completion_tokens=max_completion_tokens,
            limit_safety_ratio=limit_safety_ratio,
            minimum_wait_time=minimum_wait_time,
            logger=logger,
        )

    _FALLBACK_MODEL = "gpt-5"

    def _init_encoders(self) -> None:
        import tiktoken

        self.encoders: dict = {}
        fallback_encoder = None
        for model_name in self.model_names:
            try:
                self.encoders[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                if fallback_encoder is None:
                    try:
                        fallback_encoder = tiktoken.encoding_for_model(self._FALLBACK_MODEL)
                    except KeyError:
                        fallback_encoder = tiktoken.get_encoding("o200k_base")
                if self.logger:
                    self.logger.warning(
                        f"Model '{model_name}' not recognised by tiktoken, "
                        f"falling back to {self._FALLBACK_MODEL} tokenizer"
                    )
                self.encoders[model_name] = fallback_encoder

    # OpenAI vision bills images at ~85 tokens (low detail) to ~765 tokens
    # (high detail, single tile). We use a conservative estimate.
    _TOKENS_PER_IMAGE = 765
    _TOKENS_PER_AUDIO = 512

    def count_prompt_tokens(self, messages: list[dict], model_name: str) -> int:
        """Count tokens using tiktoken. Handles both text-only and multimodal content."""
        if model_name not in self.encoders:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.encoders.keys())}"
            )
        encoding = self.encoders[model_name]
        total = 2  # conversation start/end tokens
        for message in messages:
            total += 4  # per-message role/content overhead
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "") if isinstance(part, dict) else ""
                    if part_type == "text":
                        total += len(encoding.encode(part.get("text", "")))
                    elif part_type == "image_url":
                        total += self._TOKENS_PER_IMAGE
                    elif part_type == "input_audio":
                        total += self._TOKENS_PER_AUDIO
                    else:
                        total += 4
        return total


# ---------------------------------------------------------------------------
# Google Gemini — character-based token estimation
# ---------------------------------------------------------------------------

class GoogleGeminiRateLimiter(BaseRateLimiter):
    """
    Rate limiter for Google Gemini API endpoints.

    Gemini uses a proprietary SentencePiece tokenizer that cannot be run
    offline without the full model weights. An exact count requires an API
    call (``client.models.count_tokens``), which would add latency and cost.
    Instead we use the widely-accepted offline approximation of **4 characters
    per token** for English-like text. This is intentionally conservative
    (slightly overestimates) to stay safely under the rate limits.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        logger=None,
    ):
        super().__init__(
            model_rate_limits=model_rate_limits,
            max_completion_tokens=max_completion_tokens,
            limit_safety_ratio=limit_safety_ratio,
            minimum_wait_time=minimum_wait_time,
            logger=logger,
        )

    def _init_encoders(self) -> None:
        pass

    # Gemini charges ~258 tokens per image (standard tile budget).
    # Audio is billed at 32 tokens/second; without duration metadata we use a
    # conservative fixed estimate. Video frames are similarly opaque offline.
    _TOKENS_PER_IMAGE = 258
    _TOKENS_PER_AUDIO_OR_VIDEO = 512

    def count_prompt_tokens(self, messages: list[dict], model_name: str) -> int:
        """
        Estimate token count for OpenAI-format messages using ~4 chars/token.

        Messages arrive via the MLflow gateway in OpenAI format::

            {"role": "user", "content": "..."}                   # text-only
            {"role": "user", "content": [{"type": "text", ...},  # multimodal
                                         {"type": "image_url", ...}]}

        Includes fixed per-image/audio/video estimates for multimodal parts.
        """
        total = 2  # conversation framing
        for message in messages:
            total += 4  # per-message role/turn overhead
            content = message.get("content", "")
            if isinstance(content, str):
                total += max(1, len(content) // self._CHARS_PER_TOKEN)
            elif isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "") if isinstance(part, dict) else ""
                    if part_type == "text":
                        total += max(1, len(part.get("text", "")) // self._CHARS_PER_TOKEN)
                    elif part_type == "image_url":
                        total += self._TOKENS_PER_IMAGE
                    elif part_type in ("input_audio", "video"):
                        total += self._TOKENS_PER_AUDIO_OR_VIDEO
                    else:
                        total += 4
        return total


# ---------------------------------------------------------------------------
# Anthropic — character-based token estimation
# ---------------------------------------------------------------------------

class AnthropicRateLimiter(BaseRateLimiter):
    """
    Rate limiter for Anthropic API endpoints.

    Claude models use a proprietary BPE tokenizer that is not available
    offline without additional dependencies.  We use the same ~4 characters
    per token approximation used by the Gemini limiter, which is
    intentionally conservative for English-like text.

    Messages arrive in OpenAI-compatible format via the MLflow gateway::

        {"role": "user", "content": "..."}                   # text-only
        {"role": "user", "content": [{"type": "text", ...},  # multimodal
                                     {"type": "image_url", ...}]}
    """

    _CHARS_PER_TOKEN = 4

    # Claude bills images based on resolution; a single 1568×1568 tile costs
    # ~1600 tokens. We use a conservative fixed estimate.
    _TOKENS_PER_IMAGE = 1600
    _TOKENS_PER_AUDIO = 512

    def __init__(
        self,
        model_rate_limits: dict[str, dict[str, int]],
        max_completion_tokens: int = 150,
        limit_safety_ratio: float = 0.98,
        minimum_wait_time: float = 3.0,
        logger=None,
    ):
        super().__init__(
            model_rate_limits=model_rate_limits,
            max_completion_tokens=max_completion_tokens,
            limit_safety_ratio=limit_safety_ratio,
            minimum_wait_time=minimum_wait_time,
            logger=logger,
        )

    def _init_encoders(self) -> None:
        pass

    def count_prompt_tokens(self, messages: list[dict], model_name: str) -> int:
        """Estimate token count using ~4 chars/token for OpenAI-format messages."""
        total = 2  # conversation framing
        for message in messages:
            total += 4  # per-message role/content overhead
            content = message.get("content", "")
            if isinstance(content, str):
                total += max(1, len(content) // self._CHARS_PER_TOKEN)
            elif isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "") if isinstance(part, dict) else ""
                    if part_type == "text":
                        total += max(1, len(part.get("text", "")) // self._CHARS_PER_TOKEN)
                    elif part_type == "image_url":
                        total += self._TOKENS_PER_IMAGE
                    elif part_type == "input_audio":
                        total += self._TOKENS_PER_AUDIO
                    else:
                        total += 4
        return total
