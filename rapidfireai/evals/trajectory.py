"""Trajectory data model and LangGraph callback for agent evaluation.

When ``capture_trajectory=True`` on :class:`RFLangGraphAgentConfig`,
RF attaches a :class:`TrajectoryCallback` before invoking the graph.
The callback records every node execution, tool call, LLM call,
error, and timing into an :class:`AgentTrajectory`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """One node execution inside a LangGraph graph."""

    node_name: str
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    """Full trajectory of a single agent invocation."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    total_time: float = 0.0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timed_out: bool = False

    # ---- derived helpers ----

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def error_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.error is not None]

    @property
    def error_rate(self) -> float:
        if not self.steps:
            return 0.0
        return len(self.error_steps) / len(self.steps)

    def finalize(self) -> None:
        """Recompute aggregates from steps."""
        self.total_llm_calls = sum(len(s.llm_calls) for s in self.steps)
        self.total_tool_calls = sum(len(s.tool_calls) for s in self.steps)
        self.total_tokens = sum(
            call.get("prompt_tokens", 0) + call.get("completion_tokens", 0)
            for s in self.steps
            for call in s.llm_calls
        )
        self.total_time = sum(s.duration_seconds for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage / JSON logging."""
        return {
            "steps": [
                {
                    "node_name": s.node_name,
                    "tool_calls": s.tool_calls,
                    "llm_calls": s.llm_calls,
                    "duration_seconds": s.duration_seconds,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "total_time": self.total_time,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "timed_out": self.timed_out,
        }


# ---------------------------------------------------------------------------
# LangGraph callback
# ---------------------------------------------------------------------------

# Token cost table (USD per 1 M tokens) — input / output
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort cost estimate.  Returns 0.0 for unknown models."""
    for key, (inp, out) in _COST_TABLE.items():
        if key in model:
            return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
    return 0.0


class TrajectoryCallback:
    """LangGraph-compatible callback that populates an :class:`AgentTrajectory`.

    Attach via ``config={"callbacks": [TrajectoryCallback(trajectory)]}``.

    The callback intercepts LangGraph lifecycle events (``on_chain_start``,
    ``on_chain_end``, ``on_tool_start``, ``on_tool_end``, ``on_llm_start``,
    ``on_llm_end``, ``on_chain_error``) and records them into the trajectory.
    """

    def __init__(self, trajectory: AgentTrajectory):
        self.trajectory = trajectory
        self._current_step: TrajectoryStep | None = None
        self._step_start: float = 0.0
        self._llm_start: float = 0.0
        self._last_state: Any = None

    @property
    def last_state(self) -> Any:
        """Last graph state seen — useful for extracting partial results on timeout."""
        return self._last_state

    # --- chain (node) events ---

    def on_chain_start(self, serialized: dict | None = None, inputs: Any = None, **kwargs: Any) -> None:
        name = "unknown"
        if serialized and isinstance(serialized, dict):
            name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        elif "name" in kwargs:
            name = kwargs["name"]

        self._current_step = TrajectoryStep(node_name=name)
        self._step_start = time.monotonic()

    def on_chain_end(self, outputs: Any = None, **kwargs: Any) -> None:
        if self._current_step is not None:
            self._current_step.duration_seconds = time.monotonic() - self._step_start
            self.trajectory.steps.append(self._current_step)
            self._current_step = None
        if outputs is not None:
            self._last_state = outputs

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_step is not None:
            self._current_step.error = str(error)
            self._current_step.duration_seconds = time.monotonic() - self._step_start
            self.trajectory.steps.append(self._current_step)
            self._current_step = None

    # --- tool events ---

    def on_tool_start(self, serialized: dict | None = None, input_str: str = "", **kwargs: Any) -> None:
        pass

    def on_tool_end(self, output: str = "", **kwargs: Any) -> None:
        tool_name = kwargs.get("name", "unknown_tool")
        if self._current_step is not None:
            self._current_step.tool_calls.append({
                "name": tool_name,
                "result": str(output)[:500],
            })

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        tool_name = kwargs.get("name", "unknown_tool")
        if self._current_step is not None:
            self._current_step.tool_calls.append({
                "name": tool_name,
                "error": str(error),
            })

    # --- LLM events ---

    def on_llm_start(self, serialized: dict | None = None, prompts: list | None = None, **kwargs: Any) -> None:
        self._llm_start = time.monotonic()

    def on_llm_end(self, response: Any = None, **kwargs: Any) -> None:
        if self._current_step is None:
            return

        llm_call: dict[str, Any] = {"duration_seconds": time.monotonic() - self._llm_start}

        # Extract token usage from LangChain response objects
        if response is not None:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage", {})
            if token_usage:
                llm_call["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
                llm_call["completion_tokens"] = token_usage.get("completion_tokens", 0)
                model = llm_output.get("model_name", "")
                llm_call["model"] = model
                llm_call["cost_usd"] = estimate_cost(
                    model,
                    llm_call.get("prompt_tokens", 0),
                    llm_call.get("completion_tokens", 0),
                )

            # Also check generations for response_metadata (newer LangChain pattern)
            generations = getattr(response, "generations", None)
            if generations and not token_usage:
                for gen_list in generations:
                    for gen in gen_list:
                        meta = getattr(gen, "generation_info", {}) or {}
                        if not meta:
                            msg = getattr(gen, "message", None)
                            meta = getattr(msg, "response_metadata", {}) or {}
                        usage = meta.get("usage", meta.get("token_usage", {}))
                        if usage:
                            llm_call["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                            llm_call["completion_tokens"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
                            model = meta.get("model", meta.get("model_name", ""))
                            llm_call["model"] = model
                            llm_call["cost_usd"] = estimate_cost(
                                model,
                                llm_call.get("prompt_tokens", 0),
                                llm_call.get("completion_tokens", 0),
                            )

        self._current_step.llm_calls.append(llm_call)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_step is not None:
            self._current_step.llm_calls.append({"error": str(error)})
