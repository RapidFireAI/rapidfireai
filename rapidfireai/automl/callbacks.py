"""Callback protocols for inter-shard decision-making during experiments.

Classes
-------
RunDecision
    Dataclass returned by ``FitShardCallback.on_shard_complete`` (fit mode).
PipelineDecision
    Dataclass returned by ``ShardCallback.on_shard_complete`` (evals mode).
FitShardCallback
    Protocol for fit-mode inter-shard pruning callbacks.
ShardCallback
    Protocol for evals-mode inter-shard pruning callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class RunDecision:
    """Decision returned by a ``FitShardCallback`` after a fit-mode shard completes.

    Attributes
    ----------
    action : ``"continue"`` or ``"prune"``
    replacement_config : dict or None
        Config-leaf dict for a replacement run, or ``None``.
    """

    action: Literal["continue", "prune"]
    replacement_config: dict[str, Any] | None = None


@dataclass
class PipelineDecision:
    """Decision returned by a ``ShardCallback`` after an evals-mode shard completes.

    Attributes
    ----------
    action : ``"continue"`` or ``"prune"``
    replacement_config : dict or None
        Config-leaf dict for a replacement pipeline, or ``None``.
    """

    action: Literal["continue", "prune"]
    replacement_config: dict[str, Any] | None = None


class FitShardCallback(Protocol):
    """Protocol for callbacks invoked after each shard in fit mode.

    Call order: ``register_runs`` → ``on_shard_complete`` (repeated) → ``finalize``.
    """

    def register_runs(self, run_id_to_config: dict[int, dict[str, Any]]) -> None:
        """Map newly created DB run IDs to their config dicts."""
        ...

    def on_shard_complete(
        self,
        run_id: int,
        shard_id: int,
        metrics: dict[str, Any],
    ) -> RunDecision:
        """Evaluate a run after it finishes a shard.

        Parameters
        ----------
        run_id : int
        shard_id : int
        metrics : dict[str, Any]

        Returns
        -------
        RunDecision
        """
        ...

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Called after the experiment loop ends."""
        ...


class ShardCallback(Protocol):
    """Protocol for callbacks invoked after each shard in evals mode.

    Call order: ``register_pipelines`` → ``on_shard_complete`` (repeated) → ``finalize``.
    """

    def register_pipelines(self, pipeline_id_to_config: dict[int, dict[str, Any]]) -> None:
        """Map newly created DB pipeline IDs to their config dicts."""
        ...

    def on_shard_complete(
        self,
        pipeline_id: int,
        shard_id: int,
        metrics: dict[str, Any],
    ) -> PipelineDecision:
        """Evaluate a pipeline after it finishes a shard.

        Parameters
        ----------
        pipeline_id : int
        shard_id : int
        metrics : dict[str, Any]

        Returns
        -------
        PipelineDecision
        """
        ...

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Called after the experiment loop ends."""
        ...
