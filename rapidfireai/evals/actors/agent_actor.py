"""Agent evaluation actor for LangGraph agent pipelines.

Handles LangGraph agent invocations with timeout, step limits,
trajectory capture, and code execution for HumanEval+-style evals.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from typing import Any

import ray

from rapidfireai.utils.constants import RF_EXPERIMENT_PATH
from rapidfireai.evals.trajectory import AgentTrajectory, TrajectoryCallback
from rapidfireai.evals.utils.logger import RFLogger
from rapidfireai.evals.utils.mlflow_utils import setup_mlflow, is_mlflow_enabled


@ray.remote
class AgentEvalActor:
    """Ray actor that evaluates LangGraph agent pipelines.

    Each actor builds/compiles the graph once per pipeline (via
    ``initialize_for_pipeline``), then processes shards row-by-row
    (agents are I/O-bound, not batchable).

    The actor supports:
    - Custom graph factories (``graph_fn``)
    - Prebuilt agents (``create_react_agent`` etc.)
    - Trajectory capture (``capture_trajectory=True``)
    - Per-invocation timeout and step limits
    - Code execution for HumanEval+-style evaluations
    """

    def __init__(
        self,
        experiment_name: str = "unknown",
        experiment_path: str = RF_EXPERIMENT_PATH,
        actor_id: int = 0,
    ):
        setup_mlflow(experiment_name)
        logging_manager = RFLogger(
            experiment_name=experiment_name,
            experiment_path=experiment_path,
        )
        self.logger = logging_manager.get_logger(f"AgentEvalActor-{actor_id}")
        self.actor_id = actor_id

        self.compiled_graph = None
        self.agent_config = None
        self.pipeline_id: int | None = None
        self.metric_run_id: str | None = None

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    def initialize_for_pipeline(
        self,
        agent_config,
        pipeline_id: int | None = None,
        model_name: str | None = None,
        metric_run_id: str | None = None,
    ) -> None:
        """Configure this actor for a specific agent pipeline.

        Builds and compiles the graph once so it can be reused across
        all rows in the shard.
        """
        try:
            self.agent_config = agent_config
            self.pipeline_id = pipeline_id
            self.metric_run_id = metric_run_id

            self.logger.info(
                f"Building graph for pipeline {pipeline_id} "
                f"(model={model_name}, max_steps={agent_config.max_steps})"
            )
            self.compiled_graph = agent_config.build_and_compile()
            self.logger.info(f"Graph compiled for pipeline {pipeline_id}")

            if self.metric_run_id and is_mlflow_enabled():
                import mlflow

                mlflow.start_run(run_id=self.metric_run_id)

        except Exception as e:
            error_type = type(e).__name__
            self.logger.exception(f"Failed to initialize agent pipeline: {error_type}: {e}")
            raise RuntimeError(
                f"Failed to initialize agent pipeline: {error_type}: {e}"
            ) from None

    # ------------------------------------------------------------------
    # Batch (shard) processing
    # ------------------------------------------------------------------

    def process_batch(
        self,
        batch_data: dict[str, list],
        preprocess_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        compute_metrics_fn: Callable | None = None,
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """Process a batch of data through the agent pipeline.

        Unlike RAG pipelines, agent invocations are sequential (each
        row is an independent agent run). The method iterates over rows,
        calls ``input_mapper`` → ``graph.invoke()`` → ``output_mapper``,
        and optionally captures full trajectories.

        Args:
            batch_data: Dictionary of lists (HF Dataset style).
            preprocess_fn: Optional pre-processing (rarely used for agents).
            postprocess_fn: Optional post-processing of the batch.
            compute_metrics_fn: Per-batch metrics function.

        Returns:
            ``(batch_data, batch_metrics)`` — same contract as
            ``QueryProcessingActor.process_batch``.
        """
        try:
            if hasattr(batch_data, "to_dict"):
                batch_data = batch_data.to_dict()

            cfg = self.agent_config
            if cfg is None or self.compiled_graph is None:
                raise RuntimeError("Actor not initialized — call initialize_for_pipeline first")

            # Optional batch-level preprocessing
            if preprocess_fn is not None:
                batch_data = preprocess_fn(batch_data)

            columns = list(batch_data.values())
            if not columns:
                return batch_data, {}
            num_rows = len(columns[0])

            agent_outputs: list[Any] = []
            trajectories: list[AgentTrajectory] = []

            for i in range(num_rows):
                row = {k: v[i] for k, v in batch_data.items()}

                # Build graph input
                if cfg.input_mapper is not None:
                    graph_input = cfg.input_mapper(row)
                else:
                    graph_input = row

                # Invoke with guardrails (timeout + step limit + trajectory)
                result = self._invoke_with_guardrails(graph_input)

                # Map output
                if cfg.output_mapper is not None:
                    answer = cfg.output_mapper(result["output"])
                else:
                    answer = result["output"]

                agent_outputs.append(answer)

                if result["trajectory"] is not None:
                    trajectories.append(result["trajectory"])

            batch_data["agent_output"] = agent_outputs
            if trajectories:
                batch_data["trajectories"] = trajectories

            # Optional batch-level postprocessing
            if postprocess_fn is not None:
                batch_data = postprocess_fn(batch_data)

            # Compute metrics
            batch_metrics: dict[str, Any] = {}
            if compute_metrics_fn is not None:
                default_metrics = {
                    "Samples Processed": {
                        "value": num_rows,
                        "is_algebraic": False,
                    },
                }
                batch_metrics = {**default_metrics, **compute_metrics_fn(batch_data)}

            return batch_data, batch_metrics

        except Exception as e:
            error_type = type(e).__name__
            self.logger.exception(f"Error processing agent batch: {error_type}: {e}")
            raise RuntimeError(
                f"Error processing agent batch: {error_type}: {e}"
            ) from None

    # ------------------------------------------------------------------
    # Graph invocation with guardrails
    # ------------------------------------------------------------------

    def _invoke_with_guardrails(self, graph_input: Any) -> dict[str, Any]:
        """Invoke the compiled graph with timeout and trajectory capture.

        Returns:
            ``{"output": <graph result or None>, "trajectory": <AgentTrajectory or None>}``
        """
        cfg = self.agent_config
        trajectory: AgentTrajectory | None = None
        callbacks: list = []

        if cfg.capture_trajectory:
            trajectory = AgentTrajectory()
            callbacks = [TrajectoryCallback(trajectory)]

        invoke_config: dict[str, Any] = {
            "recursion_limit": cfg.max_steps,
        }
        if callbacks:
            invoke_config["callbacks"] = callbacks

        result = None
        start = time.monotonic()

        try:
            result = _invoke_with_timeout(
                self.compiled_graph,
                graph_input,
                invoke_config,
                timeout=cfg.timeout_seconds,
            )
        except _GraphTimeoutError:
            self.logger.warning(
                f"Agent timed out after {cfg.timeout_seconds}s "
                f"(pipeline {self.pipeline_id})"
            )
            if trajectory is not None:
                trajectory.timed_out = True
            if callbacks:
                result = callbacks[0].last_state
        except Exception as e:
            self.logger.warning(f"Agent invocation error: {e}")
            if trajectory is not None:
                trajectory.steps.append(
                    __import__("rapidfireai.evals.trajectory", fromlist=["TrajectoryStep"]).TrajectoryStep(
                        node_name="__error__",
                        error=str(e),
                        duration_seconds=time.monotonic() - start,
                    )
                )
            if callbacks:
                result = callbacks[0].last_state

        if trajectory is not None:
            trajectory.total_time = time.monotonic() - start
            trajectory.finalize()

        return {"output": result, "trajectory": trajectory}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release resources."""
        self.compiled_graph = None
        self.agent_config = None


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


class _GraphTimeoutError(Exception):
    pass


def _invoke_with_timeout(graph, graph_input, config: dict, timeout: float):
    """Invoke a compiled graph with a wall-clock timeout.

    Uses ``signal.alarm`` on POSIX (works inside Ray workers).
    Falls back to direct invocation when signals are unavailable.
    """
    if timeout <= 0:
        return graph.invoke(graph_input, config=config)

    def _handler(signum, frame):
        raise _GraphTimeoutError(f"Graph invocation exceeded {timeout}s timeout")

    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))
        result = graph.invoke(graph_input, config=config)
        signal.alarm(0)
        return result
    except _GraphTimeoutError:
        raise
    except AttributeError:
        # signal.SIGALRM not available (Windows)
        return graph.invoke(graph_input, config=config)
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
