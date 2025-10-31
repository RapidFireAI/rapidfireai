"""
Interactive Control Handler for RF-Inferno Experiments.

Handles dynamic pipeline management operations during experiment execution:
- Stop: Pause pipeline execution
- Resume: Continue stopped pipeline
- Delete: Permanently remove pipeline
- Clone: Create new pipeline from existing context
"""

import json
import time

from rapidfireai.evals.actors.rate_limiter_actor import RateLimiterActor

from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.metrics.aggregator import Aggregator
from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler
from rapidfireai.evals.utils.config import OpenAIAPIModelConfig, VLLMModelConfig
from rapidfireai.evals.utils.constants import ICOperation, ICStatus, PipelineStatus
from rapidfireai.evals.utils.logger import RFLogger


class InteractiveControlHandler:
    """
    Handler for processing interactive control operations on running experiments.

    This class encapsulates all logic for stop/resume/delete/clone operations,
    keeping the Controller class focused on its core orchestration responsibilities.
    """

    def __init__(self, experiment_name: str, experiment_path: str, context_cache: dict):
        """
        Initialize the IC handler.

        Args:
            experiment_name: Name of the experiment
            experiment_path: Path to experiment logs/artifacts
            context_cache: Controller's context cache (maps context_hash -> (context_id, ObjectRef))
        """
        # Initialize logger
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        self.logger = logging_manager.get_logger("InteractiveControl")

        # Reference to controller's context cache
        self._context_cache = context_cache

    def check_and_process_requests(
        self,
        scheduler: PipelineScheduler,
        db: RFDatabase,
        num_shards: int,
        dataset,
        pipeline_aggregators: dict,
        pipeline_results: dict,
        pipeline_id_to_config: dict,
        pipeline_to_rate_limiter: dict = None,
        online_strategy_kwargs: dict = None,
        progress_display=None,
    ) -> None:
        """
        Check for and process pending interactive control operations.

        This method polls the database for pending IC operations and executes them,
        allowing users to dynamically control pipelines during execution.

        Args:
            scheduler: PipelineScheduler instance
            db: Database instance
            num_shards: Total number of shards (for validation)
            dataset: Dataset being processed
            pipeline_aggregators: Dict mapping pipeline_id to Aggregator
            pipeline_results: Dict mapping pipeline_id to results/metrics
            pipeline_id_to_config: Dict mapping pipeline_id to (name, config)
            online_strategy_kwargs: Optional online aggregation strategy parameters
            progress_display: Optional PipelineProgressDisplay instance for updating UI
        """
        pending_ops = db.get_pending_ic_operations()
        if not pending_ops:
            return

        for op in pending_ops:
            ic_id = op["ic_id"]
            operation = op["operation"]
            pipeline_id = op["pipeline_id"]

            try:
                # Mark as processing
                db.update_ic_operation_status(ic_id, ICStatus.PROCESSING.value)

                if operation == ICOperation.STOP.value:
                    self._handle_stop(pipeline_id, scheduler, db, progress_display)

                elif operation == ICOperation.RESUME.value:
                    self._handle_resume(pipeline_id, scheduler, db, num_shards, progress_display)

                elif operation == ICOperation.DELETE.value:
                    self._handle_delete(pipeline_id, scheduler, db, pipeline_results, progress_display)

                elif operation == ICOperation.CLONE.value:
                    self._handle_clone(
                        op["request_data"],
                        scheduler,
                        db,
                        num_shards,
                        dataset,
                        pipeline_aggregators,
                        pipeline_results,
                        pipeline_id_to_config,
                        pipeline_to_rate_limiter,
                        online_strategy_kwargs,
                        progress_display,
                    )

                else:
                    raise ValueError(f"Unknown operation: {operation}")

                # Mark as completed
                db.update_ic_operation_status(ic_id, ICStatus.COMPLETED.value)
                self.logger.info(f"Completed IC operation {ic_id}: {operation} (pipeline {pipeline_id})")

                # add delay to prevent retry storms
                time.sleep(0.5)

            except Exception as e:
                error_msg = f"Failed to process IC operation {ic_id}: {str(e)}"
                self.logger.exception(error_msg)
                db.update_ic_operation_status(ic_id, ICStatus.FAILED.value, str(e))

                # add delay to prevent retry storms
                time.sleep(0.5)

    def _handle_stop(
        self, pipeline_id: int, scheduler: PipelineScheduler, db: RFDatabase, progress_display=None
    ) -> None:
        """
        Stop a pipeline (remove from scheduling, save progress).

        Args:
            pipeline_id: ID of pipeline to stop
            scheduler: PipelineScheduler instance
            db: Database instance
            progress_display: Optional progress display to update
        """
        # Remove from scheduler (returns shards completed)
        shards_completed = scheduler.remove_pipeline(pipeline_id)

        # Update database status
        db.set_pipeline_status(pipeline_id, PipelineStatus.STOPPED)

        # Update display
        if progress_display:
            progress_display.update_pipeline(pipeline_id, status="STOPPED")

        self.logger.info(f"Stopped pipeline {pipeline_id} at {shards_completed} shards completed")

    def _handle_resume(
        self, pipeline_id: int, scheduler: PipelineScheduler, db: RFDatabase, num_shards: int, progress_display=None
    ) -> None:
        """
        Resume a stopped pipeline (re-add to scheduler with saved progress).

        Args:
            pipeline_id: ID of pipeline to resume
            scheduler: PipelineScheduler instance
            db: Database instance
            num_shards: Total number of shards
            progress_display: Optional progress display to update
        """
        # Get pipeline info from database
        pipeline = db.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found in database")

        # Get saved progress
        shards_completed = pipeline["shards_completed"]

        # Validate pipeline was stopped
        if pipeline["status"] != PipelineStatus.STOPPED.value:
            raise ValueError(f"Pipeline {pipeline_id} is not stopped (status: {pipeline['status']})")

        # Re-add to scheduler with existing progress
        scheduler.add_pipeline(pipeline_id, shards_completed)

        # Update database status
        db.set_pipeline_status(pipeline_id, PipelineStatus.ONGOING)

        # Update display
        if progress_display:
            progress_display.update_pipeline(pipeline_id, status="ONGOING")

        self.logger.info(f"Resumed pipeline {pipeline_id} from shard {shards_completed}/{num_shards}")

    def _handle_delete(
        self,
        pipeline_id: int,
        scheduler: PipelineScheduler,
        db: RFDatabase,
        pipeline_results: dict,
        progress_display=None,
    ) -> None:
        """
        Delete a pipeline permanently (remove from scheduler and mark deleted).

        Args:
            pipeline_id: ID of pipeline to delete
            scheduler: PipelineScheduler instance
            db: Database instance
            pipeline_results: Dict mapping pipeline_id to results/metrics
            progress_display: Optional progress display to update
        """
        # Remove from scheduler
        scheduler.remove_pipeline(pipeline_id)

        # Update database status
        db.set_pipeline_status(pipeline_id, PipelineStatus.DELETED)

        # Clear pipeline results (optional - could keep for audit)
        pipeline_results.pop(pipeline_id, None)

        # Update display
        if progress_display:
            progress_display.update_pipeline(pipeline_id, status="DELETED")

        self.logger.info(f"Deleted pipeline {pipeline_id}")

    def _handle_clone(
        self,
        request_data: str,
        scheduler: PipelineScheduler,
        db: RFDatabase,
        num_shards: int,
        dataset,
        pipeline_aggregators: dict,
        pipeline_results: dict,
        pipeline_id_to_config: dict,
        pipeline_to_rate_limiter: dict = None,
        online_strategy_kwargs: dict = None,
        progress_display=None,
    ) -> int:
        """
        Clone a new pipeline using an existing context.

        Args:
            request_data: JSON string with clone parameters
            scheduler: PipelineScheduler instance
            db: Database instance
            num_shards: Total number of shards
            dataset: Dataset being processed
            pipeline_aggregators: Dict mapping pipeline_id to Aggregator
            pipeline_results: Dict mapping pipeline_id to results/metrics
            pipeline_id_to_config: Dict mapping pipeline_id to (name, config)
            online_strategy_kwargs: Optional online aggregation strategy parameters
            progress_display: Optional progress display to update

        Returns:
            pipeline_id of the newly created pipeline
        """
        # Parse request data
        data = json.loads(request_data)
        context_id = data["context_id"]
        pipeline_name = data.get("pipeline_name", f"cloned_{int(time.time())}")
        pipeline_type = data.get("pipeline_type", "vllm")  # Default to vllm for backwards compatibility

        # Get context info from database
        context = db.get_context(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found in database")

        context_hash = context["context_hash"]

        # Validate context exists in controller cache
        if context_hash not in self._context_cache:
            raise ValueError(f"Context {context_hash} not loaded in controller cache. Cannot clone pipeline.")

        # Get ContextGenerator object from cache
        _, _, context_obj = self._context_cache[context_hash]

        # Create appropriate ModelConfig based on pipeline type
        if pipeline_type.lower() == "vllm":
            model_config_dict = data["model_config"]
            sampling_params_dict = data["sampling_params"]

            model_config = VLLMModelConfig(
                model_config=model_config_dict, sampling_params=sampling_params_dict, context_generator=context_obj
            )

        elif pipeline_type.lower() == "openai":
            client_config = data["client_config"]
            model_config_dict = data["model_config"]

            # Note: Rate limiting is now configured at the experiment level
            model_config = OpenAIAPIModelConfig(
                client_config=client_config,
                model_config=model_config_dict,
                context_generator=context_obj,
            )

        else:
            raise ValueError(f"Unknown pipeline_type: {pipeline_type}. Supported types: 'vllm', 'openai'")

        # Register pipeline using existing Controller method
        # Note: This requires calling back to Controller's _register_pipelines
        # For now, we'll do the registration manually here to avoid circular dependency
        pipeline_id = db.create_pipeline(
            context_id=context_id,
            pipeline_name=pipeline_name,
            pipeline_type=pipeline_type,
            model_config_json=json.dumps(
                model_config.model_config
                if hasattr(model_config, "model_config")
                else {"model": model_config_dict.get("model", "unknown")}
            ),
            sampling_params_json=json.dumps(model_config.sampling_params_to_dict()),
            status=PipelineStatus.NEW,
        )

        # Add to pipeline_id_to_config mapping
        pipeline_id_to_config[pipeline_id] = (pipeline_name, model_config)

        # Reuse experiment-wide rate limiter actor for OpenAI pipelines
        # Since rate limiting is now at the experiment level, all OpenAI pipelines share the same rate limiter
        if isinstance(model_config, OpenAIAPIModelConfig) and pipeline_to_rate_limiter is not None:
            # Get the shared rate limiter actor from any existing OpenAI pipeline
            existing_rate_limiter = None
            for pid, rate_limiter_actor in pipeline_to_rate_limiter.items():
                if rate_limiter_actor is not None:
                    existing_rate_limiter = rate_limiter_actor
                    break

            if existing_rate_limiter:
                # Reuse the experiment-wide rate limiter
                pipeline_to_rate_limiter[pipeline_id] = existing_rate_limiter
                self.logger.info(f"Cloned OpenAI pipeline {pipeline_id} will use experiment-wide rate limiter")
            else:
                # This should not happen - experiment should always have a rate limiter for OpenAI pipelines
                raise RuntimeError(
                    f"Cannot clone OpenAI pipeline {pipeline_id}: no experiment-wide rate limiter found. "
                    "This suggests the experiment was not properly configured with OpenAI rate limits."
                )

        # Initialize aggregator for the new pipeline
        aggregator = Aggregator()
        if online_strategy_kwargs:
            aggregator.set_online_strategy(**online_strategy_kwargs)
        aggregator.set_total_population_size(len(dataset))
        pipeline_aggregators[pipeline_id] = aggregator

        # Initialize results tracking
        pipeline_results[pipeline_id] = {"results": {}, "metrics": {}, "start_time": None}

        # Add to scheduler (starts from shard 0)
        scheduler.add_pipeline(pipeline_id, shards_completed=0)

        # Update status to ONGOING (will be scheduled in next iteration)
        db.set_pipeline_status(pipeline_id, PipelineStatus.ONGOING)

        # Extract model name for display
        if hasattr(model_config, "model_config") and "model" in model_config.model_config:
            model_name = model_config.model_config["model"]
        else:
            model_name = model_config_dict.get("model", "Unknown")

        # Add to progress display
        if progress_display:
            progress_display.add_pipeline(pipeline_id, pipeline_name, model_name, status="ONGOING")

        self.logger.info(
            f"Cloned new pipeline {pipeline_id} ({pipeline_name}, type={pipeline_type}) "
            f"using context {context_id} ({context_hash[:8]})"
        )

        return pipeline_id
