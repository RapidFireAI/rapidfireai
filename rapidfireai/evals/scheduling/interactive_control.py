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
from rapidfireai.automl import RFOpenAIAPIModelConfig, RFvLLMModelConfig
from rapidfireai.evals.utils.constants import ICOperation, ICStatus, PipelineStatus
from rapidfireai.evals.utils.logger import RFLogger


class InteractiveControlHandler:
    """
    Handler for processing interactive control operations on running experiments.

    This class encapsulates all logic for stop/resume/delete/clone operations,
    keeping the Controller class focused on its core orchestration responsibilities.
    """

    def __init__(self, experiment_name: str, experiment_path: str, context_cache: dict, metric_manager=None):
        """
        Initialize the IC handler.

        Args:
            experiment_name: Name of the experiment
            experiment_path: Path to experiment logs/artifacts
            context_cache: Controller's context cache (maps context_hash -> (context_id, ObjectRef))
            metric_manager: Optional MetricLogger instance for creating metric runs on clone
        """
        # Initialize logger
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        self.logger = logging_manager.get_logger("InteractiveControl")

        # Reference to controller's context cache
        self._context_cache = context_cache

        # Store metric manager for creating metric runs on clone
        self.metric_manager = metric_manager

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
        pipeline_to_max_completion_tokens: dict = None,
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
                        pipeline_to_max_completion_tokens,
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

        # Delete run from MetricLogger (MLflow) - mirrors fit mode behavior
        if self.metric_manager:
            try:
                pipeline = db.get_pipeline(pipeline_id)
                if pipeline and pipeline.get("metric_run_id"):
                    self.metric_manager.delete_run(pipeline["metric_run_id"])
                    self.logger.info(f"Deleted MLflow run {pipeline['metric_run_id']} for pipeline {pipeline_id}")
            except Exception as e:
                self.logger.warning(f"Failed to delete MLflow run for pipeline {pipeline_id}: {e}")

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
        pipeline_to_max_completion_tokens: dict = None,
        progress_display=None,
    ) -> int:
        """
        Clone a new pipeline from a parent pipeline with edited configuration.

        The clone inherits the parent's context_id, RAG, and prompt_manager.
        Only the JSON-editable parameters (model config, sampling params, etc.) are modified.

        Args:
            request_data: JSON string with {"parent_pipeline_id": int, "config_json": {...}}
            scheduler: PipelineScheduler instance
            db: Database instance
            num_shards: Total number of shards
            dataset: Dataset being processed
            pipeline_aggregators: Dict mapping pipeline_id to Aggregator
            pipeline_results: Dict mapping pipeline_id to results/metrics
            pipeline_id_to_config: Dict mapping pipeline_id to (name, config)
            pipeline_to_rate_limiter: Optional dict for OpenAI rate limiters
            progress_display: Optional progress display to update

        Returns:
            pipeline_id of the newly created pipeline
        """
        # Parse request data
        data = json.loads(request_data)
        parent_pipeline_id = data["parent_pipeline_id"]
        edited_json = data["config_json"]

        # Get parent pipeline from database to inherit context_id
        parent_pipeline_db = db.get_pipeline(parent_pipeline_id)
        if not parent_pipeline_db:
            raise ValueError(f"Parent pipeline {parent_pipeline_id} not found in database")

        context_id = parent_pipeline_db["context_id"]

        # Validate that the context exists in the database
        context = db.get_context(context_id)
        if not context:
            raise ValueError(
                f"Context {context_id} from parent pipeline {parent_pipeline_id} not found in database. "
                f"The context may have been deleted."
            )

        self.logger.info(f"Validated context {context_id} exists (status: {context.get('status', 'unknown')})")

        # Get parent's FULL deserialized configuration from database
        # This includes rag and prompt_manager Python objects (stored via dill serialization)
        parent_full_config = parent_pipeline_db.get("pipeline_config")
        if not parent_full_config:
            raise ValueError(
                f"Parent pipeline {parent_pipeline_id} has no stored configuration. "
                f"Cannot clone without parent's config."
            )

        parent_model_config = parent_full_config["pipeline"]

        # Directly inherit RAG and prompt_manager from parent's deserialized config
        rag = getattr(parent_model_config, "rag", None)
        prompt_manager = getattr(parent_model_config, "prompt_manager", None)

        self.logger.info(
            f"Retrieved parent config from database for pipeline {parent_pipeline_id}: "
            f"rag={'present' if rag else 'None'}, "
            f"prompt_manager={'present' if prompt_manager else 'None'}"
        )

        # Apply RAG configuration changes if present in edited_json
        # This mirrors the extract_rag_params logic in serialize.py but in reverse:
        # applies the flattened rag_config back to the nested RAG object structure
        if rag is not None and "rag_config" in edited_json:
            rag_config = edited_json["rag_config"]
            self.logger.info(f"Applying RAG config changes: {rag_config}")

            # Map flattened rag_config keys to their locations in the RAG object
            # This mapping corresponds to extract_rag_params() in serialize.py

            # Direct attribute: search_type
            if "search_type" in rag_config:
                rag.search_type = rag_config["search_type"]

            # search_kwargs dict: k, filter, fetch_k, lambda_mult
            if rag.search_kwargs is None:
                rag.search_kwargs = {}
            for key in ["k", "filter", "fetch_k", "lambda_mult"]:
                if key in rag_config:
                    rag.search_kwargs[key] = rag_config[key]

            # reranker_kwargs dict: top_n
            if rag.reranker_kwargs is None:
                rag.reranker_kwargs = {}
            if "top_n" in rag_config:
                rag.reranker_kwargs["top_n"] = rag_config["top_n"]

            # text_splitter attributes: chunk_size, chunk_overlap
            if rag.text_splitter is not None:
                if "chunk_size" in rag_config:
                    rag.text_splitter._chunk_size = rag_config["chunk_size"]
                if "chunk_overlap" in rag_config:
                    rag.text_splitter._chunk_overlap = rag_config["chunk_overlap"]

            self.logger.info(f"RAG config updated successfully")

        # Extract pipeline type from edited JSON (or inherit from parent)
        pipeline_type = edited_json.get("pipeline_type")
        if not pipeline_type:
            # If not specified in JSON, infer from parent
            if isinstance(parent_model_config, RFvLLMModelConfig):
                pipeline_type = "vllm"
            elif isinstance(parent_model_config, RFOpenAIAPIModelConfig):
                pipeline_type = "openai"
            else:
                raise ValueError("Cannot determine pipeline type from parent")

        # Apply JSON edits on top of parent's configuration
        if pipeline_type.lower() == "vllm":
            # Get parent's baseline config from _user_params (original dicts, not converted objects)
            parent_model_config_dict = parent_model_config._user_params.get("model_config", {})
            parent_sampling_params = parent_model_config._user_params.get("sampling_params", {})

            # Apply edits from JSON using key-by-key merging (user values override parent values)
            # This preserves all parent keys and only overrides the keys specified by the user
            model_config_dict = {**parent_model_config_dict, **edited_json.get("model_config", {})}
            sampling_params_dict = {**parent_sampling_params, **edited_json.get("sampling_params", {})}

            model_config = RFvLLMModelConfig(
                model_config=model_config_dict,
                sampling_params=sampling_params_dict,
                rag=rag,  # Inherited from parent (with rag_config modifications applied if specified)
                prompt_manager=prompt_manager,  # Inherited from parent
            )

        elif pipeline_type.lower() == "openai":
            # Get parent's baseline config from _user_params (original dicts)
            parent_client_config = parent_model_config._user_params.get("client_config", {})
            parent_model_config_dict = parent_model_config._user_params.get("model_config", {})
            parent_rpm = parent_model_config._user_params.get("rpm_limit", 500)
            parent_tpm = parent_model_config._user_params.get("tpm_limit", 500_000)
            parent_max_completion_tokens = parent_model_config._user_params.get("max_completion_tokens", None)

            # Apply edits from JSON using key-by-key merging (user values override parent values)
            # This preserves all parent keys and only overrides the keys specified by the user
            client_config = {**parent_client_config, **edited_json.get("client_config", {})}
            model_config_dict = {**parent_model_config_dict, **edited_json.get("model_config", {})}

            model_config = RFOpenAIAPIModelConfig(
                client_config=client_config,
                model_config=model_config_dict,
                rag=rag,  # Inherited from parent (with rag_config modifications applied if specified)
                prompt_manager=prompt_manager,  # Inherited from parent
                rpm_limit=edited_json.get("rpm_limit", parent_rpm),
                tpm_limit=edited_json.get("tpm_limit", parent_tpm),
                max_completion_tokens=edited_json.get("max_completion_tokens", parent_max_completion_tokens),
            )

        else:
            raise ValueError(f"Unknown pipeline_type: {pipeline_type}. Supported types: 'vllm', 'openai'")

        # Build complete pipeline_config structure (inherit from parent if not in JSON)
        parent_batch_size = parent_full_config.get("batch_size", 32)
        parent_online_strategy = parent_full_config.get("online_strategy_kwargs")

        # Inherit function keys from parent (these cannot be edited via JSON)
        parent_preprocess_fn = parent_full_config.get("preprocess_fn")
        parent_postprocess_fn = parent_full_config.get("postprocess_fn")
        parent_compute_metrics_fn = parent_full_config.get("compute_metrics_fn")
        parent_accumulate_metrics_fn = parent_full_config.get("accumulate_metrics_fn")

        pipeline_config_dict = {
            "pipeline": model_config,
            "batch_size": edited_json.get("batch_size", parent_batch_size),
        }

        # Inherit function keys from parent (required for batch processing)
        if parent_preprocess_fn is not None:
            pipeline_config_dict["preprocess_fn"] = parent_preprocess_fn
        if parent_postprocess_fn is not None:
            pipeline_config_dict["postprocess_fn"] = parent_postprocess_fn
        if parent_compute_metrics_fn is not None:
            pipeline_config_dict["compute_metrics_fn"] = parent_compute_metrics_fn
        if parent_accumulate_metrics_fn is not None:
            pipeline_config_dict["accumulate_metrics_fn"] = parent_accumulate_metrics_fn

        # Add online_strategy_kwargs if present (merge with parent using key-by-key)
        if parent_online_strategy:
            # Merge parent online_strategy_kwargs with user edits
            pipeline_config_dict["online_strategy_kwargs"] = {
                **parent_online_strategy,
                **edited_json.get("online_strategy_kwargs", {})
            }
        elif "online_strategy_kwargs" in edited_json:
            # No parent strategy, use user's strategy as-is
            pipeline_config_dict["online_strategy_kwargs"] = edited_json["online_strategy_kwargs"]

        # Create new pipeline in database
        new_pipeline_id = db.create_pipeline(
            context_id=context_id,  # Inherited from parent
            pipeline_type=pipeline_type,
            pipeline_config=pipeline_config_dict,
            status=PipelineStatus.NEW,
        )

        # Add to pipeline_id_to_config mapping
        pipeline_id_to_config[new_pipeline_id] = pipeline_config_dict

        # Create metric run for the cloned pipeline (mirrors _register_pipelines and fit mode's _create_models)
        if self.metric_manager:
            try:
                pipeline_name = edited_json.get("pipeline_name", f"Pipeline {new_pipeline_id}")
                metric_run_id = self.metric_manager.create_run(f"{pipeline_name}_{new_pipeline_id}")
                db.set_pipeline_metric_run_id(new_pipeline_id, metric_run_id)

                # Log parent-run param (consistent with fit mode)
                self.metric_manager.log_param(metric_run_id, "parent-run", str(parent_pipeline_id))

                # Log model param
                if hasattr(model_config, "model_config") and model_config.model_config:
                    model_name = model_config.model_config.get("model", "unknown")
                    self.metric_manager.log_param(metric_run_id, "model", model_name)

                # Log RAG params
                if rag and hasattr(rag, "search_type"):
                    self.metric_manager.log_param(metric_run_id, "rag_search_type", str(rag.search_type))
                if rag and hasattr(rag, "search_kwargs") and rag.search_kwargs:
                    k = rag.search_kwargs.get("k")
                    if k is not None:
                        self.metric_manager.log_param(metric_run_id, "rag_k", str(k))

                # Log sampling params
                if hasattr(model_config, "sampling_params") and model_config.sampling_params:
                    sampling_str = json.dumps(model_config.sampling_params) if isinstance(model_config.sampling_params, dict) else str(model_config.sampling_params)
                    self.metric_manager.log_param(metric_run_id, "sampling_params", sampling_str)

                self.logger.info(f"Created metric run {metric_run_id} for cloned pipeline {new_pipeline_id}")
            except Exception as e:
                self.logger.warning(f"Failed to create MLflow run for cloned pipeline {new_pipeline_id}: {e}")

        # Reuse experiment-wide rate limiter actor for OpenAI pipelines
        # Since rate limiting is now at the experiment level, all OpenAI pipelines share the same rate limiter
        if isinstance(model_config, RFOpenAIAPIModelConfig) and pipeline_to_rate_limiter is not None:
            # Get the shared rate limiter actor from any existing OpenAI pipeline
            existing_rate_limiter = None
            for pid, rate_limiter_actor in pipeline_to_rate_limiter.items():
                if rate_limiter_actor is not None:
                    existing_rate_limiter = rate_limiter_actor
                    break

            if existing_rate_limiter:
                # Reuse the experiment-wide rate limiter
                pipeline_to_rate_limiter[new_pipeline_id] = existing_rate_limiter
                self.logger.info(f"Cloned OpenAI pipeline {new_pipeline_id} will use experiment-wide rate limiter")
            else:
                # This should not happen - experiment should always have a rate limiter for OpenAI pipelines
                raise RuntimeError(
                    f"Cannot clone OpenAI pipeline {new_pipeline_id}: no experiment-wide rate limiter found. "
                    "This suggests the experiment was not properly configured with OpenAI rate limits."
                )

            # Register max_completion_tokens for the cloned OpenAI pipeline
            if pipeline_to_max_completion_tokens is not None:
                max_completion_tokens = model_config.model_config.get("max_completion_tokens", 150)
                pipeline_to_max_completion_tokens[new_pipeline_id] = max_completion_tokens

        # Initialize aggregator for the new pipeline
        aggregator = Aggregator()
        pipeline_config = pipeline_id_to_config[new_pipeline_id]
        if hasattr(pipeline_config["pipeline"], "online_strategy"):
            aggregator.set_online_strategy(**pipeline_config["pipeline"].online_strategy)
        aggregator.set_total_population_size(len(dataset))
        pipeline_aggregators[new_pipeline_id] = aggregator

        # Initialize results tracking
        pipeline_results[new_pipeline_id] = {"results": {}, "metrics": {}, "start_time": None}

        # Add to scheduler (starts from shard 0)
        scheduler.add_pipeline(new_pipeline_id, shards_completed=0)

        # Update status to ONGOING (will be scheduled in next iteration)
        db.set_pipeline_status(new_pipeline_id, PipelineStatus.ONGOING)

        # Extract model name and metadata for display
        pipeline = pipeline_config["pipeline"]

        # Extract model name
        if isinstance(model_config, RFOpenAIAPIModelConfig):
            # For OpenAI, model name is in model_config.model_config
            model_name = model_config.model_config.get("model", "Unknown")
        elif isinstance(model_config, RFvLLMModelConfig):
            # For vLLM, model name is in model_config.model_config
            model_name = model_config.model_config.get("model", "Unknown")
        elif hasattr(pipeline, "model_config") and pipeline.model_config is not None:
            if "model" in pipeline.model_config:
                model_name = pipeline.model_config["model"]
            else:
                model_name = "Unknown"
        else:
            model_name = "Unknown"

        # Extract metadata fields (similar to controller.py)
        search_type = None
        rag_k = None
        top_n = None
        chunk_size = None
        chunk_overlap = None
        sampling_params = None
        prompt_manager_k = None
        model_config_dict = None

        if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
            # Extract full model config (excluding the model name)
            model_config_copy = pipeline.model_config.copy()
            model_config_copy.pop("model", None)
            if model_config_copy:  # Only assign if there are other configs
                model_config_dict = model_config_copy

        # Extract RAG-related fields
        if hasattr(pipeline, "rag") and pipeline.rag is not None:
            search_type = getattr(pipeline.rag, "search_type", None)
            if hasattr(pipeline.rag, "search_kwargs") and pipeline.rag.search_kwargs is not None:
                rag_k = pipeline.rag.search_kwargs.get("k", None)
            if hasattr(pipeline.rag, "reranker_kwargs") and pipeline.rag.reranker_kwargs is not None:
                top_n = pipeline.rag.reranker_kwargs.get("top_n", None)
            if hasattr(pipeline.rag, "text_splitter") and pipeline.rag.text_splitter is not None:
                chunk_size = getattr(pipeline.rag.text_splitter, "_chunk_size", None)
                chunk_overlap = getattr(pipeline.rag.text_splitter, "_chunk_overlap", None)

        # Extract sampling params from _user_params (dict, not SamplingParams object)
        if hasattr(pipeline, "sampling_params") and pipeline.sampling_params is not None:
            sampling_params = pipeline._user_params.get("sampling_params", None)

        # Extract prompt_manager fields
        if hasattr(pipeline, "prompt_manager") and pipeline.prompt_manager is not None:
            prompt_manager_k = getattr(pipeline.prompt_manager, "k", None)

        # Add to progress display
        if progress_display:
            progress_display.add_pipeline(
                pipeline_id=new_pipeline_id,
                pipeline_config=pipeline_config,
                model_name=model_name,
                status="ONGOING",
                search_type=search_type,
                rag_k=rag_k,
                top_n=top_n,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                sampling_params=sampling_params,
                prompt_manager_k=prompt_manager_k,
                model_config=model_config_dict,
            )

        self.logger.info(
            f"Cloned new pipeline {new_pipeline_id} (type={pipeline_type}) "
            f"using context {context_id} from parent pipeline {parent_pipeline_id}"
        )

        return new_pipeline_id
