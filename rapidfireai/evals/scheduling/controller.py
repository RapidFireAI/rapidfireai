import hashlib
import json
import time
from collections.abc import Callable
from typing import Any
import ray

from rapidfireai.utils.constants import ColabConfig, RF_EXPERIMENT_PATH
from rapidfireai.evals.actors.doc_actor import DocProcessingActor
from rapidfireai.evals.actors.inference_engines import InferenceEngine
from rapidfireai.evals.actors.query_actor import QueryProcessingActor
from rapidfireai.evals.data.dataset import DataLoader
from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.metrics.aggregator import Aggregator
from rapidfireai.evals.scheduling.interactive_control import InteractiveControlHandler
from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler
from rapidfireai.evals.scheduling.scheduler import Scheduler
from rapidfireai.automl import ModelConfig, RFvLLMModelConfig
from rapidfireai.evals.utils.constants import (
    NUM_CPUS_PER_DOC_ACTOR,
    NUM_QUERY_PROCESSING_ACTORS,
    ContextStatus,
    ExperimentStatus,
    PipelineStatus,
    TaskStatus,
)
from rapidfireai.evals.utils.logger import RFLogger
from rapidfireai.evals.utils.progress_display import ContextBuildingDisplay, PipelineProgressDisplay
from rapidfireai.automl import RFGridSearch, RFRandomSearch
from rapidfireai.automl import get_runs, get_flattened_config_leaf
from rapidfireai.evals.utils.serialize import extract_pipeline_config_json

class Controller:
    """
    Controller for orchestrating distributed inference pipeline.

    Manages data loading, scheduling, aggregation, and actor creation.
    Handles all the complexity of RAG initialization, actor creation, and batch processing.
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_path: str = RF_EXPERIMENT_PATH,
        metric_manager=None,
    ):
        """
        Initialize the controller.

        Args:
            experiment_name: Name of the experiment
            experiment_path: Path to experiment logs/artifacts
            metric_manager: Optional MetricLogger instance for logging metrics
        """
        self.aggregator = Aggregator()
        self.dataloader = DataLoader()
        self.scheduler = Scheduler(strategy="round_robin")
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        self.metric_manager = metric_manager

        # Initialize logger
        logging_manager = RFLogger(experiment_name=self.experiment_name, experiment_path=self.experiment_path)
        self.logger = logging_manager.get_logger("Controller")

        # Cache for RAG contexts (persists only during Controller lifetime)
        # Maps context_hash -> (context_id, components_ref)
        self._context_cache: dict[str, tuple[int, ray.ObjectRef]] = {}

        # Initialize interactive control handler
        self.ic_handler = InteractiveControlHandler(
            experiment_name=self.experiment_name,
            experiment_path=self.experiment_path,
            context_cache=self._context_cache,
            metric_manager=self.metric_manager,
        )

    @staticmethod
    def _sanitize_for_json(obj: Any) -> dict[str, Any]:
        """
        Sanitize an object for JSON serialization by removing non-serializable fields.

        Removes functions, lambdas, GPU references, Ray ObjectRefs, and other
        non-serializable objects while preserving serializable primitive types.

        Args:
            obj: Object to sanitize (typically has __dict__ attribute)

        Returns:
            Dictionary with only JSON-serializable fields
        """
        if obj is None:
            return {}

        # Get object attributes if it has __dict__, otherwise return empty dict
        if not hasattr(obj, "__dict__"):
            return {}

        sanitized = {}
        for key, value in obj.__dict__.items():
            # Skip private attributes
            if key.startswith("_"):
                continue

            # Skip non-serializable types
            if callable(value):  # Functions, methods, lambdas
                continue
            if isinstance(value, type):  # Classes
                continue
            if hasattr(value, "__module__") and "ray" in value.__module__:  # Ray objects
                continue
            if hasattr(value, "__class__") and "torch" in value.__class__.__module__:  # PyTorch objects
                continue

            # Try to serialize the value
            try:
                json.dumps(value)  # Test if serializable
                sanitized[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue

        return sanitized

    def _log_pipeline_params(
        self,
        pipeline_config: dict,
        run_id: str,
        manager: Any,
        manager_name: str = "tracking",
    ) -> None:
        """
        Log comprehensive pipeline parameters to MLflow/Trackio.

        Args:
            pipeline_config: The pipeline configuration dict
            run_id: The MLflow/Trackio run ID
            manager: The MLflow or Trackio manager instance
            manager_name: Name of the manager for logging (default: "tracking")
        """
        pipeline = pipeline_config.get("pipeline")

        # Log batch_size from config
        batch_size = pipeline_config.get("batch_size")
        if batch_size is not None:
            manager.log_param(run_id, "batch_size", str(batch_size))

        # Log online_strategy_kwargs
        online_strategy = pipeline_config.get("online_strategy_kwargs")
        if online_strategy:
            for key, value in online_strategy.items():
                manager.log_param(run_id, f"online_strategy.{key}", str(value))

        if not pipeline:
            return

        # Log model_config params (comprehensive)
        if hasattr(pipeline, "model_config") and pipeline.model_config:
            model_config = pipeline.model_config
            for key, value in model_config.items():
                if value is not None:
                    # Truncate long values for MLflow (max 500 chars)
                    str_value = str(value)
                    if len(str_value) > 500:
                        str_value = str_value[:497] + "..."
                    manager.log_param(run_id, f"model.{key}", str_value)

        # Log client_config params (excluding sensitive data like api_key)
        if hasattr(pipeline, "client_config") and pipeline.client_config:
            client_config = pipeline.client_config
            for key, value in client_config.items():
                if key.lower() in ["api_key", "secret", "token", "password"]:
                    continue  # Skip sensitive params
                if value is not None:
                    manager.log_param(run_id, f"client.{key}", str(value))

        # Log rate limits
        if hasattr(pipeline, "rpm_limit") and pipeline.rpm_limit:
            manager.log_param(run_id, "rpm_limit", str(pipeline.rpm_limit))
        if hasattr(pipeline, "tpm_limit") and pipeline.tpm_limit:
            manager.log_param(run_id, "tpm_limit", str(pipeline.tpm_limit))

        # Log prompt_manager params
        if hasattr(pipeline, "prompt_manager") and pipeline.prompt_manager:
            pm = pipeline.prompt_manager
            if hasattr(pm, "k") and pm.k is not None:
                manager.log_param(run_id, "fewshot_k", str(pm.k))
            if hasattr(pm, "instructions") and pm.instructions:
                # Truncate long instructions
                instructions = pm.instructions[:200] + "..." if len(pm.instructions) > 200 else pm.instructions
                manager.log_param(run_id, "instructions", instructions)

        # Log RAG params
        if hasattr(pipeline, "rag") and pipeline.rag:
            rag = pipeline.rag
            if hasattr(rag, "search_type"):
                manager.log_param(run_id, "rag.search_type", str(rag.search_type))
            if hasattr(rag, "search_kwargs") and rag.search_kwargs:
                for key, value in rag.search_kwargs.items():
                    manager.log_param(run_id, f"rag.{key}", str(value))
            if hasattr(rag, "chunk_size"):
                manager.log_param(run_id, "rag.chunk_size", str(rag.chunk_size))
            if hasattr(rag, "chunk_overlap"):
                manager.log_param(run_id, "rag.chunk_overlap", str(rag.chunk_overlap))

        # Log sampling params (for vLLM)
        if hasattr(pipeline, "sampling_params") and pipeline.sampling_params:
            sampling = pipeline.sampling_params
            if isinstance(sampling, dict):
                for key, value in sampling.items():
                    manager.log_param(run_id, f"sampling.{key}", str(value))
            else:
                manager.log_param(run_id, "sampling_params", str(sampling)[:500])

        # Log max_completion_tokens
        if hasattr(pipeline, "max_completion_tokens") and pipeline.max_completion_tokens:
            manager.log_param(run_id, "max_completion_tokens", str(pipeline.max_completion_tokens))

    @staticmethod
    def _collect_unique_contexts(
        config_leaves: Any,
    ) -> dict[str, tuple[Any, Any]]:
        """
        Collect unique RAG contexts from pipeline configs.

        Multiple pipelines may share the same RAG configuration. This method
        deduplicates them by hash to avoid building the same RAG components multiple times.

        Args:
            config_leaves: List of pipeline config dictionaries

        Returns:
            Dictionary mapping context_hash -> (rag_spec, prompt_manager) for unique contexts
        """
        unique_contexts = {}

        for config_leaf in config_leaves:
            # Check if pipeline has a RAG spec or prompt_manager
            pipeline_config = config_leaf["pipeline"]
            has_rag_attr = hasattr(pipeline_config, "rag")

            # Get RAG spec and prompt_manager
            rag_spec = getattr(pipeline_config, "rag", None) if has_rag_attr else None
            prompt_manager = getattr(pipeline_config, "prompt_manager", None)

            # Skip if neither RAG nor prompt_manager (no context to build)
            if not rag_spec and not prompt_manager:
                continue

            # Compute context hash
            rag_hash = rag_spec.get_hash() if rag_spec else None
            prompt_hash = prompt_manager.get_hash() if prompt_manager else None

            # Create combined context hash
            if rag_hash and prompt_hash:
                context_hash = hashlib.sha256(f"{rag_hash}:{prompt_hash}".encode()).hexdigest()
            elif rag_hash:
                context_hash = rag_hash
            elif prompt_hash:
                context_hash = prompt_hash
            else:
                continue  # Should not happen, but safety check

            if context_hash not in unique_contexts:
                unique_contexts[context_hash] = (rag_spec, prompt_manager)

        return unique_contexts

    def _setup_context_generators(
        self,
        config_leaves: Any,
        db: RFDatabase,
    ) -> None:
        """
        Setup RAG contexts: build if needed and cache in Ray object store.

        This method orchestrates the entire context generation lifecycle:
        1. Collect unique RAG contexts from all pipeline configs
        2. Check controller cache for already-built contexts (current session)
        3. Build new contexts in parallel using DocProcessingActors
        4. Store all contexts in Ray object store for sharing
        5. Track metadata in database for analytics

        All built contexts are stored in self._context_cache.

        Args:
            config_leaves: List of pipeline config dictionaries
            db: Database instance
        """
        # Step 1: Collect unique RAG contexts
        unique_contexts = self._collect_unique_contexts(config_leaves)

        if not unique_contexts:
            self.logger.info("No RAG contexts found in pipeline configs")
            return

        self.logger.info(f"Found {len(unique_contexts)} unique RAG context(s)")

        # Step 2: Identify contexts that need to be built
        contexts_to_build = []
        for context_hash, (rag_spec, prompt_manager) in unique_contexts.items():
            # Check if already built in this Controller session
            if context_hash in self._context_cache:
                self.logger.info(f"Context {context_hash[:8]}... already in cache (session), reusing")
                continue

            # Need to build new context
            self.logger.info(f"Will build new context {context_hash[:8]}...")

            # Create context record in DB for tracking/analytics
            context_id = db.create_context(
                context_hash=context_hash,
                rag_config_json=json.dumps(self._sanitize_for_json(rag_spec)),
                prompt_config_json=json.dumps(self._sanitize_for_json(prompt_manager)),
                status=ContextStatus.ONGOING,
            )
            db.set_context_start_time(context_id, time.time())

            contexts_to_build.append(
                {
                    "context_hash": context_hash,
                    "context_id": context_id,
                    "rag": rag_spec,
                    "prompt_manager": prompt_manager,
                    "start_time": time.time(),
                }
            )

        if not contexts_to_build:
            self.logger.info("All contexts already in cache, no building needed")
            return

        # Step 3: Build all contexts in parallel
        self.logger.info(f"Building {len(contexts_to_build)} context(s) in parallel...")
        try:
            self.build_rag_components(contexts_to_build, db)
        except Exception:
            self.logger.exception("Failed to build contexts in parallel")
            raise

    def build_rag_components(
        self,
        contexts_to_build: list[dict],
        db: RFDatabase,
    ) -> None:
        """
        Build multiple RAG components in parallel using DocProcessingActors.

        Creates one DocProcessingActor per context and processes them all in parallel.
        Updates database and cache for each context as they complete.

        Args:
            contexts_to_build: List of dicts with keys: context_hash, context_id,
                              rag, prompt_manager, start_time
            db: Database instance for tracking build status
        """
        if not contexts_to_build:
            return

        num_contexts = len(contexts_to_build)
        self.logger.info(f"Creating {num_contexts} DocProcessingActor(s) for parallel processing")

        # Prepare context info for display (add enable_gpu flag)
        for context_info in contexts_to_build:
            rag_spec = context_info["rag"]
            context_info["enable_gpu"] = rag_spec.enable_gpu_search if rag_spec else False

        # Initialize progress display for context building
        context_display = ContextBuildingDisplay(contexts_to_build)
        context_display.start()

        # Step 1: Create all DocProcessingActors and submit all build tasks
        actor_tasks = []
        for context_info in contexts_to_build:
            rag_spec = context_info["rag"]
            prompt_manager = context_info["prompt_manager"]

            # Skip if neither RAG nor prompt_manager (shouldn't happen, but safety check)
            if not rag_spec and not prompt_manager:
                continue

            # Allocate resources based on GPU needs:
            # - If GPU search enabled: 1 GPU + 2 CPUs
            # - If CPU only: 0 GPUs + 2 CPUs
            # - If prompt-only with CUDA device requested: 1 GPU + 2 CPUs
            # - If prompt-only without CUDA: 0 GPUs + 2 CPUs
            needs_gpu = False
            if rag_spec and rag_spec.enable_gpu_search:
                needs_gpu = True
            elif rag_spec:
                # Check if rag_spec embeddings or reranker request CUDA (even if enable_gpu_search=False)
                if rag_spec.embedding_kwargs:
                    model_kwargs = rag_spec.embedding_kwargs.get("model_kwargs", {})
                    device = model_kwargs.get("device", "") if isinstance(model_kwargs, dict) else ""
                    if device and str(device).startswith("cuda"):
                        needs_gpu = True
                if not needs_gpu and rag_spec.reranker_kwargs:
                    reranker_model_kwargs = rag_spec.reranker_kwargs.get("model_kwargs", {})
                    device = reranker_model_kwargs.get("device", "") if isinstance(reranker_model_kwargs, dict) else ""
                    if device and str(device).startswith("cuda"):
                        needs_gpu = True
            elif prompt_manager and prompt_manager.embedding_kwargs:
                # Check if prompt_manager requests CUDA device
                model_kwargs = prompt_manager.embedding_kwargs.get("model_kwargs", {})
                device = model_kwargs.get("device", "") if isinstance(model_kwargs, dict) else ""
                if device and str(device).startswith("cuda"):
                    needs_gpu = True
            system_num_gpus = ray.available_resources().get("GPU", 0)
            if system_num_gpus > 1:
                num_gpus_for_actor = 1
            elif system_num_gpus > 0:
                num_gpus_for_actor = 0.5
            else:
                num_gpus_for_actor = 0
            num_cpus_for_actor = NUM_CPUS_PER_DOC_ACTOR

            # Create DocProcessingActor
            # Ray will queue actors if resources aren't immediately available
            doc_actor = DocProcessingActor.options(
                num_gpus=num_gpus_for_actor,
                num_cpus=num_cpus_for_actor,
            ).remote(
                experiment_name=self.experiment_name,
                experiment_path=self.experiment_path,
            )

            # Submit build task (non-blocking)
            components_future = doc_actor.build_rag_components.remote(rag_spec, prompt_manager)

            actor_tasks.append(
                {
                    "actor": doc_actor,
                    "future": components_future,
                    "context_info": context_info,
                }
            )

        self.logger.info(f"Submitted {len(actor_tasks)} build task(s) in parallel")

        # Step 2: Wait for all tasks to complete and process results
        for task in actor_tasks:
            context_info = task["context_info"]
            context_hash = context_info["context_hash"]
            context_id = context_info["context_id"]
            start_time = context_info["start_time"]

            try:
                # Wait for this specific build to complete
                components = ray.get(task["future"])
                end_time = time.time()
                duration = end_time - start_time

                # Put CPU-serializable components in Ray object store (shared memory)
                context_generator_ref = ray.put(components)

                # Update database
                db.set_context_end_time(context_id, end_time, duration)
                db.set_context_status(context_id, ContextStatus.ONGOING)
                self.logger.info(f"Built context {context_id} ({context_hash[:8]}...) successfully in {duration:.2f}s")

                # Cache for session-level reuse
                self._context_cache[context_hash] = (context_id, context_generator_ref)

                # Update display
                context_display.update_context(context_hash, status="complete", duration=duration)

            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time

                db.set_context_status(context_id, ContextStatus.FAILED)
                db.set_context_error(context_id, str(e))
                self.logger.exception(f"Failed to build context {context_id} ({context_hash[:8]}...)")

                # Update display
                context_display.update_context(context_hash, status="failed", duration=duration)

                # HALT: Context creation is critical - stop the entire experiment
                context_display.stop()
                error_message = (
                    f"\n{'='*80}\n"
                    f"❌ CRITICAL ERROR: RAG Source Preprocessing Failed\n"
                    f"{'='*80}\n"
                    f"RAG Source ID: {context_id}\n"
                    f"Context Hash: {context_hash[:16]}...\n"
                    f"Error: {str(e)}\n"
                    f"{'='*80}\n"
                    f"\nThe experiment has been halted. Please fix the error and try again.\n"
                )
                print(error_message)
                raise RuntimeError(f"Context creation failed for context {context_id}") from e

            finally:
                # Clean up DocProcessingActor
                ray.kill(task["actor"])

        # Stop the context building display
        context_display.stop()

        self.logger.info(f"Completed parallel context building for {num_contexts} context(s)")

    def create_query_actors(
        self,
        engine_class: type[InferenceEngine],
        engine_kwargs: dict[str, Any],
        context_generator_ref: ray.ObjectRef | None,
    ) -> list:
        """
        Create query processing actors with the specified inference engine.

        Args:
            engine_class: The inference engine class to instantiate (VLLMInferenceEngine or OpenAIInferenceEngine)
            engine_kwargs: Kwargs to pass to engine constructor
            context_generator_ref: Ray ObjectRef to shared RAG components in object store
            gpus_per_actor: GPUs per actor
            cpus_per_actor: CPUs per actor

        Returns:
            List of Ray actor handles
        """
        num_actors = self.num_actors or NUM_QUERY_PROCESSING_ACTORS
        gpus_per_actor = self.num_gpus // num_actors
        cpus_per_actor = self.num_cpus // num_actors

        assert gpus_per_actor > 0, (
            "Not enough GPUs available. Got {self.num_gpus} GPUs and {num_actors} actors, need at least 1 GPU per actor"
        )
        assert cpus_per_actor > 0, (
            "Not enough CPUs available. Got {self.num_cpus} CPUs and {num_actors} actors, need at least 1 CPU per actor"
        )

        actors = []
        for i in range(num_actors):
            actor = QueryProcessingActor.options(num_gpus=gpus_per_actor, num_cpus=cpus_per_actor).remote(
                engine_class=engine_class,
                engine_kwargs=engine_kwargs,
                context_generator_ref=context_generator_ref,
                experiment_name=self.experiment_name,
                experiment_path=self.experiment_path,
                actor_id=i,
            )
            actors.append(actor)

        return actors

    def _register_pipelines(
        self,
        config_leaves: list[Any],
        db: RFDatabase,
    ) -> tuple[list[int], dict[int, dict]]:
        """
        Register pipelines in database.

        Args:
            pipeline_configs: List of (pipeline_name, model_config) tuples
            db: Database instance

        Returns:
            Tuple of (pipeline_ids, pipeline_id_to_config mapping)
        """
        pipeline_id_to_config = {}
        pipeline_ids = []

        for pipeline_config in config_leaves:
            # Determine context_id for this pipeline
            context_id = None
            pipeline = pipeline_config["pipeline"]
            has_rag_attr = hasattr(pipeline, "rag")
            rag_spec = getattr(pipeline, "rag", None) if has_rag_attr else None
            prompt_manager = getattr(pipeline, "prompt_manager", None)

            # Check if pipeline has RAG or prompt_manager to look up context
            if rag_spec or prompt_manager:
                # Get RAG hash if present
                rag_hash = rag_spec.get_hash() if rag_spec else None

                # Get prompt_manager hash if present
                prompt_hash = prompt_manager.get_hash() if prompt_manager else None

                # Create combined context hash (matches logic in _collect_unique_contexts)
                if rag_hash and prompt_hash:
                    context_hash = hashlib.sha256(f"{rag_hash}:{prompt_hash}".encode()).hexdigest()
                elif rag_hash:
                    context_hash = rag_hash
                elif prompt_hash:
                    context_hash = prompt_hash
                else:
                    context_hash = None

                if context_hash and context_hash in self._context_cache:
                    context_id, _ = self._context_cache[context_hash]

            # Generate flattened config for IC Ops panel display
            # First extract JSON-serializable config, then flatten it
            json_config = extract_pipeline_config_json(pipeline_config)
            flattened_config = get_flattened_config_leaf(json_config) if json_config else {}

            pipeline_id = db.create_pipeline(
                context_id=context_id,
                pipeline_type="vllm",
                pipeline_config=pipeline_config,
                status=PipelineStatus.NEW,
                flattened_config=flattened_config,
            )

            # Create MetricLogger run for this pipeline
            metric_run_id = None
            if self.metric_manager:
                try:
                    pipeline_name = pipeline_config.get("pipeline_name", f"Pipeline {pipeline_id}")
                    metric_run_id = self.metric_manager.create_run(f"{pipeline_name}_{pipeline_id}")
                    db.set_pipeline_metric_run_id(pipeline_id, metric_run_id)

                    pipeline = pipeline_config.get("pipeline")
                    if pipeline:
                        if hasattr(pipeline, "model_config") and pipeline.model_config:
                            model_name = pipeline.model_config.get("model", "unknown")
                            self.metric_manager.log_param(metric_run_id, "model", model_name)

                        if hasattr(pipeline, "rag") and pipeline.rag:
                            if hasattr(pipeline.rag, "search_type"):
                                self.metric_manager.log_param(metric_run_id, "rag_search_type", str(pipeline.rag.search_type))
                            if hasattr(pipeline.rag, "search_kwargs") and pipeline.rag.search_kwargs:
                                k = pipeline.rag.search_kwargs.get("k")
                                if k is not None:
                                    self.metric_manager.log_param(metric_run_id, "rag_k", str(k))

                        # Extract sampling params
                        if hasattr(pipeline, "sampling_params") and pipeline.sampling_params:
                            import json
                            sampling_str = json.dumps(pipeline.sampling_params) if isinstance(pipeline.sampling_params, dict) else str(pipeline.sampling_params)
                            self.metric_manager.log_param(metric_run_id, "sampling_params", sampling_str)

                    self.logger.debug(f"Created Metrics run {metric_run_id} for pipeline {pipeline_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to create Metrics run for pipeline {pipeline_id}: {e}")

            pipeline_ids.append(pipeline_id)
            pipeline_id_to_config[pipeline_id] = pipeline_config

        return pipeline_ids, pipeline_id_to_config

    def _compute_final_metrics_for_pipelines(
        self,
        pipeline_ids: list[int],
        pipeline_id_to_config: dict[int, dict],
        pipeline_aggregators: dict[int, Aggregator],
        pipeline_results: dict[int, dict],
        db: RFDatabase,
        progress_display=None,
        pipeline_id_to_info: dict[int, dict] = None,
        total_dataset_size: int = None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Compute final metrics for each pipeline and update database.

        Includes pipelines with statuses: COMPLETED, STOPPED, ONGOING (with partial results).
        Excludes pipelines with statuses: DELETED, FAILED.

        Args:
            pipeline_ids: List of pipeline IDs
            pipeline_id_to_config: Mapping of pipeline_id to (name, config)
            pipeline_aggregators: Mapping of pipeline_id to Aggregator
            pipeline_results: Mapping of pipeline_id to results/metrics dict
            db: Database instance
            progress_display: Optional progress display to update
            pipeline_id_to_info: Optional mapping of pipeline_id to pipeline info dict

        Returns:
            Dict mapping pipeline_id to (aggregated_results, cumulative_metrics) for
            COMPLETED, STOPPED, and ONGOING pipelines (excludes DELETED and FAILED)
        """
        self.logger.info("Computing final metrics for all pipelines...")

        final_results = {}
        for pipeline_id in pipeline_ids:
            pipeline_config = pipeline_id_to_config[pipeline_id]
            pipeline_name = pipeline_config.get("pipeline_name", f"Pipeline {pipeline_id}")

            # Check pipeline status
            pipeline_status = db.get_pipeline(pipeline_id)["status"]

            # Skip DELETED and FAILED pipelines
            if pipeline_status in [PipelineStatus.DELETED.value, PipelineStatus.FAILED.value]:
                if pipeline_status == PipelineStatus.FAILED.value:
                    self.logger.warning(f"Pipeline {pipeline_id} failed, skipping final metrics")
                else:
                    self.logger.info(f"Pipeline {pipeline_id} deleted, skipping final metrics")
                continue

            # Skip pipelines with no results (cloned but never processed)
            if not pipeline_results[pipeline_id]["results"]:
                self.logger.info(f"Pipeline {pipeline_id} has no results, skipping final metrics")
                continue

            aggregator = pipeline_aggregators[pipeline_id]
            start_time = pipeline_results[pipeline_id]["start_time"]
            end_time = time.time()

            # Get metrics functions from pipeline config
            compute_metrics_fn = pipeline_config.get("compute_metrics_fn", None)
            accumulate_metrics_fn = pipeline_config.get("accumulate_metrics_fn", None)
            cumulative_metrics = aggregator.compute_final_metrics(
                aggregated_results=pipeline_results[pipeline_id]["results"],
                aggregated_metrics=pipeline_results[pipeline_id]["metrics"],
                compute_metrics_fn=compute_metrics_fn,
                accumulate_metrics_fn=accumulate_metrics_fn,
                start_time=start_time,
                end_time=end_time,
            )

            # Add confidence intervals to final metrics before storing
            samples_processed = sum(
                m.get("value", 0)
                for m in pipeline_results[pipeline_id]["metrics"].get("Samples Processed", [{}])
            )
            if aggregator.online_strategy and samples_processed > 0:
                cumulative_metrics = aggregator.online_strategy.add_confidence_interval_info(
                    cumulative_metrics, samples_processed
                )

            # Add pipeline info to cumulative_metrics (use passed pipeline_id_to_info if available)
            if pipeline_id_to_info and pipeline_id in pipeline_id_to_info:
                pipeline_info_dict = pipeline_id_to_info[pipeline_id]
                for key, value in pipeline_info_dict.items():
                    cumulative_metrics[key] = {"value": value}

            # Reorder cumulative_metrics: run_id, model_name, hyperparams, Samples Processed, then metrics
            ordered_metrics = {}

            ordered_metrics["run_id"] = {"value": pipeline_id}

            if "model_name" in cumulative_metrics:
                ordered_metrics["model_name"] = cumulative_metrics["model_name"]

            hyperparam_keys = ["search_type", "rag_k", "top_n", "chunk_size", "chunk_overlap",
                             "sampling_params", "prompt_manager_k", "model_config"]
            for key in hyperparam_keys:
                if key in cumulative_metrics:
                    ordered_metrics[key] = cumulative_metrics[key]

            if "Samples Processed" in cumulative_metrics:
                ordered_metrics["Samples Processed"] = cumulative_metrics["Samples Processed"]

            excluded_keys = {"run_id", "model_name", "Samples Processed"} | set(hyperparam_keys)
            for key, value in cumulative_metrics.items():
                if key not in excluded_keys:
                    ordered_metrics[key] = value

            final_results[pipeline_id] = (
                pipeline_results[pipeline_id]["results"],
                ordered_metrics,
            )

            # Update pipeline status
            db.set_pipeline_status(pipeline_id, PipelineStatus.COMPLETED)
            if progress_display:
                # Update display with final metrics to ensure all metrics are shown
                # Convert ordered_metrics format to display format with CI info
                display_metrics = {}
                for metric_name, metric_data in ordered_metrics.items():
                    if isinstance(metric_data, dict):
                        display_metrics[metric_name] = {
                            "value": metric_data.get("value", 0),
                            "lower_bound": metric_data.get("lower_bound"),
                            "upper_bound": metric_data.get("upper_bound"),
                            "margin_of_error": metric_data.get("margin_of_error"),
                            "is_algebraic": metric_data.get("is_algebraic", False),
                        }
                    else:
                        display_metrics[metric_name] = {"value": metric_data}

                progress_display.update_pipeline(
                    pipeline_id,
                    status="COMPLETED",
                    metrics=display_metrics
                )

            # Log final metrics to MetricLogger
            if self.metric_manager:
                try:
                    pipeline = db.get_pipeline(pipeline_id)
                    metric_run_id = pipeline.get("metric_run_id") if pipeline else None
                    if metric_run_id:
                        total_samples = pipeline.get("total_samples_processed", 0)
                        if total_dataset_size and total_dataset_size > 0:
                            percentage_processed = (total_samples / total_dataset_size * 100)
                        else:
                            percentage_processed = 100  # Assume complete if dataset size unknown
                        step = int(percentage_processed)

                        for metric_name, metric_data in ordered_metrics.items():
                            if metric_name in ["run_id", "model_name", "Samples Processed", "Processing Time", "Samples Per Second"]:
                                continue

                            if isinstance(metric_data, dict):
                                metric_value = metric_data.get("value", 0)
                                lower_bound = metric_data.get("lower_bound")
                                upper_bound = metric_data.get("upper_bound")
                            else:
                                metric_value = metric_data
                                lower_bound = None
                                upper_bound = None

                            # Log main metric value
                            if isinstance(metric_value, (int, float)):
                                try:
                                    self.metric_manager.log_metric(metric_run_id, metric_name, float(metric_value), step=step)
                                except Exception as e:
                                    self.logger.debug(f"Failed to log final metric {metric_name} to MetricLogger: {e}")

                            # Log lower_bound if available
                            if lower_bound is not None and isinstance(lower_bound, (int, float)):
                                try:
                                    self.metric_manager.log_metric(metric_run_id, f"{metric_name}_lower_bound", float(lower_bound), step=step)
                                except Exception as e:
                                    self.logger.debug(f"Failed to log final metric {metric_name}_lower_bound to MetricLogger: {e}")

                            # Log upper_bound if available
                            if upper_bound is not None and isinstance(upper_bound, (int, float)):
                                try:
                                    self.metric_manager.log_metric(metric_run_id, f"{metric_name}_upper_bound", float(upper_bound), step=step)
                                except Exception as e:
                                    self.logger.debug(f"Failed to log final metric {metric_name}_upper_bound to MetricLogger: {e}")

                        try:
                            self.metric_manager.end_run(metric_run_id)
                        except Exception as e:
                            self.logger.debug(f"Failed to end MetricLogger run {metric_run_id}: {e}")
                except Exception as e:
                    self.logger.debug(f"Failed to log final metrics to MetricLogger for pipeline {pipeline_id}: {e}")

            self.logger.info(f"Pipeline {pipeline_id} ({pipeline_name}) completed successfully")

        if progress_display:
            progress_display.stop()
        return final_results

    def run_multi_pipeline_inference(
        self,
        experiment_id: int,
        config_group: RFGridSearch | RFRandomSearch,
        dataset,
        num_shards: int,
        seed: int = 42,
        num_actors: int = None,
        num_gpus: int = None,
        num_cpus: int = None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Run multi-pipeline inference with fair round-robin scheduling.

        This orchestrates multiple inference pipelines processing shards in a time-sharing manner.
        Each pipeline is scheduled fairly using generation-based round-robin scheduling.

        Args:
            experiment_id: Experiment ID (created in experiment.py)
            config_group: Grid search or random search configuration group
            dataset: Dataset to process
            num_shards: Number of shards to split the dataset into
            seed: Random seed for reproducibility (default: 42)
            num_actors: Total number of actors
            num_gpus: Total number of GPUs available
            num_cpus: Total number of CPUs available

        Returns:
            Dict mapping pipeline_id to (aggregated_results, cumulative_metrics) tuple
        """
        # Initialize database
        db = RFDatabase()

        # PHASE 1: Shard the dataset
        shards = self.dataloader.get_shards_from_data(dataset, num_shards)
        shard_sizes = [len(shard) for shard in shards]
        self.logger.info(f"Dataset sharded into {num_shards} shard(s). Shard sizes: {shard_sizes}")

        config_leaves = get_runs(config_group, seed)

        # PHASE 2: Receive pipeline configurations from user
        self.logger.info(f"Received {len(config_leaves)} pipeline configuration(s)")

        # PHASE 3: Setup context generators (collect unique, check DB, build if needed)
        self._setup_context_generators(config_leaves, db)

        # PHASE 4: Create query processing actors (shared pool)
        # Actors are created without any pipeline or context information
        # They will receive pipeline-specific context when scheduled
        query_actors = []
        gpus_per_actor = num_gpus // num_actors if num_actors > 0 else 0
        cpus_per_actor = num_cpus // num_actors if num_actors > 0 else 1

        for i in range(num_actors):
            actor = QueryProcessingActor.options(num_gpus=gpus_per_actor, num_cpus=cpus_per_actor).remote(
                experiment_name=self.experiment_name,
                experiment_path=self.experiment_path,
                actor_id=i,
            )
            query_actors.append(actor)

        self.logger.info(f"Created {num_actors} query processing actors (generic pool)")

        # PHASE 5: Register pipelines in database
        pipeline_ids, pipeline_id_to_config = self._register_pipelines(config_leaves, db)

        # PHASE 6: Initialize PipelineScheduler
        scheduler = PipelineScheduler(
            pipeline_ids=pipeline_ids,
            num_actors=num_actors,
            num_shards=num_shards,
        )
        self.logger.info(
            f"Initialized scheduler with {len(pipeline_ids)} pipelines, {num_actors} actors, {num_shards} shards"
        )

        # Set up aggregators for each pipeline
        pipeline_aggregators = {}
        pipeline_results = {}  # {pipeline_id: {"results": {}, "metrics": {}}}
        total_dataset_size = len(dataset)  # Store for MetricLogger percentage calculation

        for pipeline_id in pipeline_ids:
            aggregator = Aggregator()
            pipeline_config = pipeline_id_to_config[pipeline_id]
            if hasattr(pipeline_config, "online_strategy"):
                aggregator.set_online_strategy(**pipeline_config.online_strategy)
            aggregator.set_total_population_size(total_dataset_size)
            pipeline_aggregators[pipeline_id] = aggregator
            pipeline_results[pipeline_id] = {"results": {}, "metrics": {}, "start_time": None}

        # Initialize progress display table
        pipeline_info = []
        pipeline_configs = [pipeline_id_to_config[pipeline_id] for pipeline_id in pipeline_ids]
        for pipeline_id, pipeline_config in zip(pipeline_ids, pipeline_configs, strict=False):
            # Initialize all variables to None
            model_name = "Unknown"
            search_type = None
            rag_k = None
            top_n = None
            chunk_size = None
            chunk_overlap = None
            sampling_params = None
            prompt_manager_k = None
            model_config = None

            # Extract model name from config
            pipeline = pipeline_config["pipeline"]
            if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
                if "model" in pipeline.model_config:
                    model_name = pipeline.model_config["model"]
                # Extract full model config (excluding the model name)
                model_config_copy = pipeline.model_config.copy()
                model_config_copy.pop("model", None)
                if model_config_copy:  # Only assign if there are other configs
                    model_config = model_config_copy

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

            # Extract sampling params
            if hasattr(pipeline, "sampling_params") and pipeline.sampling_params is not None:
                sampling_params = pipeline._user_params.get("sampling_params", None)

            # Extract prompt_manager fields
            if hasattr(pipeline, "prompt_manager") and pipeline.prompt_manager is not None:
                prompt_manager_k = getattr(pipeline.prompt_manager, "k", None)

            # Build pipeline info dict with all fields (only include non-None values)
            pipeline_info_dict = {
                "pipeline_id": pipeline_id,
                "pipeline_config": pipeline_config,
                "model_name": model_name,
            }

            # Add optional fields only if they're not None
            if search_type is not None:
                pipeline_info_dict["search_type"] = search_type
            if rag_k is not None:
                pipeline_info_dict["rag_k"] = rag_k
            if top_n is not None:
                pipeline_info_dict["top_n"] = top_n
            if chunk_size is not None:
                pipeline_info_dict["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                pipeline_info_dict["chunk_overlap"] = chunk_overlap
            if sampling_params is not None:
                pipeline_info_dict["sampling_params"] = sampling_params
            if prompt_manager_k is not None:
                pipeline_info_dict["prompt_manager_k"] = prompt_manager_k
            if model_config is not None:
                pipeline_info_dict["model_config"] = model_config

            pipeline_info.append(pipeline_info_dict)

        progress_display = PipelineProgressDisplay(pipeline_info, num_shards)

        rate_limiter_actor = None
        pipeline_to_rate_limiter = {}
        has_openai_pipeline = False
        model_rate_limits = {}
        max_completion_tokens_by_model = {}
        pipeline_to_max_completion_tokens = {}

        for pipeline_id, pipeline_config in pipeline_id_to_config.items():
            from rapidfireai.automl import RFOpenAIAPIModelConfig
            pipeline = pipeline_config["pipeline"]
            if hasattr(pipeline, "model_config") and isinstance(pipeline, RFOpenAIAPIModelConfig):
                has_openai_pipeline = True
                model_config = pipeline.model_config
                model_name = model_config.get("model", "gpt-3.5-turbo")

                if pipeline.rpm_limit is None or pipeline.tpm_limit is None:
                    raise ValueError(
                        f"OpenAI pipeline {pipeline_id} (model: {model_name}) is missing rate limits. "
                        f"Please provide rpm_limit and tpm_limit to RFOpenAIAPIModelConfig."
                    )

                if model_name not in model_rate_limits:
                    model_rate_limits[model_name] = {
                        "rpm": pipeline.rpm_limit,
                        "tpm": pipeline.tpm_limit,
                    }
                    max_completion_tokens_by_model[model_name] = model_config.get("max_completion_tokens", 150)
                elif (model_rate_limits[model_name]["rpm"] != pipeline.rpm_limit or
                      model_rate_limits[model_name]["tpm"] != pipeline.tpm_limit):
                    self.logger.warning(
                        f"Model {model_name} has inconsistent rate limits across pipelines. "
                        f"Using first encountered values: {model_rate_limits[model_name]}"
                    )

                pipeline_to_max_completion_tokens[pipeline_id] = model_config.get(
                    "max_completion_tokens",
                    max_completion_tokens_by_model.get(model_name, 150)
                )
                pipeline_to_rate_limiter[pipeline_id] = None

        if has_openai_pipeline:
            from rapidfireai.evals.actors.rate_limiter_actor import RateLimiterActor

            max_max_tokens = max(max_completion_tokens_by_model.values()) if max_completion_tokens_by_model else 150

            rate_limiter_actor = RateLimiterActor.remote(
                model_rate_limits=model_rate_limits,
                max_completion_tokens=max_max_tokens,
                limit_safety_ratio=0.90,
                minimum_wait_time=1.0,
                experiment_name=self.experiment_name,
                experiment_path=self.experiment_path,
            )

            for pipeline_id in pipeline_to_rate_limiter:
                pipeline_to_rate_limiter[pipeline_id] = rate_limiter_actor
                if pipeline_id not in pipeline_to_max_completion_tokens:
                    pipeline_config = pipeline_id_to_config[pipeline_id]
                    pipeline = pipeline_config["pipeline"]
                    model_name = pipeline.model_config.get("model", "gpt-3.5-turbo")
                    pipeline_to_max_completion_tokens[pipeline_id] = max_completion_tokens_by_model.get(
                        model_name,
                        pipeline.model_config.get("max_completion_tokens", 150)
                    )

            limits_summary = ", ".join([
                f"{model}: {limits['rpm']} RPM, {limits['tpm']} TPM"
                for model, limits in model_rate_limits.items()
            ])
            self.logger.info(
                f"Created experiment-wide rate limiter actor for {len(pipeline_to_rate_limiter)} OpenAI pipeline(s) "
                f"with per-model limits ({limits_summary})"
            )

        # PHASE 7: Main scheduling loop
        self.logger.info("Starting multi-pipeline inference scheduling...")

        # Track active tasks: {actor_id: {"futures": [...], "pipeline_id": int, ...}}
        active_tasks = {}

        # Track start time for each pipeline (for throughput calculation)
        pipeline_start_times = {}

        # Start the progress display
        progress_display.start()

        loop_iteration = 0
        while True:
            loop_iteration += 1

            # Check for completed tasks
            completed_actor_ids = []
            for actor_id, task_info in list(active_tasks.items()):
                futures = task_info["futures"]
                pipeline_id = task_info["pipeline_id"]
                shard_id = task_info["shard_id"]
                task_id = task_info["task_id"]

                # Check if all batches are done
                ready_futures, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=0)

                if len(ready_futures) == len(futures):
                    # All batches completed
                    try:
                        # Aggregate results for this shard
                        aggregator = pipeline_aggregators[pipeline_id]
                        pipeline_config = pipeline_id_to_config[pipeline_id]
                        pipeline = pipeline_config["pipeline"]
                        shard_results, shard_metrics = aggregator.aggregate_with_progress(
                            futures=ready_futures,
                            accumulate_metrics_fn=pipeline_config["accumulate_metrics_fn"],
                        )

                        # Merge into pipeline's overall results
                        for key in shard_results:
                            if key in pipeline_results[pipeline_id]["results"]:
                                pipeline_results[pipeline_id]["results"][key].extend(shard_results[key])
                            else:
                                pipeline_results[pipeline_id]["results"][key] = shard_results[key].copy()

                        for key in shard_metrics:
                            if key in pipeline_results[pipeline_id]["metrics"]:
                                pipeline_results[pipeline_id]["metrics"][key].extend(shard_metrics[key])
                            else:
                                pipeline_results[pipeline_id]["metrics"][key] = shard_metrics[key].copy()

                        # Update database
                        end_time = time.time()
                        duration = end_time - task_info["start_time"]

                        db.set_actor_task_end_time(task_id, end_time, duration)
                        db.set_actor_task_status(task_id, TaskStatus.COMPLETED)

                        # Update pipeline progress
                        shards_completed = shard_id + 1
                        samples_processed = shards_completed * len(shards[0])  # Approximate
                        db.set_pipeline_progress(pipeline_id, shard_id + 1, shards_completed, samples_processed)

                        # Check if pipeline completed all shards
                        if shards_completed >= num_shards:
                            # Mark as completed (metrics will be finalized in Phase 8)
                            db.set_pipeline_status(pipeline_id, PipelineStatus.COMPLETED)
                            progress_display.update_pipeline(pipeline_id, status="COMPLETED")
                            self.logger.info(
                                f"Pipeline {pipeline_id} completed all {num_shards} shards"
                            )

                        # Compute current metrics with confidence intervals
                        confidence_value = None
                        display_metrics = {}

                        if pipeline_config["accumulate_metrics_fn"] and aggregator.online_strategy:
                            # Accumulate metrics from all completed shards
                            try:
                                cumulative_metrics = pipeline_config["accumulate_metrics_fn"](pipeline_results[pipeline_id]["metrics"])
                                # Add confidence interval information
                                metrics_with_ci = aggregator.online_strategy.add_confidence_interval_info(
                                    cumulative_metrics, samples_processed
                                )

                                # Extract all metrics for display with full CI info
                                for metric_name, metric_data in metrics_with_ci.items():
                                    if isinstance(metric_data, dict):
                                        # Include full metric data with CI info (value, lower_bound, upper_bound, margin_of_error)
                                        display_metrics[metric_name] = {
                                            "value": metric_data.get("value", 0),
                                            "lower_bound": metric_data.get("lower_bound"),
                                            "upper_bound": metric_data.get("upper_bound"),
                                            "margin_of_error": metric_data.get("margin_of_error"),
                                            "is_algebraic": metric_data.get("is_algebraic", False),
                                        }
                                        # For the first algebraic metric, use its margin of error as confidence display
                                        if confidence_value is None and metric_data.get("is_algebraic", False):
                                            if "margin_of_error" in metric_data:
                                                confidence_value = metric_data["margin_of_error"]
                                    else:
                                        # Simple metric format
                                        display_metrics[metric_name] = {"value": metric_data}

                                # Calculate throughput (samples/second)
                                elapsed_time = time.time() - pipeline_start_times.get(pipeline_id, time.time())
                                if elapsed_time > 0:
                                    throughput = samples_processed / elapsed_time
                                    display_metrics["Throughput"] = {"value": throughput}

                                pipeline = db.get_pipeline(pipeline_id)
                                metric_run_id = pipeline.get("metric_run_id") if pipeline else None

                                if self.metric_manager and metric_run_id:
                                    try:
                                        actual_samples_processed = pipeline.get("total_samples_processed", samples_processed)
                                        percentage_processed = (actual_samples_processed / total_dataset_size * 100) if total_dataset_size > 0 else 0
                                        step = int(percentage_processed)

                                        for metric_name, metric_data in metrics_with_ci.items():
                                            if metric_name in ["run_id", "model_name", "Samples Processed", "Processing Time", "Samples Per Second"]:
                                                continue

                                            if isinstance(metric_data, dict):
                                                metric_value = metric_data.get("value", 0)
                                                lower_bound = metric_data.get("lower_bound")
                                                upper_bound = metric_data.get("upper_bound")
                                            else:
                                                metric_value = metric_data
                                                lower_bound = None
                                                upper_bound = None

                                            # Log main metric value
                                            if isinstance(metric_value, (int, float)):
                                                try:
                                                    self.metric_manager.log_metric(metric_run_id, metric_name, float(metric_value), step=step)
                                                except Exception as e:
                                                    self.logger.debug(f"Failed to log metric {metric_name} to MetricLogger: {e}")

                                            # Log lower_bound if available
                                            if lower_bound is not None and isinstance(lower_bound, (int, float)):
                                                try:
                                                    self.metric_manager.log_metric(metric_run_id, f"{metric_name}_lower_bound", float(lower_bound), step=step)
                                                except Exception as e:
                                                    self.logger.debug(f"Failed to log metric {metric_name}_lower_bound to MetricLogger: {e}")

                                            # Log upper_bound if available
                                            if upper_bound is not None and isinstance(upper_bound, (int, float)):
                                                try:
                                                    self.metric_manager.log_metric(metric_run_id, f"{metric_name}_upper_bound", float(upper_bound), step=step)
                                                except Exception as e:
                                                    self.logger.debug(f"Failed to log metric {metric_name}_upper_bound to MetricLogger: {e}")

                                        if "Throughput" in display_metrics:
                                            throughput_value = display_metrics["Throughput"]["value"]
                                            if isinstance(throughput_value, (int, float)):
                                                try:
                                                    self.metric_manager.log_metric(metric_run_id, "Throughput", float(throughput_value), step=step)
                                                except Exception as e:
                                                    self.logger.debug(f"Failed to log Throughput to MetricLogger: {e}")
                                    except Exception as e:
                                        self.logger.debug(f"Failed to log metrics to MetricLogger: {e}")
                            except Exception as e:
                                self.logger.debug(f"Could not compute live metrics: {e}")

                        # Update progress display
                        progress_display.update_pipeline(
                            pipeline_id=pipeline_id,
                            shard=shards_completed,
                            confidence=confidence_value,
                            metrics=display_metrics,
                        )

                        self.logger.info(
                            f"Pipeline {pipeline_id} completed shard {shard_id} "
                            f"({task_info['batch_count']} batches, {duration:.2f}s)"
                        )

                        # Mark for cleanup
                        completed_actor_ids.append(actor_id)

                    except Exception as e:
                        # Task failed - mark pipeline as FAILED but continue with other pipelines
                        error_msg = str(e)
                        self.logger.exception(f"Pipeline {pipeline_id} failed on shard {shard_id}")

                        # Update database
                        db.set_actor_task_status(task_id, TaskStatus.FAILED)
                        db.set_actor_task_error(task_id, error_msg)
                        db.set_pipeline_status(pipeline_id, PipelineStatus.FAILED)
                        db.set_pipeline_error(pipeline_id, error_msg)

                        # Display error in notebook (but don't halt the experiment)
                        pipeline_config = pipeline_id_to_config.get(pipeline_id)
                        pipeline_name = pipeline_config.get("pipeline_name", f"Pipeline {pipeline_id}") if pipeline_config else f"Pipeline {pipeline_id}"
                        error_display = (
                            f"\n{'='*80}\n"
                            f"⚠️  Run {pipeline_id} ({pipeline_name}) FAILED\n"
                            f"{'='*80}\n"
                            f"Shard: {shard_id + 1}/{num_shards}\n"
                            f"Error: {error_msg}\n"
                            f"{'='*80}\n"
                            f"This run has been marked as FAILED. The experiment will continue with other runs.\n"
                        )
                        print(error_display)

                        # Update progress display
                        progress_display.update_pipeline(pipeline_id, status="FAILED")

                        # Remove pipeline from scheduler (no more tasks will be scheduled for it)
                        scheduler.remove_pipeline(pipeline_id)

                        # Mark for cleanup - actor is now free to process other pipelines
                        completed_actor_ids.append(actor_id)

            # Remove completed tasks and update scheduler
            for actor_id in completed_actor_ids:
                del active_tasks[actor_id]
                scheduler.set_completed_task(actor_id)

            # Check for interactive control requests (stop/resume/delete/clone)
            self.ic_handler.check_and_process_requests(
                scheduler=scheduler,
                db=db,
                num_shards=num_shards,
                dataset=dataset,
                pipeline_aggregators=pipeline_aggregators,
                pipeline_results=pipeline_results,
                pipeline_id_to_config=pipeline_id_to_config,
                pipeline_to_rate_limiter=pipeline_to_rate_limiter,
                pipeline_to_max_completion_tokens=pipeline_to_max_completion_tokens,
                # online_strategy_kwargs=online_strategy_kwargs,
                progress_display=progress_display,
            )

            # Get next schedule
            schedule = scheduler.schedule()

            # Check termination
            if schedule["pipeline_id"] is None:
                self.logger.info("All pipelines completed all shards!")
                break

            # Check if all actors busy
            if schedule["pipeline_id"] == -1:
                if loop_iteration % 10 == 0:  # Log occasionally
                    status = scheduler.get_status()
                    self.logger.debug(
                        f"All actors busy. Active: {status['active_pipelines']}, "
                        f"Busy actors: {status['busy_actors']}, "
                        f"Gen: {status['current_generation']}"
                    )
                time.sleep(0.5)
                continue

            # Execute schedule
            pipeline_id = schedule["pipeline_id"]
            actor_id = schedule["actor_id"]
            shard_id = schedule["shard_id"]

            pipeline_config = pipeline_id_to_config[pipeline_id]
            pipeline = pipeline_config["pipeline"]
            pipeline_name = pipeline_config.get("pipeline_name", f"Pipeline {pipeline_id}")

            # Update pipeline status
            if pipeline_results[pipeline_id]["start_time"] is None:
                start_time = time.time()
                pipeline_results[pipeline_id]["start_time"] = start_time
                pipeline_start_times[pipeline_id] = start_time
                db.set_pipeline_status(pipeline_id, PipelineStatus.ONGOING)

            # Get shard data and split into batches
            batch_size = pipeline_config["batch_size"]#TODO: set default batch size
            shard_data = shards[shard_id]
            batches = self.dataloader.get_batches(shard_data, batch_size)

            self.logger.info(
                f"Scheduling pipeline {pipeline_id} ({pipeline_name}) on actor {actor_id} "
                f"for shard {shard_id} ({len(batches)} batches)"
            )

            # Create task in database
            task_id = db.create_actor_task(
                pipeline_id=pipeline_id,
                actor_id=actor_id,
                shard_id=shard_id,
                status=TaskStatus.SCHEDULED,
            )

            # Submit all batches to this specific actor
            # NOTE: For now, we submit to one actor sequentially
            # In future, we could parallelize batches across actors for each (pipeline, shard)
            actor = query_actors[actor_id]

            # Initialize actor for this pipeline
            # Get context_generator_ref for this pipeline
            context_generator_ref = None
            pipeline_config = pipeline_id_to_config[pipeline_id]
            pipeline = pipeline_config["pipeline"]
            has_rag_attr = hasattr(pipeline, "rag")
            rag_spec = getattr(pipeline, "rag", None) if has_rag_attr else None
            prompt_manager = getattr(pipeline, "prompt_manager", None)

            # Check if pipeline has RAG or prompt_manager to look up context
            if rag_spec or prompt_manager:
                # Get RAG hash if present
                rag_hash = rag_spec.get_hash() if rag_spec else None

                # Get prompt_manager hash if present
                prompt_hash = prompt_manager.get_hash() if prompt_manager else None

                # Create combined context hash (matches logic in _collect_unique_contexts)
                if rag_hash and prompt_hash:
                    context_hash = hashlib.sha256(f"{rag_hash}:{prompt_hash}".encode()).hexdigest()
                elif rag_hash:
                    context_hash = rag_hash
                elif prompt_hash:
                    context_hash = prompt_hash
                else:
                    context_hash = None

                if context_hash and context_hash in self._context_cache:
                    _, context_generator_ref = self._context_cache[context_hash]

            engine_kwargs = pipeline.get_engine_kwargs()

            if pipeline_id in pipeline_to_rate_limiter:
                rate_limiter = pipeline_to_rate_limiter[pipeline_id]
                if rate_limiter is None:
                    raise ValueError(
                        f"Rate limiter actor is None for OpenAI pipeline {pipeline_id}. "
                        f"This should not happen - the rate limiter should be initialized "
                        f"for all OpenAI pipelines."
                    )
                engine_kwargs["rate_limiter_actor"] = rate_limiter
                engine_kwargs["max_completion_tokens"] = pipeline_to_max_completion_tokens[pipeline_id]

            ray.get(
                actor.initialize_for_pipeline.remote(
                    engine_class=pipeline.get_engine_class(),
                    engine_kwargs=engine_kwargs,
                    context_generator_ref=context_generator_ref,
                )
            )

            self.logger.debug(f"Initialized actor {actor_id} for pipeline {pipeline_id} ({pipeline_name})")

            futures = []
            preprocess_fn = pipeline_config.get("preprocess_fn")
            postprocess_fn = pipeline_config.get("postprocess_fn")
            compute_metrics_fn = pipeline_config.get("compute_metrics_fn")
            accumulate_metrics_fn = pipeline_config.get("accumulate_metrics_fn")
            for batch in batches:
                future = actor.process_batch.remote(
                    batch,
                    preprocess_fn=preprocess_fn,
                    postprocess_fn=postprocess_fn,
                    compute_metrics_fn=compute_metrics_fn if accumulate_metrics_fn else None,
                )
                futures.append(future)

            # Track task
            task_start_time = time.time()
            active_tasks[actor_id] = {
                "futures": futures,
                "pipeline_id": pipeline_id,
                "shard_id": shard_id,
                "task_id": task_id,
                "batch_count": len(batches),
                "start_time": task_start_time,
            }

            # Update task status to in-progress
            db.set_actor_task_start_time(task_id, task_start_time)
            db.set_actor_task_status(task_id, TaskStatus.IN_PROGRESS)
            db.set_pipeline_current_shard(pipeline_id, shard_id)

        # PHASE 8: Compute final metrics for each pipeline
        pipeline_id_to_info = {}
        for info_dict in pipeline_info:
            pipeline_id = info_dict["pipeline_id"]
            info_copy = {k: v for k, v in info_dict.items() if k not in ["pipeline_config", "pipeline_id"]}
            pipeline_id_to_info[pipeline_id] = info_copy

        final_results = self._compute_final_metrics_for_pipelines(
            pipeline_ids,
            pipeline_id_to_config,
            pipeline_aggregators,
            pipeline_results,
            db,
            progress_display,
            pipeline_id_to_info,
            total_dataset_size=total_dataset_size,
        )


        # Cleanup actors
        for actor in query_actors:
            ray.kill(actor)

        return final_results
