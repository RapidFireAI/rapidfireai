import json
import time
from collections.abc import Callable
from typing import Any

import ray

from rapidfireai.evals.actors.doc_actor import DocProcessingActor
from rapidfireai.evals.actors.inference_engines import InferenceEngine
from rapidfireai.evals.actors.query_actor import QueryProcessingActor
from rapidfireai.evals.data.dataset import DataLoader
from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.metrics.aggregator import Aggregator
from rapidfireai.evals.rag.context_generator import ContextGenerator
from rapidfireai.evals.scheduling.interactive_control import InteractiveControlHandler
from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler
from rapidfireai.evals.scheduling.scheduler import Scheduler
from rapidfireai.evals.utils.config import ModelConfig, VLLMModelConfig
from rapidfireai.evals.utils.constants import (
    NUM_CPUS_PER_DOC_ACTOR,
    NUM_QUERY_PROCESSING_ACTORS,
    ContextStatus,
    ExperimentStatus,
    PipelineStatus,
    TaskStatus,
)
from rf_infrapidfireai.inferrno.utils.logger import RFLogger
from rapidfireai.evals.utils.progress_display import ContextBuildingDisplay, PipelineProgressDisplay


class Controller:
    """
    Controller for orchestrating distributed inference pipeline.

    Manages data loading, scheduling, aggregation, and actor creation.
    Handles all the complexity of RAG initialization, actor creation, and batch processing.
    """

    def __init__(
        self,
        num_actors: int,
        num_gpus: int,
        num_cpus: int,
        experiment_name: str,
        experiment_path: str,
        openai_rpm_limit: int | None = None,
        openai_tpm_limit: int | None = None,
        openai_max_completion_tokens: int = 150,
    ):
        """
        Initialize the controller.

        Args:
            num_actors: Total number of actors
            num_gpus: Total number of GPUs available
            num_cpus: Total number of CPUs available
            experiment_name: Name of the experiment
            experiment_path: Path to experiment logs/artifacts
            openai_rpm_limit: OpenAI API requests per minute limit (experiment-wide)
            openai_tpm_limit: OpenAI API tokens per minute limit (experiment-wide)
            openai_max_completion_tokens: Maximum completion tokens per OpenAI request
        """
        self.aggregator = Aggregator()
        self.dataloader = DataLoader()
        self.scheduler = Scheduler(strategy="round_robin")
        self.num_actors = num_actors
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path

        # OpenAI rate limiting (experiment-wide)
        self.openai_rpm_limit = openai_rpm_limit
        self.openai_tpm_limit = openai_tpm_limit
        self.openai_max_completion_tokens = openai_max_completion_tokens

        # Initialize logger
        logging_manager = RFLogger(experiment_name=self.experiment_name, experiment_path=self.experiment_path)
        self.logger = logging_manager.get_logger("Controller")

        # Cache for context generators (persists only during Controller lifetime)
        # Maps context_hash -> (context_id, components_ref, context_generator)
        self._context_cache: dict[str, tuple[int, ray.ObjectRef, Any]] = {}

        # Initialize interactive control handler
        self.ic_handler = InteractiveControlHandler(
            experiment_name=self.experiment_name,
            experiment_path=self.experiment_path,
            context_cache=self._context_cache,
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

    @staticmethod
    def _collect_unique_context_generators(
        pipeline_configs: list[tuple[str, ModelConfig]],
    ) -> dict[str, ContextGenerator]:
        """
        Collect unique context generators from pipeline configs.

        Multiple pipelines may share the same context generator. This method
        deduplicates them by hash to avoid building the same RAG components multiple times.

        Args:
            pipeline_configs: List of (pipeline_name, model_config) tuples

        Returns:
            Dictionary mapping context_hash -> ContextGenerator for unique contexts
        """
        unique_contexts = {}

        for _, model_config in pipeline_configs:
            # Check if pipeline has a context generator
            if not hasattr(model_config, "context_generator") or not model_config.context_generator:
                continue

            context_generator = model_config.context_generator

            # Skip if no RAG spec (prompt-only context)
            if not context_generator.rag_spec:
                continue

            # Get hash and deduplicate
            context_hash = context_generator.get_hash()
            if context_hash not in unique_contexts:
                unique_contexts[context_hash] = context_generator

        return unique_contexts

    def _setup_context_generators(
        self,
        pipeline_configs: list[tuple[str, ModelConfig]],
        db: RFDatabase,
    ) -> None:
        """
        Setup context generators: build if needed and cache in Ray object store.

        This method orchestrates the entire context generation lifecycle:
        1. Collect unique context generators from all pipeline configs
        2. Check controller cache for already-built contexts (current session)
        3. Build new contexts in parallel using DocProcessingActors
        4. Store all contexts in Ray object store for sharing
        5. Track metadata in database for analytics

        All built contexts are stored in self._context_cache.

        Args:
            pipeline_configs: List of (pipeline_name, model_config) tuples
            db: Database instance
        """
        # Step 1: Collect unique context generators
        unique_contexts = self._collect_unique_context_generators(pipeline_configs)

        if not unique_contexts:
            self.logger.info("No context generators found in pipeline configs")
            return

        self.logger.info(f"Found {len(unique_contexts)} unique context generator(s)")

        # Step 2: Identify contexts that need to be built
        contexts_to_build = []
        for context_hash, context_generator in unique_contexts.items():
            # Check if already built in this Controller session
            if context_hash in self._context_cache:
                self.logger.info(f"Context {context_hash[:8]}... already in cache (session), reusing")
                continue

            # Need to build new context
            self.logger.info(f"Will build new context {context_hash[:8]}...")

            # Create context record in DB for tracking/analytics
            context_id = db.create_context(
                context_hash=context_hash,
                rag_config_json=json.dumps(self._sanitize_for_json(context_generator.rag_spec)),
                prompt_config_json=json.dumps(self._sanitize_for_json(context_generator.prompt_manager)),
                status=ContextStatus.ONGOING,
            )
            db.set_context_start_time(context_id, time.time())

            contexts_to_build.append(
                {
                    "context_hash": context_hash,
                    "context_id": context_id,
                    "context_generator": context_generator,
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
                              context_generator, start_time
            db: Database instance for tracking build status
        """
        if not contexts_to_build:
            return

        num_contexts = len(contexts_to_build)
        self.logger.info(f"Creating {num_contexts} DocProcessingActor(s) for parallel processing")

        # Prepare context info for display (add enable_gpu flag)
        for context_info in contexts_to_build:
            rag_spec = context_info["context_generator"].rag_spec
            context_info["enable_gpu"] = rag_spec.enable_gpu_search if rag_spec else False

        # Initialize progress display for context building
        context_display = ContextBuildingDisplay(contexts_to_build)
        context_display.start()

        # Step 1: Create all DocProcessingActors and submit all build tasks
        actor_tasks = []
        for context_info in contexts_to_build:
            context_generator = context_info["context_generator"]
            rag_spec = context_generator.rag_spec
            prompt_manager = context_generator.prompt_manager

            if not rag_spec:
                continue

            # Allocate resources based on GPU needs:
            # - If GPU search enabled: 1 GPU + 2 CPUs
            # - If CPU only: 0 GPUs + 2 CPUs
            num_gpus_for_actor = 1 if rag_spec.enable_gpu_search else 0
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

                # Cache for session-level reuse (store context_generator object for cloning)
                context_generator = context_info["context_generator"]
                self._context_cache[context_hash] = (context_id, context_generator_ref, context_generator)

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
        pipeline_configs: list[tuple[str, VLLMModelConfig]],
        db: RFDatabase,
    ) -> tuple[list[int], dict[int, tuple[str, VLLMModelConfig]]]:
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

        for pipeline_name, model_config in pipeline_configs:
            # Determine context_id for this pipeline
            context_id = None
            if (
                hasattr(model_config, "context_generator")
                and model_config.context_generator
                and model_config.context_generator.rag_spec
            ):
                context_hash = model_config.context_generator.get_hash()
                if context_hash in self._context_cache:
                    context_id, _, _ = self._context_cache[context_hash]

            pipeline_id = db.create_pipeline(
                context_id=context_id,
                pipeline_name=pipeline_name,
                pipeline_type="vllm",
                model_config_json=json.dumps(model_config.model_config),
                sampling_params_json=json.dumps(model_config.sampling_params_to_dict()),
                status=PipelineStatus.NEW,
            )
            pipeline_ids.append(pipeline_id)
            pipeline_id_to_config[pipeline_id] = (pipeline_name, model_config)
            self.logger.info(f"Registered pipeline {pipeline_id}: {pipeline_name}")

        return pipeline_ids, pipeline_id_to_config

    def _compute_final_metrics_for_pipelines(
        self,
        pipeline_ids: list[int],
        pipeline_id_to_config: dict[int, tuple[str, VLLMModelConfig]],
        pipeline_aggregators: dict[int, Aggregator],
        pipeline_results: dict[int, dict],
        compute_metrics_fn: Callable,
        accumulate_metrics_fn: Callable,
        db: RFDatabase,
        progress_display=None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Compute final metrics for each pipeline and update database.

        Args:
            pipeline_ids: List of pipeline IDs
            pipeline_id_to_config: Mapping of pipeline_id to (name, config)
            pipeline_aggregators: Mapping of pipeline_id to Aggregator
            pipeline_results: Mapping of pipeline_id to results/metrics dict
            compute_metrics_fn: Metrics computation function
            accumulate_metrics_fn: Metrics accumulation function
            db: Database instance
            progress_display: Optional progress display to update

        Returns:
            Dict mapping pipeline_id to (aggregated_results, cumulative_metrics)
        """
        self.logger.info("Computing final metrics for all pipelines...")

        final_results = {}
        for pipeline_id in pipeline_ids:
            pipeline_name, _ = pipeline_id_to_config[pipeline_id]

            # Check pipeline status
            pipeline_status = db.get_pipeline(pipeline_id)["status"]

            # Skip pipelines that didn't complete successfully
            if pipeline_status != PipelineStatus.COMPLETED.value:
                if pipeline_status == PipelineStatus.FAILED.value:
                    self.logger.warning(f"Pipeline {pipeline_id} ({pipeline_name}) failed, skipping final metrics")
                else:
                    self.logger.info(
                        f"Pipeline {pipeline_id} ({pipeline_name}) has status {pipeline_status}, skipping final metrics"
                    )
                continue

            # Skip pipelines with no results (cloned but never processed)
            if not pipeline_results[pipeline_id]["results"]:
                self.logger.info(f"Pipeline {pipeline_id} ({pipeline_name}) has no results, skipping final metrics")
                continue

            aggregator = pipeline_aggregators[pipeline_id]
            start_time = pipeline_results[pipeline_id]["start_time"]
            end_time = time.time()

            cumulative_metrics = aggregator.compute_final_metrics(
                aggregated_results=pipeline_results[pipeline_id]["results"],
                aggregated_metrics=pipeline_results[pipeline_id]["metrics"],
                compute_metrics_fn=compute_metrics_fn,
                accumulate_metrics_fn=accumulate_metrics_fn,
                start_time=start_time,
                end_time=end_time,
            )

            final_results[pipeline_id] = (
                pipeline_results[pipeline_id]["results"],
                cumulative_metrics,
            )

            # Update pipeline status
            db.set_pipeline_status(pipeline_id, PipelineStatus.COMPLETED)
            if progress_display:
                progress_display.update_pipeline(pipeline_id, status="COMPLETED")
            self.logger.info(f"Pipeline {pipeline_id} ({pipeline_name}) completed successfully")

        if progress_display:
            progress_display.stop()
        return final_results

    def run_multi_pipeline_inference(
        self,
        experiment_id: int,
        pipeline_configs: list[tuple[str, ModelConfig]],
        dataset,
        batch_size: int,
        num_shards: int,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        compute_metrics_fn: Callable = None,
        accumulate_metrics_fn: Callable = None,
        online_strategy_kwargs: dict[str, Any] = None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Run multi-pipeline inference with fair round-robin scheduling.

        This orchestrates multiple inference pipelines processing shards in a time-sharing manner.
        Each pipeline is scheduled fairly using generation-based round-robin scheduling.

        Args:
            experiment_id: Experiment ID (created in experiment.py)
            pipeline_configs: List of (pipeline_name, model_config) tuples
            dataset: Dataset to process
            batch_size: Batch size for processing
            num_shards: Number of shards to split the dataset into
            preprocess_fn: Optional preprocessing function
            postprocess_fn: Optional postprocessing function
            compute_metrics_fn: Optional metrics computation function
            accumulate_metrics_fn: Optional metrics accumulation function
            online_strategy_kwargs: Optional online aggregation strategy parameters

        Returns:
            Dict mapping pipeline_id to (aggregated_results, cumulative_metrics) tuple
        """
        # Initialize database
        db = RFDatabase()

        # PHASE 1: Shard the dataset
        shards = self.dataloader.get_shards_from_data(dataset, num_shards)
        shard_sizes = [len(shard) for shard in shards]
        self.logger.info(f"Dataset sharded into {num_shards} shard(s). Shard sizes: {shard_sizes}")

        # PHASE 2: Receive pipeline configurations from user
        self.logger.info(f"Received {len(pipeline_configs)} pipeline configuration(s)")

        # PHASE 3: Setup context generators (collect unique, check DB, build if needed)
        self._setup_context_generators(pipeline_configs, db)

        # PHASE 4: Create query processing actors (shared pool)
        # Actors are created without any pipeline or context information
        # They will receive pipeline-specific context when scheduled
        query_actors = []
        gpus_per_actor = self.num_gpus // self.num_actors if self.num_actors > 0 else 0
        cpus_per_actor = self.num_cpus // self.num_actors if self.num_actors > 0 else 1

        for i in range(self.num_actors):
            actor = QueryProcessingActor.options(num_gpus=gpus_per_actor, num_cpus=cpus_per_actor).remote(
                experiment_name=self.experiment_name,
                experiment_path=self.experiment_path,
                actor_id=i,
            )
            query_actors.append(actor)

        self.logger.info(f"Created {self.num_actors} query processing actors (generic pool)")

        # PHASE 5: Register pipelines in database
        pipeline_ids, pipeline_id_to_config = self._register_pipelines(pipeline_configs, db)

        # PHASE 6: Initialize PipelineScheduler
        scheduler = PipelineScheduler(
            pipeline_ids=pipeline_ids,
            num_actors=self.num_actors,
            num_shards=num_shards,
        )
        self.logger.info(
            f"Initialized scheduler with {len(pipeline_ids)} pipelines, {self.num_actors} actors, {num_shards} shards"
        )

        # Set up aggregators for each pipeline
        pipeline_aggregators = {}
        pipeline_results = {}  # {pipeline_id: {"results": {}, "metrics": {}}}

        for pipeline_id in pipeline_ids:
            aggregator = Aggregator()
            if online_strategy_kwargs:
                aggregator.set_online_strategy(**online_strategy_kwargs)
            aggregator.set_total_population_size(len(dataset))
            pipeline_aggregators[pipeline_id] = aggregator
            pipeline_results[pipeline_id] = {"results": {}, "metrics": {}, "start_time": None}

        # Initialize progress display table
        pipeline_info = []
        for pipeline_id, (pipeline_name, model_config) in zip(pipeline_ids, pipeline_configs, strict=False):
            # Extract model name from config
            if hasattr(model_config, "model_config") and "model" in model_config.model_config:
                model_name = model_config.model_config["model"]
            else:
                model_name = "Unknown"
            pipeline_info.append((pipeline_id, pipeline_name, model_name))

        progress_display = PipelineProgressDisplay(pipeline_info, num_shards)

        # PHASE 6.5: Create single rate limiter actor for OpenAI pipelines (experiment-wide)
        rate_limiter_actor = None
        pipeline_to_rate_limiter = {}  # {pipeline_id: actor_handle}

        # Check if any pipeline uses OpenAI
        has_openai_pipeline = False
        openai_model_names = []
        for pipeline_id, (pipeline_name, model_config) in pipeline_id_to_config.items():
            from rapidfireai.evals.utils.config import OpenAIAPIModelConfig

            if isinstance(model_config, OpenAIAPIModelConfig):
                has_openai_pipeline = True
                model_name = model_config.model_config.get("model", "gpt-3.5-turbo")
                if model_name not in openai_model_names:
                    openai_model_names.append(model_name)
                # Map this pipeline to the shared rate limiter (will be created below)
                pipeline_to_rate_limiter[pipeline_id] = None  # Placeholder, will be set after creation

        # Create single rate limiter actor for all OpenAI pipelines if needed
        if has_openai_pipeline:
            if self.openai_rpm_limit is None or self.openai_tpm_limit is None:
                raise ValueError(
                    "OpenAI pipelines detected but rate limits not configured. "
                    "Please provide openai_rpm_limit and openai_tpm_limit to Experiment constructor."
                )

            from rapidfireai.evals.actors.rate_limiter_actor import RateLimiterActor

            rate_limiter_actor = RateLimiterActor.remote(
                model_names=openai_model_names,
                rpm_limit=self.openai_rpm_limit,
                tpm_limit=self.openai_tpm_limit,
                max_completion_tokens=self.openai_max_completion_tokens,
                limit_safety_ratio=0.90,  # Use 90% of limit to avoid 429 errors
                minimum_wait_time=1.0,  # Minimum 1s wait when rate limited
            )

            # Update all OpenAI pipeline mappings
            for pipeline_id in pipeline_to_rate_limiter:
                pipeline_to_rate_limiter[pipeline_id] = rate_limiter_actor

            self.logger.info(
                f"Created experiment-wide rate limiter actor for {len(pipeline_to_rate_limiter)} OpenAI pipeline(s) "
                f"(RPM: {self.openai_rpm_limit}, TPM: {self.openai_tpm_limit}, max_tokens: {self.openai_max_completion_tokens})"
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
                        shard_results, shard_metrics = aggregator.aggregate_with_progress(
                            futures=ready_futures,
                            accumulate_metrics_fn=accumulate_metrics_fn,
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
                                f"Pipeline {pipeline_id} ({pipeline_name}) completed all {num_shards} shards"
                            )

                        # Compute current metrics with confidence intervals
                        confidence_value = None
                        display_metrics = {}

                        if accumulate_metrics_fn and aggregator.online_strategy:
                            # Accumulate metrics from all completed shards
                            try:
                                cumulative_metrics = accumulate_metrics_fn(pipeline_results[pipeline_id]["metrics"])
                                # Add confidence interval information
                                metrics_with_ci = aggregator.online_strategy.add_confidence_interval_info(
                                    cumulative_metrics, samples_processed
                                )

                                # Extract metrics for display
                                if "Accuracy" in metrics_with_ci:
                                    acc_data = metrics_with_ci["Accuracy"]
                                    if isinstance(acc_data, dict):
                                        display_metrics["Accuracy"] = {"value": acc_data.get("value", 0)}
                                        # Use margin of error as confidence display
                                        if "margin_of_error" in acc_data:
                                            confidence_value = acc_data["margin_of_error"]

                                # Calculate throughput (samples/second)
                                elapsed_time = time.time() - pipeline_start_times.get(pipeline_id, time.time())
                                if elapsed_time > 0:
                                    throughput = samples_processed / elapsed_time
                                    display_metrics["Throughput"] = {"value": throughput}
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
                        pipeline_name = pipeline_id_to_config.get(pipeline_id, (f"Pipeline {pipeline_id}", None))[0]
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
                online_strategy_kwargs=online_strategy_kwargs,
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

            pipeline_name, model_config = pipeline_id_to_config[pipeline_id]

            # Update pipeline status
            if pipeline_results[pipeline_id]["start_time"] is None:
                start_time = time.time()
                pipeline_results[pipeline_id]["start_time"] = start_time
                pipeline_start_times[pipeline_id] = start_time
                db.set_pipeline_status(pipeline_id, PipelineStatus.ONGOING)

            # Get shard data and split into batches
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
            if (
                hasattr(model_config, "context_generator")
                and model_config.context_generator
                and model_config.context_generator.rag_spec
            ):
                context_hash = model_config.context_generator.get_hash()
                if context_hash in self._context_cache:
                    _, context_generator_ref, _ = self._context_cache[context_hash]

            # Configure the actor for this specific pipeline
            engine_kwargs = model_config.get_engine_kwargs()

            # Inject rate limiter actor and max_completion_tokens for OpenAI pipelines
            if pipeline_id in pipeline_to_rate_limiter:
                engine_kwargs["rate_limiter_actor"] = pipeline_to_rate_limiter[pipeline_id]
                engine_kwargs["max_completion_tokens"] = self.openai_max_completion_tokens

            ray.get(
                actor.initialize_for_pipeline.remote(
                    engine_class=model_config.get_engine_class(),
                    engine_kwargs=engine_kwargs,
                    context_generator_ref=context_generator_ref,
                )
            )

            self.logger.debug(f"Initialized actor {actor_id} for pipeline {pipeline_id} ({pipeline_name})")

            futures = []
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
        final_results = self._compute_final_metrics_for_pipelines(
            pipeline_ids,
            pipeline_id_to_config,
            pipeline_aggregators,
            pipeline_results,
            compute_metrics_fn,
            accumulate_metrics_fn,
            db,
            progress_display,
        )

        # Update experiment status
        db.set_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
        self.logger.info(f"Experiment {experiment_id} completed!")

        # Cleanup actors
        for actor in query_actors:
            ray.kill(actor)

        return final_results
