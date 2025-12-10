"""
Query processing actor for parallel inference.

This actor handles batch processing using pluggable inference engines
(VLLM or OpenAI). It receives pre-initialized RAG components from Ray's
object store and processes batches of queries through the configured engine.
"""

import hashlib
import os
from collections.abc import Callable
from typing import Any
import ray

from rapidfireai.utils.constants import RF_EXPERIMENT_PATH
from rapidfireai.evals.actors.inference_engines import InferenceEngine
from rapidfireai.evals.rag.rag_pipeline import LangChainRagSpec
from rapidfireai.evals.rag.prompt_manager import PromptManager
from rapidfireai.evals.utils.logger import RFLogger


@ray.remote
class QueryProcessingActor:
    """
    Unified query processing actor that uses different inference engines.

    This actor:
    - Receives pre-initialized RAG components from Ray object store (shared memory)
    - Uses a pluggable inference engine (VLLM or OpenAI) for generation
    - Processes batches through preprocessing → generation → postprocessing
    - Computes batch-level metrics
    """

    def __init__(
        self,
        experiment_name: str = "unknown",
        experiment_path: str = RF_EXPERIMENT_PATH,
        actor_id: int = 0,
    ):
        """
        Initialize a generic query processing actor.

        The actor is created without any pipeline-specific configuration.
        Configuration is injected later via initialize_for_pipeline() when
        the scheduler assigns this actor to a specific pipeline.

        Args:
            experiment_name: Name of the experiment
            experiment_path: Path to experiment logs/artifacts
            actor_id: Index of this actor (for logging and identification)
        """
        # AWS Fix: Initialize CUDA context early to prevent CUBLAS_STATUS_NOT_INITIALIZED
        # This must happen BEFORE any torch operations (including embedding/LLM model loading)
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
            try:
                import torch
                if torch.cuda.is_available():
                    # Force CUDA initialization by performing a simple operation
                    _ = torch.zeros(1, device='cuda')
                    torch.cuda.synchronize()
            except Exception:
                # Silently continue if CUDA initialization fails (will use CPU)
                pass

        # Initialize logger with actor ID
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        self.logger = logging_manager.get_logger(f"QueryProcessingActor-{actor_id}")
        self.actor_id = actor_id

        # Pipeline-specific components (initialized later)
        self.inference_engine = None
        self.rag_spec = None  # RAG specification with retriever, template, etc.
        self.prompt_manager = None  # Prompt manager for few-shot examples
        self.current_engine_config_hash = None  # Track currently loaded model

    def initialize_for_pipeline(
        self,
        engine_class: type[InferenceEngine],
        engine_kwargs: dict[str, Any],
        context_generator_ref: dict[str, Any] = None,
    ):
        """
        Configure this actor for a specific pipeline.

        This method is called by the scheduler when assigning this actor
        to process batches for a specific pipeline. It sets up the inference
        engine and context generator.

        Args:
            engine_class: The inference engine class to instantiate
            engine_kwargs: Kwargs for instantiating the inference engine
            context_generator_ref: RAG components dict (automatically dereferenced by Ray when passed as ObjectRef)

        Raises:
            RuntimeError: If any error occurs during initialization. The original exception
                         is converted to RuntimeError to ensure it can be properly serialized by Ray.
        """
        try:
            # Create a hash of the engine configuration to check if we need to reinitialize
            # For VLLM, only model_config requires reload - sampling_params can vary per request
            # Use repr() to handle non-serializable objects
            if engine_class.__name__ == "VLLMInferenceEngine":
                # Only hash model_config - sampling_params don't require model reload
                model_config = engine_kwargs.get("model_config", {})
                config_str = f"{engine_class.__name__}:{repr(sorted(model_config.items()))}"
            else:
                # For other engines, hash everything
                config_str = f"{engine_class.__name__}:{repr(sorted(engine_kwargs.items()))}"

            config_hash = hashlib.md5(config_str.encode()).hexdigest()

            # Only reinitialize if the engine config has changed
            if self.current_engine_config_hash != config_hash:
                # Clean up old engine if it exists
                if self.inference_engine is not None:
                    self.logger.info(f"Cleaning up old inference engine (hash: {self.current_engine_config_hash[:8]})")
                    try:
                        self.inference_engine.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Error during engine cleanup: {e}")

                    # Force garbage collection and GPU memory release
                    del self.inference_engine
                    import gc

                    gc.collect()

                    # For CUDA, explicitly empty cache
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            self.logger.info("GPU memory cache cleared")
                    except ImportError:
                        pass

                self.logger.info(f"Initializing new inference engine (config hash: {config_hash[:8]})")
                self.inference_engine = engine_class(**engine_kwargs)
                self.current_engine_config_hash = config_hash
            else:
                self.logger.info(f"Reusing existing inference engine (config hash: {config_hash[:8]})")

            # Recreate RAG spec and prompt manager from shared components (if provided)
            self.rag_spec = None
            self.prompt_manager = None
            if context_generator_ref is not None:
                # Ray automatically dereferences ObjectRefs passed to remote actors
                # So we receive the dict directly, not an ObjectRef

                # Check if RAG components are present (context might only have prompt_manager)
                has_rag_components = "faiss_index_bytes" in context_generator_ref

                # Recreate RAG spec if RAG components are present
                if has_rag_components:
                    # NOTE: Keep FAISS on CPU for query actors to avoid GPU memory conflicts
                    # The DocProcessingActor builds the index on GPU (fast), transfers to CPU,
                    # and shares it. Query actors use CPU FAISS which is still fast for retrieval
                    # and avoids memory management issues with multiple actors sharing GPUs.

                    self.logger.info("Using CPU-based FAISS for retrieval (avoids GPU memory conflicts)")

                    # Deserialize FAISS index to create an independent copy for this actor
                    # FAISS indices are not thread-safe, so each actor needs its own copy
                    import pickle
                    from langchain_community.vectorstores import FAISS

                    self.logger.info("Deserializing FAISS index for this actor...")
                    faiss_index = pickle.loads(context_generator_ref["faiss_index_bytes"])
                    docstore = pickle.loads(context_generator_ref["docstore_bytes"])
                    index_to_docstore_id = pickle.loads(context_generator_ref["index_to_docstore_id_bytes"])

                    # Recreate the embedding function
                    embedding_cls = context_generator_ref["embedding_cls"]
                    embedding_kwargs = context_generator_ref["embedding_kwargs"]
                    embedding_function = embedding_cls(**embedding_kwargs)
                    self.logger.info(f"Recreated embedding function: {embedding_cls.__name__}")

                    # Create a new FAISS vector store with the deserialized components
                    vector_store = FAISS(
                        embedding_function=embedding_function,
                        index=faiss_index,
                        docstore=docstore,
                        index_to_docstore_id=index_to_docstore_id
                    )
                    self.logger.info("Created independent FAISS vector store for this actor")

                    # Create the retriever
                    search_type = context_generator_ref["search_type"]
                    search_kwargs = context_generator_ref["search_kwargs"]
                    retriever = vector_store.as_retriever(
                        search_type=search_type,
                        search_kwargs=search_kwargs
                    )
                    self.logger.info(f"Recreated retriever with search_type={search_type}")

                    # Recreate RAG spec with query-time components
                    # We don't need document_loader or text_splitter for query-time operations,
                    # so we use None/placeholder values
                    self.rag_spec = LangChainRagSpec(
                        document_loader=None,  # Not needed for query-time
                        text_splitter=None,  # Not needed for query-time
                        embedding_cls=embedding_cls,
                        embedding_kwargs=embedding_kwargs,
                        retriever=retriever,
                        vector_store=vector_store,
                        search_type=search_type,
                        search_kwargs=search_kwargs,
                        reranker_cls=context_generator_ref.get("reranker_cls"),
                        reranker_kwargs=context_generator_ref.get("reranker_kwargs"),
                        enable_gpu_search=False,  # Query actors always use CPU
                        document_template=context_generator_ref.get("template"),
                    )
                    # Manually set the embedding and template since we bypassed normal initialization
                    self.rag_spec.embedding = embedding_function
                    self.rag_spec.template = context_generator_ref.get("template")
                    self.logger.info("Recreated RAG spec with retriever and template")

                # Set up PromptManager if provided (reinitialize after deserialization)
                # This can exist with or without RAG components
                prompt_manager = context_generator_ref.get("prompt_manager")
                if prompt_manager:
                    # Check if we need to recreate the embedding and fewshot generator
                    # (they are set to None during pickling to avoid unpicklable locks)
                    if not hasattr(prompt_manager, "fewshot_generator") or prompt_manager.fewshot_generator is None:
                        # Recreate embedding_function and fewshot_generator
                        prompt_manager.setup_examples()
                    self.prompt_manager = prompt_manager
                    self.logger.info("Recreated prompt manager")

        except Exception as e:
            # Convert any exception to RuntimeError to ensure it can be properly serialized by Ray.
            error_type = type(e).__name__
            error_message = str(e)

            # Log the original exception details for debugging
            self.logger.exception(f"Failed to initialize pipeline: {error_type}: {error_message}")

            # Convert to RuntimeError with descriptive message
            raise RuntimeError(
                f"Failed to initialize pipeline: {error_type}: {error_message}"
            ) from None  # Don't chain to avoid serialization issues

    def _has_gpu(self) -> bool:
        """
        Check if this actor has GPU allocated by Ray.

        Ray sets CUDA_VISIBLE_DEVICES to control GPU visibility.
        The first visible GPU is always accessed as device 0.

        Returns:
            True if GPU is available, False otherwise
        """
        return "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != ""

    def process_batch(
        self,
        batch_data: dict[str, list],
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        compute_metrics_fn: Callable = None,
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """
        Process a batch of data through the inference pipeline.

        Pipeline stages:
        1. Preprocessing: Build prompts with context/few-shot examples
        2. Generation: Run inference using the configured engine
        3. Postprocessing: Extract answers, format results
        4. Metrics: Compute batch-level metrics

        Args:
            batch_data: Dictionary containing batch data (e.g., questions, prompts)
                       Can also be a HuggingFace Dataset (will be converted to dict)
            preprocess_fn: Optional function to preprocess batch before inference
            postprocess_fn: Optional function to postprocess results
            compute_metrics_fn: Optional function to compute batch-level metrics

        Returns:
            Tuple of (processed_batch_data, batch_metrics)
        """
        try:
            # Convert HuggingFace Dataset to dictionary if needed
            if hasattr(batch_data, "to_dict"):
                batch_data = batch_data.to_dict()

            # Stage 1: Preprocess - build prompts with context/examples
            if preprocess_fn:
                batch_data = preprocess_fn(batch_data, self.rag_spec, self.prompt_manager)

            # Stage 2: Generate using the inference engine
            prompts = batch_data["prompts"]
            generated_texts = self.inference_engine.generate(prompts)
            batch_data["generated_text"] = generated_texts

            # Stage 3: Postprocess - extract answers, format results
            if postprocess_fn:
                batch_data = postprocess_fn(batch_data)

            # Stage 4: Compute metrics
            batch_metrics = {}
            if compute_metrics_fn:
                default_metrics = {
                    "Samples Processed": {
                        "value": len(batch_data["generated_text"]),
                        "is_algebraic": False,
                    }
                }
                batch_metrics = {**default_metrics, **compute_metrics_fn(batch_data)}
            return batch_data, batch_metrics

        except Exception as e:
            # Convert any exception to RuntimeError to ensure it can be properly serialized by Ray.
            # This is especially important for exceptions like APIStatusError from OpenAI that
            # require specific keyword arguments that can't be pickled.
            error_type = type(e).__name__
            error_message = str(e)

            # Log the original exception details for debugging
            self.logger.exception(f"Error processing batch in QueryProcessingActor: {error_type}: {error_message}")

            # Convert to RuntimeError with descriptive message
            raise RuntimeError(
                f"Error processing batch: {error_type}: {error_message}"
            ) from None  # Don't chain to avoid serialization issues

    def cleanup(self):
        """Clean up inference engine resources."""
        self.inference_engine.cleanup()


# Export for external use
__all__ = ["QueryProcessingActor"]
