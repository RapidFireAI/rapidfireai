"""
Document processing actor for building RAG components.

This actor runs once using all available resources for fast initialization
of embeddings and FAISS indexes. After building, components are placed in
Ray's object store for sharing across query processing actors.
"""

import os
from typing import Any

import faiss
import ray

from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_pinecone import PineconeVectorStore

from rapidfireai.evals.rag.prompt_manager import PromptManager
from rapidfireai.evals.rag.rag_pipeline import LangChainRagSpec
from rapidfireai.evals.utils.logger import RFLogger


@ray.remote
class DocProcessingActor:
    """
    Actor responsible for building RAG components (FAISS index, embeddings).

    This actor uses all available GPU/CPU resources for maximum speed during
    the one-time initialization phase. After building, components are placed
    in Ray's object store for sharing across query processing actors.
    """

    def __init__(self, experiment_name: str, experiment_path: str):
        """
        Initialize the document processing actor.

        Args:
            experiment_name: Name of the experiment
            experiment_path: Path to experiment logs/artifacts
        """
        # AWS Fix: Initialize CUDA context early to prevent CUBLAS_STATUS_NOT_INITIALIZED
        # This must happen BEFORE any torch operations (including embedding model loading)
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

        # Initialize logger
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        self.logger = logging_manager.get_logger("DocProcessingActor")

        # Multimodal summarizer engine state. The same actor is reused across
        # text -> image -> table modalities; the config-hash check in
        # initialize_summarizer lets us short-circuit when consecutive
        # modalities share an engine config. Note that cleanup_summarizer is
        # type-aware: stateless engines (APIInferenceEngine) are kept alive
        # between modalities so the hash check actually fires, while GPU-bound
        # engines (vLLM) are always torn down to free VRAM.
        self.summarizer_engine = None
        self.summarizer_engine_config_hash: str | None = None

        self.logger.info("DocProcessingActor initialized")

    def build_rag_components(
        self,
        rag_spec: LangChainRagSpec | None,
        prompt_manager: PromptManager | None = None,
        summarizer_rate_limiters: dict[str, Any] | None = None,
        summarizer_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Build RAG components and/or prompt manager and return them for sharing.

        This method performs the heavy lifting of:
        - Loading and splitting documents (if RAG spec provided)
        - Generating embeddings (if RAG spec provided)
        - Building FAISS index (on GPU if enabled, if RAG spec provided)
        - Transferring GPU index to CPU for serialization (if RAG spec provided)
        - Initializing prompt manager (if provided)

        Args:
            rag_spec: Optional RAG specification with document loader, embeddings config, etc.
                     Can be None for prompt-only pipelines.
            prompt_manager: Optional prompt manager for few-shot examples
            summarizer_rate_limiters: Optional ``{provider: RateLimiterActor}`` mapping
                used by API summarizer generators inside ``rag_spec.multimodal_processor``.
                Forwarded to ``rag_spec.build_pipeline``; ignored if no multimodal
                summarization is configured.
            summarizer_batch_size: Optional hard ceiling on the number of prompts
                sent per ``summarize_batch`` call inside multimodal summarization.
                Forwarded to ``rag_spec.build_pipeline``; ``None`` means "send
                each modality in a single call" (engine-internal batching only).

        Returns:
            Dictionary containing initialized components ready for sharing:
                - faiss_index_bytes, docstore_bytes, index_to_docstore_id_bytes: (if vector store was built)
                - embedding_cfg: Config dict with "class" + kwargs (used by query actors)
                - search_cfg: Config dict with "type" + kwargs (used by query actors)
                - reranker_cfg: Not included — passed pipeline-specifically by the controller
                - retriever: Retriever object (if provided)
                - template: Document formatting template
                - enable_gpu_search: Whether GPU FAISS was used during build
                - prompt_manager: Initialized prompt manager (if provided)

        Raises:
            RuntimeError: If any error occurs during RAG component building. The original exception
                         is converted to RuntimeError to ensure it can be properly serialized by Ray.
        """
        self.logger.info("DocProcessingActor: Starting context initialization...")

        try:
            # Serialize FAISS index when built; share flattened RAG config for query actors
            import pickle

            components = {}

            # Build RAG (embeddings, FAISS index) if RAG spec provided
            # If enable_gpu_search=True, this builds on GPU
            if rag_spec:
                self.logger.info("Building document index...")
                # Pass self as the summarizer_runner so multimodal summarization
                # (if configured on rag_spec) routes back to this actor's
                # initialize_summarizer / summarize_batch / cleanup_summarizer
                # methods. These are called as plain Python methods (not via
                # .remote()) since we're already inside the actor.
                rag_spec.build_pipeline(
                    summarizer_runner=self,
                    summarizer_rate_limiters=summarizer_rate_limiters,
                    summarizer_batch_size=summarizer_batch_size,
                )
                self.logger.info("Document index built successfully")

                # context_generator_ref contains only index/embedding data that is
                # shared across all pipelines using this context. Retrieval config
                # (search_cfg, reranker_cfg) is pipeline-specific and is passed
                # separately by the controller at pipeline-initialization time, so
                # that clone-modified pipelines can override it independently.
                components.update({
                    "rag_spec_exists": True,
                    "embedding_cls": rag_spec.embedding_cls,
                    "embedding_kwargs": rag_spec.embedding_kwargs,
                    "template": rag_spec.template if rag_spec.template is not None else None,
                    "enable_gpu_search": rag_spec.enable_gpu_search if rag_spec.enable_gpu_search is not None else False,
                    "artifact_storage_cfg": rag_spec.artifact_storage_cfg,
                })
                
                if rag_spec.vector_store is not None and isinstance(rag_spec.vector_store, FAISS):
                    if rag_spec.enable_gpu_search:
                        # Transfer GPU index to CPU for serialization (if GPU was used and we have a FAISS vector store)
                        self.logger.info("Transferring FAISS index from GPU to CPU for serialization...")
                        # Transfer the GPU index to CPU
                        cpu_index = faiss.index_gpu_to_cpu(rag_spec.vector_store.index)
                        # Replace GPU index with CPU version
                        rag_spec.vector_store.index = cpu_index
                        self.logger.info("FAISS index transferred to CPU successfully")

                    self.logger.info("Serializing FAISS index for cross-actor sharing...")
                    faiss_index_bytes = pickle.dumps(rag_spec.vector_store.index)
                    docstore_bytes = pickle.dumps(rag_spec.vector_store.docstore)
                    index_to_docstore_id_bytes = pickle.dumps(rag_spec.vector_store.index_to_docstore_id)
                    components.update({
                        "vector_store_type": "faiss",
                        "faiss_index_bytes": faiss_index_bytes,
                        "docstore_bytes": docstore_bytes,
                        "index_to_docstore_id_bytes": index_to_docstore_id_bytes,
                    })
                    self.logger.info(f"FAISS index serialized: {len(faiss_index_bytes)} bytes")
                
                elif rag_spec.vector_store is not None and isinstance(rag_spec.vector_store, PGVector):
                    self.logger.info("Serializing PGVector store for cross-actor sharing...")
                    components.update({
                        "vector_store_type": "pgvector",
                        "pgvector_connection": rag_spec.pgvector_connection,
                        "pgvector_collection_name": rag_spec.pgvector_collection_name,
                    })
                    self.logger.info(f"PGVector collection name: {rag_spec.pgvector_collection_name}")

                elif rag_spec.vector_store is not None and isinstance(rag_spec.vector_store, PineconeVectorStore):
                    self.logger.info("Serializing PineconeVector store for cross-actor sharing...")
                    components.update({
                        "vector_store_type": "pinecone",
                        "pinecone_index_name": rag_spec.pinecone_index_name,
                        "pinecone_api_key": os.environ.get("PINECONE_API_KEY", None),
                        "pinecone_namespace": rag_spec.pinecone_namespace,
                        "pinecone_text_key": rag_spec.pinecone_text_key,
                        "pinecone_distance_strategy": rag_spec.pinecone_distance_strategy,
                    })
                    self.logger.info(f"Pinecone index name: {rag_spec.pinecone_index_name}")
                    
                else:
                    try:
                        vector_store_bytes = pickle.dumps(rag_spec.vector_store)
                        retriever_bytes = pickle.dumps(rag_spec.retriever)
                        components.update({
                            "vector_store": vector_store_bytes,
                            "retriever": retriever_bytes,
                        })
                    except Exception as e:
                        self.logger.exception(f"Failed to serialize vector store or retriever: {e}")
                        raise RuntimeError(f"Failed to serialize vector store or retriever: {e}") from e

            # Set up PromptManager if provided
            if prompt_manager:
                self.logger.info("Setting up PromptManager...")
                prompt_manager.setup_examples()
                self.logger.info("PromptManager setup successfully")
                components.update({
                    "prompt_manager_exists": True,
                    "prompt_manager": prompt_manager,
                })

            self.logger.info("DocProcessingActor: Context components ready for sharing")
            return components

        except Exception as e:
            # Convert any exception to RuntimeError to ensure it can be properly serialized by Ray.
            error_type = type(e).__name__
            error_message = str(e)

            # Log the original exception details for debugging
            self.logger.exception(f"Failed to build RAG components: {error_type}: {error_message}")

            # Convert to RuntimeError with descriptive message
            # Include the original exception type and message for better error reporting
            raise RuntimeError(
                f"Failed to build RAG components: {error_type}: {error_message}"
            ) from None  # Don't chain to avoid serialization issues

    # ------------------------------------------------------------------
    # Multimodal summarizer engine lifecycle
    #
    # These three methods are invoked by ``LangChainRagSpec._describe_multi_modal_documents``
    # via the ``summarizer_runner`` handle (which is ``self`` when called from
    # within ``build_rag_components``). They mirror the engine-instantiation
    # pattern in ``QueryProcessingActor.initialize_for_pipeline`` so that
    # config-hash-based reuse, GPU teardown, and Ray-safe error handling
    # behave identically across the two phases.
    # ------------------------------------------------------------------

    def initialize_summarizer(
        self,
        generator: Any,
        rate_limiter_actor: Any | None = None,
    ) -> None:
        """
        Instantiate (or reuse) the inference engine for one summarizer.

        Computes a config hash from the generator's engine class + kwargs (with
        the same vLLM-only ``model_config`` special-case as ``QueryProcessingActor``,
        so sampling-param tweaks don't trigger a model reload). When the hash
        matches the previously initialized engine, the existing engine is reused.
        Otherwise the previous engine is cleaned up (with GPU cache flush) and
        a fresh one is built.

        For ``APIInferenceEngine``, ``rate_limiter_actor`` is injected into the
        engine kwargs (the engine constructor requires it). For vLLM engines
        the argument is ignored.

        Args:
            generator: A ``ModelConfig`` instance from a summarizer cfg.
            rate_limiter_actor: Required for API generators, unused for vLLM.

        Raises:
            RuntimeError: If engine instantiation fails. The original exception
                is converted so it can be serialized by Ray.
        """
        try:
            import hashlib
            engine_class = generator.get_engine_class()
            engine_kwargs = generator.get_engine_kwargs()

            # API engine constructor requires a rate limiter; vLLM doesn't.
            if engine_class.__name__ == "APIInferenceEngine":
                engine_kwargs = {**engine_kwargs, "rate_limiter_actor": rate_limiter_actor}

            # Mirror query_actor's hashing strategy so sampling_params changes
            # don't unnecessarily reload the model on vLLM.
            if engine_class.__name__ == "VLLMInferenceEngine":
                model_config = engine_kwargs.get("model_config", {})
                config_str = f"{engine_class.__name__}:{repr(sorted(model_config.items()))}"
            else:
                config_str = f"{engine_class.__name__}:{repr(sorted(engine_kwargs.items()))}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()

            if self.summarizer_engine_config_hash == config_hash and self.summarizer_engine is not None:
                self.logger.info(
                    f"Reusing existing summarizer engine (config hash: {config_hash[:8]})"
                )
                return

            # Different config — tear the old engine down before allocating a new one.
            if self.summarizer_engine is not None:
                self.logger.info(
                    f"Cleaning up old summarizer engine "
                    f"(hash: {(self.summarizer_engine_config_hash or '?')[:8]}) before init"
                )
                self._teardown_summarizer_engine()

            self.logger.info(
                f"Initializing summarizer engine {engine_class.__name__} "
                f"(config hash: {config_hash[:8]})"
            )
            self.summarizer_engine = engine_class(**engine_kwargs)
            self.summarizer_engine_config_hash = config_hash

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            self.logger.exception(f"Failed to initialize summarizer: {error_type}: {error_message}")
            raise RuntimeError(
                f"Failed to initialize summarizer: {error_type}: {error_message}"
            ) from None

    def summarize_batch(self, prompts: list) -> list[str]:
        """
        Generate summaries for a batch of prompts using the active summarizer engine.

        Args:
            prompts: List of chat-message lists (as built by
                ``LangChainRagSpec._build_summarization_prompts``).

        Returns:
            List of generated summary strings, one per prompt.

        Raises:
            RuntimeError: If no engine has been initialized, or if generation
                fails. Always converted to ``RuntimeError`` for Ray serialization.
        """
        try:
            if self.summarizer_engine is None:
                raise RuntimeError(
                    "summarize_batch called before initialize_summarizer. "
                    "Call initialize_summarizer() with a generator first."
                )
            return self.summarizer_engine.generate(prompts)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            self.logger.exception(f"Error in summarize_batch: {error_type}: {error_message}")
            raise RuntimeError(
                f"Error in summarize_batch: {error_type}: {error_message}"
            ) from None

    def cleanup_summarizer(self) -> None:
        """
        Conditionally tear down the current summarizer engine.

        Called by ``LangChainRagSpec._describe_multi_modal_documents`` in a
        ``finally`` block after every modality. Behavior depends on the engine
        type:

        - **GPU-bound engines** (e.g. ``VLLMInferenceEngine``) are always torn
          down so model weights are evicted from VRAM before the next modality
          loads its own engine. This holds even if generation raised partway
          through.
        - **Stateless engines** (e.g. ``APIInferenceEngine``) are *not* torn
          down. Their ``cleanup()`` is a no-op and they hold no GPU memory, so
          discarding them only defeats the config-hash short-circuit in
          :meth:`initialize_summarizer` — which was the entire point of
          tracking ``summarizer_engine_config_hash`` across modalities. Keeping
          the engine + hash alive lets the next modality reuse the same client
          when its generator config matches.

        Safe to call when no engine is active — short-circuits in that case.
        """
        if self.summarizer_engine is None:
            return
        # Stateless API clients have nothing to release and we want to preserve
        # the cached engine + config hash so the next modality's
        # initialize_summarizer call can short-circuit when configs match.
        if type(self.summarizer_engine).__name__ == "APIInferenceEngine":
            self.logger.debug(
                f"Skipping teardown for stateless summarizer engine "
                f"(hash: {(self.summarizer_engine_config_hash or '?')[:8]}); "
                "kept alive for cross-modality reuse"
            )
            return
        self._teardown_summarizer_engine()

    def _teardown_summarizer_engine(self) -> None:
        """Internal: cleanup engine + force GC + empty CUDA cache (mirrors query_actor)."""
        try:
            self.summarizer_engine.cleanup()
        except Exception as e:
            self.logger.warning(f"Error during summarizer engine cleanup: {e}")

        del self.summarizer_engine
        self.summarizer_engine = None
        self.summarizer_engine_config_hash = None

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("Summarizer engine GPU memory cache cleared")
        except ImportError:
            pass


# Export for external use
__all__ = ["DocProcessingActor"]
