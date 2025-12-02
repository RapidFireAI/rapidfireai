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

        self.logger.info("DocProcessingActor initialized")

    def build_rag_components(
        self,
        rag_spec: LangChainRagSpec | None,
        prompt_manager: PromptManager | None = None,
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

        Returns:
            Dictionary containing initialized components ready for sharing:
                - faiss_index_bytes: Serialized FAISS index (if RAG spec provided)
                - docstore_bytes: Serialized docstore (if RAG spec provided)
                - index_to_docstore_id_bytes: Serialized mapping (if RAG spec provided)
                - embedding_cls: Embedding class (if RAG spec provided)
                - embedding_kwargs: Embedding kwargs (if RAG spec provided)
                - search_type: Search type (if RAG spec provided)
                - search_kwargs: Search parameters (if RAG spec provided)
                - template: Document formatting template (if RAG spec provided)
                - prompt_manager: Initialized prompt manager (if provided)
                - enable_gpu_search: Flag indicating if GPU search was used during build (if RAG spec provided)
                - reranker: Reranker function (if RAG spec provided and reranker exists)

        Raises:
            RuntimeError: If any error occurs during RAG component building. The original exception
                         is converted to RuntimeError to ensure it can be properly serialized by Ray.
        """
        self.logger.info("DocProcessingActor: Starting context initialization...")

        try:
            # Build RAG (embeddings, FAISS index) if RAG spec provided
            # If enable_gpu_search=True, this builds on GPU
            if rag_spec:
                self.logger.info("Building FAISS index...")
                rag_spec.build_index()
                self.logger.info("FAISS index built successfully")

                # Transfer GPU index to CPU for serialization (if GPU was used)
                if rag_spec.enable_gpu_search:
                    self.logger.info("Transferring FAISS index from GPU to CPU for serialization...")

                    # Transfer the GPU index to CPU
                    cpu_index = faiss.index_gpu_to_cpu(rag_spec.vector_store.index)

                    # Replace GPU index with CPU version
                    rag_spec.vector_store.index = cpu_index
                    self.logger.info("FAISS index transferred to CPU successfully")

            # Set up PromptManager if provided
            if prompt_manager:
                self.logger.info("Setting up PromptManager...")
                prompt_manager.setup_examples()
                self.logger.info("PromptManager setup successfully")

            # Serialize FAISS index to bytes for independent deserialization in each actor (if RAG spec provided)
            # FAISS indices are not thread-safe across processes, so each actor needs its own copy
            import pickle

            # Initialize components dict
            components = {}

            if rag_spec:
                # Get the FAISS index and the docstore
                faiss_index = rag_spec.vector_store.index
                docstore = rag_spec.vector_store.docstore
                index_to_docstore_id = rag_spec.vector_store.index_to_docstore_id

                # Serialize FAISS index to bytes
                self.logger.info("Serializing FAISS index for cross-actor sharing...")
                faiss_index_bytes = pickle.dumps(faiss_index)
                docstore_bytes = pickle.dumps(docstore)
                index_to_docstore_id_bytes = pickle.dumps(index_to_docstore_id)

                self.logger.info(f"FAISS index serialized: {len(faiss_index_bytes)} bytes")

                # Package RAG components for sharing
                components.update({
                    "faiss_index_bytes": faiss_index_bytes,  # Serialized FAISS index
                    "docstore_bytes": docstore_bytes,  # Serialized docstore
                    "index_to_docstore_id_bytes": index_to_docstore_id_bytes,  # Serialized mapping
                    "embedding_cls": rag_spec.embedding_cls,  # Class to recreate embedding
                    "embedding_kwargs": rag_spec.embedding_kwargs,  # Kwargs to recreate embedding
                    "search_type": rag_spec.search_type,
                    "search_kwargs": rag_spec.search_kwargs,
                    "template": rag_spec.template,
                    "enable_gpu_search": rag_spec.enable_gpu_search,  # Track GPU usage
                    "reranker_cls": rag_spec.reranker_cls,  # Reranker class (if any)
                    "reranker_kwargs": rag_spec.reranker_kwargs,  # Reranker kwargs (if any)
                })

            # Add prompt_manager if provided (works with or without RAG)
            if prompt_manager:
                components["prompt_manager"] = prompt_manager

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


# Export for external use
__all__ = ["DocProcessingActor"]
