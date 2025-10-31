"""
Document processing actor for building RAG components.

This actor runs once using all available resources for fast initialization
of embeddings and FAISS indexes. After building, components are placed in
Ray's object store for sharing across query processing actors.
"""

import time
from typing import Any

import faiss
import ray

from rapidfireai.infer.rag.prompt_manager import PromptManager
from rapidfireai.infer.rag.rag_pipeline import LangChainRagSpec
from rapidfireai.infer.utils.constants import MAX_RATE_LIMIT_RETRIES, RATE_LIMIT_BACKOFF_BASE
from rapidfireai.infer.utils.error_utils import is_rate_limit_error
from rapidfireai.infer.utils.logger import RFLogger


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
        # Initialize logger
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=experiment_path)
        self.logger = logging_manager.get_logger("DocProcessingActor")

        self.logger.info("DocProcessingActor initialized")

    def build_rag_components(
        self,
        rag_spec: LangChainRagSpec,
        prompt_manager: PromptManager = None,
    ) -> dict[str, Any]:
        """
        Build RAG components and return them for sharing.

        This method performs the heavy lifting of:
        - Loading and splitting documents
        - Generating embeddings
        - Building FAISS index (on GPU if enabled)
        - Transferring GPU index to CPU for serialization
        - Initializing prompt manager (if provided)

        Args:
            rag_spec: RAG specification with document loader, embeddings config, etc.
            prompt_manager: Optional prompt manager for few-shot examples

        Returns:
            Dictionary containing initialized components ready for sharing:
                - vector_store: FAISS vector store (with CPU index for serialization)
                - retriever: Configured retriever
                - embedding: Embedding model instance
                - search_type: Search type (similarity, mmr, etc.)
                - search_kwargs: Search parameters
                - template: Document formatting template
                - prompt_manager: Initialized prompt manager (if provided)
                - enable_gpu_search: Flag indicating if GPU search was used during build
        """
        self.logger.info("DocProcessingActor: Starting RAG initialization...")

        # Build RAG (embeddings, FAISS index) with retry logic for rate limits
        # If enable_gpu_search=True, this builds on GPU
        self.logger.info("Building FAISS index...")

        retry_count = 0
        last_error = None

        while retry_count < MAX_RATE_LIMIT_RETRIES:
            try:
                rag_spec.build_index()
                self.logger.info("FAISS index built successfully")
                break  # Success!
            except Exception as e:
                last_error = e

                # Check if it's a rate limit error
                if is_rate_limit_error(e):
                    retry_count += 1
                    if retry_count < MAX_RATE_LIMIT_RETRIES:
                        # Exponential backoff: 2, 4, 8, 16, 32 seconds
                        wait_time = RATE_LIMIT_BACKOFF_BASE ** retry_count
                        self.logger.warning(
                            f"Rate limit hit during FAISS index building. "
                            f"Retry {retry_count}/{MAX_RATE_LIMIT_RETRIES} in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Max retries ({MAX_RATE_LIMIT_RETRIES}) exceeded for rate limit errors")
                        raise
                else:
                    # Not a rate limit error - fail immediately
                    self.logger.error(f"Non-rate-limit error during FAISS index building: {type(e).__name__}")
                    raise

        if last_error and retry_count >= MAX_RATE_LIMIT_RETRIES:
            raise last_error

        # Transfer GPU index to CPU for serialization (if GPU was used)
        if rag_spec.enable_gpu_search:
            self.logger.info("Transferring FAISS index from GPU to CPU for serialization...")

            # Transfer the GPU index to CPU
            cpu_index = faiss.index_gpu_to_cpu(rag_spec.vector_store.index)

            # Replace GPU index with CPU version
            rag_spec.vector_store.index = cpu_index
            self.logger.info("FAISS index transferred to CPU successfully")

        # Set up PromptManager if provided (with retry logic for rate limits)
        if prompt_manager:
            self.logger.info("Setting up PromptManager...")

            retry_count = 0
            last_error = None

            while retry_count < MAX_RATE_LIMIT_RETRIES:
                try:
                    prompt_manager.setup_examples()
                    self.logger.info("PromptManager setup successfully")
                    break  # Success!
                except Exception as e:
                    last_error = e

                    # Check if it's a rate limit error
                    if is_rate_limit_error(e):
                        retry_count += 1
                        if retry_count < MAX_RATE_LIMIT_RETRIES:
                            # Exponential backoff: 2, 4, 8, 16, 32 seconds
                            wait_time = RATE_LIMIT_BACKOFF_BASE ** retry_count
                            self.logger.warning(
                                f"Rate limit hit during PromptManager setup. "
                                f"Retry {retry_count}/{MAX_RATE_LIMIT_RETRIES} in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            self.logger.error(f"Max retries ({MAX_RATE_LIMIT_RETRIES}) exceeded for rate limit errors")
                            raise
                    else:
                        # Not a rate limit error - fail immediately
                        self.logger.error(f"Non-rate-limit error during PromptManager setup: {type(e).__name__}")
                        raise

            if last_error and retry_count >= MAX_RATE_LIMIT_RETRIES:
                raise last_error

        # Serialize FAISS index to bytes for independent deserialization in each actor
        # FAISS indices are not thread-safe across processes, so each actor needs its own copy
        import pickle

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

        # Package components for sharing
        # Note: Pass serialized FAISS index, not the vector_store object
        components = {
            "faiss_index_bytes": faiss_index_bytes,  # Serialized FAISS index
            "docstore_bytes": docstore_bytes,  # Serialized docstore
            "index_to_docstore_id_bytes": index_to_docstore_id_bytes,  # Serialized mapping
            "embedding_cls": rag_spec.embedding_cls,  # Class to recreate embedding
            "embedding_kwargs": rag_spec.embedding_kwargs,  # Kwargs to recreate embedding
            "search_type": rag_spec.search_type,
            "search_kwargs": rag_spec.search_kwargs,
            "template": rag_spec.template,
            "prompt_manager": prompt_manager,
            "enable_gpu_search": rag_spec.enable_gpu_search,  # Track GPU usage
        }

        self.logger.info("DocProcessingActor: RAG components ready for sharing (FAISS index serialized)")
        return components


# Export for external use
__all__ = ["DocProcessingActor"]
