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
                - faiss_index_bytes, docstore_bytes, index_to_docstore_id_bytes: (if vector store was built)
                - embedding_cfg: Config dict with "class" + kwargs (used by query actors)
                - search_cfg: Config dict with "type" + kwargs (used by query actors)
                - reranker_cfg: Config dict with "class" + kwargs, or None (used by query actors)
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
                rag_spec.build_pipeline()
                self.logger.info("Document index built successfully")

                components.update({
                    "rag_spec_exists": True,
                    "embedding_cls": rag_spec.embedding_cls,
                    "embedding_kwargs": rag_spec.embedding_kwargs,
                    "search_cfg": {
                        "type": rag_spec.search_type, **rag_spec.search_kwargs
                        } if rag_spec.search_type else None,
                    "reranker_cfg": {
                        "class": rag_spec.reranker_cls, **rag_spec.reranker_kwargs
                        } if rag_spec.reranker_cls is not None else None,
                    "template": rag_spec.template if rag_spec.template is not None else None,
                    "enable_gpu_search": rag_spec.enable_gpu_search if rag_spec.enable_gpu_search is not None else False,
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
                    raise NotImplementedError("Pinecone store not implemented for cross-actor sharing")
                    
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


# Export for external use
__all__ = ["DocProcessingActor"]
