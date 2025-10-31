"""
Context Generator for RAG (Retrieval-Augmented Generation) applications.

Lightweight wrapper providing unified interface to RAG retrieval and prompting components.
Components are passed pre-initialized from DocProcessingActor via Ray's object store.
"""

import asyncio
import hashlib
import json
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rapidfireai.infer.rag.prompt_manager import PromptManager
from rapidfireai.infer.rag.rag_pipeline import LangChainRagSpec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextGenerator:
    """
    Lightweight wrapper providing unified interface to RAG retrieval and prompting.

    Supports two usage patterns:
    1. Configuration (notebooks): Pass rag_spec (unbuilt) for Controller to build
    2. Runtime (actors): Pass retriever + document_template (pre-built components)

    Attributes:
        rag_spec (LangChainRagSpec): RAG spec (unbuilt, stored for Controller to build).
        retriever (BaseRetriever): Retriever for document retrieval.
        document_template (callable): Template function for formatting documents.
        prompt_manager (PromptManager): Prompt manager for instructions/examples.
    """

    def __init__(
        self,
        rag_spec: LangChainRagSpec = None,
        prompt_manager: PromptManager = None,
        retriever: BaseRetriever = None,
        document_template: callable = None,
    ) -> None:
        """
        Initialize with either rag_spec (config) or pre-built components (runtime).

        Two usage patterns:
        1. Configuration (notebook): Pass rag_spec (unbuilt) for Controller to build
        2. Runtime (actor): Pass retriever + document_template (already built)

        Args:
            rag_spec: RAG specification (unbuilt, for Controller to build)
            prompt_manager: Optional prompt manager for instructions/examples
            retriever: Pre-built retriever (for runtime use in actors)
            document_template: Pre-built document template function

        Raises:
            ValueError: If all of rag_spec, retriever, and prompt_manager are None.
        """
        if not rag_spec and not retriever and not prompt_manager:
            raise ValueError("either rag_spec, retriever, or prompt_manager is required")

        self.rag_spec = rag_spec  # Store unbuilt spec (if provided)
        self.prompt_manager = prompt_manager

        # For runtime use: pre-built components passed directly
        if retriever:
            self.retriever = retriever
            self.document_template = document_template or self._default_template
        # For config use: extract from rag_spec (if built)
        elif rag_spec:
            self.retriever = rag_spec.retriever if rag_spec else None
            self.document_template = rag_spec.template if (rag_spec and rag_spec.template) else self._default_template
        else:
            self.retriever = None
            self.document_template = self._default_template

    def _default_template(self, doc: Document) -> str:
        """Default document formatting template."""
        metadata = "; ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        return f"{metadata}:\n{doc.page_content}"

    def get_instructions(self) -> str:
        """Get the instructions text."""
        if not self.prompt_manager:
            raise ValueError("prompt_manager not configured")
        return self.prompt_manager.get_instructions()

    def get_fewshot_examples(self, user_queries: list[str]) -> list[str]:
        """Get few-shot examples for the given queries."""
        if not self.prompt_manager:
            raise ValueError("prompt_manager not configured")

        import concurrent.futures

        async def gather_examples():
            """Async helper to gather all examples concurrently"""
            tasks = [self.prompt_manager.get_fewshot_examples(user_query=query) for query in user_queries]
            return await asyncio.gather(*tasks)

        # Check if we're already in an event loop (e.g., Ray async actor)
        try:
            asyncio.get_running_loop()

            # We're in an event loop, need to run in a new thread
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(gather_examples())
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(gather_examples())

    def get_hash(self) -> str:
        """
        Generate a unique hash for this context configuration.

        Used for deduplicating contexts in the database - if two pipelines have
        identical RAG and prompt configurations, they can share the same context.

        Returns:
            SHA256 hash string
        """
        config_dict = {}

        # Hash RAG spec if present
        if self.rag_spec:
            rag_dict = {}

            # Document loader configuration
            if hasattr(self.rag_spec.document_loader, "path"):
                rag_dict["documents_path"] = self.rag_spec.document_loader.path

            # Text splitter configuration
            if hasattr(self.rag_spec, "text_splitter"):
                text_splitter = self.rag_spec.text_splitter
                rag_dict["chunk_size"] = getattr(text_splitter, "_chunk_size", None)
                rag_dict["chunk_overlap"] = getattr(text_splitter, "_chunk_overlap", None)
                rag_dict["text_splitter_type"] = type(text_splitter).__name__

            # Embedding configuration
            rag_dict["embedding_cls"] = self.rag_spec.embedding_cls.__name__ if self.rag_spec.embedding_cls else None
            rag_dict["embedding_kwargs"] = self.rag_spec.embedding_kwargs  # Contains model_name, device, etc.

            # Search configuration
            rag_dict["search_type"] = self.rag_spec.search_type
            rag_dict["search_kwargs"] = self.rag_spec.search_kwargs  # Contains k and other search params
            rag_dict["enable_gpu_search"] = self.rag_spec.enable_gpu_search
            rag_dict["has_reranker"] = self.rag_spec.reranker is not None

            config_dict["rag_spec"] = rag_dict

        # Hash prompt manager if present
        if self.prompt_manager:
            prompt_dict = {
                "instructions": self.prompt_manager.instructions,
                "k": self.prompt_manager.k,  # Number of fewshot examples to retrieve
                "embedding_cls": self.prompt_manager.embedding_cls.__name__
                if self.prompt_manager.embedding_cls
                else None,
                "embedding_kwargs": self.prompt_manager.embedding_kwargs,  # Model name and config
                "example_selector_cls": self.prompt_manager.example_selector_cls.__name__
                if self.prompt_manager.example_selector_cls
                else None,
                "num_examples": len(self.prompt_manager.examples) if self.prompt_manager.examples else 0,
                # Hash the examples themselves to detect changes
                "examples_hash": hashlib.sha256(
                    json.dumps(self.prompt_manager.examples, sort_keys=True).encode()
                ).hexdigest()
                if self.prompt_manager.examples
                else None,
            }
            config_dict["prompt_manager"] = prompt_dict

        # Convert to JSON string and hash
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def get_context(self, batch_queries: list[str]) -> list[str]:
        """
        Retrieve and serialize relevant context documents for batch queries.

        Args:
            batch_queries: List of query strings

        Returns:
            List of formatted context strings
        """
        if not self.retriever:
            raise ValueError("retriever not configured")

        # Batch retrieval
        batch_docs = self.retriever.batch(batch_queries)

        # Serialize documents
        separator = "\n\n"
        return [separator.join([self.document_template(doc) for doc in docs]) for docs in batch_docs]
