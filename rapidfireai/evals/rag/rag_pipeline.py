"""
RAG (Retrieval-Augmented Generation) Specification using LangChain components.

This module provides a comprehensive RAG implementation that combines document loading,
text splitting, embedding generation, vector storage, and retrieval functionality.
Uses FAISS by default for both CPU and GPU similarity search with optimized indexes:
- GPU: IndexFlatL2 moved to GPU using StandardGpuResources for exact L2 distance search
- CPU: IndexHNSWFlat for approximate nearest neighbor search with HNSW algorithm
"""

import copy
from collections.abc import Callable
from typing import Any, Optional, List as list
import hashlib
import json

import faiss
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from langchain_core.documents import BaseDocumentCompressor
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

def _default_document_template(doc: Document) -> str:
    """
    Default document formatting template.
    
    Args:
        doc: A langchain Document to format.
        
    Returns:
        Formatted string with metadata and content.
    """
    metadata = "; ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
    return f"{metadata}:\n{doc.page_content}"


class LangChainRagSpec:
    """
    A comprehensive RAG (Retrieval-Augmented Generation) implementation using LangChain.

    This class provides a complete pipeline for document processing, embedding generation,
    vector storage, and batch context retrieval for RAG applications. Uses FAISS by default
    for both CPU and GPU similarity search with optimized flat indexes for maximum performance.

    Attributes:
        document_loader (BaseLoader): The document loader for loading source documents.
        text_splitter (TextSplitter): The text splitter for chunking documents.
        embedding_cls (Type[Embeddings]): The embedding class to instantiate.
        embedding_kwargs (Dict[str, Any]): Dictionary containing all parameters needed to initialize the embedding class.
        embedding (Optional[Embeddings]): The instantiated embedding model (created in initialize()).
        vector_store (Optional[VectorStore]): The vector store for storing and searching embeddings.
        retriever (BaseRetriever): The retriever for finding relevant documents.
        search_type (str): The search algorithm type ('similarity', 'similarity_score_threshold', 'mmr').
        search_kwargs (Dict[str, Any]): Additional search parameters including k, filter, fetch_k, lambda_mult.
        reranker (Optional[Callable]): Optional reranking function for batch results.
        enable_gpu_search (bool): Whether to use GPU FAISS (IndexFlatL2 on GPU) or CPU FAISS (IndexHNSWFlat).

    Vector Store Configuration:
        - GPU Mode (enable_gpu_search=True): Creates IndexFlatL2 and moves to GPU for exact L2 distance search
        - CPU Mode (enable_gpu_search=False): Uses IndexHNSWFlat for approximate search with HNSW

    Requirements:
        - faiss-gpu package (provides both GPU and CPU FAISS implementations)
        - CUDA-compatible GPU (for GPU mode)
    """

    def __init__(
        self,
        document_loader: BaseLoader,
        text_splitter: TextSplitter,
        embedding_cls: type[Embeddings],  # Class like HuggingFaceEmbeddings, OpenAIEmbeddings, etc
        embedding_kwargs: Optional[dict[str, Any]] | None = None,
        retriever: Optional[BaseRetriever] | None = None,
        vector_store: Optional[VectorStore] | None = None,
        search_type: str = "similarity",
        search_kwargs: dict | None = None,
        reranker_cls: Optional[type[BaseDocumentCompressor]] | None = None,
        reranker_kwargs: Optional[dict[str, Any]] | None = None,
        enable_gpu_search: bool = False,
        document_template: Optional[Callable[[Document], str]] | None = None,
    ) -> None:
        """
        Initialize the RAG specification with LangChain components.

        Args:
            document_loader: The document loader for loading source documents.
            text_splitter: The text splitter for chunking documents into smaller pieces.
            embedding_cls: The embedding class (e.g., HuggingFaceEmbeddings) to instantiate.
            embedding_kwargs: Dictionary containing all parameters needed to initialize the embedding class.
                The user must provide the correct parameters for their chosen embedding class.
                For example, HuggingFaceEmbeddings might need {'model_name': 'sentence-transformers/all-mpnet-base-v2', 'model_kwargs': {'device': 'cuda'}}.
            retriever: Optional custom retriever. If not provided, one will be created
                      from the FAISS vector_store.
            vector_store: Optional vector store for storing embeddings. If not provided,
                         a new FAISS vector store will be created automatically.
            search_type: The search algorithm type. Options include:
                        - "similarity": Standard similarity search
                        - "similarity_score_threshold": Similarity with score threshold
                        - "mmr": Maximum Marginal Relevance search
            search_kwargs: Additional parameters for search configuration including:
                          - k: Number of documents to retrieve (default: 5)
                          - filter: Filter criteria for search (function)
                          - fetch_k: Number of documents to fetch for MMR (default: 20)
                          - lambda_mult: Diversity parameter for MMR (default: 0.5)
            reranker: Optional function to rerank retrieved documents.
                     Should accept a single list of Documents and return a reranked list.
                     Will be applied to each query's results individually.
            enable_gpu_search: Whether to use GPU-accelerated FAISS (IndexFlatL2 on GPU)
                                  or CPU-based FAISS (IndexHNSWFlat) for similarity search.
                                  Requires faiss-gpu package and CUDA-compatible GPU for GPU mode.
                                  Default: False (uses CPU-based FAISS with HNSW algorithm).
            document_template: Optional function to format documents.
                            Should accept a single langchain_core.documents.Document and return a string.
                            Default style is "metadata:\ncontent". If not provided, a default template is used.
                            Multiple documents are separated by double newlines.

        Raises:
            ValueError: If invalid search_type is provided or required parameters are missing.
            ImportError: If faiss-gpu package is not available.
            RuntimeError: If GPU mode is requested but CUDA GPU is not available.
        """
        # Validate required parameters
        # Note: document_loader and text_splitter can be None if retriever/vector_store
        # are provided (for query-time reconstruction from serialized components)
        if not document_loader and not retriever and not vector_store:
            raise ValueError("document_loader is required unless retriever or vector_store is provided")
        if not text_splitter and not retriever and not vector_store:
            raise ValueError("text_splitter is required unless retriever or vector_store is provided")
        if not embedding_cls:
            raise ValueError("embedding_cls is required")

        # Validate search_type
        valid_search_types = {"similarity", "similarity_score_threshold", "mmr"}
        if search_type not in valid_search_types:
            raise ValueError(f"search_type must be one of {valid_search_types}, got: {search_type}")

        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.embedding_cls = embedding_cls
        self.embedding_kwargs = embedding_kwargs or {}
        self.embedding: Embeddings | None = None  # Will be created in initialize()
        self.search_type = search_type
        if document_template:
            self.template = document_template
        else:
            self.template = _default_document_template

        # Default search kwargs with type safety
        self.search_kwargs: dict[str, Any] = {
            "k": 5,
            "filter": None,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        }
        if search_kwargs:
            self.search_kwargs.update(search_kwargs)

        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker_cls = reranker_cls
        self.reranker_kwargs = reranker_kwargs or {}
        self.enable_gpu_search = enable_gpu_search
        self.reranker = None

    @staticmethod
    def default_template(doc: Document) -> str:
        """
        Default document formatting template.
        
        Args:
            doc: A langchain Document to format.
            
        Returns:
            Formatted string with metadata and content.
        """
        return _default_document_template(doc)

    @property
    def document_template(self) -> Callable[[Document], str]:
        """
        Get the document template function.
        
        Returns:
            The document template callable.
        """
        return self.template

    def build_index(self) -> None:
        """
        Build the FAISS index and set up retrieval components.

        This method must be called after instantiation to:
        1. Create the embedding model instance using the provided class and parameters
        2. Initialize the vector store with appropriate index type (GPU or CPU)
        3. Build the document index by loading, splitting, and adding documents to FAISS
        4. Create the retriever from the vector store

        The FAISS index selection depends on the enable_gpu_search flag:
        - If True: Creates IndexFlatL2 and moves to GPU for exact L2 distance search
        - If False: Uses FAISS IndexHNSWFlat for approximate nearest neighbor search on CPU

        FAISS Index Configuration:
        - GPU: IndexFlatL2 moved to GPU using StandardGpuResources, exact L2 distance search
        - CPU (IndexHNSWFlat): HNSW algorithm with M=32, efConstruction=128, efSearch=64

        Raises:
            Exception: If embedding instantiation fails due to invalid parameters.
            ImportError: If faiss-gpu package is not available.
        """
        # Create embedding instance with provided configuration
        self.embedding = self.embedding_cls(**self.embedding_kwargs)

        if self.reranker_cls:
            if self.reranker_cls is CrossEncoderReranker:
                hf_model_name = self.reranker_kwargs.pop("model_name", "cross-encoder/ms-marco-MiniLM-L6-v2")
                hf_model_kwargs = self.reranker_kwargs.pop("model_kwargs", {})
                
                self.reranker = self.reranker_cls(
                    model=HuggingFaceCrossEncoder(
                        model_name=hf_model_name,
                        model_kwargs=hf_model_kwargs
                    ),
                    **self.reranker_kwargs
                )
            else:
                self.reranker = self.reranker_cls(**self.reranker_kwargs)

        # Initialize vector store and retriever based on provided parameters
        if not self.retriever and not self.vector_store:
            try:
                if self.enable_gpu_search:
                    # Use GPU-accelerated FAISS vector store with exact search (L2 distance) by default
                    # FAISS vector store will be built (adding documents) in _build_vector_store() method
                    self.vector_store = FAISS(
                        embedding_function=self.embedding,
                        index=faiss.IndexFlatL2(len(self.embedding.embed_query("RapidFire AI is awesome!"))),
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={},
                    )

                else:
                    # Use CPU-based Implementation of FAISS with approximate search HNSW
                    # TODO: move these to constants.py
                    M = 16  # good default value: controls the number of bidirectional connections of each node
                    ef_construction = 64  # 4-8x of M: Size of dynamic candidate list during construction
                    ef_search = 32  # 2-4x of M: Size of dynamic candidate list during search

                    hnsw_index = faiss.IndexHNSWFlat(len(self.embedding.embed_query("RapidFire AI is awesome!")), M)
                    hnsw_index.hnsw.efConstruction = ef_construction
                    hnsw_index.hnsw.efSearch = ef_search

                    self.vector_store = FAISS(
                        embedding_function=self.embedding,
                        index=hnsw_index,
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={},
                    )

            except ImportError as e:
                raise ImportError(
                    "FAISS is required for GPU similarity search. Install it with: pip install faiss-gpu"
                ) from e

            self._build_vector_store()
            self.retriever = self.vector_store.as_retriever(
                search_type=self.search_type, search_kwargs=self.search_kwargs
            )
        elif not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type=self.search_type, search_kwargs=self.search_kwargs
            )

    def copy(self) -> "LangChainRagSpec":
        """
        Create a deep copy of the LangChainRagSpec object.

        This method creates a new instance with the same configuration but independent
        vector store and retriever instances. Useful for creating variations of a
        RAG setup without affecting the original.

        Returns:
            LangChainRagSpec: A new instance with the same configuration.
        """
        # Create new instance with same base configuration
        new_rag = LangChainRagSpec(
            document_loader=self.document_loader,  # Shared reference
            text_splitter=self.text_splitter,  # Shared reference
            embedding_cls=self.embedding_cls,  # Shared reference
            embedding_kwargs=self.embedding_kwargs,  # Shared reference
            retriever=copy.deepcopy(self.retriever),  # Will be created fresh in initialize() if not provided
            vector_store=self.vector_store,  # Will be created fresh in initialize() if not provided
            search_type=self.search_type,  # Shared reference
            search_kwargs=copy.deepcopy(self.search_kwargs),  # Deep copy to avoid shared refs
            reranker_cls=self.reranker_cls,  # Shared reference
            reranker_kwargs=copy.deepcopy(self.reranker_kwargs),  # Deep copy to avoid shared refs
            enable_gpu_search=self.enable_gpu_search,  # Include GPU setting
        )

        return new_rag

    def _load_documents(self) -> list[Document]:
        """
        Load documents using the configured document loader.

        Returns:
            List[Document]: A list of loaded documents.
        """
        return self.document_loader.load()

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks using the configured text splitter.

        Args:
            documents: A list of documents to be split.

        Returns:
            List[Document]: A list of document chunks.
        """
        return self.text_splitter.split_documents(documents)

    def _build_vector_store(self) -> None:
        """
        Build the vector store by loading, splitting, and adding documents.

        This method orchestrates the document processing pipeline by:
        1. Loading documents from the source
        2. Splitting them into chunks
        3. Creating and populating the appropriate vector store (CPU or GPU-accelerated)

        The FAISS vector store must already be initialized with the appropriate index type
        (GpuIndexFlatL2 for GPU or IndexHNSWFlat for CPU) before calling this method.
        Documents are added to the existing FAISS index using the add_documents() method.
        """
        all_splits = self._split_documents(documents=self._load_documents())
        self.vector_store.add_documents(documents=all_splits)

    def _retrieve_from_vector_store(self, batch_queries: list[str]) -> list[list[Document]]:
        """
        Retrieve relevant documents from the vector store for batch queries.

        Args:
            batch_queries: A list of search query strings to process in batch.

        Returns:
            List[List[Document]]: A list where each element is a list of relevant
                                documents for the corresponding query.
        """
        return self.retriever.batch(batch_queries)

    def _serialize_docs(self, batch_docs: list[list[Document]]) -> list[str]:
        """
        Serialize batch documents into formatted strings for context injection.

        Args:
            batch_docs: A batch of document lists, where each inner list contains
                       Document objects for a single query.

        Returns:
            List[str]: A list of formatted strings where each string contains
                      all documents for one query. Documents are formatted according to the template specified.
        """

        separator = "\n\n"
        return [separator.join([self.template(d) for d in docs]) for docs in batch_docs]

    def retrieve_documents(self, batch_queries: list[str]) -> list[str]:
        """
        Retrieve, optionally rerank, and serialize relevant documents for batch queries.

        This is the main public method for getting relevant context as formatted strings
        that can be directly injected into prompts for RAG applications. Supports efficient
        batch processing for improved performance when processing multiple queries.

        Args:
            batch_queries: A list of search query strings to find relevant documents for.
                          Can contain a single query for individual processing or
                          multiple queries for batch processing.

        Returns:
            List[str]: A list of formatted strings containing relevant documents with
                      their metadata for each query. Documents are potentially reranked
                      if a reranker function was provided. Each document is formatted as
                      "metadata:\ncontent" and documents are separated by double newlines.
        """
        batch_docs = self._retrieve_from_vector_store(batch_queries=batch_queries)
        batch_docs = self._rerank_docs(batch_docs=batch_docs)
        context = self._serialize_docs(batch_docs=batch_docs)
        return context

    def _rerank_docs(self, batch_docs: list[list[Document]]) -> list[list[Document]]:
        """
        Optionally rerank batch documents using the configured reranker function.

        The reranker function is applied to each query's document list individually,
        maintaining the intuitive API where rerankers work on single document lists.

        Args:
            batch_docs: A batch of document lists where each inner list contains
                       documents for a single query.

        Returns:
            List[List[Document]]: The batch of documents, reranked if a reranker
                                 is configured, otherwise returned as-is. Maintains
                                 the same structure as input.
        """
        if self.reranker:
            # Apply reranker to each query's documents individually
            return [self.reranker(docs) for docs in batch_docs]
        return batch_docs

    def serialize_documents(self, batch_docs: list[list[Document]]) -> list[str]:
        """
        Serialize batch documents into formatted strings for context injection.
        """
        separator = "\n\n"
        return [separator.join([self.template(d) for d in docs]) for docs in batch_docs]

    def get_context(self, batch_queries: list[str], use_reranker: bool = True, serialize: bool = True) -> list[str]:
        """
        Retrieve and serialize relevant context documents for batch queries.
        
        This is a convenience method that retrieves context documents. By default,
        it uses reranking if a reranker is configured. Set use_reranker=False to
        skip reranking and just retrieve and serialize documents.
        
        Args:
            batch_queries: List of query strings to retrieve context for.
            use_reranker: Whether to apply reranking if a reranker is configured.
                         Default: True. Set to False to skip reranking.
        
        Returns:
            List of formatted context strings, one per query.
            
        Raises:
            ValueError: If retriever is not configured (build_index() not called).
        """
        if not self.retriever:
            raise ValueError("retriever not configured. Call build_index() first.")
        
        # Batch retrieval
        batch_docs = self.retriever.batch(batch_queries)
        
        # Optionally rerank
        if use_reranker:
            batch_docs = self._rerank_docs(batch_queries=batch_queries, batch_docs=batch_docs)
        
        # Serialize documents
        if serialize:
            return self.serialize_documents(batch_docs=batch_docs)
        else:
            return batch_docs

    def get_hash(self) -> str:
        """
        Generate a unique hash for this RAG configuration.

        Used for deduplicating contexts in the database - if two pipelines have
        identical RAG configurations, they can share the same context.

        Returns:
            SHA256 hash string
        """
        rag_dict = {}

        # Document loader configuration
        if hasattr(self.document_loader, "path"):
            rag_dict["documents_path"] = self.document_loader.path

        # Text splitter configuration
        if hasattr(self, "text_splitter"):
            text_splitter = self.text_splitter
            rag_dict["chunk_size"] = getattr(text_splitter, "_chunk_size", None)
            rag_dict["chunk_overlap"] = getattr(text_splitter, "_chunk_overlap", None)
            rag_dict["text_splitter_type"] = type(text_splitter).__name__

        # Embedding configuration
        rag_dict["embedding_cls"] = self.embedding_cls.__name__ if self.embedding_cls else None
        rag_dict["embedding_kwargs"] = self.embedding_kwargs  # Contains model_name, device, etc.

        # Search configuration
        rag_dict["search_type"] = self.search_type
        rag_dict["search_kwargs"] = self.search_kwargs  # Contains k and other search params
        rag_dict["enable_gpu_search"] = self.enable_gpu_search
        rag_dict["has_reranker"] = self.reranker_cls is not None and self.reranker_kwargs is not None

        # Convert to JSON string and hash
        rag_json = json.dumps(rag_dict, sort_keys=True)
        return hashlib.sha256(rag_json.encode()).hexdigest()

    def _rerank_docs(self, batch_queries: list[str], batch_docs: list[list[Document]]) -> list[list[Document]]:
        """
        Optionally rerank batch documents using the configured BaseDocumentCompressor.
        
        The reranker (BaseDocumentCompressor) is applied to each query's document list 
        individually using the compress_documents() method, which requires both the 
        query and documents as input.
        
        Args:
            batch_queries: A list of query strings corresponding to each document list.
            batch_docs: A batch of document lists where each inner list contains
                       documents for a single query.
            
        Returns:
            List[List[Document]]: The batch of documents, reranked if a reranker
                                 (BaseDocumentCompressor) is configured, otherwise 
                                 returned as-is. Maintains the same structure as input.
        """
        if self.reranker:
            # Apply reranker to each query's documents individually
            return [self.reranker.compress_documents(docs, query) 
                    for query, docs in zip(batch_queries, batch_docs)]
        return batch_docs