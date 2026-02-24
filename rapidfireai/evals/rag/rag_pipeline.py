"""
RAG (Retrieval-Augmented Generation) Specification using LangChain components.

"""

import copy
from collections.abc import Callable
from typing import Any, Optional
import hashlib
import json

import faiss
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
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
    RAG (Retrieval-Augmented Generation) implementation using LangChain.

    This module provides a RAG implementation that combines document loading (optional),
    text splitting (optional), embedding generation, vector storage (optional), and
    retrieval. All retrieval paths are normalized to a LangChain BaseRetriever; that
    retriever is what is used for query-time retrieval.

    Component flow:
        - If a retriever is provided, it is used as-is.
        - If no retriever but a vector store is provided, a retriever is created from the
        vector store (via as_retriever()) and used. The embedding class is still required
        to embed queries; it must match the embedding model used to build the vector store.
        - If neither retriever nor vector store is provided, a FAISS vector store (and
        retriever) is built from documents. In this case a document loader is required.
        A text splitter is optional: you may embed documents without chunking if you omit it.

    Embedding config is optional:
    - Required only for embedding queries as vectors, and for building the vector store. 
    - For full-text search, provide the retriever directly and embedding config is not required.

    FAISS defaults when building an index:
    - GPU: IndexFlatL2 on GPU for exact L2 distance search
    - CPU: IndexHNSWFlat for approximate nearest neighbor search with HNSW

    Note: The retriever (and any vector store used to create it) must be serializable
    for use on Ray.
    """

    def __init__(
        self,
        document_loader: Optional[BaseLoader] | None = None,
        text_splitter: Optional[TextSplitter] | None = None,
        embedding_cfg: Optional[dict[str, Any]] | None = None,
        vector_store: Optional[VectorStore] | None = None,
        retriever: Optional[BaseRetriever] | None = None,
        search_cfg: Optional[dict[str, Any]] | None = None,
        reranker_cfg: Optional[dict[str, Any]] | None = None,
        enable_gpu_search: bool = False,
        document_template: Optional[Callable[[Document], str]] | None = None,
    ) -> None:
        """
        Initialize the RAG specification with LangChain components.

        Config dicts couple class/type with their kwargs:
        - embedding_cfg: Must have "class" key (embedding class, e.g. HuggingFaceEmbeddings); rest are kwargs.
        - search_cfg: Must have "type" key ("similarity" | "similarity_score_threshold" | "mmr"); rest are kwargs (e.g. k, fetch_k, lambda_mult).
        - reranker_cfg: Must have "class" key (reranker class, e.g. CrossEncoderReranker); rest are kwargs.

        Args:
            document_loader: Optional. Required only when neither retriever nor vector_store is provided.
            text_splitter: Optional. When building from documents, chunks before embedding; if None, embed whole pages.
            embedding_cfg: Dict with "class" (embedding class) and remaining keys as kwargs. If None, uses HuggingFaceEmbeddings with no extra kwargs.
            vector_store: Optional. If retriever not provided, a retriever is created from it. Embedding must match.
            retriever: Optional. If provided, used directly.
            search_cfg: Dict with "type" (search algorithm) and remaining keys as search kwargs (k, filter, fetch_k, lambda_mult). If None, defaults to similarity with k=5.
            reranker_cfg: Dict with "class" (reranker class) and remaining keys as kwargs. If None, no reranker.
            enable_gpu_search: Use GPU FAISS when building index; default False.
            document_template: Optional function (Document -> str) to format documents.

        Raises:
            ValueError: If embedding_cfg is missing "class", search_cfg has invalid "type", or document_loader missing when building from documents.
        """
        # Document loader required only when we build index from documents (no retriever, no vector_store)
        if not document_loader and not retriever and not vector_store:
            raise ValueError(
                "document_loader is required when neither retriever nor vector_store is provided "
                "(a FAISS index will be built from loaded documents)."
            )

        # Parse embedding_cfg: "class" -> embedding_cls, rest -> embedding_kwargs
        if not embedding_cfg:
            self.embedding_cls = HuggingFaceEmbeddings
            self.embedding_kwargs = {}
        else:
            cfg = dict(embedding_cfg)
            self.embedding_cls = cfg.pop("class", None)
            if self.embedding_cls is None:
                raise ValueError("embedding_cfg must contain a 'class' key (the embedding class)")
            self.embedding_kwargs = dict(cfg)

        # Parse search_cfg: "type" -> search_type, rest -> search_kwargs
        default_search_kwargs: dict[str, Any] = {
            "k": 5,
            "filter": None,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        }
        if not search_cfg:
            self.search_type = "similarity"
            self.search_kwargs = dict(default_search_kwargs)
        else:
            cfg = dict(search_cfg)
            self.search_type = cfg.pop("type", "similarity")
            valid_search_types = {"similarity", "similarity_score_threshold", "mmr"}
            if self.search_type not in valid_search_types:
                raise ValueError(f"search_cfg['type'] must be one of {valid_search_types}, got: {self.search_type}")
            self.search_kwargs = dict(default_search_kwargs)
            self.search_kwargs.update(cfg)

        # Parse reranker_cfg: "class" -> reranker_cls, rest -> reranker_kwargs
        if not reranker_cfg:
            self.reranker_cls = None
            self.reranker_kwargs = {}
        else:
            cfg = dict(reranker_cfg)
            self.reranker_cls = cfg.pop("class", None)
            self.reranker_kwargs = dict(cfg)

        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.embedding: Embeddings | None = None  # Will be created in build_index()
        if document_template:
            self.template = document_template
        else:
            self.template = _default_document_template
        self.vector_store = vector_store
        self.retriever = retriever
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
        Create the embedding instance and ensure a retriever is available.

        - If a retriever was provided at init, it is used as-is (embedding is still
          created for any callers that need it).
        - If a vector_store was provided but no retriever, a retriever is created from
          it via as_retriever(search_type=..., search_kwargs=...).
        - If neither retriever nor vector_store was provided, a FAISS vector store is
          created, documents are loaded (and optionally split if text_splitter is set),
          added to FAISS, and a retriever is created from the vector store.

        FAISS index type when building:
        - enable_gpu_search=True: IndexFlatL2 on GPU (exact L2 search).
        - enable_gpu_search=False: IndexHNSWFlat on CPU (approximate HNSW search).

        Raises:
            Exception: If embedding instantiation fails.
            ImportError: If faiss (or faiss-gpu) is not available when building index.
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
        # Build config dicts from current instance for the copy
        embedding_cfg = {"class": self.embedding_cls, **self.embedding_kwargs}
        search_cfg = {"type": self.search_type, **self.search_kwargs}
        reranker_cfg = (
            {"class": self.reranker_cls, **copy.deepcopy(self.reranker_kwargs)}
            if self.reranker_cls else None
        )
        new_rag = LangChainRagSpec(
            document_loader=self.document_loader,
            text_splitter=self.text_splitter,
            embedding_cfg=embedding_cfg,
            vector_store=copy.deepcopy(self.vector_store) if self.vector_store else None,
            retriever=copy.deepcopy(self.retriever) if self.retriever else None,
            search_cfg=search_cfg,
            reranker_cfg=reranker_cfg,
            enable_gpu_search=self.enable_gpu_search,
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
        Build the vector store by loading documents, optionally splitting, and adding.

        Loads documents from the document_loader. If a text_splitter is configured,
        documents are chunked before embedding; otherwise they are embedded as whole
        documents. The FAISS vector store must already be initialized (in build_index)
        before this method is called.
        """
        documents = self._load_documents()
        if self.text_splitter:
            documents = self._split_documents(documents)
        self.vector_store.add_documents(documents=documents)

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
        if self.document_loader is not None and hasattr(self.document_loader, "path"):
            rag_dict["documents_path"] = self.document_loader.path

        # Text splitter configuration (optional)
        if self.text_splitter is not None:
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