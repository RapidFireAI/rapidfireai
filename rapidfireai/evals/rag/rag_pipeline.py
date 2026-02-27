"""
RAG (Retrieval-Augmented Generation) Specification using LangChain components.

"""

import copy
import warnings
from collections.abc import Callable
from typing import Any, Optional
import hashlib
import json

import faiss
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

class LangChainRagSpec:
    """
    RAG (Retrieval-Augmented Generation) implementation using LangChain.

    This module provides a RAG implementation that combines document loading (optional),
    text splitting (optional), embedding generation, vector storage (optional), and
    retrieval. All retrieval paths are normalized to a LangChain BaseRetriever; that
    retriever is what is used for query-time retrieval.

    Component flow:
        - If a retriever is provided, it is used as-is.
        - If no retriever but a vector_store_cfg dict is provided, a new vector store of
        the specified type ("faiss", "pgvector", or "pinecone") is created and populated
        from documents. A document loader is required in this case.
        - If neither retriever nor vector_store_cfg is provided, a FAISS vector store (and
        retriever) is built from documents. A document loader is required.
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
        vector_store_cfg: Optional[dict[str, Any]] | None = None,
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
            document_loader: Optional. Required only when neither retriever nor vector_store_cfg is provided.
            text_splitter: Optional. When building from documents, chunks before embedding; if None, embed whole pages.
            embedding_cfg: Dict with "class" (embedding class) and remaining keys as kwargs. If None, uses HuggingFaceEmbeddings with no extra kwargs.
            vector_store_cfg: Optional config dict with a "type" key ("faiss", "pgvector", "pinecone")
                plus any type-specific keys (e.g. "connection", "collection_name" for pgvector).
                The store is built from documents at build_pipeline() time; document_loader is required.
            retriever: Optional. If provided, used directly.
            search_cfg: Dict with "type" (search algorithm) and remaining keys as search kwargs (k, filter, fetch_k, lambda_mult). If None, defaults to similarity with k=5.
            reranker_cfg: Dict with "class" (reranker class) and remaining keys as kwargs. If None, no reranker.
            enable_gpu_search: Use GPU FAISS when building index; default False.
            document_template: Optional function (Document -> str) to format documents.

        Raises:
            ValueError: If embedding_cfg is missing "class", search_cfg has invalid "type",
                or document_loader is missing when building from documents.
        """
        if not document_loader and not vector_store_cfg and not retriever:
            raise ValueError(
                "document_loader is required when neither retriever nor vector_store_cfg is provided "
                "(a FAISS index will be built from loaded documents)."
            )

        if vector_store_cfg and retriever:
            warnings.warn(
                "vector_store_cfg will be ignored because a retriever was provided. "
                "The retriever will be used directly.",
                UserWarning,
                stacklevel=2,
            )

        # if vector_store_cfg and not retriever and not document_loader:
        #     raise ValueError(
        #         "document_loader is required when vector_store_cfg is provided "
        #         "(the vector store will be built from loaded documents)."
        #     )

        # Embedding config can be provided in embedding_cfg or vector_store_cfg
        if not embedding_cfg and (not vector_store_cfg or not vector_store_cfg.get("embedding_cfg", None)):
            # No embedding config provided
            self.embedding_cls = None
            self.embedding_kwargs = {}
        else:
            # If embedding config is provided in vector_store_cfg, use it
            # Otherwise, use embedding_cfg
            if vector_store_cfg and vector_store_cfg.get("embedding_cfg", None):
                cfg = copy.deepcopy(vector_store_cfg.pop("embedding_cfg"))
            else:
                cfg = copy.deepcopy(embedding_cfg)
            self.embedding_cfg = copy.deepcopy(cfg)
            self.embedding_cls = cfg.pop("class", None)
            if self.embedding_cls is None:
                raise ValueError("embedding_cfg or vector_store_cfg['embedding_cfg'] must contain a 'class' key (the embedding class)")
            self.embedding_kwargs = dict(cfg) if cfg else {}

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
        self.embedding: Embeddings | None = None  # Will be created in build_pipeline()
        if document_template:
            self.template = document_template
        else:
            self.template = LangChainRagSpec.default_template
        self.vector_store_cfg = vector_store_cfg
        self.vector_store: VectorStore | None = None
        self.retriever = retriever
        self.enable_gpu_search = enable_gpu_search
        self.reranker = None
        self.experiment_name: str | None = None  # Injected by Controller before use

    @staticmethod
    def default_template(doc: Document) -> str:
        """
        Default document formatting template.
        
        Args:
            doc: A langchain Document to format.
            
        Returns:
            Formatted string with metadata and content.
        """
        metadata = "; ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        return f"{metadata}:\n{doc.page_content}"

    def _load_documents(self) -> list[Document]:
        """
        Load documents using the configured document loader.

        Returns:
            List[Document]: A list of loaded documents.
        """
        return self.document_loader.load() if self.document_loader is not None else []

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks using the configured text splitter.

        Args:
            documents: A list of documents to be split.

        Returns:
            List[Document]: A list of document chunks.
        """
        if documents and self.text_splitter:
            return self.text_splitter.split_documents(documents)
        else:
            return documents

    def _create_vector_store(self, type: str) -> VectorStore:
        """
        Create a vector store instance from the config provided
        """

        if type == "faiss":
            try:
                if self.enable_gpu_search:
                    # Use GPU-accelerated FAISS vector store with exact search (L2 distance) by default
                    # FAISS vector store will be built (adding documents) in _build_vector_store() method
                    return FAISS(
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

                    return FAISS(
                        embedding_function=self.embedding,
                        index=hnsw_index,
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={},
                    )
            except ImportError as e:
                raise ImportError(
                    "FAISS is required for GPU similarity search. Install it with: pip install faiss-gpu"
                ) from e
            
        elif type == "pgvector":
            pgvector_kwargs = copy.deepcopy(self.vector_store_cfg)
            pgvector_kwargs.pop("type", None)
            pgvector_kwargs.pop("embeddings", None)
            pgvector_kwargs.pop("batch_size", None)
            self.pgvector_connection = pgvector_kwargs.pop("connection", None)
            if self.pgvector_connection is None:
                raise ValueError("vector_store_cfg must include a 'connection' key for pgvector")
            # If user provided a collection name, use it; otherwise, use a hash of the RAG configuration
            # This ensures that each RAG configuration has a unique collection name
            self.pgvector_collection_name = pgvector_kwargs.pop("collection_name", self.get_hash())
            return PGVector(
                embeddings=self.embedding,
                collection_name=self.pgvector_collection_name,
                connection=self.pgvector_connection,
                **pgvector_kwargs,
            )

        elif type == "pinecone":
            #TODO: implement pinecone vector store
            raise NotImplementedError(f"Pinecone vector store not implemented")
        else:
            raise NotImplementedError(f"Vector store type {type} not implemented")
            
    def _build_vector_store(self) -> None:
        """
        Build the vector store by loading documents, optionally splitting, and adding.

        Loads documents from the document_loader. If a text_splitter is configured,
        documents are chunked before embedding; otherwise they are embedded as whole
        documents. The vector store must already be initialized (via _create_vector_store
        in build_pipeline) before this method is called.
        """
        documents = self._load_documents()
        if self.text_splitter:
            documents = self._split_documents(documents)
        batch_size = self.vector_store_cfg.get("batch_size", 128) if self.vector_store_cfg else 128
        for i in range(0, len(documents), batch_size):
            self.vector_store.add_documents(documents=documents[i: i + batch_size])

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

    def build_pipeline(self) -> None:
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
        if self.embedding_cls is not None:
            self.embedding = self.embedding_cls(**self.embedding_kwargs)
        else:
            self.embedding = None                                       

        if self.reranker_cls:
            if self.reranker_cls is CrossEncoderReranker:
                hf_model_name = self.reranker_kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L6-v2")
                hf_model_kwargs = self.reranker_kwargs.get("model_kwargs", {})
                extra_kwargs = {k: v for k, v in self.reranker_kwargs.items() if k not in ("model_name", "model_kwargs")}
                self.reranker = self.reranker_cls(
                    model=HuggingFaceCrossEncoder(
                        model_name=hf_model_name,
                        model_kwargs=hf_model_kwargs
                    ),
                    **extra_kwargs
                )
            else:
                self.reranker = self.reranker_cls(**self.reranker_kwargs)

        # Initialize vector store and retriever based on provided parameters
        if not self.retriever and not self.vector_store_cfg:
            # No config provided — default to FAISS built from documents
            self.vector_store = self._create_vector_store(type="faiss")
            self._build_vector_store()
            self.retriever = self.vector_store.as_retriever(
                search_type=self.search_type, search_kwargs=self.search_kwargs
            )
        elif not self.retriever:
            # Config dict provided — create the vector store and populate it from documents
            self.vector_store = self._create_vector_store(type=self.vector_store_cfg.get("type", "faiss"))
            self._build_vector_store()
            self.retriever = self.vector_store.as_retriever(
                search_type=self.search_type, search_kwargs=self.search_kwargs
            )

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
            ValueError: If retriever is not configured (build_pipeline() not called).
        """
        if not self.retriever:
            raise ValueError("retriever not configured. Call build_pipeline() first.")
        
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

    def serialize_documents(self, batch_docs: list[list[Document]]) -> list[str]:
        """
        Serialize batch documents into formatted strings for context injection.
        """
        separator = "\n\n"
        return [separator.join([self.template(d) for d in docs]) for docs in batch_docs]
    
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
            vector_store_cfg=copy.deepcopy(self.vector_store_cfg) if self.vector_store_cfg else None,
            retriever=copy.deepcopy(self.retriever) if self.retriever else None,
            search_cfg=search_cfg,
            reranker_cfg=reranker_cfg,
            enable_gpu_search=self.enable_gpu_search,
        )

        return new_rag

    def get_hash(self) -> str:
        """
        Generate a unique hash for this RAG configuration.

        Used for deduplicating contexts in the database - if two pipelines have
        identical RAG configurations, they can share the same context.

        Returns:
            SHA256 hash string
        """
        rag_dict = {}
        rag_dict["experiment_name"] = self.experiment_name or "unknown"

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

        # Vector store configuration
        rag_dict["vector_store_cfg"] = self.vector_store_cfg

        # enable_gpu_search changes the FAISS index structure (IndexFlatL2 vs IndexHNSWFlat)
        # so it is part of the indexing stage, not retrieval
        rag_dict["enable_gpu_search"] = self.enable_gpu_search

        # Convert to JSON string and hash
        rag_json = json.dumps(rag_dict, sort_keys=True)
        return hashlib.sha256(rag_json.encode()).hexdigest()