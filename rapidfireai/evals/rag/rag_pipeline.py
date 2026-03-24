"""
RAG (Retrieval-Augmented Generation) Specification using LangChain components.

"""

import copy
import warnings
from collections.abc import Callable
from typing import Any, Optional
import hashlib
import json
import os
import mlflow
from mlflow.entities import SpanType

from rapidfireai.evals.utils.constants import SEARCH_DEFAULTS, VALID_SEARCH_TYPES

import faiss
from pinecone import Pinecone, ServerlessSpec, PodSpec, ByocSpec
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone._utilities import DistanceStrategy
from langchain_core.embeddings import Embeddings
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

        if not vector_store_cfg and not retriever:
            vector_store_cfg = {"type": "faiss"}

        # Parse search_cfg: "type" -> search_type, rest -> search_kwargs.
        # Defaults and allowed keys are gated by search type to avoid passing
        # irrelevant kwargs to LangChain (e.g. fetch_k only makes sense for MMR).
        if not search_cfg:
            self.search_type = "similarity"
            self.search_kwargs = dict(SEARCH_DEFAULTS["similarity"])
        else:
            cfg = dict(search_cfg)
            self.search_type = cfg.pop("type", "similarity")
            if self.search_type not in VALID_SEARCH_TYPES:
                raise ValueError(f"search_cfg['type'] must be one of {VALID_SEARCH_TYPES}, got: {self.search_type}")
            # Start from type-specific defaults then apply user overrides
            self.search_kwargs = dict(SEARCH_DEFAULTS[self.search_type])
            self.search_kwargs.update(cfg)

        # Parse reranker_cfg: "class" -> reranker_cls, rest -> reranker_kwargs
        # "class" may be an actual class object (from Python code) or a string
        # __qualname__ (from JSON / the clone-modify dialog).  Always resolve to
        # the live class via RERANKER_CLASS_REGISTRY so the stored config never
        # contains a bare string where dill expects a type.
        if not reranker_cfg:
            self.reranker_cls = None
            self.reranker_kwargs = {}
        else:
            cfg = dict(reranker_cfg)
            cls_value = cfg.pop("class", None)
            if isinstance(cls_value, str):
                from rapidfireai.evals.utils.constants import RERANKER_CLASS_REGISTRY
                resolved = RERANKER_CLASS_REGISTRY.get(cls_value)
                if resolved is None:
                    raise ValueError(
                        f"Unknown reranker class '{cls_value}'. "
                        f"Supported classes: {sorted(RERANKER_CLASS_REGISTRY.keys())}. "
                        f"To add a new reranker, register it in RERANKER_CLASS_REGISTRY "
                        f"in rapidfireai/evals/utils/constants.py."
                    )
                self.reranker_cls = resolved
            else:
                self.reranker_cls = cls_value  # already a live class (or None)
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
        self.pipeline_id: int | None = None       # Injected by Actor before use
        self.model_name: str | None = None        # Injected by Actor before use

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
            pinecone_kwargs = copy.deepcopy(self.vector_store_cfg)
            pinecone_kwargs.pop("type", None)       # ignore
            pinecone_kwargs.pop("dimension", None)  # Computed from embedding
            pinecone_kwargs.pop("batch_size", None) # Used in _build_vector_store()

            if pinecone_kwargs.pop("name", None) is not None:
                raise ValueError("vector_store_cfg must use the 'index_namespace' key to specify the index namespace pair.")

            pinecone_api_key = pinecone_kwargs.pop("pinecone_api_key", os.environ.get("PINECONE_API_KEY", None))
            if pinecone_api_key is None:
                raise ValueError("vector_store_cfg must include a 'pinecone_api_key' key for pinecone or set the PINECONE_API_KEY environment variable")
            os.environ["PINECONE_API_KEY"] = pinecone_api_key

            self.pinecone_text_key = pinecone_kwargs.pop("text_key", "text")
            # Pinecone index names must be <= 45 chars and start with a letter.
            # Prefix "rf-" (3 chars) + first 42 chars of SHA256 = 45 chars total.
            _default_index_namespace = ("rf-" + self.get_hash()[:42], "")
            _distance_strategy_by_metric = {
                "cosine": DistanceStrategy.COSINE,
                "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
                "dotproduct": DistanceStrategy.MAX_INNER_PRODUCT,
            }

            pc = Pinecone(api_key=pinecone_api_key)

            if "index_namespace" in pinecone_kwargs:
                # read_or_update mode: connect to an existing index
                self.pinecone_index_name, self.pinecone_namespace = tuple(pinecone_kwargs.pop("index_namespace"))
                if not pc.has_index(self.pinecone_index_name):
                    raise ValueError(f"Pinecone index {self.pinecone_index_name} does not exist")
                metric = pc.describe_index(self.pinecone_index_name).metric
            else:
                # create mode: provision a new index from spec
                pinecone_spec = pinecone_kwargs.pop("spec", None)
                if pinecone_spec is None:
                    raise ValueError("vector_store_cfg must include a 'spec' key for pinecone")
                metric = pinecone_kwargs.pop("metric", "cosine") # "cosine" or "euclidean" or "dotproduct"
                if metric not in ["cosine", "euclidean", "dotproduct"]:
                    raise ValueError("metric must be one of 'cosine', 'euclidean', or 'dotproduct'")
                self.pinecone_index_name, self.pinecone_namespace = _default_index_namespace
                if not pc.has_index(self.pinecone_index_name):
                    pc.create_index(
                        name=self.pinecone_index_name,
                        dimension=len(self.embedding.embed_query("RapidFire AI is awesome!")),
                        metric=metric,
                        spec=pinecone_spec,
                        **pinecone_kwargs
                    )

            self.pinecone_distance_strategy = _distance_strategy_by_metric[metric]
            return PineconeVectorStore(
                index=pc.Index(self.pinecone_index_name),
                embedding=self.embedding,
                namespace=self.pinecone_namespace,
                text_key=self.pinecone_text_key,
                distance_strategy=self.pinecone_distance_strategy
            )
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

    @mlflow.trace(name="get_context", span_type=SpanType.CHAIN)
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

        span = mlflow.get_current_active_span()
        if span is not None:
            if self.pipeline_id is not None:
                span.set_attribute("pipeline_id", self.pipeline_id)
            if self.model_name is not None:
                span.set_attribute("model_name", self.model_name)
        
        batch_docs = self._retrieve_docs(batch_queries=batch_queries)
        
        # Optionally rerank
        if use_reranker and self.reranker:
            batch_docs = self._rerank_docs(batch_queries=batch_queries, batch_docs=batch_docs)
        
        # Serialize documents
        if serialize:
            return self.serialize_documents(batch_docs=batch_docs)
        else:
            return batch_docs

    @mlflow.trace(name="retrieve_documents")
    def _retrieve_docs(self, batch_queries: list[str]) -> list[list[Document]]:
        """
        Retrieve documents for batch queries.
        """
        return self.retriever.batch(batch_queries)

    @mlflow.trace(name="rerank_documents")
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
        results = []
        for query, docs in zip(batch_queries, batch_docs):
            with mlflow.start_span(name="rerank", span_type=SpanType.RETRIEVER) as span:
                span.set_inputs({
                    "query": query,
                    "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                })
                reranked = self.reranker.compress_documents(docs, query)
                span.set_outputs([
                    {"page_content": d.page_content, "metadata": {**d.metadata, "rank": i}}
                    for i, d in enumerate(reranked)
                ])
            results.append(reranked)
        return results
    
    @mlflow.trace(name="serialize_documents", span_type=SpanType.RETRIEVER)
    def serialize_documents(self, batch_docs: list[list[Document]]) -> list[str]:
        """
        Serialize batch documents into formatted strings for context injection.
        """
        results = []
        for docs in batch_docs:
            with mlflow.start_span(name="serialize", span_type=SpanType.RETRIEVER) as span:
                span.set_inputs(
                    {"documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]}
                )
                serialized = "\n\n".join([self.template(d) for d in docs])
                span.set_outputs({"serialized_context": serialized})
            results.append(serialized)
        return results
    
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
        if self.embedding_cls is not None:
            embedding_cfg = {"class": self.embedding_cls, **self.embedding_kwargs}
        else:
            embedding_cfg = None
        if self.reranker_cls is not None:
            reranker_cfg = {"class": self.reranker_cls, **copy.deepcopy(self.reranker_kwargs)}
        else:
            reranker_cfg = None
        if self.search_type:
            search_cfg = {"type": self.search_type, **self.search_kwargs}
        else:
            search_cfg = None
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

    def get_text_splitter_cfg(self) -> dict[str, Any] | None:
        """
        Return a configuration dict describing the text splitter, or ``None``
        if no splitter is set.

        The returned dict always contains:

        - ``type``: class name (e.g. ``"RecursiveCharacterTextSplitter"``)
        - ``chunk_size``, ``chunk_overlap``, ``keep_separator``,
          ``add_start_index``, ``strip_whitespace``: splitter parameters

        And optionally:

        - ``tokenizer``: present when the splitter was created via
          ``from_tiktoken_encoder`` (``"tiktoken:<encoding>"``) or
          ``from_huggingface_tokenizer`` (``"hf:<name_or_path>"``).

        The tokenizer key is extracted by inspecting the ``_length_function``
        closure, since neither LangChain factory method stores the tokenizer
        as a public attribute. For ``from_tiktoken_encoder``, the value uses
        the resolved encoding name (e.g. ``model_name="gpt-4"`` →
        ``"tiktoken:cl100k_base"``); this is intentional because two model
        names that share an encoding tokenize identically and should produce
        the same hash.
        """
        if self.text_splitter is None:
            return None
        class_name = type(self.text_splitter).__name__
        cfg = {
            "type": class_name,
            "chunk_size": getattr(self.text_splitter, "_chunk_size", None),
            "chunk_overlap": getattr(self.text_splitter, "_chunk_overlap", None),
            "keep_separator": getattr(self.text_splitter, "_keep_separator", False),
            "add_start_index": getattr(self.text_splitter, "_add_start_index", False),
            "strip_whitespace": getattr(self.text_splitter, "_strip_whitespace", True),
        }
        fn = getattr(self.text_splitter, "_length_function", None)
        if fn is not None and callable(fn) and getattr(fn, "__closure__", None):
            free_vars = getattr(fn.__code__, "co_freevars", ())
            closure_map = {}
            for var_name, cell in zip(free_vars, fn.__closure__):
                try:
                    closure_map[var_name] = cell.cell_contents
                except ValueError:
                    pass
            enc = closure_map.get("enc")
            if enc is not None and hasattr(enc, "name"):
                cfg["tokenizer"] = f"tiktoken:{enc.name}"
            else:
                tokenizer = closure_map.get("tokenizer")
                if tokenizer is not None and hasattr(tokenizer, "name_or_path") and tokenizer.name_or_path:
                    cfg["tokenizer"] = f"hf:{tokenizer.name_or_path}"
        return cfg

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
            rag_dict["text_splitter"] = self.get_text_splitter_cfg()

        # Embedding configuration
        rag_dict["embedding_cls"] = self.embedding_cls.__name__ if self.embedding_cls else None
        rag_dict["embedding_kwargs"] = self.embedding_kwargs  # Contains model_name, device, etc.

        # Vector store configuration
        rag_dict["vector_store_cfg"] = self.vector_store_cfg

        # enable_gpu_search changes the FAISS index structure (IndexFlatL2 vs IndexHNSWFlat)
        # so it is part of the indexing stage, not retrieval
        rag_dict["enable_gpu_search"] = self.enable_gpu_search

        # Convert to JSON string and hash.
        # Use a fallback encoder for objects like ServerlessSpec that aren't
        # natively JSON serializable — try __dict__ first, then str().
        def _default(obj):
            if hasattr(obj, "asdict"):
                return obj.asdict()
            try:
                return vars(obj)
            except TypeError:
                return str(obj)

        rag_json = json.dumps(rag_dict, sort_keys=True, default=_default)
        return hashlib.sha256(rag_json.encode()).hexdigest()