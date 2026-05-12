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
import uuid
import mlflow
from mlflow.entities import SpanType

from rapidfireai.evals.utils.constants import SEARCH_DEFAULTS, VALID_SEARCH_TYPES, PINECONE_SOURCE_TAG

import faiss
from pinecone import Pinecone, ServerlessSpec, PodSpec, ByocSpec
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
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
from rapidfireai.evals.utils.storage_utils import CloudStorage



#     "artifact_storage_cfg": {
#         "backend": "gcs",
#         "bucket": "rapidfire-sandbox-us-west1",
#         "prefix": "mm-rag/artifacts"
#     }


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

    # Mapping of internal modality name -> (multimodal_processor key, *_source metadata key).
    # Iteration order is significant: text first, then image, then table — so vLLM users
    # never have more than one summarizer model resident in VRAM at a time.
    _SUMMARIZER_MODALITIES: tuple[tuple[str, str, str], ...] = (
        ("text", "text_summarizer_cfg", "text_source"),
        ("image", "image_summarizer_cfg", "image_source"),
        ("table", "table_summarizer_cfg", "table_source"),
    )

    # Accepted values for ``artifact_storage_cfg["backend"]``. Passed straight
    # through to ``CloudStorage`` without translation.
    _VALID_STORAGE_BACKENDS: frozenset[str] = frozenset({"s3", "gcs"})

    def __init__(
        self,
        document_loader: BaseLoader | list[Optional[BaseLoader]] | None = None,
        multimodal_processor: Optional[dict[str, Any]] | None = None,
        text_splitter: Optional[TextSplitter] | None = None,
        embedding_cfg: Optional[dict[str, Any]] | None = None,
        vector_store_cfg: Optional[dict[str, Any]] | None = None,
        retriever: Optional[BaseRetriever] | None = None,
        search_cfg: Optional[dict[str, Any]] | None = None,
        reranker_cfg: Optional[dict[str, Any]] | None = None,
        artifact_storage_cfg: Optional[dict[str, Any]] | None = None,
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
            document_loader: Optional. May be a single BaseLoader, a list of BaseLoaders (with
                None entries allowed and skipped), or None. A single loader is normalized to a
                one-element list internally; documents from all loaders are concatenated.
                Required only when neither retriever nor vector_store_cfg is provided.
            multimodal_processor: Optional dict configuring per-modality summarizers used by
                ``_describe_multi_modal_documents`` to turn raw text/table/image chunks into
                LLM-generated descriptions. Recognized keys are ``text_summarizer_cfg``,
                ``image_summarizer_cfg``, and ``table_summarizer_cfg``; each maps to a sub-dict
                with ``instructions`` (str) and ``generator`` (a ``ModelConfig`` instance, e.g.
                ``RFAPIModelConfig`` or ``RFvLLMModelConfig``). Each cfg is optional — omit a
                modality to skip summarization for it. If ``None``, no multimodal summarization
                is performed.
            text_splitter: Optional. When building from documents, chunks before embedding; if None, embed whole pages.
            embedding_cfg: Dict with "class" (embedding class) and remaining keys as kwargs. If None, uses HuggingFaceEmbeddings with no extra kwargs.
            vector_store_cfg: Optional config dict with a "type" key ("faiss", "pgvector", "pinecone")
                plus any type-specific keys (e.g. "connection", "collection_name" for pgvector).
                The store is built from documents at build_pipeline() time; document_loader is required.
            retriever: Optional. If provided, used directly.
            search_cfg: Dict with "type" (search algorithm) and remaining keys as search kwargs (k, filter, fetch_k, lambda_mult). If None, defaults to similarity with k=5.
            reranker_cfg: Dict with "class" (reranker class) and remaining keys as kwargs. If None, no reranker.
            artifact_storage_cfg: Optional dict configuring an object-store backend for offloading
                large per-document artifacts (raw images, table HTML) out of the vector DB record.
                Required keys: ``backend`` (``"s3"`` or ``"gcs"``), ``bucket`` (str).
                Optional key: ``prefix`` (str, default ``"artifacts"``). The cfg is kept flat on
                the rag spec; the actual ``CloudStorage`` client is instantiated on demand by
                :meth:`init_storage_client` and is not part of the persisted rag config. If
                ``None``, no artifact offloading is performed.
            enable_gpu_search: Use GPU FAISS when building index; default False.
            document_template: Optional function (Document -> str) to format documents.

        Raises:
            ValueError: If embedding_cfg is missing "class", search_cfg has invalid "type",
                or document_loader is missing when building from documents.
        """
        # Normalize document_loader to a list[Optional[BaseLoader]] | None so
        # downstream code (load, hash, copy) can treat it uniformly. Accept:
        #   - None                     -> stays None
        #   - a single BaseLoader      -> wrapped in a one-element list
        #   - a list of BaseLoader/None-> kept as-is
        if document_loader is not None and not isinstance(document_loader, list):
            if not isinstance(document_loader, BaseLoader):
                raise TypeError(
                    "document_loader must be a BaseLoader, a list of BaseLoaders (with optional "
                    f"None entries), or None; got {type(document_loader).__name__}."
                )
            document_loader = [document_loader]

        if not document_loader and not vector_store_cfg and not retriever:
            raise ValueError(
                "document_loader is required when neither retriever nor vector_store_cfg is provided "
                "(a FAISS index will be built from loaded documents)."
            )

        # Light validation of multimodal_processor: catch typos in summarizer
        # keys and missing 'generator' entries early, before any documents are
        # processed. Each summarizer cfg itself is optional.
        if multimodal_processor is not None:
            if not isinstance(multimodal_processor, dict):
                raise TypeError(
                    "multimodal_processor must be a dict or None; "
                    f"got {type(multimodal_processor).__name__}."
                )
            valid_keys = {"text_summarizer_cfg", "image_summarizer_cfg", "table_summarizer_cfg"}
            unknown = set(multimodal_processor) - valid_keys
            if unknown:
                raise ValueError(
                    f"multimodal_processor has unknown keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid_keys)}."
                )
            for key, cfg in multimodal_processor.items():
                if not isinstance(cfg, dict):
                    raise TypeError(
                        f"multimodal_processor['{key}'] must be a dict; got {type(cfg).__name__}."
                    )
                if "generator" not in cfg:
                    raise ValueError(
                        f"multimodal_processor['{key}'] must contain a 'generator' key "
                        "(a ModelConfig instance such as RFAPIModelConfig or RFvLLMModelConfig)."
                    )

        # Light validation of artifact_storage_cfg: catch typos in backend
        # names and missing bucket entries early, before any documents are
        # processed. The actual SDK is imported lazily by CloudStorage.
        if artifact_storage_cfg is not None:
            if not isinstance(artifact_storage_cfg, dict):
                raise TypeError(
                    "artifact_storage_cfg must be a dict or None; "
                    f"got {type(artifact_storage_cfg).__name__}."
                )
            backend = artifact_storage_cfg.get("backend")
            if backend not in self._VALID_STORAGE_BACKENDS:
                raise ValueError(
                    f"artifact_storage_cfg['backend'] must be one of "
                    f"{sorted(self._VALID_STORAGE_BACKENDS)}; got {backend!r}."
                )
            bucket = artifact_storage_cfg.get("bucket")
            if not isinstance(bucket, str) or not bucket:
                raise ValueError(
                    "artifact_storage_cfg['bucket'] must be a non-empty string."
                )
            prefix = artifact_storage_cfg.get("prefix", "artifacts")
            if not isinstance(prefix, str):
                raise TypeError(
                    "artifact_storage_cfg['prefix'] must be a string when provided."
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
        self.multimodal_processor = multimodal_processor
        # Kept flat (no copy) so callers can introspect/mutate the cfg in place.
        # The live CloudStorage handle is created on demand via init_storage_client()
        # and must never be hashed/serialized as part of the rag spec — it's
        # runtime state, not configuration.
        self.artifact_storage_cfg = artifact_storage_cfg
        self.storage_client: CloudStorage | None = None
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
        # Multimodal summarizer plumbing — injected by build_pipeline() at
        # context-build time. Kept as private attrs so they don't accidentally
        # leak into hashing/copying logic. All default to None so query-time
        # rag specs (which never summarize) and unit tests work unchanged.
        self._summarizer_runner: Any | None = None
        self._summarizer_rate_limiters: dict[str, Any] | None = None
        self._summarizer_batch_size: int | None = None

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

    def _categorize_multi_modal_documents(self, documents: list[Document]) -> list[Document]:
        """
        Normalize Unstructured multi-modal chunks into a single typed list of Documents.

        Walks chunks produced by an ``UnstructuredBaseLoader`` and tags each output
        Document with a ``document_type`` of ``"text"``, ``"table"``, or ``"image"``.
        Mutates input documents in place where possible; only image Documents are
        newly minted (since a single Composite or Table chunk may contain multiple
        embedded images, each of which becomes its own Document inheriting the
        parent's metadata).

        Per-category handling, keyed off ``metadata["category"]``:

        - ``"CompositeElement"`` -> emitted as a text Document. Any ``Image`` or
          ``Table`` elements found in ``metadata["orig_elements"]`` (gzipped-JSON
          encoded Unstructured elements) are emitted as additional image Documents
          carrying a copy of the parent's metadata.
        - ``"Table"`` -> emitted as a table Document. If the chunk also has an
          ``image_base64`` rendering, an additional image Document is emitted.
        - ``"Image"`` -> emitted as an image Document.
        - Any other category is skipped.

        For every emitted Document:

        - ``orig_elements``, ``image_base64``, and ``text_as_html`` are stripped
          from metadata (heavy / volatile fields not needed downstream).
        - A fresh ``rf_doc_id`` (UUID4 string) is assigned.
        - A ``*_source`` field is set per type:
          ``text_source`` = ``page_content`` for text,
          ``table_source`` = original ``text_as_html`` for tables,
          ``image_source`` = original ``image_base64`` for images.

        Args:
            documents: Raw Documents from an ``UnstructuredBaseLoader.load()``.

        Returns:
            A single list of typed Documents (text, table, and image, in input order),
            ready for downstream embedding / vector-store ingestion.
        """
        from unstructured.staging.base import elements_from_base64_gzipped_json

        DROP_KEYS = ("orig_elements", "image_base64", "text_as_html")

        def _strip(metadata: dict) -> None:
            for key in DROP_KEYS:
                metadata.pop(key, None)

        def _new_image_doc(image_b64: str, parent_metadata: dict) -> Document:
            new_meta = {k: v for k, v in parent_metadata.items() if k not in DROP_KEYS}
            new_meta["document_type"] = "image"
            new_meta["image_source"] = image_b64
            new_meta["rf_doc_id"] = str(uuid.uuid4())
            return Document(page_content="", metadata=new_meta)

        result: list[Document] = []

        for chunk in documents:
            category = chunk.metadata.get("category")

            if category == "Table":
                # A table chunk may also have a rendered image; surface it as a
                # separate image Document so it can be embedded / described.
                table_image_b64 = chunk.metadata.get("image_base64")
                if table_image_b64:
                    result.append(_new_image_doc(table_image_b64, chunk.metadata))

                table_html = chunk.metadata.get("text_as_html")
                _strip(chunk.metadata)
                chunk.metadata["document_type"] = "table"
                if table_html is not None:
                    chunk.metadata["table_source"] = table_html
                chunk.metadata["rf_doc_id"] = str(uuid.uuid4())
                # Authoritative content now lives in metadata["table_source"];
                # clear page_content to avoid storing it twice.
                chunk.page_content = ""
                result.append(chunk)

            elif category == "Image":
                image_b64 = chunk.metadata.get("image_base64")
                _strip(chunk.metadata)
                chunk.metadata["document_type"] = "image"
                if image_b64 is not None:
                    chunk.metadata["image_source"] = image_b64
                chunk.metadata["rf_doc_id"] = str(uuid.uuid4())
                # Authoritative content now lives in metadata["image_source"];
                # clear page_content to avoid storing it twice.
                chunk.page_content = ""
                result.append(chunk)

            elif category == "CompositeElement":
                # Pull nested Image/Table base64 out of orig_elements *before*
                # stripping the parent metadata, so each becomes its own image
                # Document carrying the composite chunk's contextual metadata
                # (filename, page_number, etc.).
                orig_b64 = chunk.metadata.get("orig_elements")
                if orig_b64:
                    for el in elements_from_base64_gzipped_json(orig_b64):
                        if type(el).__name__ in ("Image", "Table"):
                            nested_b64 = getattr(el.metadata, "image_base64", None)
                            if nested_b64:
                                result.append(_new_image_doc(nested_b64, chunk.metadata))

                text_content = chunk.page_content
                _strip(chunk.metadata)
                chunk.metadata["document_type"] = "text"
                chunk.metadata["text_source"] = text_content
                chunk.metadata["rf_doc_id"] = str(uuid.uuid4())
                # Authoritative content now lives in metadata["text_source"];
                # clear page_content to avoid storing it twice.
                chunk.page_content = ""
                result.append(chunk)

        return result

    def _describe_multi_modal_documents(self, documents: list[Document]) -> list[Document]:
        """
        Generate per-modality summaries and write them back as ``page_content``.

        For each modality (in ``text -> image -> table`` order), this method:

        1. Partitions the input list by ``metadata["document_type"]``.
        2. Builds prompts via :meth:`_build_summarization_prompts` using the
           appropriate ``*_source`` metadata field as model input.
        3. Initializes a single inference engine on the configured
           ``summarizer_runner`` (typically the enclosing ``DocProcessingActor``),
           calls ``summarize_batch``, and tears the engine down before moving
           on to the next modality. Sequential initialization avoids holding
           multiple vLLM models in GPU memory at once.
        4. Writes each generated summary to the corresponding ``Document``'s
           ``page_content`` (in place — no new Documents are allocated).

        No-ops (returns ``documents`` unchanged) when:

        - ``self.multimodal_processor`` is ``None`` (no summarization configured),
        - ``documents`` is empty, or
        - ``self._summarizer_runner`` was not injected (e.g. when the rag spec
          is constructed at query-time inside ``QueryProcessingActor``, where
          summarization is not applicable).

        Args:
            documents: Categorized Documents from
                :meth:`_categorize_multi_modal_documents`. Each must carry a
                ``document_type`` of ``"text"``, ``"table"``, or ``"image"``
                and the corresponding ``text_source`` / ``table_source`` /
                ``image_source`` metadata field.

        Returns:
            The same list, with each Document's ``page_content`` overwritten by
            its generated summary (or left as ``""`` for modalities without a
            configured summarizer).
        """
        if not self.multimodal_processor or not documents or self._summarizer_runner is None:
            return documents

        by_type: dict[str, list[Document]] = {"text": [], "image": [], "table": []}
        for doc in documents:
            doc_type = doc.metadata.get("document_type")
            if doc_type in by_type:
                by_type[doc_type].append(doc)

        for modality, cfg_key, source_key in self._SUMMARIZER_MODALITIES:
            cfg = self.multimodal_processor.get(cfg_key)
            modality_docs = by_type[modality]
            if not cfg or not modality_docs:
                continue

            generator = cfg["generator"]
            instructions = cfg.get("instructions", "")
            prompts = self._build_summarization_prompts(
                modality_docs, instructions, modality, source_key
            )
            rate_limiter = self._lookup_summarizer_rate_limiter(generator)

            # Hard ceiling on prompts per summarize_batch() call. When unset
            # (None) or non-positive, fall back to "send everything in one
            # call" and rely on the engine's own internal batching.
            batch_size = self._summarizer_batch_size
            if not batch_size or batch_size <= 0:
                batch_size = len(prompts)

            self._summarizer_runner.initialize_summarizer(generator, rate_limiter)
            try:
                # Stream summaries straight back to docs so we never hold a
                # full duplicate list of summaries in memory, and so partial
                # progress survives if a later batch fails (page_content for
                # earlier docs has already been written).
                for batch_start in range(0, len(prompts), batch_size):
                    batch_end = batch_start + batch_size
                    batch_prompts = prompts[batch_start:batch_end]
                    batch_docs = modality_docs[batch_start:batch_end]
                    batch_summaries = self._summarizer_runner.summarize_batch(batch_prompts)
                    for doc, summary in zip(batch_docs, batch_summaries):
                        doc.page_content = summary
                        # Offload the original artifact (text body / base64 image /
                        # table HTML) to object storage and replace its in-metadata
                        # value with the returned URI. Critical for text: the
                        # splitter fans each text doc into N children that all
                        # inherit the same parent metadata, so a 20 KB text_source
                        # gets duplicated N times into the vector store unless we
                        # offload first. No-op when artifact_storage_cfg was not
                        # configured.
                        if self.storage_client is None:
                            continue
                        if modality == "text":
                            doc.metadata["text_source"] = self.storage_client.put_text(
                                doc.metadata["rf_doc_id"], doc.metadata["text_source"]
                            )
                        elif modality == "image":
                            doc.metadata["image_source"] = self.storage_client.put_image(
                                doc.metadata["rf_doc_id"], doc.metadata["image_source"]
                            )
                        elif modality == "table":
                            doc.metadata["table_source"] = self.storage_client.put_html(
                                doc.metadata["rf_doc_id"], doc.metadata["table_source"]
                            )
            finally:
                # Always tear the engine down before the next modality so
                # vLLM weights are evicted from GPU memory even if generation
                # raised partway through.
                self._summarizer_runner.cleanup_summarizer()

        return documents

    def _build_summarization_prompts(
        self,
        documents: list[Document],
        instructions: str,
        modality: str,
        source_key: str,
    ) -> list[list[dict[str, Any]]]:
        """
        Build OpenAI chat-message prompts for a batch of same-modality documents.

        - For ``text`` and ``table`` modalities the prompt is a single user
          message concatenating ``instructions`` and the raw ``*_source``
          string.
        - For ``image`` the prompt is the OpenAI multimodal format with a
          text part (the instructions) followed by an ``image_url`` part
          carrying the base64-encoded image as a ``data:`` URL — the same
          shape used in ``multi-modal-testing/unstructured-langchain.ipynb``.

        Args:
            documents: Documents of a single modality.
            instructions: Free-form prompt prefix from the summarizer cfg.
            modality: ``"text"``, ``"image"``, or ``"table"``.
            source_key: Metadata key whose value is fed to the model
                (``text_source`` / ``image_source`` / ``table_source``).

        Returns:
            One chat-message list per document, suitable for
            ``InferenceEngine.generate``.
        """
        prompts: list[list[dict[str, Any]]] = []
        for doc in documents:
            source = doc.metadata.get(source_key, "") or ""
            if modality == "image":
                prompts.append([{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{source}"},
                        },
                    ],
                }])
            else:
                prompts.append([{
                    "role": "user",
                    "content": f"{instructions}\n\n{source}".strip(),
                }])
        return prompts

    def _lookup_summarizer_rate_limiter(self, generator: Any) -> Any | None:
        """
        Resolve the RateLimiterActor handle for a summarizer generator, if any.

        API generators (``RFAPIModelConfig``) require a rate-limiter actor; the
        Controller pre-creates one per provider and injects them via
        :meth:`build_pipeline`. vLLM generators don't need one — we return
        ``None`` and let ``initialize_summarizer`` skip the injection.

        Args:
            generator: The ``ModelConfig`` instance from the summarizer cfg.

        Returns:
            A Ray ActorHandle for the matching provider's RateLimiterActor,
            or ``None`` if no rate-limiters were injected or the generator is
            not API-based.
        """
        if not self._summarizer_rate_limiters:
            return None
        endpoint_config = getattr(generator, "endpoint_config", None)
        if not isinstance(endpoint_config, dict):
            return None
        provider = endpoint_config.get("provider", "openai")
        return self._summarizer_rate_limiters.get(provider)

    def _process_multi_modal_documents(self, documents: list[Document]) -> list[Document]:
        """
        Process multi-modal documents into text, table, and image documents.
        """
        _docs = self._categorize_multi_modal_documents(documents)
        _docs = self._describe_multi_modal_documents(_docs)
        return _docs

    def init_storage_client(self) -> None:
        """
        Idempotently instantiate ``self.storage_client`` from ``self.artifact_storage_cfg``.

        Safe to call any number of times from any caller. The method is a no-op
        when:

        - ``self.storage_client`` is already set (already initialized), or
        - ``self.artifact_storage_cfg`` is ``None`` (artifact offloading not
          configured).

        Otherwise, instantiates a :class:`CloudStorage` using the flat cfg
        stored on the rag spec. The credential chain is the underlying SDK's
        default — see ``rapidfireai.evals.utils.storage_utils`` for details.

        Lifecycle:

        - **Build path** (``DocProcessingActor``): :meth:`_load_documents`
          calls this on entry and resets ``self.storage_client = None`` once
          loading is finished, so the SDK handle does not outlive ingestion.
        - **Query path** (``QueryProcessingActor``): the actor calls this once
          after reconstructing the rag spec and lets the handle live for the
          rest of the actor's lifetime, making it available to user-supplied
          ``preprocess_fn`` callbacks via ``rag_spec.storage_client``.
        """
        if self.storage_client is not None:
            return
        if self.artifact_storage_cfg is None:
            return

        backend = self.artifact_storage_cfg["backend"]
        bucket = self.artifact_storage_cfg["bucket"]
        prefix = self.artifact_storage_cfg.get("prefix", "artifacts")
        self.storage_client = CloudStorage(backend=backend, bucket=bucket, prefix=prefix)

    def _load_documents(self) -> list[Document]:
        """
        Load documents from all configured document loaders.

        Iterates through ``self.document_loader`` (a list of optional BaseLoader
        instances), invoking ``load()`` on each non-None loader and concatenating
        the results.

        Storage-client lifecycle: if ``self.artifact_storage_cfg`` is set, a
        :class:`CloudStorage` handle is created via :meth:`init_storage_client`
        before any loader runs (so loaders / multimodal processing can offload
        artifacts via ``self.storage_client``) and is torn down to ``None``
        once all loaders complete, so build-time SDK handles do not outlive
        ingestion. Idempotency means subsequent calls into
        ``init_storage_client`` are cheap no-ops.

        Returns:
            List[Document]: A list of loaded documents from all loaders.
        """
        if not self.document_loader:
            return []

        self.init_storage_client()
        try:
            documents: list[Document] = []
            for loader in self.document_loader:
                if loader is not None:
                    _docs = loader.load()
                    wrapped_cls = getattr(loader, "loader_cls", None)
                    if isinstance(loader, UnstructuredBaseLoader) or (
                        isinstance(wrapped_cls, type)
                        and issubclass(wrapped_cls, UnstructuredBaseLoader)
                    ):
                        # Look for multi-modal documents and process them as text
                        _docs = self._process_multi_modal_documents(_docs)
                    documents.extend(_docs)
            return documents
        finally:
            # Build-phase only: drop the handle so it doesn't live past
            # ingestion. The query path keeps its own handle alive separately.
            self.storage_client = None

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

            pc = Pinecone(
                api_key=pinecone_api_key, 
                source_tag=PINECONE_SOURCE_TAG,
            )

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

    def build_pipeline(
        self,
        summarizer_runner: Any | None = None,
        summarizer_rate_limiters: dict[str, Any] | None = None,
        summarizer_batch_size: int | None = None,
    ) -> None:
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

        Args:
            summarizer_runner: Optional handle exposing
                ``initialize_summarizer(generator, rate_limiter_actor)``,
                ``summarize_batch(prompts) -> list[str]``, and
                ``cleanup_summarizer()``. Used by
                :meth:`_describe_multi_modal_documents` when
                ``self.multimodal_processor`` is set. Typically the enclosing
                ``DocProcessingActor`` passes ``self`` here. When ``None``,
                multimodal summarization is silently skipped — appropriate for
                query-time rag specs and for ingestion without a
                ``multimodal_processor``.
            summarizer_rate_limiters: Optional ``{provider: RateLimiterActor}``
                mapping. Used by API summarizer generators
                (``RFAPIModelConfig``); vLLM generators don't need it. Keyed
                by ``endpoint_config["provider"]``.
            summarizer_batch_size: Optional hard ceiling on the number of
                prompts sent to ``summarizer_runner.summarize_batch()`` in a
                single call. When ``None`` (default), the entire modality is
                sent in one call and the underlying engine handles its own
                internal batching. Typically supplied by the Controller as
                the maximum ``batch_size`` across all pipelines that share
                this context.

        Raises:
            Exception: If embedding instantiation fails.
            ImportError: If faiss (or faiss-gpu) is not available when building index.
        """
        # Stash so _describe_multi_modal_documents (called transitively via
        # _load_documents) can see them.
        self._summarizer_runner = summarizer_runner
        self._summarizer_rate_limiters = summarizer_rate_limiters
        self._summarizer_batch_size = summarizer_batch_size

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
            multimodal_processor=copy.deepcopy(self.multimodal_processor) if self.multimodal_processor else None,
            artifact_storage_cfg=copy.deepcopy(self.artifact_storage_cfg) if self.artifact_storage_cfg else None,
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

        - ``type``: human-readable splitter label.  For plain splitters this
          is just the class name (e.g. ``"RecursiveCharacterTextSplitter"``).
          When a custom tokenizer is detected the label is extended, e.g.
          ``"RecursiveCharacterTextSplitter(tiktoken:cl100k_base)"`` or
          ``"RecursiveCharacterTextSplitter(hf:bert-base-uncased)"``.
        - ``chunk_size``, ``chunk_overlap``, ``keep_separator``,
          ``add_start_index``, ``strip_whitespace``: splitter parameters

        The tokenizer suffix is extracted by inspecting the ``_length_function``
        closure, since neither LangChain factory method stores the tokenizer
        as a public attribute. For ``from_tiktoken_encoder``, the resolved
        encoding name is used (e.g. ``model_name="gpt-4"`` →
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
                cfg["type"] = f"{class_name}(tiktoken:{enc.name})"
            else:
                tokenizer = closure_map.get("tokenizer")
                if tokenizer is not None and hasattr(tokenizer, "name_or_path") and tokenizer.name_or_path:
                    cfg["type"] = f"{class_name}(hf:{tokenizer.name_or_path})"
        return cfg

    @staticmethod
    def _hash_loader(loader: BaseLoader) -> str:
        """
        Compute a deterministic hash of a document loader's full configuration.

        LangChain's ``BaseLoader`` does not expose a built-in hash, but most
        loader subclasses store their configuration as plain instance
        attributes (e.g. ``DirectoryLoader`` keeps ``path``, ``glob``,
        ``loader_cls``, ``loader_kwargs``, ``recursive``, ``sample_seed``,
        etc. on ``self``). This method serializes ``vars(loader)`` to a
        canonical JSON form and SHA256-hashes it, so two loaders with
        identical configurations produce the same hash.

        Non-JSON-serializable values are converted as follows:
        - Class objects (e.g. ``loader_cls=JSONLoader``) → ``"module.qualname"``.
        - Functions / bound methods → ``"module.qualname"``.
        - Objects with an ``asdict()`` method → that dict.
        - Other objects → ``vars(obj)`` if available, else ``str(obj)``.

        Args:
            loader: A LangChain ``BaseLoader`` instance.

        Returns:
            SHA256 hash hex string.
        """
        def _default(obj):
            if isinstance(obj, type):
                return f"{obj.__module__}.{obj.__qualname__}"
            if callable(obj) and hasattr(obj, "__qualname__"):
                return f"{getattr(obj, '__module__', '')}.{obj.__qualname__}"
            if hasattr(obj, "asdict"):
                return obj.asdict()
            try:
                return vars(obj)
            except TypeError:
                return str(obj)

        loader_state = {
            "class": f"{type(loader).__module__}.{type(loader).__qualname__}",
            **vars(loader),
        }
        loader_json = json.dumps(loader_state, sort_keys=True, default=_default)
        return hashlib.sha256(loader_json.encode()).hexdigest()

    @staticmethod
    def _hash_multimodal_processor(processor: dict[str, Any]) -> str:
        """
        Compute a deterministic hash of a multimodal processor configuration.

        The processor dict drives summarization of text/table/image chunks, so
        any change to a summarizer's instructions or generator (model) must
        invalidate cached descriptions. For each known summarizer key
        (``text_summarizer_cfg``, ``image_summarizer_cfg``, ``table_summarizer_cfg``)
        this captures:

        - ``instructions``: the raw prompt string (or whatever the user passed).
        - ``generator_class``: the fully-qualified class name of the generator.
        - ``generator_state``: ``vars(generator)`` (constructor kwargs etc.),
          serialized via the same fallback encoder used by ``_hash_loader``.

        Missing summarizer keys are skipped, so configurations that differ only
        in which modalities are summarized produce different hashes naturally.

        Args:
            processor: The validated multimodal_processor dict.

        Returns:
            SHA256 hash hex string.
        """
        def _default(obj):
            if isinstance(obj, type):
                return f"{obj.__module__}.{obj.__qualname__}"
            if callable(obj) and hasattr(obj, "__qualname__"):
                return f"{getattr(obj, '__module__', '')}.{obj.__qualname__}"
            if hasattr(obj, "asdict"):
                return obj.asdict()
            try:
                return vars(obj)
            except TypeError:
                return str(obj)

        state: dict[str, Any] = {}
        for key in ("text_summarizer_cfg", "image_summarizer_cfg", "table_summarizer_cfg"):
            cfg = processor.get(key)
            if cfg is None:
                continue
            generator = cfg.get("generator")
            state[key] = {
                "instructions": cfg.get("instructions"),
                "generator_class": (
                    f"{type(generator).__module__}.{type(generator).__qualname__}"
                    if generator is not None
                    else None
                ),
                "generator_state": vars(generator) if generator is not None else None,
            }
        payload = json.dumps(state, sort_keys=True, default=_default)
        return hashlib.sha256(payload.encode()).hexdigest()

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

        # Hash each loader and sort the hashes to make the order-insensitive
        if self.document_loader:
            loader_hashes = sorted(
                self._hash_loader(loader)
                for loader in self.document_loader
                if loader is not None
            )
            if loader_hashes:
                rag_dict["document_loaders"] = loader_hashes

        # Multimodal processor: changing instructions or generator model
        # invalidates cached summaries, so it must contribute to the hash.
        if self.multimodal_processor:
            rag_dict["multimodal_processor"] = self._hash_multimodal_processor(
                self.multimodal_processor
            )

        # Artifact storage cfg: bucket/prefix changes mean URIs baked into
        # docstore metadata point at a different location, so a fresh build
        # is required. The live storage_client handle is excluded — it's
        # runtime state, not config.
        if self.artifact_storage_cfg:
            rag_dict["artifact_storage_cfg"] = {
                "backend": self.artifact_storage_cfg.get("backend"),
                "bucket": self.artifact_storage_cfg.get("bucket"),
                "prefix": self.artifact_storage_cfg.get("prefix", "artifacts"),
            }

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