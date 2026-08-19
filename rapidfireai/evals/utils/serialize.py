import base64
import json
from typing import Any

import dill

from rapidfireai.automl import RFvLLMModelConfig, RFAPIModelConfig
from rapidfireai.evals.utils.constants import SEARCH_TYPE_KEYS


def encode_payload(payload: object) -> str:
    """Encode the payload for the database"""
    return base64.b64encode(dill.dumps(payload)).decode("utf-8")


def decode_db_payload(payload: str) -> object:
    """Decode the payload from the database"""
    return dill.loads(base64.b64decode(payload))


_JSON_SCALARS = (str, int, float, bool, type(None))


def describe_value(value: Any) -> Any:
    """Coerce an arbitrary config value into a JSON-safe knob representation.

    Mirrors the rules already used for config hashing in
    ``LangChainRagSpec._json_default_encoder`` so a knob is described the same way
    wherever it is read:

    - classes and callables become their ``__qualname__`` (``document_template``,
      ``embedding_cfg["class"]``)
    - objects exposing ``asdict()`` become that dict (Pinecone's ``ServerlessSpec``)
    - objects carrying ``_user_params`` recurse into their declared constructor
      kwargs, which is what expands a nested generator model config
      (``multimodal_processor[...]["generator"]``) into real knobs

    Any other object becomes just its class name. We deliberately do *not* fall
    back to ``vars(obj)`` the way the hashing encoder does: dumping ``__dict__``
    is what leaks runtime handles and internal attribute names into config views.
    """
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, dict):
        return {str(k): describe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [describe_value(v) for v in value]
    if isinstance(value, type):
        return value.__qualname__
    if callable(value) and hasattr(value, "__qualname__"):
        return value.__qualname__
    asdict = getattr(value, "asdict", None)
    if callable(asdict):
        try:
            return describe_value(asdict())
        except Exception:
            return type(value).__qualname__
    user_params = getattr(value, "_user_params", None)
    if isinstance(user_params, dict):
        described = {
            str(k): describe_value(v)
            for k, v in user_params.items()
            if v is not None
        }
        return {"class": type(value).__qualname__, **described}
    return type(value).__qualname__


def extract_pipeline_config_json(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract JSON-serializable data from a pipeline config dictionary.

    Extracts only serializable parameters (dicts, strings, ints, etc.) and ignores
    functions, classes, and other non-serializable objects. This is used for storing
    a JSON representation in the database for analytics/display purposes.

    The actual pipeline config (with functions and classes) should be stored using
    encode_payload/decode_db_payload in the pipeline_config column.

    Args:
        pipeline_config: Pipeline config dict with keys:
            - "pipeline": RFvLLMModelConfig or RFAPIModelConfig instance
            - "batch_size": int
            - "preprocess_fn": function (skipped)
            - "postprocess_fn": function (skipped)
            - "compute_metrics_fn": function (skipped)
            - "accumulate_metrics_fn": function (skipped)
            - "online_strategy_kwargs": dict (optional)

    Returns:
        Dictionary with only JSON-serializable data from the pipeline config
    """
    json_config = {}

    # Extract batch_size if present
    if "batch_size" in pipeline_config:
        json_config["batch_size"] = pipeline_config["batch_size"]

    # Extract online_strategy_kwargs if present
    if "online_strategy_kwargs" in pipeline_config:
        json_config["online_strategy_kwargs"] = pipeline_config[
            "online_strategy_kwargs"
        ]

    # Extract pipeline type and model-specific params
    if "pipeline" in pipeline_config:
        pipeline = pipeline_config["pipeline"]

        # Helper function to extract RAG retrieval params for the clone-modify dialog.
        # Only exposes retrieval-stage params (search_cfg, reranker_cfg) in the same
        # format as LangChainRagSpec constructor args. Indexing-stage params
        # (embedding_cfg, vector_store_cfg, text_splitter) are intentionally excluded
        # because cloned pipelines always reuse the parent's pre-built index.
        def extract_rag_params(rag_spec):
            if rag_spec is None:
                return None

            rag_config = {}

            # search_cfg: mirrors the search_cfg constructor arg {"type": ..., <type-specific kwargs>}
            # Only include kwargs relevant to the chosen search type.
            search_type = getattr(rag_spec, "search_type", None)
            search_kwargs = getattr(rag_spec, "search_kwargs", None) or {}
            if search_type is not None:
                allowed_keys = SEARCH_TYPE_KEYS.get(search_type, set(search_kwargs.keys()))
                rag_config["search_cfg"] = {
                    "type": search_type,
                    **{k: v for k, v in search_kwargs.items() if k in allowed_keys and v is not None},
                }

            # reranker_cfg: full constructor-style dict including "class" and all kwargs.
            # The reranker is now instantiated per-pipeline at query time (not at index time),
            # so the user can change both the class and its kwargs via clone-modify.
            reranker_cls = getattr(rag_spec, "reranker_cls", None)
            reranker_kwargs = getattr(rag_spec, "reranker_kwargs", None) or {}
            if reranker_cls is not None:
                rag_config["reranker_cfg"] = {
                    "class": reranker_cls.__qualname__,
                    **{k: v for k, v in reranker_kwargs.items() if v is not None},
                }

            return rag_config if rag_config else None

        if isinstance(pipeline, RFvLLMModelConfig):
            json_config["pipeline_type"] = "vllm"

            # Extract model_config (dict)
            if hasattr(pipeline, "model_config") and pipeline.model_config is not None:
                json_config["model_config"] = pipeline.model_config

            # Extract sampling_params from _user_params (original dict, not SamplingParams object)
            if hasattr(pipeline, "_user_params") and "sampling_params" in pipeline._user_params:
                json_config["sampling_params"] = pipeline._user_params["sampling_params"]

            # Extract RAG params if present
            if hasattr(pipeline, "rag") and pipeline.rag is not None:
                rag_config = extract_rag_params(pipeline.rag)
                if rag_config:
                    json_config["rag_config"] = rag_config

        elif isinstance(pipeline, RFAPIModelConfig):
            json_config["pipeline_type"] = "api"

            # Extract client_config (dict) - filter out sensitive keys
            if (
                hasattr(pipeline, "client_config")
                and pipeline.client_config is not None
            ):
                sensitive_keys = {"api_key", "secret", "token", "password", "key"}
                json_config["client_config"] = {
                    k: v for k, v in pipeline.client_config.items()
                    if k.lower() not in sensitive_keys
                }

            # Extract endpoint_config - filter out api_key
            if hasattr(pipeline, "endpoint_config") and pipeline.endpoint_config is not None:
                endpoint_cfg = dict(pipeline.endpoint_config)
                endpoint_cfg.pop("api_key", None)
                json_config["endpoint_config"] = endpoint_cfg

            # Extract model_config (sampling parameters)
            if hasattr(pipeline, "model_config") and pipeline.model_config:
                json_config["model_config"] = pipeline.sampling_params_to_dict()

            # Extract rate limiting params.
            # Providers use one of two schemes:
            #   * combined-tpm scheme:  tpm_limit only          (e.g. OpenAI, Gemini)
            #   * split scheme:         itpm_limit + otpm_limit (Anthropic)
            # Serialize whichever scheme the pipeline was configured with so the
            # JSON snapshot in the database faithfully reflects the live config.
            if hasattr(pipeline, "rpm_limit") and pipeline.rpm_limit is not None:
                json_config["rpm_limit"] = pipeline.rpm_limit
            if hasattr(pipeline, "tpm_limit") and pipeline.tpm_limit is not None:
                json_config["tpm_limit"] = pipeline.tpm_limit
            if hasattr(pipeline, "itpm_limit") and pipeline.itpm_limit is not None:
                json_config["itpm_limit"] = pipeline.itpm_limit
            if hasattr(pipeline, "otpm_limit") and pipeline.otpm_limit is not None:
                json_config["otpm_limit"] = pipeline.otpm_limit
            if (
                hasattr(pipeline, "max_completion_tokens")
                and pipeline.max_completion_tokens is not None
            ):
                json_config["max_completion_tokens"] = pipeline.max_completion_tokens

            # Extract RAG params if present
            if hasattr(pipeline, "rag") and pipeline.rag is not None:
                rag_config = extract_rag_params(pipeline.rag)
                if rag_config:
                    json_config["rag_config"] = rag_config

    # NOTE: the prompt manager is intentionally NOT emitted here. Its knobs
    # (embedding_cfg, example_selector, k, instructions, ...) are all
    # indexing-stage: LangChain builds the example selector's FAISS index at
    # creation time, so none of them are clone-modifyable. ``extract_pipeline_config_json``
    # feeds the clone-modify dialog, where only retrieval/generator knobs are
    # editable. The prompt manager's knobs are still surfaced for display via
    # ``build_pipeline_knobs`` (MLflow params + notebook display table) and
    # the dispatcher's contexts-table ``prompt_config_json``.

    # Validate JSON serializability
    try:
        json.dumps(json_config)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize pipeline config to JSON: {e}") from e

    return json_config


# Maximum length of the ``instructions`` knob value before it is truncated.
# Instructions are free text and would make an unreadable axis value; the
# first 50 chars (with a trailing ellipsis when truncated) preserve enough to
# distinguish distinct instruction sets without flooding the axes/winner
# grids with a paragraph.
_INSTRUCTIONS_PREVIEW_LEN = 50


def describe_prompt_manager(pm: Any) -> dict[str, Any]:
    """Describe a ``RFPromptManager`` as a JSON-safe ``prompt_manager_config``.

    Mirrors :func:`describe_value` and the extended-label style of
    :meth:`RFLangChainRagSpec.get_text_splitter_cfg` so the fewshot prompt
    manager's knobs are described the same way the RAG indexing knobs are. The
    returned dict is meant to be flattened under ``prompt_manager_config.*``
    and shown in the axes-of-experimentation / winning-configuration grids.

    Knobs surfaced:

    - ``k``: number of examples retrieved per query.
    - ``instructions``: first ``_INSTRUCTIONS_PREVIEW_LEN`` chars (with a
      trailing ``"..."`` when longer) — enough to identify the instruction
      set without dumping a paragraph into an axis value.
    - ``example_selector``: the example selector class qualname (e.g.
      ``"SemanticSimilarityExampleSelector"``).
    - ``embedding_cfg``: ``{class, model_name, model_kwargs.device,
      encode_kwargs.normalize_embeddings, encode_kwargs.batch_size}`` — the
      same shape as ``rag_config.embedding_cfg`` so the two embedding knobs
      read side by side. The model identifier is normalised to ``model_name``
      regardless of which kwarg name the embedding class uses:
      ``HuggingFaceEmbeddings`` takes ``model_name`` while provider classes
      such as ``OpenAIEmbeddings`` / ``GoogleGenerativeAIEmbeddings`` take
      ``model`` — both are surfaced under ``model_name`` so the axis column
      is stable across providers.
    - ``example_prompt_template``: stringified template object —
      ``"{PromptTemplate(input_variables=[...], template='...')}"`` (qual
      name + input values), so distinct templates are distinguishable without
      leaking the full template body as an axis value.
    - ``num_examples``: count of provided examples (a knob the user can sweep).
    """
    cfg: dict[str, Any] = {}

    k = getattr(pm, "k", None)
    if k is not None:
        cfg["k"] = k

    instructions = getattr(pm, "instructions", None)
    if instructions:
        preview = instructions[:_INSTRUCTIONS_PREVIEW_LEN]
        cfg["instructions"] = f"{preview}..." if len(instructions) > _INSTRUCTIONS_PREVIEW_LEN else instructions

    example_selector_cls = getattr(pm, "example_selector_cls", None)
    if example_selector_cls is not None:
        cfg["example_selector"] = describe_value(example_selector_cls)

    embedding_cls = getattr(pm, "embedding_cls", None)
    if embedding_cls is not None:
        emb: dict[str, Any] = {"class": describe_value(embedding_cls)}
        embedding_kwargs = getattr(pm, "embedding_kwargs", None) or {}
        # Embedding classes name their model kwarg differently:
        # ``HuggingFaceEmbeddings`` takes ``model_name`` while provider classes
        # (``OpenAIEmbeddings``, ``GoogleGenerativeAIEmbeddings``) take ``model``.
        # Normalise both to the ``model_name`` column so the fewshot axes/winner
        # grids show the model under one stable key regardless of provider.
        model_name = embedding_kwargs.get("model_name")
        if model_name is None:
            model_name = embedding_kwargs.get("model")
        if model_name is not None:
            emb["model_name"] = model_name
        model_kwargs = embedding_kwargs.get("model_kwargs") or {}
        if "device" in model_kwargs:
            emb["model_kwargs.device"] = model_kwargs["device"]
        encode_kwargs = embedding_kwargs.get("encode_kwargs") or {}
        if "normalize_embeddings" in encode_kwargs:
            emb["encode_kwargs.normalize_embeddings"] = encode_kwargs["normalize_embeddings"]
        if "batch_size" in encode_kwargs:
            emb["encode_kwargs.batch_size"] = encode_kwargs["batch_size"]
        cfg["embedding_cfg"] = emb

    template = getattr(pm, "example_prompt_template", None)
    if template is not None:
        cls_name = type(template).__qualname__
        input_variables = list(getattr(template, "input_variables", []) or [])
        template_str = getattr(template, "template", "")
        cfg["example_prompt_template"] = (
            f"{cls_name}(input_variables={input_variables}, template={template_str!r})"
        )

    examples = getattr(pm, "examples", None)
    if examples is not None:
        cfg["num_examples"] = len(examples)

    return cfg


def build_pipeline_knobs(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """Full knob view of a pipeline: cloneable config plus indexing-stage knobs.

    ``extract_pipeline_config_json`` omits indexing-stage knobs on purpose: its
    consumer is the clone-modify dialog, where they aren't editable because a
    clone always reuses the parent's pre-built index. Display surfaces do need
    them, so this wrapper adds them back rather than widening the extractor,
    which would change clone behaviour.

    Everything RAG lives under ``rag_config.*`` with the user's ``*_cfg`` kwarg
    names regardless of which stage it belongs to, so no knob is reachable by two
    different paths.

    Knobs are read from the spec's internals rather than its ``_user_params``
    because the clone path writes retrieval edits straight to
    ``search_type`` / ``search_kwargs`` / ``reranker_cls`` / ``reranker_kwargs``;
    a cloned spec's ``_user_params`` still holds the parent's values.
    """
    knobs = extract_pipeline_config_json(pipeline_config)

    pipeline = pipeline_config.get("pipeline")
    rag = getattr(pipeline, "rag", None) if pipeline is not None else None

    # A fewshot prompt manager is an alternative to (or companion of) a RAG
    # spec: an evals-mode run may carry either, both, or neither. Extract it
    # independently of ``rag`` so a fewshot-only run (rag=None) still surfaces
    # its prompt_manager_config.* knobs in the axes/winner grids.
    prompt_manager = getattr(pipeline, "prompt_manager", None) if pipeline is not None else None
    if prompt_manager is not None:
        pm_cfg = describe_prompt_manager(prompt_manager)
        if pm_cfg:
            knobs["prompt_manager_config"] = pm_cfg

    if rag is None:
        return knobs

    rag_config = knobs.setdefault("rag_config", {})

    if getattr(rag, "text_splitter", None) is not None:
        text_splitter_cfg = rag.get_text_splitter_cfg()
        if text_splitter_cfg:
            rag_config["text_splitter_cfg"] = describe_value(text_splitter_cfg)

    embedding_cls = getattr(rag, "embedding_cls", None)
    if embedding_cls is not None:
        rag_config["embedding_cfg"] = {
            "class": describe_value(embedding_cls),
            **describe_value(getattr(rag, "embedding_kwargs", None) or {}),
        }

    vector_store_cfg = getattr(rag, "vector_store_cfg", None)
    if vector_store_cfg:
        rag_config["vector_store_cfg"] = describe_value(dict(vector_store_cfg))

    rag_config["enable_gpu_search"] = bool(getattr(rag, "enable_gpu_search", False))

    # ``__init__`` substitutes a default for these two when the user leaves them
    # out (a local-disk storage dict rooted at RF_HOME, and the built-in document
    # template), so the internals can't tell "unset" from "explicitly configured
    # that way". Reporting the substituted default would add a constant axis to
    # every report that no user chose. ``_user_params`` still holds what was
    # actually written, and unlike the retrieval knobs neither of these is
    # rewritten by the clone path, so it is safe to consult here.
    declared = getattr(rag, "_user_params", None)
    if not isinstance(declared, dict):
        declared = {}

    # ``False`` is an explicit opt-out (artifacts dropped rather than stored),
    # which is a different choice from leaving it unset, so test against None.
    artifact_storage_cfg = declared.get("artifact_storage_cfg")
    if artifact_storage_cfg is not None:
        rag_config["artifact_storage_cfg"] = describe_value(artifact_storage_cfg)

    multimodal_processor = getattr(rag, "multimodal_processor", None)
    if multimodal_processor:
        rag_config["multimodal_processor"] = describe_value(multimodal_processor)

    document_loader = getattr(rag, "document_loader", None)
    if document_loader:
        described = describe_value(document_loader)
        # Normalized to a list internally; collapse the common single-loader case
        # so the knob reads as one class name instead of a one-element list.
        if isinstance(described, list) and len(described) == 1:
            described = described[0]
        rag_config["document_loader"] = described

    document_template = declared.get("document_template")
    if document_template is not None:
        rag_config["document_template"] = describe_value(document_template)

    # The extractor validates its own output, but that runs before the merge
    # above, so the indexing knobs would otherwise go unchecked.
    try:
        json.dumps(knobs)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize pipeline knobs to JSON: {e}") from e

    return knobs