"""Optuna-based hyperparameter optimization integrated with RapidFire's chunk/shard loop.

Classes
-------
RFOptuna
    User-facing ``AutoMLAlgorithm`` subclass.  Drop-in replacement for
    ``RFGridSearch`` / ``RFRandomSearch``.
OptunaChunkCallback
    ``ChunkCallback`` implementation for fit mode — prunes/replaces runs
    between training chunks.
OptunaShardCallback
    ``ShardCallback`` implementation for evals mode — prunes/replaces
    pipelines between evaluation shards.

Helper functions handle search-space extraction, Optuna trial sampling,
config-leaf expansion, and metric resolution.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import re
import statistics
import uuid
import warnings
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any

import optuna

from rapidfireai.automl.base import AutoMLAlgorithm
from rapidfireai.automl.callbacks import (
    ChunkCallback,
    PipelineDecision,
    RunDecision,
    ShardCallback,
)
from rapidfireai.automl.datatypes import List, Range
from rapidfireai.fit.utils.exceptions import AutoMLException

# Module logger for the adapted median pruner's comparison trace. One INFO line
# is emitted per prune-check so the runtime ordering / peer-availability gap
# (the fastest pipeline reaching each shard first and finding no peers) is
# visible in the logs. See OptunaFitShardCallback / OptunaShardCallback.
_log = logging.getLogger("rapidfireai.automl.optuna")


def _log_prune_compare(
    logger: logging.Logger | logging.LoggerAdapter,
    trial_num: int,
    step: int,
    direction: str,
    current: float,
    peers: list[float],
    median: float | None,
    *,
    prune: bool,
    reason: str,
) -> None:
    """Emit a single readable line describing one adapted-median prune decision.

    Parameters
    ----------
    logger : logging.Logger or logging.LoggerAdapter
        Logger to emit through. The callbacks pass their injected
        ``SafeLoggerAdapter`` when the controller has wired one in, so the line
        picks up the ``[<experiment>:<name>]`` prefix that the dashboard's log
        viewer filters on (dispatcher.py:1096). Falls back to the module logger
        ``rapidfireai.automl.optuna`` otherwise.
    trial_num : int
        Optuna trial number (not the DB run/pipeline id).
    step : int
        The intermediate step the comparison ran at (cumulative shard count in
        fit mode, raw shard id in evals mode).
    direction : str
        ``"MINIMIZE"`` / ``"MAXIMIZE"`` -- controls which side of the median is
        "worse".
    current : float
        The current trial's value at *step* (best-across-steps in fit mode,
        latest-step value in evals mode).
    peers : list[float]
        Values of every other trial that has reported at *step* so far. Empty
        when the current trial is the first to reach *step* -- the structural
        "leader is un-prunable" case.
    median : float or None
        Median of *peers*, or ``None`` when *peers* is empty.
    prune : bool
        Whether the adapted pruner decided to prune.
    reason : str
        Short tag: ``no_peers_at_step``, ``current_is_nan``, ``worse_than_median``,
        ``better_than_median``.
    """
    peers_str = ", ".join(f"{v:.4f}" for v in peers) if peers else "(none)"
    median_str = f"{median:.4f}" if median is not None else "n/a"
    logger.info(
        "[RFOptuna prune-check] trial=%s step=%s dir=%s current=%.4f "
        "peers=[%s] median=%s -> %s (%s)",
        trial_num, step, direction, current, peers_str, median_str,
        "PRUNE" if prune else "continue", reason,
    )


# ---------------------------------------------------------------------------
# Optuna Trial helpers (API compatibility across Optuna versions)
# ---------------------------------------------------------------------------


def _trial_state_from_storage(study: optuna.Study, trial: optuna.Trial) -> optuna.trial.TrialState:
    """Return the stored state for *trial*.

    ``Trial`` instances returned by :meth:`~optuna.study.Study.ask` do not always
    expose a ``state`` attribute (e.g. recent Optuna releases); use frozen trials
    from the study storage instead.
    """
    for frozen in study.get_trials(deepcopy=False):
        if frozen.number == trial.number:
            return frozen.state
    raise AutoMLException(
        f"Could not resolve Optuna trial state for trial number {trial.number}"
    )


# When the primary objective (e.g. eval_loss) is never logged — common on tiny
# runs where eval may not fire — try common Trainer / MLflow key aliases.
_OBJECTIVE_ALIAS_KEYS: dict[str, tuple[str, ...]] = {
    "eval_loss": ("eval/loss", "eval-loss", "validation_loss", "train_loss", "loss"),
}


def _ordered_objective_keys(primary: str) -> tuple[str, ...]:
    keys = [primary]
    seen = {primary}
    for alias in _OBJECTIVE_ALIAS_KEYS.get(primary, ()):
        if alias not in seen:
            seen.add(alias)
            keys.append(alias)
    return tuple(keys)


def _float_from_logged_metric_value(raw: Any) -> float | None:
    """Parse a scalar from MLflow-style history or a plain numeric.  Returns ``None`` on failure."""
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        last = raw[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            return float(last[1])
        if isinstance(last, dict) and "value" in last:
            return float(last["value"])
        if isinstance(last, (int, float)):
            return float(last)
        return None
    if isinstance(raw, dict) and "value" in raw:
        return float(raw["value"])
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _normalize_metric_key(key: str) -> str:
    """Normalize a metric key for case/underscore/whitespace-insensitive match."""
    return key.lower().replace("_", "").replace(" ", "")


def _resolve_scalar_for_objective(metrics: dict[str, Any], objective_metric: str) -> float | None:
    """Return a scalar for *objective_metric*, trying known aliases as fallbacks.

    Lookup order: the primary key, then registered aliases (e.g.
    ``eval_loss`` → ``train_loss``), then a case/underscore/whitespace-
    insensitive scan of all keys.  The fuzzy scan catches MLflow variants
    like ``"Eval Loss"`` or ``"eval-loss"`` that are not in the alias table.
    """
    for key in _ordered_objective_keys(objective_metric):
        val = _float_from_logged_metric_value(metrics.get(key))
        if val is not None:
            return val
    target = _normalize_metric_key(objective_metric)
    for key, raw in metrics.items():
        if _normalize_metric_key(key) != target:
            continue
        val = _float_from_logged_metric_value(raw)
        if val is not None:
            return val
    return None


def _resolve_metric_history(metrics: dict[str, Any], objective_metric: str) -> list[tuple[int, float]]:
    """Return the full ``(step, value)`` history for the objective metric.

    Tries the primary key first, then known aliases.  Returns an empty list
    when no history is available.  Handles MLflow-style ``[(step, value), ...]``
    lists, plain numeric scalars, and bare lists of numbers.
    """
    for key in _ordered_objective_keys(objective_metric):
        raw = metrics.get(key)
        if raw is None:
            continue
        if isinstance(raw, list) and raw:
            history: list[tuple[int, float]] = []
            for entry in raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    history.append((int(entry[0]), float(entry[1])))
                elif isinstance(entry, (int, float)):
                    history.append((len(history), float(entry)))
            if history:
                return sorted(history, key=lambda x: x[0])
        if isinstance(raw, (int, float)):
            return [(0, float(raw))]
    return []


# ---------------------------------------------------------------------------
# Multi-objective helpers
# ---------------------------------------------------------------------------


def _pareto_dominates(a: list[float], b: list[float], directions: list[str]) -> bool:
    """Return True if solution *a* Pareto-dominates solution *b*.

    *a* dominates *b* when it is at least as good in every objective and
    strictly better in at least one.
    """
    strictly_better = False
    for va, vb, d in zip(a, b, directions):
        if d == "minimize":
            if va > vb:
                return False
            if va < vb:
                strictly_better = True
        else:
            if va < vb:
                return False
            if va > vb:
                strictly_better = True
    return strictly_better


def _resolve_multi_objectives(
    metrics: dict[str, Any],
    objective_metrics: list[str],
) -> list[float] | None:
    """Resolve a value for each objective metric.  Returns ``None`` if any is missing."""
    values: list[float] = []
    for metric in objective_metrics:
        v = _resolve_scalar_for_objective(metrics, metric)
        if v is None:
            return None
        values.append(v)
    return values


# ---------------------------------------------------------------------------
# Search-space extraction and sampling
# ---------------------------------------------------------------------------


def _extract_search_space(
    obj: Any,
    prefix: str = "",
) -> list[tuple[str, Range | List]]:
    """Walk a config template and collect all Range/List parameters.

    Returns a flat list of ``(dotted_path, Range_or_List)`` tuples.  The
    traversal mirrors ``recursive_expand_gridsearch`` so the same config
    structures that work with ``RFGridSearch`` / ``RFRandomSearch`` also work
    here (including ``RFModelConfig`` dataclass templates with nested
    ``peft_config`` / ``training_args`` objects).

    Only *unconditional* entries appear here.  A ``List`` is terminal: knobs
    nested inside its members depend on which member Optuna draws, so they are
    conditional parameters registered at suggest time by
    :func:`_sample_list_member` rather than listed up front.  Inspect
    ``trial.params`` -- not this function -- to see everything a trial actually
    sampled.
    """
    params: list[tuple[str, Range | List]] = []

    if isinstance(obj, (Range, List)):
        params.append((prefix, obj))
    elif hasattr(obj, "_user_params"):
        params.extend(_extract_search_space(obj._user_params, prefix))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            params.extend(_extract_search_space(value, child_prefix))
    elif is_dataclass(obj) and not isinstance(obj, type):
        # RFModelConfig and other templates are dataclasses without _user_params;
        # nested Range/List live under peft_config / training_args / dict fields.
        for f in fields(obj):
            value = getattr(obj, f.name)
            child_prefix = f"{prefix}.{f.name}" if prefix else f.name
            params.extend(_extract_search_space(value, child_prefix))
    # Primitive or non-searchable -- skip
    return params


def _seed_ranges(obj: Any, seed: int, _seen: set[int] | None = None) -> None:
    """Stamp *seed* onto every ``Range`` reachable from *obj*, in place.

    ``Range`` owns a seeded generator, but a template written by a user leaves
    it unseeded.  ``RFOptuna`` and ``RFRandomSearch`` inject their run ``seed``
    here so the values a run explores are reproducible.  This is purely about
    reproducibility: ``Range`` no longer memoizes a value set, so seeding has
    nothing to do with making two call sites agree -- the value set a coverage
    enumeration draws is cached by ``RFOptuna`` and reused by suggest.

    ``List`` itself is not seeded: it is a categorical of ordered choices
    (``List.sample()`` uses the already-seeded global RNG).  Its members are
    still walked so a ``Range`` nested inside a choice is not left on an
    unseeded generator.

    ``_seen`` guards against shared or cyclic references re-seeding one range
    twice.
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))

    if isinstance(obj, Range):
        obj.set_seed(seed)
    elif isinstance(obj, List):
        for value in obj.values:
            _seed_ranges(value, seed, _seen)
    elif hasattr(obj, "_user_params"):
        _seed_ranges(obj._user_params, seed, _seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            _seed_ranges(value, seed, _seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _seed_ranges(value, seed, _seen)
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            _seed_ranges(getattr(obj, f.name), seed, _seen)


def _find_unsampled_params(obj: Any, prefix: str = "") -> list[str]:
    """Return dotted paths of any ``Range`` / ``List`` still present in *obj*.

    Safety net used after sampling: anything reported here would otherwise be
    resolved by ``recursive_expand_randomsearch`` via ``Range.sample(1)[0]`` /
    ``List.sample()`` on the seeded/global RNG, i.e. silently randomized
    outside Optuna's view.

    Traverses everything :func:`_extract_search_space` does, plus ``list`` /
    ``tuple`` members, so it catches structures the sampler cannot reach
    (``_set_nested`` splits on ``.`` and has no integer indexing).

    Known gap: arbitrary ``__dict__`` is deliberately not walked.  Evals
    configs hold heavy objects (FAISS indices, embedders, Ray refs) and
    traversing them would be slow and fragile, so a ``Range`` stashed directly
    on a plain object with no ``_user_params`` still slips through.
    """
    if isinstance(obj, (Range, List)):
        return [prefix or "<root>"]
    if hasattr(obj, "_user_params"):
        return _find_unsampled_params(obj._user_params, prefix)

    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_unsampled_params(value, child_prefix))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_find_unsampled_params(value, f"{prefix}[{index}]"))
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            child_prefix = f"{prefix}.{f.name}" if prefix else f.name
            found.extend(_find_unsampled_params(getattr(obj, f.name), child_prefix))
    return found


# Subtrees whose contents feed the RAG index / prompt-manager context hash, and
# the subtrees within them that do not.  ``LangChainRagSpec.get_hash`` excludes
# ``search_cfg`` and ``reranker_cfg`` precisely so retrieval knobs can vary
# without forcing a rebuild, and ``PromptManager.get_hash`` covers its whole
# config, so the prompt-manager subtree is index-affecting throughout.
_CONTEXT_ROOT_SEGMENTS = ("rag", "prompt_manager")
_RETRIEVAL_ONLY_SEGMENTS = ("search_cfg", "reranker_cfg")

_LIST_INDEX_SUFFIX = re.compile(r"\[\d+\]$")


def _is_index_affecting_path(path: str) -> bool:
    """Return ``True`` if a knob at *path* changes which RAG index is needed.

    A path qualifies when it traverses a ``rag`` or ``prompt_manager`` segment
    and does not sit under ``search_cfg`` / ``reranker_cfg``.  ``List`` member
    namespacing (``api_config[1].rag.embedding_cfg``) and the multi-template
    prefix (``_t0.``) are both tolerated, since only segment names matter.
    """
    segments = [_LIST_INDEX_SUFFIX.sub("", part) for part in path.split(".")]
    if not any(part in _CONTEXT_ROOT_SEGMENTS for part in segments):
        return False
    return not any(part in _RETRIEVAL_ONLY_SEGMENTS for part in segments)


_PRIMITIVE_TYPES = (type(None), bool, int, float, str)


def _attrs_for_labeling(obj: Any) -> dict[str, Any] | None:
    """Return a primitive-only attribute dict describing *obj* for label building.

    Returns ``None`` when *obj* has no labellable structure (e.g. ``None``,
    a list/tuple, or a built-in type without a ``__dict__``) -- the caller
    falls back to ``repr(obj)`` for those.

    - ``dict``                       -> shallow copy of primitive entries
    - object with ``__dict__``       -> ``vars(obj)`` filtered to primitives
    - everything else                -> ``None``
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        src: dict[Any, Any] = obj
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return None
    else:
        src = getattr(obj, "__dict__", None)
        if not src:
            return None
    attrs: dict[str, Any] = {}
    for key, val in src.items():
        if isinstance(val, _PRIMITIVE_TYPES):
            attrs[str(key).lstrip("_")] = val
        elif isinstance(val, type):
            attrs[str(key).lstrip("_")] = val.__name__
    return attrs


def _label_prefix(obj: Any) -> str:
    """Return the class-name-like prefix used when building a label for *obj*."""
    if obj is None:
        return "None"
    if isinstance(obj, dict):
        return "cfg"
    return type(obj).__name__


def _object_labels(objects: list[Any]) -> list[str]:
    """Build concise labels showing only the attributes that differ across *objects*.

    For example, two ``RecursiveCharacterTextSplitter`` instances that only
    differ in ``chunk_size`` produce::

        ["RecursiveCharacterTextSplitter(chunk_size=256)",
         "RecursiveCharacterTextSplitter(chunk_size=128)"]

    Shared defaults (``keep_separator``, ``strip_whitespace``, etc.) are omitted
    so the labels stay short and meaningful in Optuna trial output.

    A single ``List([...])`` may freely mix:

    - regular objects (labelled by class name + differing attrs)
    - plain ``dict`` configs (labelled ``cfg(...)``)
    - ``list`` / ``tuple`` literals (labelled via ``repr``)
    - ``None`` (labelled ``"None"``, e.g. an optional reranker)
    """
    per_obj: list[tuple[str, dict[str, Any] | None]] = [
        (_label_prefix(obj), _attrs_for_labeling(obj)) for obj in objects
    ]

    all_keys: set[str] = set()
    for _, attrs in per_obj:
        if attrs:
            all_keys.update(attrs)

    varying = {
        k for k in all_keys
        if len({(attrs or {}).get(k) for _, attrs in per_obj}) > 1
    }
    if not varying:
        varying = all_keys

    labels: list[str] = []
    for i, (prefix, attrs) in enumerate(per_obj):
        obj = objects[i]
        if attrs is None:
            labels.append("None" if obj is None else repr(obj))
            continue
        parts = [f"{k}={attrs[k]!r}" for k in sorted(varying) if k in attrs]
        labels.append(f"{prefix}({', '.join(parts)})" if parts else prefix)
    return labels


def _sample_list_member(
    trial: optuna.Trial,
    list_name: str,
    idx: int,
    member: Any,
    range_cache: dict[int, list] | None = None,
) -> Any:
    """Resolve any Range/List nested inside a chosen ``List`` member.

    These knobs are *conditional* on the categorical draw, so they are
    registered only for the member Optuna actually picked.  Names are
    namespaced by member index (``api_config[1].rag.embedding_cfg``) so two
    members declaring the same path with different distributions never
    collide under one Optuna parameter name.

    The deep copy is required: ``search_space`` entries are extracted from the
    *original* template rather than the copy made in ``_sample_from_trial``,
    so ``param.values[idx]`` is the shared template object.  Mutating it in
    place would corrupt the template for every subsequent trial.

    *range_cache* is ``RFOptuna``'s ``id(Range) → value-set`` map, threaded
    through so an index-affecting ``Range`` reuses the values coverage
    enumeration already drew instead of drawing a second, incompatible set.
    """
    if isinstance(member, (Range, List)):
        # A search-space object nested directly as a choice. Sample it in place
        # rather than recursing, since it has no dotted path to write back to.
        member_name = f"{list_name}[{idx}]"
        return _suggest_value(
            trial,
            member_name,
            member,
            index_affecting=_is_index_affecting_path(member_name),
            range_cache=range_cache,
        )

    nested = _extract_search_space(member)
    if not nested:
        return member
    member = copy.deepcopy(member)
    for path, nested_param in nested:
        nested_name = f"{list_name}[{idx}].{path}"
        value = _suggest_value(
            trial,
            nested_name,
            nested_param,
            index_affecting=_is_index_affecting_path(nested_name),
            range_cache=range_cache,
        )
        _set_nested(member, path, value)
    return member


def _suggest_value(
    trial: optuna.Trial,
    name: str,
    param: Range | List,
    index_affecting: bool = False,
    range_cache: dict[int, list] | None = None,
) -> Any:
    """Use an Optuna trial to sample a single value for *param*.

    Maps ``Range`` → ``suggest_int`` / ``suggest_float`` and
    ``List`` → ``suggest_categorical``.

    When *index_affecting* is set, a ``Range`` becomes a categorical over the
    value set coverage enumeration already drew.  A continuous knob that
    selects a RAG index is not searchable in practice: the set of indexes has
    to be finite and known before any query runs, so the knob is discretised
    to the same values :meth:`RFOptuna.get_context_coverage_leaves`
    pre-builds.  ``Range`` itself no longer memoizes, so the value set is
    read from *range_cache* (keyed by ``id(param)``); when no cache is
    supplied it is drawn on the spot via ``param.sample(param.sample_n)``.
    Retrieval-only knobs keep their full continuous resolution.

    For a ``List`` of non-primitive members, any ``Range`` / ``List`` nested
    inside the chosen member is also registered, as a conditional parameter
    namespaced under ``{name}[{idx}].``.  See :func:`_sample_list_member`.
    """
    if isinstance(param, Range):
        if index_affecting:
            values = (
                range_cache.get(id(param))
                if range_cache is not None and id(param) in range_cache
                else None
            )
            if values is None:
                values = param.sample(param.sample_n)
                if range_cache is not None:
                    range_cache[id(param)] = values
            return trial.suggest_categorical(name, values)
        if param.dtype == "int":
            kwargs: dict[str, Any] = {}
            if param.step is not None:
                kwargs["step"] = int(param.step)
            if param.log:
                kwargs["log"] = True
            return trial.suggest_int(name, int(param.start), int(param.end), **kwargs)
        else:
            kwargs = {}
            if param.step is not None:
                kwargs["step"] = float(param.step)
            if param.log:
                kwargs["log"] = True
            return trial.suggest_float(name, float(param.start), float(param.end), **kwargs)
    elif isinstance(param, List):
        if all(isinstance(v, _PRIMITIVE_TYPES) for v in param.values):
            return trial.suggest_categorical(name, param.values)
        # Mixed primitive / dict / object choices (including ``None`` for an
        # optional component such as a reranker) -- always go through
        # ``_object_labels``, which handles every type uniformly and never calls
        # ``vars()`` on a value that lacks ``__dict__``.
        labels = _object_labels(param.values)
        if len(set(labels)) < len(labels):
            labels = [f"{lbl}#{i}" for i, lbl in enumerate(labels)]
        chosen_idx = labels.index(trial.suggest_categorical(name, labels))
        return _sample_list_member(
            trial, name, chosen_idx, param.values[chosen_idx],
            range_cache=range_cache,
        )
    raise AutoMLException(f"Unsupported search-space type: {type(param)}")


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a value inside a nested dict / ``_user_params`` object by dotted path."""
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if hasattr(obj, "_user_params"):
            obj = obj._user_params
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)

    last = parts[-1]
    if hasattr(obj, "_user_params"):
        obj = obj._user_params
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def _sample_from_trial(
    trial: optuna.Trial,
    search_space: list[tuple[str, Range | List]],
    config_template: Any,
    param_prefix: str = "",
    range_cache: dict[int, list] | None = None,
) -> Any:
    """Deep-copy *config_template* and replace each Range/List with a sampled value.

    *param_prefix* is prepended to Optuna parameter names (used for multi-template
    namespacing so identically-named params in different templates stay distinct).

    *range_cache* is ``RFOptuna``'s ``id(Range) → value-set`` map so
    index-affecting ``Range`` knobs reuse the values coverage enumeration drew
    rather than drawing a second, incompatible set.
    """
    config = copy.deepcopy(config_template)
    for dotted_path, param in search_space:
        optuna_name = f"{param_prefix}{dotted_path}" if param_prefix else dotted_path
        value = _suggest_value(
            trial,
            optuna_name,
            param,
            index_affecting=_is_index_affecting_path(dotted_path),
            range_cache=range_cache,
        )
        _set_nested(config, dotted_path, value)
    return config


def _sample_from_trial_multi(
    trial: optuna.Trial,
    config_templates: list[Any],
    search_spaces: list[list[tuple[str, Range | List]]],
    range_cache: dict[int, list] | None = None,
) -> Any:
    """Pick a template via Optuna categorical (if >1), then sample its search space.

    Single-template case is identical to ``_sample_from_trial`` (no extra
    categorical, no parameter prefix) for full backward compatibility.

    *range_cache* is threaded through so index-affecting ``Range`` knobs reuse
    the value set coverage enumeration already drew.

    Raises
    ------
    AutoMLException
        If any ``Range`` / ``List`` survives sampling.  This is the single
        choke point for every sampling entry point (``RFOptuna.get_runs`` and
        both callbacks' ``_maybe_suggest_replacement``), so the check cannot be
        bypassed.  ``get_runs`` runs before any worker or API spend, so an
        unreachable search-space entry fails at launch rather than after a
        full-cost run.
    """
    if len(config_templates) == 1:
        sampled = _sample_from_trial(
            trial, search_spaces[0], config_templates[0],
            range_cache=range_cache,
        )
    else:
        tidx = trial.suggest_categorical(
            "_config_template_idx", list(range(len(config_templates))),
        )
        sampled = _sample_from_trial(
            trial,
            search_spaces[tidx],
            config_templates[tidx],
            param_prefix=f"_t{tidx}.",
            range_cache=range_cache,
        )

    residual = _find_unsampled_params(sampled)
    if residual:
        raise AutoMLException(
            "RFOptuna could not register these search-space entries as Optuna "
            f"parameters: {residual}. They would be silently randomized outside "
            "Optuna's view. Move them into a dict, a config object exposing "
            "_user_params, or a dataclass field -- Range/List nested inside a "
            "plain list or tuple literal cannot be addressed."
        )
    return sampled


# ---------------------------------------------------------------------------
# Helpers to expand a sampled config template into a config leaf
# (mirrors the expansion in grid_search / random_search)
# ---------------------------------------------------------------------------


def _template_to_leaf_fit(config_obj: Any, trainer_type: str) -> dict[str, Any]:
    """Convert a sampled ``RFModelConfig`` into a flat config-leaf dict for the controller."""
    from rapidfireai.automl.random_search import recursive_expand_randomsearch

    peft_params = (
        {}
        if config_obj.peft_config is None
        else recursive_expand_randomsearch(config_obj.peft_config._user_params)
    )
    training_params = (
        {}
        if config_obj.training_args is None
        else recursive_expand_randomsearch(config_obj.training_args._user_params)
    )
    model_kwargs = (
        {}
        if config_obj.model_kwargs is None
        else recursive_expand_randomsearch(config_obj.model_kwargs)
    )
    ref_model_kwargs = (
        {}
        if config_obj.ref_model_kwargs is None
        else recursive_expand_randomsearch(config_obj.ref_model_kwargs)
    )
    reward_funcs = (
        {}
        if config_obj.reward_funcs is None
        else recursive_expand_randomsearch(config_obj.reward_funcs)
    )

    excluded_attrs = {
        "model_name",
        "tokenizer",
        "tokenizer_kwargs",
        "model_type",
        "model_kwargs",
        "peft_config",
        "training_args",
        "ref_model_name",
        "ref_model_type",
        "ref_model_kwargs",
        "reward_funcs",
        "num_gpus",
    }
    additional_kwargs = {
        k: v
        for k, v in config_obj.__dict__.items()
        if k not in excluded_attrs and v is not None
    }

    leaf: dict[str, Any] = {
        "trainer_type": trainer_type,
        "training_args": training_params,
        "peft_params": peft_params,
        "model_name": config_obj.model_name,
        "tokenizer": config_obj.tokenizer,
        "tokenizer_kwargs": config_obj.tokenizer_kwargs,
        "model_type": config_obj.model_type,
        "model_kwargs": model_kwargs,
        "additional_kwargs": additional_kwargs,
    }
    num_gpus = getattr(config_obj, "num_gpus", None)
    if num_gpus is not None:
        leaf["num_gpus"] = num_gpus

    if trainer_type == "DPO":
        leaf["ref_model_config"] = {
            "model_name": config_obj.ref_model_name,
            "model_type": config_obj.ref_model_type,
            "model_kwargs": ref_model_kwargs,
        }
    elif trainer_type == "GRPO":
        leaf["reward_funcs"] = reward_funcs

    return leaf


_PIPELINE_KEY_ALIASES = (
    "pipeline",
    "vllm_config",
    "api_config",
    "openai_config",
    "gemini_config",
)


def _template_to_leaf_evals(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a sampled evals config dict into a config-leaf dict for the controller.

    Mirrors :func:`grid_search._get_runs_evals` /
    :func:`random_search._get_runs_evals` by recognising any of the historical
    pipeline keys (``pipeline``, ``vllm_config``, ``api_config``,
    ``openai_config``, ``gemini_config``) and normalising the result to a
    single ``"pipeline"`` key -- the controller looks up ``config_leaf["pipeline"]``
    unconditionally, so omitting an alias here used to crash sharded evals
    runs that built their config with ``api_config=...``.
    """
    from rapidfireai.automl.random_search import recursive_expand_randomsearch

    pipeline_key = None
    for key in _PIPELINE_KEY_ALIASES:
        if key in config_dict:
            pipeline_key = key
            break

    if pipeline_key is None:
        return config_dict

    pipeline = config_dict[pipeline_key]
    pipeline_instance = recursive_expand_randomsearch(pipeline)

    additional = {
        k: recursive_expand_randomsearch(v)
        for k, v in config_dict.items()
        if k not in _PIPELINE_KEY_ALIASES and v is not None
    }

    return {"pipeline": pipeline_instance, **additional}


# ---------------------------------------------------------------------------
# RAG index coverage enumeration (build_all_indexes)
# ---------------------------------------------------------------------------

# Building a RAG index means chunking and embedding the whole corpus, so the
# up-front set has to stay small enough to be a deliberate choice.
_MAX_PREBUILT_CONTEXTS = 64

# Retry cap for drawing a replacement whose RAG index actually exists, in the
# spirit of RFRandomSearch's ``max_attempts = num_runs * 10``.
_MAX_REPLACEMENT_ATTEMPTS = 10


def _expand_for_index_coverage(
    item: Any,
    path: str = "",
    range_cache: dict[int, list] | None = None,
) -> Iterator[Any]:
    """Enumerate every variant of *item* that needs its own RAG index.

    Mirrors ``recursive_expand_gridsearch`` -- reconstructing config objects
    from their ``_user_params`` so the result is a real spec that can be hashed
    and built -- with one difference: knobs that do not feed the context hash
    are collapsed to a single representative value instead of being expanded.
    Without that, FiQA's ``search_cfg["k"]`` and ``reranker_cfg["top_n"]`` would
    turn 2 distinct indexes into 16 identical candidates.

    Kept local rather than parameterising the grid-search walker, so
    ``RFGridSearch`` behaviour is untouched.

    *range_cache* is ``RFOptuna``'s ``id(Range) → value-set`` map.  When a
    ``Range`` is enumerated, its drawn value set is stored in the cache so that
    :func:`_suggest_value` later reuses the exact same values for Optuna's
    categorical choices -- ``Range`` no longer memoizes, so this is the single
    place the value set is drawn.  Coverage enumeration runs before
    ``get_runs`` (see ``RFOptuna.get_context_coverage_leaves``), guaranteeing
    the cache is populated before any suggest call reads it.
    """
    from rapidfireai.automl.grid_search import _accepts_verbose

    if isinstance(item, Range):
        values = item.sample(item.sample_n)
        if range_cache is not None:
            range_cache[id(item)] = values
        yield from values if _is_index_affecting_path(path) else values[:1]
    elif isinstance(item, List):
        members = item.values if _is_index_affecting_path(path) else item.values[:1]
        for member in members:
            yield from _expand_for_index_coverage(member, path, range_cache)
    elif hasattr(item, "_user_params"):
        suppress = _accepts_verbose(item.__class__)
        for params in _expand_for_index_coverage(item._user_params, path, range_cache):
            if suppress:
                params = {**params, "verbose": False}
            yield item.__class__(**params)
    elif isinstance(item, dict):
        keys = list(item.keys())
        value_lists = [
            list(_expand_for_index_coverage(item[k], f"{path}.{k}" if path else str(k), range_cache))
            for k in keys
        ]
        for values in product(*value_lists):
            yield copy.deepcopy(dict(zip(keys, values, strict=False)))
    else:
        yield item


def _context_coverage_leaves(
    pipeline: Any,
    range_cache: dict[int, list] | None = None,
) -> list[dict[str, Any]]:
    """Return context-only config leaves covering every index *pipeline* can reach.

    Only the ``rag`` and ``prompt_manager`` subtrees are expanded.  The pipeline
    object itself is deliberately not reconstructed per combination: that would
    re-provision the MLflow gateway for each candidate, and the consumers need
    nothing else from it -- ``_collect_unique_contexts`` reads
    ``config_leaf["pipeline"]`` and then only ``.rag`` / ``.prompt_manager``,
    and ``_setup_context_generators`` only stamps ``experiment_name`` on those
    two objects.

    *range_cache* is populated here (see :func:`_expand_for_index_coverage`) so
    the value sets drawn during coverage enumeration are reused by suggest.
    """
    params = getattr(pipeline, "_user_params", None)
    if isinstance(params, dict):
        rag = params.get("rag")
        prompt_manager = params.get("prompt_manager")
    else:
        rag = getattr(pipeline, "rag", None)
        prompt_manager = getattr(pipeline, "prompt_manager", None)

    if rag is None and prompt_manager is None:
        return []

    rag_variants = (
        list(_expand_for_index_coverage(rag, "rag", range_cache)) if rag is not None else [None]
    )
    pm_variants = (
        list(_expand_for_index_coverage(prompt_manager, "prompt_manager", range_cache))
        if prompt_manager is not None
        else [None]
    )

    return [
        {"pipeline": SimpleNamespace(rag=rag_variant, prompt_manager=pm_variant)}
        for rag_variant in rag_variants
        for pm_variant in pm_variants
    ]


# ---------------------------------------------------------------------------
# Sampler / pruner factories
# ---------------------------------------------------------------------------

_SAMPLERS: dict[str, Any] = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "cmaes": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}

_PRUNERS: dict[str, Any] = {
    "median": lambda n_startup: optuna.pruners.MedianPruner(n_startup_trials=n_startup),
}


# ---------------------------------------------------------------------------
# Optuna callback implementations
# ---------------------------------------------------------------------------


class OptunaChunkCallback:
    """``ChunkCallback`` implementation for Optuna-based pruning in fit mode.

    Created by :meth:`RFOptuna.get_callback`.  After each training chunk the
    controller calls ``on_chunk_complete`` which reports metrics to Optuna
    and returns a ``RunDecision`` (continue / prune with optional replacement).

    Parameters
    ----------
    study : optuna.Study
    search_spaces : list[list[tuple[str, Range | List]]]
        Per-template search spaces.
    config_templates : list[Any]
        Original ``RFModelConfig`` template objects.
    trainer_type : str
        ``"SFT"`` / ``"DPO"`` / ``"GRPO"``.
    budget : int
        Max total trials (initial + replacements).
    objective_metric : str
        Primary metric key (e.g. ``"eval_loss"``).
    granularity : str
        ``"chunk"`` or ``"epoch"``.
    num_chunks : int or None
        Total chunks per epoch; required when ``granularity="epoch"``.
    objective_metrics : list[str] or None
        All metric keys (multi-objective).
    directions : list[str] or None
        ``"minimize"`` / ``"maximize"`` per metric.

    Methods
    -------
    on_chunk_complete(run_id, chunk_id, metrics) -> RunDecision
        Evaluate a run after a chunk.
    finalize(final_metrics)
        Tell remaining RUNNING trials their final objective values.
    _remap_pending_trial(db_run_id)
        Swap a placeholder key with the real DB run ID after replacement.
    """

    def __init__(
        self,
        study: optuna.Study,
        search_spaces: list[list[tuple[str, Range | List]]],
        config_templates: list[Any],
        trainer_type: str,
        budget: int,
        objective_metric: str,
        granularity: str = "chunk",
        num_chunks: int | None = None,
        *,
        objective_metrics: list[str] | None = None,
        directions: list[str] | None = None,
        range_cache: dict[int, list] | None = None,
    ):
        if granularity not in ("chunk", "epoch"):
            raise AutoMLException(
                f"granularity must be 'chunk' or 'epoch', got '{granularity}'"
            )
        if granularity == "epoch" and (num_chunks is None or num_chunks < 1):
            raise AutoMLException(
                "num_chunks must be a positive integer when granularity='epoch'"
            )

        self._study = study
        self._search_spaces = search_spaces
        self._config_templates = config_templates
        self._trainer_type = trainer_type
        self._budget = budget
        self._objective_metric = objective_metric
        self._objective_metrics = objective_metrics or [objective_metric]
        self._directions = directions or ["minimize"]
        self._is_multi_objective = len(self._objective_metrics) > 1
        self._granularity = granularity
        self._num_chunks = num_chunks
        self._range_cache = range_cache
        self._trials: dict[int, optuna.trial.Trial] = {}
        self._spawned = 0
        self._cumulative_step: dict[int, int] = {}
        self._chunks_since_last_eval: dict[int, int] = {}
        self._multi_intermediates: dict[int, dict[int, list[float]]] = {}
        self._pruned_run_ids: set[int] = set()
        # Logger for prune-check trace lines. Set by the controller via
        # set_logger() so the lines pick up the SafeLoggerAdapter's
        # [<experiment>:<name>] prefix and pass the dashboard log filter
        # (dispatcher.py:1096). Falls back to the module logger until then.
        self._rf_logger: logging.Logger | logging.LoggerAdapter | None = None

    def set_logger(self, logger: logging.Logger | logging.LoggerAdapter) -> None:
        """Install the controller's logger so prune-check lines reach the dashboard.

        The dashboard's ``get_experiment_logs`` endpoint only returns lines
        containing ``[<experiment_name>:`` or ``| <experiment_name> |``
        (dispatcher.py:1096). The controller's ``SafeLoggerAdapter`` adds that
        prefix; the module-level ``rapidfireai.automl.optuna`` logger does not,
        so without this call the prune-check lines are written to the file but
        hidden in the dashboard.
        """
        self._rf_logger = logger

    @property
    def _prune_log(self) -> logging.Logger | logging.LoggerAdapter:
        """Logger used for prune-check lines: the injected one if set, else module."""
        return self._rf_logger if self._rf_logger is not None else _log

    # -- bookkeeping kept by RFOptuna before handing off --

    def _set_initial_trials(self, trial_map: dict[int, optuna.trial.Trial], spawned: int) -> None:
        """Populate the ``run_id → trial`` mapping and set the spawned count."""
        self._trials.update(trial_map)
        self._spawned = spawned

    # -- ChunkCallback protocol --

    def register_runs(self, run_id_to_config: dict[int, dict[str, Any]]) -> None:
        """No-op — initial mapping is handled via ``_set_initial_trials``."""
        pass

    def on_chunk_complete(
        self,
        run_id: int,
        chunk_id: int,
        metrics: dict[str, Any],
    ) -> RunDecision:
        """Evaluate a run after a training chunk.

        Parameters
        ----------
        run_id : int
            DB run identifier.
        chunk_id : int
            Zero-based chunk index.
        metrics : dict[str, Any]
            Metric values (flat scalars, MLflow step histories, or
            dict-wrapped values).

        Returns
        -------
        RunDecision
        """
        trial = self._trials.get(run_id)
        if trial is None:
            return RunDecision(action="continue")

        if self._is_multi_objective:
            return self._on_chunk_complete_multi(run_id, chunk_id, metrics, trial)

        # Report one value per chunk at a monotonic, batch-size-independent
        # step (cumulative chunks completed, 0-indexed).  The anchor is a
        # per-run counter rather than the optimizer step, so trials with
        # different batch sizes remain comparable; it also keeps increasing
        # across epochs (chunk_id cycles 0..n-1 each epoch).  The reported
        # value is the last (most recent) entry in the metric history, i.e.
        # the metric state at the chunk boundary.
        step = self._cumulative_step.get(run_id, 0)
        self._cumulative_step[run_id] = step + 1
        metric_value = self._resolve_metric(metrics)
        if metric_value is None:
            return RunDecision(action="continue")
        trial.report(metric_value, step=step)

        if self._granularity == "epoch":
            self._chunks_since_last_eval[run_id] = (
                self._chunks_since_last_eval.get(run_id, 0) + 1
            )
            if self._chunks_since_last_eval[run_id] < self._num_chunks:
                return RunDecision(action="continue")
            self._chunks_since_last_eval[run_id] = 0

        if isinstance(self._study.pruner, optuna.pruners.NopPruner):
            return RunDecision(action="continue")

        if trial.should_prune() or self._should_prune_concurrent(trial):
            self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            replacement = self._maybe_suggest_replacement()
            return RunDecision(action="prune", replacement_config=replacement)

        return RunDecision(action="continue")

    def _on_chunk_complete_multi(
        self,
        run_id: int,
        chunk_id: int,
        metrics: dict[str, Any],
        trial: optuna.Trial,
    ) -> RunDecision:
        """Multi-objective variant of on_chunk_complete.

        Optuna's built-in pruners and ``trial.report()`` don't support
        multi-objective studies, so we track intermediate values ourselves
        and use Pareto-dominance-based pruning.
        """
        # Key intermediates by the cumulative chunks-completed counter (see
        # on_chunk_complete) so Pareto pruning compares trials at a monotonic,
        # batch-size-independent step that keeps increasing across epochs.
        step = self._cumulative_step.get(run_id, 0)
        self._cumulative_step[run_id] = step + 1
        values = _resolve_multi_objectives(metrics, self._objective_metrics)
        if values is None:
            return RunDecision(action="continue")

        intermediates = self._multi_intermediates.setdefault(run_id, {})
        intermediates[step] = values

        if self._granularity == "epoch":
            self._chunks_since_last_eval[run_id] = (
                self._chunks_since_last_eval.get(run_id, 0) + 1
            )
            if self._chunks_since_last_eval[run_id] < self._num_chunks:
                return RunDecision(action="continue")
            self._chunks_since_last_eval[run_id] = 0

        if self._should_prune_pareto(run_id, step):
            self._pruned_run_ids.add(run_id)
            self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            replacement = self._maybe_suggest_replacement()
            return RunDecision(action="prune", replacement_config=replacement)

        return RunDecision(action="continue")

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Tell all remaining RUNNING trials their final objective values.

        Parameters
        ----------
        final_metrics : dict[int, dict[str, Any]]
            ``run_id → final metrics dict``.
        """
        for run_id, trial in self._trials.items():
            if not isinstance(run_id, int):
                continue
            if _trial_state_from_storage(self._study, trial) == optuna.trial.TrialState.RUNNING:
                run_metrics = final_metrics.get(run_id, {})
                if self._is_multi_objective:
                    values = _resolve_multi_objectives(run_metrics, self._objective_metrics)
                    if values is not None:
                        self._study.tell(trial, values=values)
                    else:
                        self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    value = self._resolve_metric(run_metrics)
                    if value is not None:
                        self._study.tell(trial, values=value)
                    else:
                        self._study.tell(trial, state=optuna.trial.TrialState.FAIL)

    # -- internals --

    def _should_prune_pareto(self, run_id: int, step: int) -> bool:
        """Pareto-dominance pruning for multi-objective studies.

        A run is pruned if it is Pareto-dominated by more than half the
        *active* (non-pruned) peers at the current step — analogous to
        single-objective median pruning.  Already-pruned runs are excluded
        so their ghost values don't block every subsequent trial.
        """
        current_vals = self._multi_intermediates.get(run_id, {}).get(step)
        if current_vals is None:
            return False

        dominating_peers = 0
        total_peers = 0
        for other_id, other_steps in self._multi_intermediates.items():
            if other_id == run_id:
                continue
            if other_id in self._pruned_run_ids:
                continue
            if step not in other_steps:
                continue
            total_peers += 1
            if _pareto_dominates(other_steps[step], current_vals, self._directions):
                dominating_peers += 1

        if total_peers == 0:
            return False
        return dominating_peers > total_peers / 2

    def _should_prune_concurrent(self, trial: optuna.Trial) -> bool:
        """RapidFire's adapted median pruner for fit mode.

        Compares the current trial's best intermediate value (across all its
        reported steps) against the median of peer values at the current
        trial's latest step, across ALL trials (RUNNING + COMPLETE), and prunes
        if the current value is strictly worse than the median.

        Optuna's built-in ``MedianPruner`` only compares against ``COMPLETE``
        trials, but in RapidFire's concurrent chunk loop every trial stays
        ``RUNNING`` until ``finalize()``, so the built-in pruner never has
        reference data.  This method is the actual pruning mechanism selected
        by ``pruner="median"``; ``trial.should_prune()`` is also called but is
        a no-op until ``finalize()``.  NaN intermediate values are filtered out
        before the median comparison.  No startup-trial threshold or minimum
        step is applied, so a pipeline can be pruned after the first chunk.
        """
        all_frozen = self._study.get_trials(deepcopy=False)

        current = None
        for ft in all_frozen:
            if ft.number == trial.number:
                current = ft
                break
        if current is None or not current.intermediate_values:
            return False

        last_step = max(current.intermediate_values.keys())
        values = [v for v in current.intermediate_values.values() if not math.isnan(v)]
        if not values:
            return False

        minimize = self._study.direction == optuna.study.StudyDirection.MINIMIZE
        best_current = min(values) if minimize else max(values)

        peer_values = []
        for ft in all_frozen:
            if ft.number == trial.number:
                continue
            if last_step in ft.intermediate_values:
                v = ft.intermediate_values[last_step]
                if not math.isnan(v):
                    peer_values.append(v)

        direction = self._study.direction.name
        if not peer_values:
            _log_prune_compare(
                self._prune_log, trial.number, last_step, direction, best_current, [], None,
                prune=False, reason="no_peers_at_step",
            )
            return False

        median_val = statistics.median(peer_values)
        will_prune = best_current > median_val if minimize else best_current < median_val
        _log_prune_compare(
            self._prune_log, trial.number, last_step, direction, best_current, peer_values, median_val,
            prune=will_prune,
            reason="worse_than_median" if will_prune else "better_than_median",
        )
        return will_prune

    def _resolve_metric(self, metrics: dict[str, Any]) -> float | None:
        """Extract the objective metric value from a metrics dict.

        Supports both flat dicts (``{"eval_loss": 0.5}``) and MLflow-style
        histories (``{"eval_loss": [(step, value), ...]}``) by taking the
        last recorded value. If the primary objective is missing, tries aliases
        (e.g. ``eval_loss`` → ``train_loss``) so small SFT runs still finalize.
        """
        return _resolve_scalar_for_objective(metrics, self._objective_metric)

    def _maybe_suggest_replacement(self) -> dict[str, Any] | None:
        """Ask Optuna for a new trial and return a config leaf, or ``None`` if budget exhausted."""
        if self._spawned >= self._budget:
            return None

        new_trial = self._study.ask()
        config_obj = _sample_from_trial_multi(
            new_trial, self._config_templates, self._search_spaces,
            range_cache=self._range_cache,
        )
        leaf = _template_to_leaf_fit(config_obj, self._trainer_type)

        placeholder_id = f"_optuna_pending_{uuid.uuid4().hex[:8]}"
        self._trials[placeholder_id] = new_trial
        self._spawned += 1
        return leaf

    def _remap_pending_trial(self, db_run_id: int) -> None:
        """Replace a placeholder trial key with the real DB run ID after replacement."""
        pending = [k for k in self._trials if isinstance(k, str) and k.startswith("_optuna_pending_")]
        if pending:
            trial = self._trials.pop(pending[0])
            self._trials[db_run_id] = trial


class OptunaShardCallback:
    """``ShardCallback`` implementation for Optuna-based pruning in evals mode.

    Evals-mode counterpart of :class:`OptunaChunkCallback`.

    Parameters
    ----------
    study : optuna.Study
    search_spaces : list[list[tuple[str, Range | List]]]
        Per-template search spaces.
    config_templates : list[dict[str, Any]]
        Original evals config template dicts.
    budget : int
        Max total trials (initial + replacements).
    objective_metric : str
        Primary metric key.
    objective_metrics : list[str] or None
        All metric keys (multi-objective).
    directions : list[str] or None
        ``"minimize"`` / ``"maximize"`` per metric.

    Methods
    -------
    on_shard_complete(pipeline_id, shard_id, metrics) -> PipelineDecision
        Evaluate a pipeline after a shard.
    finalize(final_metrics)
        Tell remaining RUNNING trials their final objective values.
    set_context_feasibility(predicate)
        Install the controller's "is this config's RAG index built?" check.
    _remap_pending_trial(db_pipeline_id)
        Swap a placeholder key with the real DB pipeline ID.
    """

    def __init__(
        self,
        study: optuna.Study,
        search_spaces: list[list[tuple[str, Range | List]]],
        config_templates: list[dict[str, Any]],
        budget: int,
        objective_metric: str,
        *,
        objective_metrics: list[str] | None = None,
        directions: list[str] | None = None,
        range_cache: dict[int, list] | None = None,
    ):
        self._study = study
        self._search_spaces = search_spaces
        self._config_templates = config_templates
        self._budget = budget
        self._objective_metric = objective_metric
        self._objective_metrics = objective_metrics or [objective_metric]
        self._directions = directions or ["minimize"]
        self._is_multi_objective = len(self._objective_metrics) > 1
        self._range_cache = range_cache
        self._trials: dict[int, optuna.trial.Trial] = {}
        self._spawned = 0
        self._multi_intermediates: dict[int, dict[int, list[float]]] = {}
        self._pruned_run_ids: set[int] = set()
        self._context_feasibility: Any = None
        self._warned_context_narrowing = False
        # Logger for prune-check trace lines; see OptunaFitShardCallback.set_logger.
        self._rf_logger: logging.Logger | logging.LoggerAdapter | None = None

    def set_logger(self, logger: logging.Logger | logging.LoggerAdapter) -> None:
        """Install the controller's logger so prune-check lines reach the dashboard.

        See OptunaFitShardCallback.set_logger for the dashboard-filter rationale.
        """
        self._rf_logger = logger

    @property
    def _prune_log(self) -> logging.Logger | logging.LoggerAdapter:
        """Logger used for prune-check lines: the injected one if set, else module."""
        return self._rf_logger if self._rf_logger is not None else _log

    def _set_initial_trials(self, trial_map: dict[int, optuna.trial.Trial], spawned: int) -> None:
        """Populate the pipeline_id → trial mapping from the initial batch."""
        self._trials.update(trial_map)
        self._spawned = spawned

    # -- ShardCallback protocol --

    def register_pipelines(self, pipeline_id_to_config: dict[int, dict[str, Any]]) -> None:
        """No-op — initial mapping is handled via ``_set_initial_trials``."""
        pass

    def on_shard_complete(
        self,
        pipeline_id: int,
        shard_id: int,
        metrics: dict[str, Any],
    ) -> PipelineDecision:
        """Evaluate a pipeline after an evaluation shard.

        Parameters
        ----------
        pipeline_id : int
            DB pipeline identifier.
        shard_id : int
            Zero-based shard index.
        metrics : dict[str, Any]
            Cumulative aggregated metrics up to this shard.

        Returns
        -------
        PipelineDecision
        """
        trial = self._trials.get(pipeline_id)
        if trial is None:
            return PipelineDecision(action="continue")

        if self._is_multi_objective:
            values = _resolve_multi_objectives(metrics, self._objective_metrics)
            if values is None:
                return PipelineDecision(action="continue")
            intermediates = self._multi_intermediates.setdefault(pipeline_id, {})
            intermediates[shard_id] = values
            if self._should_prune_pareto(pipeline_id, shard_id):
                self._pruned_run_ids.add(pipeline_id)
                self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                replacement = self._maybe_suggest_replacement()
                return PipelineDecision(action="prune", replacement_config=replacement)
            return PipelineDecision(action="continue")

        metric_value = self._resolve_metric(metrics)
        if metric_value is None:
            return PipelineDecision(action="continue")

        trial.report(metric_value, step=shard_id)

        if isinstance(self._study.pruner, optuna.pruners.NopPruner):
            return PipelineDecision(action="continue")

        if trial.should_prune() or self._should_prune_concurrent(trial):
            self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            replacement = self._maybe_suggest_replacement()
            return PipelineDecision(action="prune", replacement_config=replacement)

        return PipelineDecision(action="continue")

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Tell all remaining RUNNING trials their final objective values.

        Parameters
        ----------
        final_metrics : dict[int, dict[str, Any]]
            ``pipeline_id → final metrics dict``.
        """
        for pipeline_id, trial in self._trials.items():
            if not isinstance(pipeline_id, int):
                continue
            if _trial_state_from_storage(self._study, trial) == optuna.trial.TrialState.RUNNING:
                pm = final_metrics.get(pipeline_id, {})
                if self._is_multi_objective:
                    values = _resolve_multi_objectives(pm, self._objective_metrics)
                    if values is not None:
                        self._study.tell(trial, values=values)
                    else:
                        self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    value = self._resolve_metric(pm)
                    if value is not None:
                        self._study.tell(trial, values=value)
                    else:
                        self._study.tell(trial, state=optuna.trial.TrialState.FAIL)

    # -- internals --

    def _should_prune_pareto(self, pipeline_id: int, step: int) -> bool:
        """Pareto-dominance pruning for multi-objective studies.

        Only compares against active (non-pruned) peers so ghost values
        from already-pruned pipelines don't block subsequent trials.
        """
        current_vals = self._multi_intermediates.get(pipeline_id, {}).get(step)
        if current_vals is None:
            return False

        dominating_peers = 0
        total_peers = 0
        for other_id, other_steps in self._multi_intermediates.items():
            if other_id == pipeline_id:
                continue
            if other_id in self._pruned_run_ids:
                continue
            if step not in other_steps:
                continue
            total_peers += 1
            if _pareto_dominates(other_steps[step], current_vals, self._directions):
                dominating_peers += 1

        if total_peers == 0:
            return False
        return dominating_peers > total_peers / 2

    def _should_prune_concurrent(self, trial: optuna.Trial) -> bool:
        """RapidFire's adapted median pruner for evals mode.

        Differs from the fit-mode (``OptunaChunkCallback``) version in two
        ways: it uses the value at the current trial's latest step rather than
        the best value across all steps, and a NaN current value prunes
        immediately (the fit-mode version filters NaN out of the median pool
        instead).  Otherwise the same adapted median comparison across ALL
        trials (RUNNING + COMPLETE) applies, with no startup-trial threshold
        or minimum step.
        """
        all_frozen = self._study.get_trials(deepcopy=False)

        current = None
        for ft in all_frozen:
            if ft.number == trial.number:
                current = ft
                break
        if current is None or not current.intermediate_values:
            return False

        last_step = max(current.intermediate_values.keys())
        current_value = current.intermediate_values[last_step]
        direction = self._study.direction.name
        if math.isnan(current_value):
            _log_prune_compare(
                self._prune_log, trial.number, last_step, direction, current_value, [], None,
                prune=True, reason="current_is_nan",
            )
            return True

        peer_values = []
        for ft in all_frozen:
            if ft.number == trial.number:
                continue
            if last_step in ft.intermediate_values:
                v = ft.intermediate_values[last_step]
                if not math.isnan(v):
                    peer_values.append(v)

        if not peer_values:
            _log_prune_compare(
                self._prune_log, trial.number, last_step, direction, current_value, [], None,
                prune=False, reason="no_peers_at_step",
            )
            return False

        median_val = statistics.median(peer_values)
        minimize = self._study.direction == optuna.study.StudyDirection.MINIMIZE
        will_prune = current_value > median_val if minimize else current_value < median_val
        _log_prune_compare(
            self._prune_log, trial.number, last_step, direction, current_value, peer_values, median_val,
            prune=will_prune,
            reason="worse_than_median" if will_prune else "better_than_median",
        )
        return will_prune

    def _resolve_metric(self, metrics: dict[str, Any]) -> float | None:
        """Extract the objective metric value from a metrics dict.

        Delegates to :func:`_resolve_scalar_for_objective`, which tries the
        primary key, registered aliases, and a case/underscore/whitespace-
        insensitive scan of all keys (e.g. ``"Eval Loss"`` → ``eval_loss``).
        """
        return _resolve_scalar_for_objective(metrics, self._objective_metric)

    def set_context_feasibility(self, predicate: Any) -> None:
        """Install a predicate deciding whether a config leaf's RAG index exists.

        Called by the evals controller with a closure over its live context
        cache.  Without it every suggestion is accepted, which is correct when
        ``build_all_indexes=True`` because every reachable index is already
        built.
        """
        self._context_feasibility = predicate

    def _maybe_suggest_replacement(self) -> dict[str, Any] | None:
        """Ask Optuna for a new trial and return an evals config leaf.

        Returns ``None`` when the budget is exhausted, or when every attempt
        needed a RAG index that was never built.  A rejected candidate is told
        ``FAIL`` and resampled: rejection happens before the leaf is returned,
        so it never reaches the database or the UI, and it does not consume
        budget -- only accepted suggestions increment ``_spawned``.
        """
        if self._spawned >= self._budget:
            return None

        for _ in range(_MAX_REPLACEMENT_ATTEMPTS):
            new_trial = self._study.ask()
            config_dict = _sample_from_trial_multi(
                new_trial, self._config_templates, self._search_spaces,
                range_cache=self._range_cache,
            )
            leaf = _template_to_leaf_evals(config_dict)

            if self._context_feasibility is not None and not self._context_feasibility(leaf):
                # No index for this config. Fail the trial rather than
                # completing it: FAIL keeps it out of best_trial and out of the
                # sampler's model, so TPE is not taught that this region scored
                # anything.
                self._study.tell(new_trial, state=optuna.trial.TrialState.FAIL)
                if not self._warned_context_narrowing:
                    self._warned_context_narrowing = True
                    warnings.warn(
                        "RFOptuna rejected a suggested config because its RAG index "
                        "was not built. With build_all_indexes=False only the indexes "
                        "needed by the initial configs exist, so replacements are "
                        "restricted to those. Set build_all_indexes=True to let "
                        "Optuna explore every index-affecting combination.",
                        stacklevel=2,
                    )
                continue

            placeholder_id = f"_optuna_pending_{uuid.uuid4().hex[:8]}"
            self._trials[placeholder_id] = new_trial
            self._spawned += 1
            return leaf

        warnings.warn(
            f"RFOptuna could not find a feasible replacement config in "
            f"{_MAX_REPLACEMENT_ATTEMPTS} attempts; every suggestion needed an "
            "unbuilt RAG index. Leaving the slot unused. Set "
            "build_all_indexes=True to make every combination available.",
            stacklevel=2,
        )
        return None

    def _remap_pending_trial(self, db_pipeline_id: int) -> None:
        """Replace a placeholder trial key with the real DB pipeline ID after replacement."""
        pending = [k for k in self._trials if isinstance(k, str) and k.startswith("_optuna_pending_")]
        if pending:
            trial = self._trials.pop(pending[0])
            self._trials[db_pipeline_id] = trial


# ---------------------------------------------------------------------------
# RFOptuna — user-facing AutoMLAlgorithm
# ---------------------------------------------------------------------------


class RFOptuna(AutoMLAlgorithm):
    """Optuna-powered hyperparameter search for RapidFire AI.

    Drop-in replacement for ``RFGridSearch`` / ``RFRandomSearch`` that uses
    Optuna's ask-and-tell API.  Supports single and multi-objective
    optimisation, adaptive pruning, and budget-controlled trial replacement.

    When a run is pruned (stopped early due to poor intermediate metrics),
    Optuna automatically generates a replacement config via ``study.ask()``
    so the GPU slot is reused with a better-informed suggestion.  This
    continues until ``budget`` total trials have been created.

    Parameters
    ----------
    configs :
        One or more config templates containing ``Range`` / ``List``
        search-space definitions.  Accepts a plain list, a ``List([...])``
        wrapper, or a single template.  When multiple templates are
        provided, Optuna treats the template choice as a categorical
        hyperparameter.

        A ``List`` of config objects (e.g.
        ``api_config=List([gemini_a, gemini_b])``) is supported: the member
        choice becomes a categorical, and any ``Range`` / ``List`` nested
        inside the chosen member becomes a conditional parameter named
        ``api_config[<idx>].<path>``.  Namespacing by member index keeps
        members with different distributions at the same path from colliding
        under one parameter name; the cost is that TPE treats
        ``api_config[0].*`` and ``api_config[1].*`` as distinct parameters and
        learns preferences separately per member (the same trade-off already
        made by the ``_t{idx}.`` prefix for multiple templates).
    trainer_type : str or None
        ``"SFT"`` / ``"DPO"`` / ``"GRPO"`` for fit mode, ``None`` for evals
        mode.
    n_initial : int
        Number of configs to generate up-front via ``study.ask()``.
    budget : int
        Maximum total trials (initial + replacements).  Clamped to
        ``max(budget, n_initial)``.  Set ``budget == n_initial`` to disable
        replacement.
    objective : str
        ``"minimize:eval_loss"`` or ``"maximize:accuracy"`` for
        single-objective.  ``"maximize:rougeL,maximize:bleu"``
        (comma-separated) for multi-objective.
    sampler : str
        ``"tpe"`` (default), ``"cmaes"``, or ``"random"``.
    pruner : str or None
        ``"median"`` (default) or ``None``.

        ``"median"`` selects RapidFire's *adapted* median pruner, not Optuna's
        stock ``MedianPruner``. Optuna's built-in pruners only compare against
        ``COMPLETE`` trials, but in RapidFire's concurrent loop every trial
        stays ``RUNNING`` until ``finalize()``, so the built-in pruner never
        has reference data. The adapted pruner (``_should_prune_concurrent``)
        instead compares the current trial's intermediate value against the
        median of all peer values at the same step, across ``RUNNING`` and
        ``COMPLETE`` trials. It applies no startup-trial threshold and no
        minimum step, so a pipeline can be pruned after the first shard/chunk.

        ``None`` disables pruning entirely: no trial is ever marked ``PRUNED``
        and no replacement is spawned.

        In multi-objective mode ``pruner`` is ignored and Pareto-dominance
        pruning (``_should_prune_pareto``) always applies.
    seed : int
        Seed for the algorithm's own stochastic state: every ``Range``
        generator, the global RNG used by ``List.sample()`` / fallback draws,
        and the Optuna study sampler.  This is the **single** seed governing
        the algorithm's reproducibility -- the run-level ``seed`` passed to
        ``run_evals`` / ``run_fit`` (and on to :meth:`get_runs` /
        :meth:`get_context_coverage_leaves`) is ignored for the algorithm's
        draws and only governs surrounding infrastructure (dataset sharding,
        etc.).  Defaults to 42 so ``RFOptuna()`` is reproducible out of the
        box; pass an explicit ``seed`` to vary which part of each ``Range``
        this run explores.
    granularity : str
        ``"chunk"`` (default) or ``"epoch"``.  Controls when pruning is
        evaluated in fit mode.  Ignored in evals mode.
    build_all_indexes : bool
        Evals mode only; ignored in fit mode.  When ``True`` (default), every
        RAG index the search space can reach is built up front, so any
        replacement Optuna suggests during the run already has its retriever
        available.  The cost is that indexes Optuna never visits are built too,
        which is the same index count ``RFGridSearch`` would build for the
        equivalent space.  When ``False``, only the indexes needed by the
        ``n_initial`` configs are built and a suggestion requiring a missing one
        is rejected and resampled -- cheaper up front, but it narrows the space
        Optuna can actually explore.

        Index-affecting ``Range`` knobs are discretised either way, since the
        set of indexes must be finite and known before any query runs: each is
        drawn down to ``Range.sample_n`` distinct values by ``sample(n)``.
        ``seed`` selects which part of the range this run explores.

    Methods
    -------
    get_runs(seed=42) -> list[dict]
        Create the Optuna study and sample ``n_initial`` config leaves.
        The ``seed`` argument is accepted for the controller contract but
        ignored -- the constructor ``seed`` governs all draws.
    get_callback(num_chunks=None) -> OptunaChunkCallback | OptunaShardCallback | None
        Return the callback wired to the study.  Call after ``get_runs()``.
    bind_initial_trials(ordered_ids)
        Map DB run/pipeline IDs to the Optuna trials from ``get_runs()``.
    get_context_coverage_leaves(seed=42) -> list[dict]
        Context-only leaves naming every RAG index the space can reach.
        The ``seed`` argument is accepted for the controller contract but
        ignored -- the constructor ``seed`` governs all draws.
    """

    def __init__(
        self,
        configs=None,
        trainer_type: str | None = None,
        n_initial: int = 16,
        budget: int = 40,
        objective: str = "minimize:eval_loss",
        sampler: str = "tpe",
        pruner: str | None = "median",
        seed: int = 42,
        granularity: str = "chunk",
        build_all_indexes: bool = True,
    ):
        if granularity not in ("chunk", "epoch"):
            raise AutoMLException(
                f"granularity must be 'chunk' or 'epoch', got '{granularity}'"
            )

        self.n_initial = n_initial
        self.budget = max(budget, n_initial)
        self.objective = objective
        self.sampler_name = sampler.lower()
        self.pruner_name = pruner.lower() if pruner else None
        self._seed = seed
        self._granularity = granularity
        self.build_all_indexes = build_all_indexes

        self._study: optuna.Study | None = None
        self._callback: OptunaChunkCallback | OptunaShardCallback | None = None
        self._config_templates: list[Any] = []
        self._search_spaces: list[list[tuple[str, Range | List]]] = []
        self._initial_trials: list[optuna.trial.Trial] = []
        # id(Range) -> value set drawn once during coverage enumeration and reused
        # by suggest. Range no longer memoizes, so RFOptuna owns this cache.
        self._range_value_cache: dict[int, list] = {}
        # Guards _seed_ranges so coverage (which now runs before get_runs in the
        # evals controller) and get_runs don't double-seed and reset the RNG.
        self._ranges_seeded = False

        # Parse objective(s) — supports single or comma-separated multi-objective
        objectives = [o.strip() for o in objective.split(",")]
        self._directions: list[str] = []
        self._objective_metrics: list[str] = []
        for obj_str in objectives:
            parts = obj_str.split(":", 1)
            if len(parts) != 2 or parts[0] not in ("minimize", "maximize"):
                raise AutoMLException(
                    f"Each objective must be 'minimize:<metric>' or "
                    f"'maximize:<metric>', got '{obj_str}'"
                )
            self._directions.append(parts[0])
            self._objective_metrics.append(parts[1])
        self._is_multi_objective = len(self._objective_metrics) > 1
        self._direction = self._directions[0]
        self._objective_metric = self._objective_metrics[0]

        super().__init__(
            configs=configs,
            trainer_type=trainer_type,
            num_runs=n_initial,
        )

    # -- AutoMLAlgorithm interface --

    def get_runs(self, seed: int = 42) -> list[dict[str, Any]]:
        """Create the Optuna study and sample ``n_initial`` config leaves.

        Parameters
        ----------
        seed : int
            Run-level seed (from ``run_evals`` / ``run_fit``).  Accepted for
            the controller contract but **ignored** -- the constructor
            ``seed`` governs the global RNG, every ``Range`` generator, and
            the Optuna study sampler.

        Returns
        -------
        list[dict[str, Any]]
            One config-leaf dict per initial trial.

        Raises
        ------
        AutoMLException
            If no config templates or no ``Range`` / ``List`` parameters
            are found.
        """
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")

        # The constructor seed is the single source for the algorithm's
        # stochastic state: the global RNG, every Range's generator, and the
        # Optuna study sampler. The run-level seed (passed in here) is ignored
        # for the algorithm's draws -- it only governs surrounding infra.
        random.seed(self._seed)

        # Seed every Range once. Coverage enumeration may already have seeded
        # (it runs before get_runs in the evals controller); the guard keeps
        # this idempotent so the RNG is not reset between coverage's draw and
        # suggest reading the cache.
        self._seed_all_ranges(self._seed)

        if self._is_multi_objective:
            self._study = optuna.create_study(
                directions=self._directions,
                sampler=self._create_sampler(self._seed),
            )
        else:
            self._study = optuna.create_study(
                direction=self._direction,
                sampler=self._create_sampler(self._seed),
                pruner=self._create_pruner(),
            )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if not self.configs:
            raise AutoMLException("At least one config template is required")

        self._config_templates = list(self.configs)
        self._search_spaces = [_extract_search_space(t) for t in self._config_templates]

        if not any(self._search_spaces):
            raise AutoMLException(
                "No Range or List parameters found in any config template. "
                "Use Range(...) and List([...]) to define the search space."
            )

        runs: list[dict[str, Any]] = []
        self._initial_trials = []

        for _ in range(self.n_initial):
            trial = self._study.ask()
            self._initial_trials.append(trial)

            sampled = _sample_from_trial_multi(
                trial, self._config_templates, self._search_spaces,
                range_cache=self._range_value_cache,
            )

            if self.mode == "fit":
                leaf = _template_to_leaf_fit(sampled, self.trainer_type)
            else:
                leaf = _template_to_leaf_evals(sampled)

            runs.append(leaf)

        return runs

    def get_callback(self, num_chunks: int | None = None) -> OptunaChunkCallback | OptunaShardCallback | None:
        """Return the callback for inter-chunk/shard pruning.  Call after ``get_runs()``.

        Parameters
        ----------
        num_chunks : int or None
            Total chunks per epoch.  Only used when ``granularity="epoch"``
            in fit mode so the callback can detect epoch boundaries.

        Returns
        -------
        OptunaChunkCallback or OptunaShardCallback or None
        """
        if self._study is None:
            return None

        if self.mode == "fit":
            cb = OptunaChunkCallback(
                study=self._study,
                search_spaces=self._search_spaces,
                config_templates=self._config_templates,
                trainer_type=self.trainer_type,
                budget=self.budget,
                objective_metric=self._objective_metric,
                granularity=self._granularity,
                num_chunks=num_chunks,
                objective_metrics=self._objective_metrics,
                directions=self._directions,
                range_cache=self._range_value_cache,
            )
        else:
            cb = OptunaShardCallback(
                study=self._study,
                search_spaces=self._search_spaces,
                config_templates=self._config_templates,
                budget=self.budget,
                objective_metric=self._objective_metric,
                objective_metrics=self._objective_metrics,
                directions=self._directions,
                range_cache=self._range_value_cache,
            )

        self._callback = cb
        return cb

    def get_context_coverage_leaves(self, seed: int = 42) -> list[dict[str, Any]]:
        """Return context-only leaves for every RAG index this search space can reach.

        The controller passes these to ``_setup_context_generators`` alongside
        the real config leaves, so an index is available for any config Optuna
        may later suggest -- not just the ``n_initial`` ones sampled up front.

        *seed* is the run-level seed (from ``run_evals``).  It is accepted for
        the controller contract but **ignored** -- the constructor ``seed``
        seeds every ``Range`` before coverage draws, so coverage and
        :meth:`get_runs` use the same values.

        Returns an empty list in fit mode or when ``build_all_indexes`` is
        ``False``.

        Raises
        ------
        AutoMLException
            If the space reaches more than ``_MAX_PREBUILT_CONTEXTS`` distinct
            candidates, rather than silently launching that many corpus builds.
        """
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")

        if self.mode == "fit" or not self.build_all_indexes:
            return []

        # Coverage enumeration draws from each Range's seeded generator, so the
        # ranges must be seeded before this draws. The evals controller calls
        # this before get_runs; seed here (idempotently) with the constructor
        # seed so coverage and suggest share the exact same value sets.
        self._seed_all_ranges(self._seed)

        leaves: list[dict[str, Any]] = []
        for template in self.configs or []:
            if not isinstance(template, dict):
                continue
            pipeline = next(
                (template[key] for key in _PIPELINE_KEY_ALIASES if key in template),
                None,
            )
            if pipeline is None:
                continue
            members = pipeline.values if isinstance(pipeline, List) else [pipeline]
            for member in members:
                leaves.extend(_context_coverage_leaves(member, self._range_value_cache))

        if len(leaves) > _MAX_PREBUILT_CONTEXTS:
            raise AutoMLException(
                f"build_all_indexes=True would pre-build up to {len(leaves)} RAG "
                f"contexts, above the limit of {_MAX_PREBUILT_CONTEXTS}. Narrow the "
                "index-affecting search space (embedding, text splitter, prompt "
                "manager), lower Range.sample_n on those knobs, or set "
                "build_all_indexes=False to build only what the initial configs "
                "need and let replacements be resampled."
            )

        if leaves:
            print(
                f"[RFOptuna] build_all_indexes=True: {len(leaves)} context "
                "candidate(s) enumerated from the search space; duplicates are "
                "deduplicated by context hash before any index is built."
            )
        return leaves

    def bind_initial_trials(self, ordered_ids: list[int]) -> None:
        """Map DB run/pipeline IDs to the Optuna trials from ``get_runs()``.

        Parameters
        ----------
        ordered_ids : list[int]
            DB IDs in the same order as the config leaves from ``get_runs()``.
        """
        if self._callback is None:
            return
        trial_map = {}
        for db_id, trial in zip(ordered_ids, self._initial_trials, strict=False):
            trial_map[db_id] = trial
        self._callback._set_initial_trials(trial_map, spawned=len(self._initial_trials))

    # -- internal helpers --

    def _seed_all_ranges(self, seed: int) -> None:
        """Seed every ``Range`` in the templates once.

        Idempotent: the first call wins and stamps *seed* onto every reachable
        ``Range``; later calls are no-ops. This lets ``get_context_coverage_leaves``
        (which the evals controller runs before ``get_runs``) and ``get_runs``
        both request seeding without one resetting the generator the other drew
        from. ``Range`` no longer memoizes, so reseeding would only matter for
        reproducibility of fallback draws when the cache is empty.
        """
        if self._ranges_seeded:
            return
        for template in self.configs:
            _seed_ranges(template, seed)
        self._ranges_seeded = True

    def _create_sampler(self, seed: int) -> optuna.samplers.BaseSampler:
        factory = _SAMPLERS.get(self.sampler_name)
        if factory is None:
            raise AutoMLException(
                f"Unknown sampler '{self.sampler_name}'. "
                f"Choose from: {', '.join(_SAMPLERS)}"
            )
        return factory(seed)

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        if self.pruner_name is None:
            return optuna.pruners.NopPruner()
        factory = _PRUNERS.get(self.pruner_name)
        if factory is None:
            raise AutoMLException(
                f"Unknown pruner '{self.pruner_name}'. "
                f"Choose from: {', '.join(_PRUNERS)}, or None"
            )
        n_startup = max(1, self.n_initial // 2)
        return factory(n_startup)
