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
import math
import statistics
import uuid
from dataclasses import fields, is_dataclass
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


def _resolve_scalar_for_objective(metrics: dict[str, Any], objective_metric: str) -> float | None:
    """Return a scalar for *objective_metric*, trying known aliases as fallbacks."""
    for key in _ordered_objective_keys(objective_metric):
        val = _float_from_logged_metric_value(metrics.get(key))
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


_PRIMITIVE_TYPES = (type(None), bool, int, float, str)


def _object_labels(objects: list[Any]) -> list[str]:
    """Build concise labels showing only the attributes that differ across *objects*.

    For example, two ``RecursiveCharacterTextSplitter`` instances that only
    differ in ``chunk_size`` produce::

        ["RecursiveCharacterTextSplitter(chunk_size=256)",
         "RecursiveCharacterTextSplitter(chunk_size=128)"]

    Shared defaults (``keep_separator``, ``strip_whitespace``, etc.) are omitted
    so the labels stay short and meaningful in Optuna trial output.
    """
    per_obj: list[tuple[str, dict[str, Any]]] = []
    for obj in objects:
        attrs = {}
        for key, val in sorted(vars(obj).items()):
            if isinstance(val, _PRIMITIVE_TYPES):
                attrs[key.lstrip("_")] = val
        per_obj.append((type(obj).__name__, attrs))

    all_keys: set[str] = set()
    for _, attrs in per_obj:
        all_keys.update(attrs)

    varying = {
        k for k in all_keys
        if len({attrs.get(k) for _, attrs in per_obj}) > 1
    }
    if not varying:
        varying = all_keys

    labels: list[str] = []
    for cls_name, attrs in per_obj:
        parts = [f"{k}={attrs[k]!r}" for k in sorted(varying) if k in attrs]
        labels.append(f"{cls_name}({', '.join(parts)})" if parts else repr(objects[len(labels)]))
    return labels


def _suggest_value(trial: optuna.Trial, name: str, param: Range | List) -> Any:
    """Use an Optuna trial to sample a single value for *param*.

    Maps ``Range`` → ``suggest_int`` / ``suggest_float`` and
    ``List`` → ``suggest_categorical``.
    """
    if isinstance(param, Range):
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
        if all(isinstance(v, (list, tuple, dict)) for v in param.values):
            labels = [repr(v) for v in param.values]
        else:
            labels = _object_labels(param.values)
        if len(set(labels)) < len(labels):
            labels = [f"{lbl}#{i}" for i, lbl in enumerate(labels)]
        chosen = trial.suggest_categorical(name, labels)
        return param.values[labels.index(chosen)]
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
) -> Any:
    """Deep-copy *config_template* and replace each Range/List with a sampled value.

    *param_prefix* is prepended to Optuna parameter names (used for multi-template
    namespacing so identically-named params in different templates stay distinct).
    """
    config = copy.deepcopy(config_template)
    for dotted_path, param in search_space:
        optuna_name = f"{param_prefix}{dotted_path}" if param_prefix else dotted_path
        value = _suggest_value(trial, optuna_name, param)
        _set_nested(config, dotted_path, value)
    return config


def _sample_from_trial_multi(
    trial: optuna.Trial,
    config_templates: list[Any],
    search_spaces: list[list[tuple[str, Range | List]]],
) -> Any:
    """Pick a template via Optuna categorical (if >1), then sample its search space.

    Single-template case is identical to ``_sample_from_trial`` (no extra
    categorical, no parameter prefix) for full backward compatibility.
    """
    if len(config_templates) == 1:
        return _sample_from_trial(trial, search_spaces[0], config_templates[0])

    tidx = trial.suggest_categorical(
        "_config_template_idx", list(range(len(config_templates))),
    )
    return _sample_from_trial(
        trial,
        search_spaces[tidx],
        config_templates[tidx],
        param_prefix=f"_t{tidx}.",
    )


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


def _template_to_leaf_evals(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a sampled evals config dict into a config-leaf dict for the controller."""
    from rapidfireai.automl.random_search import recursive_expand_randomsearch

    pipeline_key = None
    for key in ("pipeline", "vllm_config", "openai_config", "gemini_config"):
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
        if k not in {"pipeline", "vllm_config", "openai_config", "gemini_config"}
        and v is not None
    }

    return {"pipeline": pipeline_instance, **additional}


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
    "hyperband": lambda n_startup: optuna.pruners.HyperbandPruner(),
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
        self._trials: dict[int, optuna.trial.Trial] = {}
        self._spawned = 0
        self._last_reported_step: dict[int, int] = {}
        self._chunks_since_last_eval: dict[int, int] = {}
        self._multi_intermediates: dict[int, dict[int, list[float]]] = {}
        self._pruned_run_ids: set[int] = set()

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

        history = _resolve_metric_history(metrics, self._objective_metric)
        if not history:
            return RunDecision(action="continue")

        last_reported = self._last_reported_step.get(run_id, -1)
        for step, value in history:
            if step > last_reported:
                trial.report(value, step=step)
                self._last_reported_step[run_id] = step

        if self._granularity == "epoch":
            self._chunks_since_last_eval[run_id] = (
                self._chunks_since_last_eval.get(run_id, 0) + 1
            )
            if self._chunks_since_last_eval[run_id] < self._num_chunks:
                return RunDecision(action="continue")
            self._chunks_since_last_eval[run_id] = 0

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
        values = _resolve_multi_objectives(metrics, self._objective_metrics)
        if values is None:
            return RunDecision(action="continue")

        intermediates = self._multi_intermediates.setdefault(run_id, {})
        intermediates[chunk_id] = values

        if self._granularity == "epoch":
            self._chunks_since_last_eval[run_id] = (
                self._chunks_since_last_eval.get(run_id, 0) + 1
            )
            if self._chunks_since_last_eval[run_id] < self._num_chunks:
                return RunDecision(action="continue")
            self._chunks_since_last_eval[run_id] = 0

        if self._should_prune_pareto(run_id, chunk_id):
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
        """Concurrent-aware pruning that compares intermediate values across
        ALL trials (RUNNING + COMPLETE).

        Optuna's built-in pruners (MedianPruner, etc.) only compare against
        COMPLETE trials, but in RapidFire's concurrent chunk loop every trial
        stays RUNNING until ``finalize()``, so the built-in pruner never has
        reference data.  This method supplements ``trial.should_prune()`` by
        checking intermediate values from all peers regardless of state.
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

        if not peer_values:
            return False

        median_val = statistics.median(peer_values)
        if minimize:
            return best_current > median_val
        return best_current < median_val

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
    ):
        self._study = study
        self._search_spaces = search_spaces
        self._config_templates = config_templates
        self._budget = budget
        self._objective_metric = objective_metric
        self._objective_metrics = objective_metrics or [objective_metric]
        self._directions = directions or ["minimize"]
        self._is_multi_objective = len(self._objective_metrics) > 1
        self._trials: dict[int, optuna.trial.Trial] = {}
        self._spawned = 0
        self._multi_intermediates: dict[int, dict[int, list[float]]] = {}
        self._pruned_run_ids: set[int] = set()

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
        """Same concurrent-aware pruning as OptunaChunkCallback."""
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
        if math.isnan(current_value):
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
            return False

        median_val = statistics.median(peer_values)
        minimize = self._study.direction == optuna.study.StudyDirection.MINIMIZE
        if minimize:
            return current_value > median_val
        return current_value < median_val

    def _resolve_metric(self, metrics: dict[str, Any]) -> float | None:
        """Extract the objective metric value from a metrics dict."""
        direct = _resolve_scalar_for_objective(metrics, self._objective_metric)
        if direct is not None:
            return direct
        raw = metrics.get(self._objective_metric)
        if raw is None:
            for key, val in metrics.items():
                if isinstance(val, dict) and "value" in val:
                    if key.lower().replace("_", "").replace(" ", "") == self._objective_metric.lower().replace("_", "").replace(" ", ""):
                        return float(val["value"])
            return None
        if isinstance(raw, dict) and "value" in raw:
            return float(raw["value"])
        if isinstance(raw, (int, float)):
            return float(raw)
        return None

    def _maybe_suggest_replacement(self) -> dict[str, Any] | None:
        """Ask Optuna for a new trial and return an evals config leaf, or ``None`` if budget exhausted."""
        if self._spawned >= self._budget:
            return None

        new_trial = self._study.ask()
        config_dict = _sample_from_trial_multi(
            new_trial, self._config_templates, self._search_spaces,
        )
        leaf = _template_to_leaf_evals(config_dict)

        placeholder_id = f"_optuna_pending_{uuid.uuid4().hex[:8]}"
        self._trials[placeholder_id] = new_trial
        self._spawned += 1
        return leaf

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
        ``"median"`` (default), ``"hyperband"``, or ``None``.  Ignored for
        multi-objective studies.
    seed : int
        Random seed for the Optuna sampler.
    granularity : str
        ``"chunk"`` (default) or ``"epoch"``.  Controls when pruning is
        evaluated in fit mode.  Ignored in evals mode.

    Methods
    -------
    get_runs(seed=42) -> list[dict]
        Create the Optuna study and sample ``n_initial`` config leaves.
    get_callback(num_chunks=None) -> OptunaChunkCallback | OptunaShardCallback | None
        Return the callback wired to the study.  Call after ``get_runs()``.
    bind_initial_trials(ordered_ids)
        Map DB run/pipeline IDs to the Optuna trials from ``get_runs()``.
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

        self._study: optuna.Study | None = None
        self._callback: OptunaChunkCallback | OptunaShardCallback | None = None
        self._config_templates: list[Any] = []
        self._search_spaces: list[list[tuple[str, Range | List]]] = []
        self._initial_trials: list[optuna.trial.Trial] = []

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
            Fallback seed (instance-level ``seed`` takes precedence).

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

        effective_seed = self._seed if self._seed is not None else seed

        if self._is_multi_objective:
            self._study = optuna.create_study(
                directions=self._directions,
                sampler=self._create_sampler(effective_seed),
            )
        else:
            self._study = optuna.create_study(
                direction=self._direction,
                sampler=self._create_sampler(effective_seed),
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
            )

        self._callback = cb
        return cb

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
