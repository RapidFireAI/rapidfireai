"""Optuna-based hyperparameter optimization integrated with RapidFire's chunk/shard loop."""

from __future__ import annotations

import copy
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
    """Parse a scalar from MLflow-style history or a plain numeric."""
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        last = raw[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            return float(last[1])
        return float(last)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _resolve_scalar_for_objective(metrics: dict[str, Any], objective_metric: str) -> float | None:
    """First matching key among *objective_metric* and known aliases."""
    for key in _ordered_objective_keys(objective_metric):
        val = _float_from_logged_metric_value(metrics.get(key))
        if val is not None:
            return val
    return None


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


def _suggest_value(trial: optuna.Trial, name: str, param: Range | List) -> Any:
    """Use an Optuna trial to sample a single value for *param*."""
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
        return trial.suggest_categorical(name, param.values)
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
) -> Any:
    """Deep-copy *config_template* and replace each Range/List with a sampled value."""
    config = copy.deepcopy(config_template)
    for dotted_path, param in search_space:
        value = _suggest_value(trial, dotted_path, param)
        _set_nested(config, dotted_path, value)
    return config


# ---------------------------------------------------------------------------
# Helpers to expand a sampled config template into a config leaf
# (mirrors the expansion in grid_search / random_search)
# ---------------------------------------------------------------------------


def _template_to_leaf_fit(config_obj: Any, trainer_type: str) -> dict[str, Any]:
    """Convert a sampled RFModelConfig into a flat config-leaf dict.

    This mirrors the leaf-building logic in ``RFGridSearch._get_runs_fit`` and
    ``RFRandomSearch._get_runs_fit`` but for a single already-concrete config.
    """
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
    """Convert a sampled evals config dict into a config-leaf dict."""
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

_SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "cmaes": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}

_PRUNERS = {
    "median": lambda: optuna.pruners.MedianPruner(),
    "hyperband": lambda: optuna.pruners.HyperbandPruner(),
}


# ---------------------------------------------------------------------------
# Optuna callback implementations
# ---------------------------------------------------------------------------


class OptunaChunkCallback:
    """ChunkCallback that uses an Optuna study for pruning and replacement in fit mode."""

    def __init__(
        self,
        study: optuna.Study,
        search_space: list[tuple[str, Range | List]],
        config_template: Any,
        trainer_type: str,
        budget: int,
        objective_metric: str,
    ):
        self._study = study
        self._search_space = search_space
        self._config_template = config_template
        self._trainer_type = trainer_type
        self._budget = budget
        self._objective_metric = objective_metric
        self._trials: dict[int, optuna.trial.Trial] = {}
        self._spawned = 0

    # -- bookkeeping kept by RFOptuna before handing off --

    def _set_initial_trials(self, trial_map: dict[int, optuna.trial.Trial], spawned: int) -> None:
        self._trials.update(trial_map)
        self._spawned = spawned

    # -- ChunkCallback protocol --

    def register_runs(self, run_id_to_config: dict[int, dict[str, Any]]) -> None:
        pass  # initial mapping handled via _set_initial_trials

    def on_chunk_complete(
        self,
        run_id: int,
        chunk_id: int,
        metrics: dict[str, Any],
    ) -> RunDecision:
        trial = self._trials.get(run_id)
        if trial is None:
            return RunDecision(action="continue")

        metric_value = self._resolve_metric(metrics)
        if metric_value is None:
            return RunDecision(action="continue")

        trial.report(metric_value, step=chunk_id)

        if trial.should_prune():
            self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            replacement = self._maybe_suggest_replacement()
            return RunDecision(action="prune", replacement_config=replacement)

        return RunDecision(action="continue")

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        for run_id, trial in self._trials.items():
            if not isinstance(run_id, int):
                continue
            if _trial_state_from_storage(self._study, trial) == optuna.trial.TrialState.RUNNING:
                run_metrics = final_metrics.get(run_id, {})
                value = self._resolve_metric(run_metrics)
                if value is not None:
                    self._study.tell(trial, values=value)
                else:
                    self._study.tell(trial, state=optuna.trial.TrialState.FAIL)

    # -- internals --

    def _resolve_metric(self, metrics: dict[str, Any]) -> float | None:
        """Extract the objective metric value from a metrics dict.

        Supports both flat dicts (``{"eval_loss": 0.5}``) and MLflow-style
        histories (``{"eval_loss": [(step, value), ...]}``) by taking the
        last recorded value. If the primary objective is missing, tries aliases
        (e.g. ``eval_loss`` → ``train_loss``) so small SFT runs still finalize.
        """
        return _resolve_scalar_for_objective(metrics, self._objective_metric)

    def _maybe_suggest_replacement(self) -> dict[str, Any] | None:
        if self._spawned >= self._budget:
            return None

        new_trial = self._study.ask()
        config_obj = _sample_from_trial(new_trial, self._search_space, self._config_template)
        leaf = _template_to_leaf_fit(config_obj, self._trainer_type)

        placeholder_id = f"_optuna_pending_{uuid.uuid4().hex[:8]}"
        self._trials[placeholder_id] = new_trial
        self._spawned += 1
        return leaf

    def _remap_pending_trial(self, db_run_id: int) -> None:
        """Replace a placeholder key with the real DB run_id after _create_models."""
        pending = [k for k in self._trials if isinstance(k, str) and k.startswith("_optuna_pending_")]
        if pending:
            trial = self._trials.pop(pending[0])
            self._trials[db_run_id] = trial


class OptunaShardCallback:
    """ShardCallback that uses an Optuna study for pruning and replacement in evals mode."""

    def __init__(
        self,
        study: optuna.Study,
        search_space: list[tuple[str, Range | List]],
        config_template: dict[str, Any],
        budget: int,
        objective_metric: str,
    ):
        self._study = study
        self._search_space = search_space
        self._config_template = config_template
        self._budget = budget
        self._objective_metric = objective_metric
        self._trials: dict[int, optuna.trial.Trial] = {}
        self._spawned = 0

    def _set_initial_trials(self, trial_map: dict[int, optuna.trial.Trial], spawned: int) -> None:
        self._trials.update(trial_map)
        self._spawned = spawned

    # -- ShardCallback protocol --

    def register_pipelines(self, pipeline_id_to_config: dict[int, dict[str, Any]]) -> None:
        pass

    def on_shard_complete(
        self,
        pipeline_id: int,
        shard_id: int,
        metrics: dict[str, Any],
    ) -> PipelineDecision:
        trial = self._trials.get(pipeline_id)
        if trial is None:
            return PipelineDecision(action="continue")

        metric_value = self._resolve_metric(metrics)
        if metric_value is None:
            return PipelineDecision(action="continue")

        trial.report(metric_value, step=shard_id)

        if trial.should_prune():
            self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            replacement = self._maybe_suggest_replacement()
            return PipelineDecision(action="prune", replacement_config=replacement)

        return PipelineDecision(action="continue")

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        for pipeline_id, trial in self._trials.items():
            if not isinstance(pipeline_id, int):
                continue
            if _trial_state_from_storage(self._study, trial) == optuna.trial.TrialState.RUNNING:
                pm = final_metrics.get(pipeline_id, {})
                value = self._resolve_metric(pm)
                if value is not None:
                    self._study.tell(trial, values=value)
                else:
                    self._study.tell(trial, state=optuna.trial.TrialState.FAIL)

    # -- internals --

    def _resolve_metric(self, metrics: dict[str, Any]) -> float | None:
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
        if self._spawned >= self._budget:
            return None

        new_trial = self._study.ask()
        config_dict = _sample_from_trial(new_trial, self._search_space, self._config_template)
        leaf = _template_to_leaf_evals(config_dict)

        placeholder_id = f"_optuna_pending_{uuid.uuid4().hex[:8]}"
        self._trials[placeholder_id] = new_trial
        self._spawned += 1
        return leaf

    def _remap_pending_trial(self, db_pipeline_id: int) -> None:
        pending = [k for k in self._trials if isinstance(k, str) and k.startswith("_optuna_pending_")]
        if pending:
            trial = self._trials.pop(pending[0])
            self._trials[db_pipeline_id] = trial


# ---------------------------------------------------------------------------
# RFOptuna — user-facing AutoMLAlgorithm
# ---------------------------------------------------------------------------


class RFOptuna(AutoMLAlgorithm):
    """Optuna-powered hyperparameter search for RapidFire AI.

    Works as a drop-in replacement for ``RFGridSearch`` / ``RFRandomSearch``.
    Initial configs are sampled via Optuna's sampler; during training the
    attached callback prunes underperforming runs and optionally replaces them
    with new Optuna suggestions.

    Args:
        configs: One or more config templates containing ``Range`` / ``List``
            search-space definitions (same format as grid/random search).
        trainer_type: ``"SFT"`` / ``"DPO"`` / ``"GRPO"`` for fit mode,
            ``None`` for evals mode.
        n_initial: Number of configs to generate up-front.
        budget: Maximum total configs across all replacements (including
            initial).  Set equal to ``n_initial`` to disable replacements.
        objective: Optimisation direction and metric name, e.g.
            ``"minimize:eval_loss"`` or ``"maximize:accuracy"``.
        sampler: Optuna sampler name — ``"tpe"``, ``"cmaes"``, or
            ``"random"``.
        pruner: Optuna pruner name — ``"median"``, ``"hyperband"``, or
            ``None`` to disable pruning.
        seed: Random seed for reproducibility.
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
    ):
        self.n_initial = n_initial
        self.budget = max(budget, n_initial)
        self.objective = objective
        self.sampler_name = sampler.lower()
        self.pruner_name = pruner.lower() if pruner else None
        self._seed = seed

        self._study: optuna.Study | None = None
        self._callback: OptunaChunkCallback | OptunaShardCallback | None = None
        self._search_space: list[tuple[str, Range | List]] = []
        self._config_template: Any = None
        self._initial_trials: list[optuna.trial.Trial] = []

        # Parse objective
        parts = objective.split(":", 1)
        if len(parts) != 2 or parts[0] not in ("minimize", "maximize"):
            raise AutoMLException(
                f"objective must be 'minimize:<metric>' or 'maximize:<metric>', got '{objective}'"
            )
        self._direction = parts[0]
        self._objective_metric = parts[1]

        super().__init__(
            configs=configs,
            trainer_type=trainer_type,
            num_runs=n_initial,
        )

    # -- AutoMLAlgorithm interface --

    def get_runs(self, seed: int = 42) -> list[dict[str, Any]]:
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")

        effective_seed = self._seed if self._seed is not None else seed

        self._study = optuna.create_study(
            direction=self._direction,
            sampler=self._create_sampler(effective_seed),
            pruner=self._create_pruner(),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if not self.configs:
            raise AutoMLException("At least one config template is required")

        self._config_template = self.configs[0]
        self._search_space = _extract_search_space(self._config_template)

        if not self._search_space:
            raise AutoMLException(
                "No Range or List parameters found in config template. "
                "Use Range(...) and List([...]) to define the search space."
            )

        runs: list[dict[str, Any]] = []
        self._initial_trials = []

        for _ in range(self.n_initial):
            trial = self._study.ask()
            self._initial_trials.append(trial)

            sampled = _sample_from_trial(trial, self._search_space, self._config_template)

            if self.mode == "fit":
                leaf = _template_to_leaf_fit(sampled, self.trainer_type)
            else:
                leaf = _template_to_leaf_evals(sampled)

            runs.append(leaf)

        return runs

    def get_callback(self) -> OptunaChunkCallback | OptunaShardCallback | None:
        """Return the callback for the controller to use between chunks/shards.

        Must be called **after** ``get_runs()``.
        """
        if self._study is None:
            return None

        if self.mode == "fit":
            cb = OptunaChunkCallback(
                study=self._study,
                search_space=self._search_space,
                config_template=self._config_template,
                trainer_type=self.trainer_type,
                budget=self.budget,
                objective_metric=self._objective_metric,
            )
        else:
            cb = OptunaShardCallback(
                study=self._study,
                search_space=self._search_space,
                config_template=self._config_template,
                budget=self.budget,
                objective_metric=self._objective_metric,
            )

        self._callback = cb
        return cb

    def bind_initial_trials(self, ordered_ids: list[int]) -> None:
        """Map the DB-assigned run/pipeline IDs to the Optuna trials created in get_runs().

        Called by the controller after ``_create_models`` / ``_register_pipelines``
        so the callback knows which trial corresponds to which run.
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
        return factory()
