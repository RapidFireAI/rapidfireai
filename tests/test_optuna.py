"""Tests for Optuna integration: search-space extraction, callbacks, RFOptuna.get_runs()."""

import copy
import types
from dataclasses import dataclass

import pytest
import optuna

from rapidfireai.automl.datatypes import List, Range
from rapidfireai.automl.optuna_search import (
    OptunaChunkCallback,
    OptunaShardCallback,
    RFOptuna,
    _extract_search_space,
    _resolve_scalar_for_objective,
    _sample_from_trial,
    _set_nested,
    _suggest_value,
    _template_to_leaf_evals,
    _trial_state_from_storage,
)
from rapidfireai.automl.callbacks import RunDecision, PipelineDecision


# ---------------------------------------------------------------------------
# Search-space extraction
# ---------------------------------------------------------------------------


class TestExtractSearchSpace:
    def test_flat_dict(self):
        template = {
            "learning_rate": Range(1e-6, 1e-3),
            "batch_size": List([4, 8, 16]),
            "epochs": 3,
        }
        space = _extract_search_space(template)
        assert len(space) == 2
        paths = {p for p, _ in space}
        assert paths == {"learning_rate", "batch_size"}

    def test_nested_dict(self):
        template = {
            "training_args": {
                "lr": Range(1e-5, 1e-3),
                "warmup": List([0, 100, 500]),
            },
            "model_name": "bert-base",
        }
        space = _extract_search_space(template)
        assert len(space) == 2
        paths = {p for p, _ in space}
        assert paths == {"training_args.lr", "training_args.warmup"}

    def test_object_with_user_params(self):
        class FakeConfig:
            def __init__(self, **kwargs):
                self._user_params = kwargs

        config = FakeConfig(lr=Range(1e-5, 1e-3), dropout=0.1, hidden=List([128, 256]))
        space = _extract_search_space(config)
        assert len(space) == 2
        paths = {p for p, _ in space}
        assert paths == {"lr", "hidden"}

    def test_empty_template(self):
        assert _extract_search_space({"a": 1, "b": "hello"}) == []

    def test_dataclass_wraps_nested_user_params(self):
        """RFModelConfig is a dataclass; Range/List under peft_config._user_params must be found."""

        class FakePeft:
            def __init__(self):
                self._user_params = {"lora_alpha": List([16, 32]), "r": 8}

        @dataclass
        class FakeModelConfig:
            model_name: str
            peft_config: object

        template = FakeModelConfig(model_name="gpt2", peft_config=FakePeft())
        space = _extract_search_space(template)
        assert len(space) == 1
        path, param = space[0]
        assert path == "peft_config.lora_alpha"
        assert isinstance(param, List)

    def test_range_log_and_step(self):
        r = Range(1e-6, 1e-3, log=True)
        assert r.log is True
        assert r.step is None
        r2 = Range(8, 64, step=8)
        assert r2.step == 8
        assert r2.log is False


def test_resolve_scalar_prefers_primary_key():
    assert _resolve_scalar_for_objective({"eval_loss": 1.0, "train_loss": 9.0}, "eval_loss") == 1.0


# ---------------------------------------------------------------------------
# Sampling from trial
# ---------------------------------------------------------------------------


class TestSuggestAndSample:
    def test_suggest_float_range(self):
        study = optuna.create_study()
        trial = study.ask()
        val = _suggest_value(trial, "lr", Range(0.001, 0.1))
        assert 0.001 <= val <= 0.1

    def test_suggest_int_range(self):
        study = optuna.create_study()
        trial = study.ask()
        val = _suggest_value(trial, "bs", Range(4, 32))
        assert 4 <= val <= 32
        assert isinstance(val, int)

    def test_suggest_categorical(self):
        study = optuna.create_study()
        trial = study.ask()
        val = _suggest_value(trial, "opt", List(["adam", "sgd", "adamw"]))
        assert val in ["adam", "sgd", "adamw"]

    def test_sample_from_trial_flat(self):
        template = {
            "lr": Range(0.0, 1.0),
            "name": "test",
            "bs": List([8, 16]),
        }
        space = _extract_search_space(template)
        study = optuna.create_study()
        trial = study.ask()
        result = _sample_from_trial(trial, space, template)

        assert isinstance(result["lr"], float)
        assert result["bs"] in [8, 16]
        assert result["name"] == "test"
        # Original template not mutated
        assert isinstance(template["lr"], Range)

    def test_sample_from_trial_nested(self):
        template = {
            "outer": {
                "inner": Range(0, 10),
                "fixed": "hello",
            }
        }
        space = _extract_search_space(template)
        study = optuna.create_study()
        trial = study.ask()
        result = _sample_from_trial(trial, space, template)
        assert isinstance(result["outer"]["inner"], int)
        assert result["outer"]["fixed"] == "hello"


class TestSetNested:
    def test_flat_dict(self):
        d = {"a": 1, "b": 2}
        _set_nested(d, "a", 99)
        assert d["a"] == 99

    def test_nested_dict(self):
        d = {"outer": {"inner": 1}}
        _set_nested(d, "outer.inner", 42)
        assert d["outer"]["inner"] == 42


# ---------------------------------------------------------------------------
# OptunaChunkCallback
# ---------------------------------------------------------------------------


def _fit_template_for_chunk_callback_tests() -> types.SimpleNamespace:
    """Minimal RFModelConfig-like object for tests that call ``_template_to_leaf_fit``."""
    return types.SimpleNamespace(
        model_name="m",
        tokenizer=None,
        tokenizer_kwargs=None,
        model_type="causal_lm",
        peft_config=None,
        training_args=None,
        model_kwargs=None,
        ref_model_kwargs=None,
        reward_funcs=None,
        ref_model_name=None,
        ref_model_type=None,
        num_gpus=None,
        formatting_func=None,
        compute_metrics=None,
        generation_config=None,
        lr=Range(0.0, 1.0),
    )


class TestOptunaChunkCallback:
    def _make_callback(self, direction="minimize", pruner=None):
        study = optuna.create_study(
            direction=direction,
            pruner=pruner or optuna.pruners.NopPruner(),
        )
        space = [("lr", Range(0.0, 1.0))]
        template = _fit_template_for_chunk_callback_tests()
        cb = OptunaChunkCallback(
            study=study,
            search_space=space,
            config_template=template,
            trainer_type="SFT",
            budget=5,
            objective_metric="eval_loss",
        )
        return cb, study

    def test_continue_when_no_prune(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        decision = cb.on_chunk_complete(1, 0, {"eval_loss": 0.5})
        assert decision.action == "continue"
        assert decision.replacement_config is None

    def test_continue_when_metric_missing(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        decision = cb.on_chunk_complete(1, 0, {"other_metric": 0.5})
        assert decision.action == "continue"

    def test_continue_when_run_unknown(self):
        cb, _ = self._make_callback()
        decision = cb.on_chunk_complete(999, 0, {"eval_loss": 0.5})
        assert decision.action == "continue"

    def test_resolve_metric_flat(self):
        cb, _ = self._make_callback()
        assert cb._resolve_metric({"eval_loss": 0.5}) == 0.5

    def test_resolve_metric_mlflow_history(self):
        cb, _ = self._make_callback()
        assert cb._resolve_metric({"eval_loss": [(0, 0.8), (1, 0.5)]}) == 0.5

    def test_resolve_metric_falls_back_when_eval_missing(self):
        """Tiny SFT jobs may log train_loss but never eval_loss."""
        cb, _ = self._make_callback()
        assert cb._resolve_metric({"train_loss": 2.5}) == 2.5
        assert cb._resolve_metric({"train_loss": [(0, 3.0), (4, 2.1)]}) == 2.1

    def test_finalize_tells_study(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        cb.finalize({1: {"eval_loss": 0.3}})
        assert _trial_state_from_storage(study, trial) == optuna.trial.TrialState.COMPLETE

    def test_finalize_fails_missing_metric(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        cb.finalize({1: {}})
        assert _trial_state_from_storage(study, trial) == optuna.trial.TrialState.FAIL

    def test_replacement_within_budget(self):
        cb, study = self._make_callback()
        cb._spawned = 3
        cb._budget = 5
        replacement = cb._maybe_suggest_replacement()
        assert replacement is not None
        assert isinstance(replacement, dict)
        assert cb._spawned == 4

    def test_no_replacement_over_budget(self):
        cb, study = self._make_callback()
        cb._spawned = 5
        cb._budget = 5
        replacement = cb._maybe_suggest_replacement()
        assert replacement is None

    def test_remap_pending_trial(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._trials["_optuna_pending_abc12345"] = trial
        cb._remap_pending_trial(42)
        assert 42 in cb._trials
        assert "_optuna_pending_abc12345" not in cb._trials


# ---------------------------------------------------------------------------
# OptunaShardCallback
# ---------------------------------------------------------------------------


class TestOptunaShardCallback:
    def _make_callback(self):
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.NopPruner(),
        )
        space = [("temperature", Range(0.0, 2.0))]
        template = {"pipeline": "fake", "temperature": Range(0.0, 2.0)}
        cb = OptunaShardCallback(
            study=study,
            search_space=space,
            config_template=template,
            budget=5,
            objective_metric="accuracy",
        )
        return cb, study

    def test_continue_decision(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({10: trial}, spawned=1)
        decision = cb.on_shard_complete(10, 0, {"accuracy": 0.85})
        assert decision.action == "continue"

    def test_resolve_metric_dict_with_value(self):
        cb, _ = self._make_callback()
        assert cb._resolve_metric({"accuracy": {"value": 0.9, "lower_bound": 0.85}}) == 0.9

    def test_resolve_metric_plain_float(self):
        cb, _ = self._make_callback()
        assert cb._resolve_metric({"accuracy": 0.75}) == 0.75

    def test_finalize(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({10: trial}, spawned=1)
        cb.finalize({10: {"accuracy": 0.92}})
        assert _trial_state_from_storage(study, trial) == optuna.trial.TrialState.COMPLETE


# ---------------------------------------------------------------------------
# RFOptuna class
# ---------------------------------------------------------------------------


class TestRFOptuna:
    def test_invalid_objective_format(self):
        with pytest.raises(Exception, match="objective must be"):
            RFOptuna(
                configs=[{"lr": Range(0.0, 1.0)}],
                objective="bad_format",
            )

    def test_invalid_sampler(self):
        rfopt = RFOptuna(
            configs=[{"lr": Range(0.0, 1.0)}],
            objective="minimize:loss",
            sampler="nonexistent",
        )
        with pytest.raises(Exception, match="Unknown sampler"):
            rfopt.get_runs(seed=42)

    def test_invalid_pruner(self):
        rfopt = RFOptuna(
            configs=[{"lr": Range(0.0, 1.0)}],
            objective="minimize:loss",
            pruner="nonexistent",
        )
        with pytest.raises(Exception, match="Unknown pruner"):
            rfopt.get_runs(seed=42)

    def test_get_runs_evals_mode(self):
        rfopt = RFOptuna(
            configs=[{"pipeline": "fake", "temperature": Range(0.0, 2.0)}],
            trainer_type=None,
            n_initial=5,
            budget=10,
            objective="maximize:accuracy",
            sampler="random",
            pruner=None,
            seed=42,
        )
        runs = rfopt.get_runs(seed=42)
        assert len(runs) == 5
        for run in runs:
            assert "pipeline" in run
            assert isinstance(run["temperature"], float)
            assert 0.0 <= run["temperature"] <= 2.0

    def test_get_runs_no_search_space_raises(self):
        rfopt = RFOptuna(
            configs=[{"fixed_param": 42}],
            objective="minimize:loss",
        )
        with pytest.raises(Exception, match="No Range or List"):
            rfopt.get_runs(seed=42)

    def test_get_callback_returns_shard_for_evals(self):
        rfopt = RFOptuna(
            configs=[{"pipeline": "fake", "temp": Range(0.0, 2.0)}],
            trainer_type=None,
            n_initial=3,
            budget=6,
            objective="maximize:acc",
            sampler="random",
            pruner=None,
        )
        rfopt.get_runs(seed=42)
        cb = rfopt.get_callback()
        assert isinstance(cb, OptunaShardCallback)

    def test_get_callback_returns_none_before_get_runs(self):
        rfopt = RFOptuna(
            configs=[{"pipeline": "fake", "temp": Range(0.0, 2.0)}],
            objective="maximize:acc",
        )
        assert rfopt.get_callback() is None

    def test_bind_initial_trials(self):
        rfopt = RFOptuna(
            configs=[{"pipeline": "fake", "temp": Range(0.0, 2.0)}],
            trainer_type=None,
            n_initial=3,
            budget=6,
            objective="maximize:acc",
            sampler="random",
            pruner=None,
        )
        rfopt.get_runs(seed=42)
        cb = rfopt.get_callback()

        rfopt.bind_initial_trials([100, 200, 300])
        assert 100 in cb._trials
        assert 200 in cb._trials
        assert 300 in cb._trials

    def test_budget_clamps_to_n_initial(self):
        rfopt = RFOptuna(
            configs=[{"x": Range(0.0, 1.0)}],
            n_initial=10,
            budget=5,
            objective="minimize:loss",
        )
        assert rfopt.budget == 10

    def test_deterministic_with_seed(self):
        def make_runs(seed):
            rfopt = RFOptuna(
                configs=[{"x": Range(0.0, 10.0), "y": List([1, 2, 3])}],
                n_initial=5,
                budget=5,
                objective="minimize:loss",
                sampler="tpe",
                pruner=None,
                seed=seed,
            )
            return rfopt.get_runs(seed=seed)

        runs_a = make_runs(42)
        runs_b = make_runs(42)
        for a, b in zip(runs_a, runs_b, strict=True):
            assert a["x"] == b["x"]
            assert a["y"] == b["y"]

    def test_base_class_get_callback_returns_none(self):
        from rapidfireai.automl import RFGridSearch
        gs = RFGridSearch(
            configs=[{"pipeline": "fake"}],
            trainer_type=None,
        )
        assert gs.get_callback() is None
