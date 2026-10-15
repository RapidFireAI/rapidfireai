"""Tests for Optuna integration: search-space extraction, callbacks, RFOptuna.get_runs()."""

import copy
import types
from dataclasses import dataclass

import pytest
import optuna

from rapidfireai.automl.datatypes import List, Range
from rapidfireai.automl.grid_search import recursive_expand_gridsearch
from rapidfireai.automl.random_search import RFRandomSearch, recursive_expand_randomsearch
from rapidfireai.automl.optuna_search import (
    OptunaFitShardCallback,
    OptunaShardCallback,
    RFOptuna,
    _context_coverage_leaves,
    _extract_search_space,
    _find_unsampled_params,
    _is_index_affecting_path,
    _MAX_REPLACEMENT_ATTEMPTS,
    _object_labels,
    _resolve_metric_history,
    _resolve_scalar_for_objective,
    _sample_from_trial,
    _sample_from_trial_multi,
    _sample_list_member,
    _seed_ranges,
    _set_nested,
    _suggest_value,
    _template_to_leaf_evals,
    _trial_state_from_storage,
)
from rapidfireai.automl.callbacks import RunDecision, PipelineDecision
from rapidfireai.fit.utils.exceptions import AutoMLException


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


class TestSeedRanges:
    """``_seed_ranges`` stamps the run seed onto every reachable Range.

    ``List`` is a categorical of ordered choices and is not itself seeded;
    its members must still be walked so a Range nested inside a choice is
    not left on an unseeded generator.
    """

    def test_seeds_range_nested_in_list_member(self):
        nested = Range(0.0, 1.0)
        sibling = Range(10, 20)
        template = {"api_config": List([{"k": nested}, {"k": sibling}])}
        assert nested.seed is None
        assert sibling.seed is None

        _seed_ranges(template, 42)

        assert nested.seed == 42
        assert sibling.seed == 42

    def test_seeds_range_nested_in_list_of_config_objects(self):
        inner = Range(32, 128, step=32)
        template = {
            "api_config": List(
                [_FakeCfg(rag=_FakeCfg(embedding_cfg={"batch_size": inner}))]
            )
        }
        _seed_ranges(template, 7)
        assert inner.seed == 7

    def test_list_of_primitives_is_a_no_op(self):
        """A List of ordered primitives has no Range to seed."""
        template = {"k": List([5, 10, 15])}
        _seed_ranges(template, 42)  # must not raise


def test_resolve_scalar_prefers_primary_key():
    assert _resolve_scalar_for_objective({"eval_loss": 1.0, "train_loss": 9.0}, "eval_loss") == 1.0


def test_resolve_scalar_fuzzy_key_match():
    """Case/underscore/whitespace-insensitive key fallback catches MLflow
    variants (``"Eval Loss"``, ``"eval-loss"``) not in the alias table."""
    assert _resolve_scalar_for_objective({"Eval Loss": 0.5}, "eval_loss") == 0.5
    assert _resolve_scalar_for_objective({"eval-loss": {"value": 0.7}}, "eval_loss") == 0.7
    assert _resolve_scalar_for_objective({"Eval_Loss": 0.3, "other": 1.0}, "eval_loss") == 0.3
    assert _resolve_scalar_for_objective({"other": 1.0}, "eval_loss") is None


class TestResolveMetricHistory:
    def test_mlflow_style_history(self):
        metrics = {"eval_loss": [(0, 0.9), (10, 0.7), (20, 0.5)]}
        assert _resolve_metric_history(metrics, "eval_loss") == [(0, 0.9), (10, 0.7), (20, 0.5)]

    def test_plain_scalar(self):
        assert _resolve_metric_history({"eval_loss": 0.42}, "eval_loss") == [(0, 0.42)]

    def test_alias_fallback(self):
        metrics = {"train_loss": [(5, 1.0), (15, 0.8)]}
        assert _resolve_metric_history(metrics, "eval_loss") == [(5, 1.0), (15, 0.8)]

    def test_no_match(self):
        assert _resolve_metric_history({"other": 1.0}, "eval_loss") == []

    def test_unsorted_input_gets_sorted(self):
        metrics = {"eval_loss": [(20, 0.5), (0, 0.9), (10, 0.7)]}
        assert _resolve_metric_history(metrics, "eval_loss") == [(0, 0.9), (10, 0.7), (20, 0.5)]

    def test_bare_number_list(self):
        metrics = {"eval_loss": [0.9, 0.7, 0.5]}
        result = _resolve_metric_history(metrics, "eval_loss")
        assert result == [(0, 0.9), (1, 0.7), (2, 0.5)]


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
# OptunaFitShardCallback
# ---------------------------------------------------------------------------


def _fit_template_for_shard_callback_tests() -> types.SimpleNamespace:
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


class TestOptunaFitShardCallback:
    def _make_callback(self, direction="minimize", pruner=None):
        study = optuna.create_study(
            direction=direction,
            pruner=pruner or optuna.pruners.NopPruner(),
        )
        space = [("lr", Range(0.0, 1.0))]
        template = _fit_template_for_shard_callback_tests()
        cb = OptunaFitShardCallback(
            study=study,
            search_spaces=[space],
            config_templates=[template],
            trainer_type="SFT",
            budget=5,
            objective_metric="eval_loss",
        )
        return cb, study

    def test_continue_when_no_prune(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        decision = cb.on_shard_complete(1, 0, {"eval_loss": 0.5})
        assert decision.action == "continue"
        assert decision.replacement_config is None

    def test_continue_when_metric_missing(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        decision = cb.on_shard_complete(1, 0, {"other_metric": 0.5})
        assert decision.action == "continue"

    def test_continue_when_run_unknown(self):
        cb, _ = self._make_callback()
        decision = cb.on_shard_complete(999, 0, {"eval_loss": 0.5})
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

    @staticmethod
    def _get_intermediate_values(study, trial):
        """Retrieve intermediate_values from the frozen trial in storage."""
        for ft in study.get_trials(deepcopy=False):
            if ft.number == trial.number:
                return ft.intermediate_values
        return {}

    def test_reports_last_value_per_shard_at_cumulative_step(self):
        """on_shard_complete reports one value per shard (the last/most-recent
        value in the metric history) at a monotonic cumulative-shards-completed
        step, not at the optimizer step."""
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        metrics = {"eval_loss": [(0, 0.9), (5, 0.8), (10, 0.7)]}
        decision = cb.on_shard_complete(1, 0, metrics)
        assert decision.action == "continue"

        reported = self._get_intermediate_values(study, trial)
        assert reported == {0: 0.7}
        assert cb._cumulative_step[1] == 1

    def test_cumulative_across_shards(self):
        """Each shard reports at the next cumulative step using its last value."""
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        cb.on_shard_complete(1, 0, {"eval_loss": [(0, 0.9), (5, 0.8)]})
        assert cb._cumulative_step[1] == 1

        cb.on_shard_complete(1, 1, {"eval_loss": [(0, 0.9), (5, 0.8), (10, 0.6), (15, 0.5)]})
        assert cb._cumulative_step[1] == 2

        reported = self._get_intermediate_values(study, trial)
        assert reported == {0: 0.8, 1: 0.5}

    def test_flat_scalar_reports_at_step_zero(self):
        """A flat scalar metric gets reported at step 0."""
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        cb.on_shard_complete(1, 0, {"eval_loss": 0.5})
        reported = self._get_intermediate_values(study, trial)
        assert reported == {0: 0.5}

    def test_remap_pending_trial(self):
        cb, study = self._make_callback()
        trial = study.ask()
        cb._trials["_optuna_pending_abc12345"] = trial
        cb._remap_pending_trial(42)
        assert 42 in cb._trials
        assert "_optuna_pending_abc12345" not in cb._trials


# ---------------------------------------------------------------------------
# OptunaFitShardCallback — epoch granularity
# ---------------------------------------------------------------------------


class TestOptunaFitShardCallbackEpochGranularity:
    """Tests for granularity='epoch': decisions only fire at epoch boundaries."""

    NUM_SHARDS = 4

    def _make_callback(self, direction="minimize", pruner=None):
        study = optuna.create_study(
            direction=direction,
            pruner=pruner or optuna.pruners.NopPruner(),
        )
        space = [("lr", Range(0.0, 1.0))]
        template = _fit_template_for_shard_callback_tests()
        cb = OptunaFitShardCallback(
            study=study,
            search_spaces=[space],
            config_templates=[template],
            trainer_type="SFT",
            budget=5,
            objective_metric="eval_loss",
            granularity="epoch",
            num_shards=self.NUM_SHARDS,
        )
        return cb, study

    def test_defers_decision_until_epoch_boundary(self):
        """Shards 0-2 should always continue; shard 3 (4th) is the epoch boundary."""
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        for shard_id in range(self.NUM_SHARDS - 1):
            decision = cb.on_shard_complete(1, shard_id, {"eval_loss": 0.9 - shard_id * 0.1})
            assert decision.action == "continue", f"expected continue at shard {shard_id}"

        decision = cb.on_shard_complete(1, self.NUM_SHARDS - 1, {"eval_loss": 0.5})
        assert decision.action == "continue"

    def test_prune_fires_at_epoch_boundary(self):
        pruner = optuna.pruners.ThresholdPruner(upper=0.1)
        cb, study = self._make_callback(pruner=pruner)
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        for shard_id in range(self.NUM_SHARDS - 1):
            decision = cb.on_shard_complete(1, shard_id, {"eval_loss": 5.0})
            assert decision.action == "continue"

        decision = cb.on_shard_complete(1, self.NUM_SHARDS - 1, {"eval_loss": 5.0})
        assert decision.action == "prune"

    def test_counter_resets_after_epoch(self):
        """After one epoch completes, the next epoch should count from 0 again."""
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        for shard_id in range(self.NUM_SHARDS):
            cb.on_shard_complete(1, shard_id, {"eval_loss": 0.5})
        assert cb._shards_since_last_eval[1] == 0

        for shard_id in range(self.NUM_SHARDS - 1):
            decision = cb.on_shard_complete(1, shard_id, {"eval_loss": 0.4})
            assert decision.action == "continue"

    def test_metrics_still_reported_every_shard(self):
        """Even with epoch granularity, one intermediate value (the last in the
        shard's history) is reported to Optuna on every shard at the cumulative
        shards-completed step so the pruner has visibility at shard boundaries."""
        cb, study = self._make_callback()
        trial = study.ask()
        cb._set_initial_trials({1: trial}, spawned=1)

        cb.on_shard_complete(1, 0, {"eval_loss": [(0, 0.9), (5, 0.8)]})
        cb.on_shard_complete(1, 1, {"eval_loss": [(0, 0.9), (5, 0.8), (10, 0.7)]})

        reported = {}
        for ft in study.get_trials(deepcopy=False):
            if ft.number == trial.number:
                reported = ft.intermediate_values
        assert reported == {0: 0.8, 1: 0.7}

    def test_independent_tracking_per_run(self):
        cb, study = self._make_callback()
        t1 = study.ask()
        t2 = study.ask()
        cb._set_initial_trials({1: t1, 2: t2}, spawned=2)

        cb.on_shard_complete(1, 0, {"eval_loss": 0.5})
        cb.on_shard_complete(1, 1, {"eval_loss": 0.4})
        cb.on_shard_complete(2, 0, {"eval_loss": 0.6})

        assert cb._shards_since_last_eval[1] == 2
        assert cb._shards_since_last_eval[2] == 1

    def test_invalid_granularity_rejected(self):
        study = optuna.create_study()
        with pytest.raises(Exception, match="granularity"):
            OptunaFitShardCallback(
                study=study,
                search_spaces=[[("x", Range(0.0, 1.0))]],
                config_templates=[{"x": Range(0.0, 1.0)}],
                trainer_type="SFT",
                budget=5,
                objective_metric="loss",
                granularity="step",
                num_shards=4,
            )

    def test_epoch_granularity_requires_num_shards(self):
        study = optuna.create_study()
        with pytest.raises(Exception, match="num_shards"):
            OptunaFitShardCallback(
                study=study,
                search_spaces=[[("x", Range(0.0, 1.0))]],
                config_templates=[{"x": Range(0.0, 1.0)}],
                trainer_type="SFT",
                budget=5,
                objective_metric="loss",
                granularity="epoch",
                num_shards=None,
            )


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
            search_spaces=[space],
            config_templates=[template],
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
# RF-OPT-02: pruner=None short-circuits pruning; pruner="median" prunes
# ---------------------------------------------------------------------------


class TestPrunerNopShortCircuit:
    """Regression tests for RF-OPT-02.

    ``pruner=None`` must disable pruning entirely (the ``NopPruner`` short-
    circuits before ``_should_prune_concurrent`` runs).  ``pruner="median"``
    must still prune via the adapted median pruner when the current trial is
    worse than the peer median.
    """

    @staticmethod
    def _make_shard_callback(pruner):
        study = optuna.create_study(direction="minimize", pruner=pruner)
        space = [("lr", Range(0.0, 1.0))]
        template = _fit_template_for_shard_callback_tests()
        cb = OptunaFitShardCallback(
            study=study,
            search_spaces=[space],
            config_templates=[template],
            trainer_type="SFT",
            budget=5,
            objective_metric="eval_loss",
        )
        return cb, study

    def test_pruner_none_never_prunes(self):
        """NopPruner short-circuits even when _should_prune_concurrent would fire."""
        cb, study = self._make_shard_callback(optuna.pruners.NopPruner())
        peer = study.ask()
        current = study.ask()
        cb._set_initial_trials({1: peer, 2: current}, spawned=2)

        # Peer reports a good (low) loss at step 0; current reports a bad
        # (high) loss at the same step, so _should_prune_concurrent would
        # return True (0.9 > median([0.2]) == 0.2) without the short-circuit.
        cb.on_shard_complete(1, 0, {"eval_loss": 0.2})
        decision = cb.on_shard_complete(2, 0, {"eval_loss": 0.9})

        assert decision.action == "continue"
        assert decision.replacement_config is None
        assert sum(
            t.state == optuna.trial.TrialState.PRUNED for t in study.trials
        ) == 0

    def test_pruner_median_prunes(self):
        """pruner='median' prunes the current trial when it is worse than the peer median."""
        cb, study = self._make_shard_callback(optuna.pruners.MedianPruner())
        peer = study.ask()
        current = study.ask()
        cb._set_initial_trials({1: peer, 2: current}, spawned=2)

        cb.on_shard_complete(1, 0, {"eval_loss": 0.2})
        decision = cb.on_shard_complete(2, 0, {"eval_loss": 0.9})

        assert decision.action == "prune"
        assert decision.replacement_config is not None
        assert _trial_state_from_storage(study, current) == optuna.trial.TrialState.PRUNED


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
            )
            return rfopt.get_runs(seed=seed)

        runs_a = make_runs(42)
        runs_b = make_runs(42)
        for a, b in zip(runs_a, runs_b, strict=True):
            assert a["x"] == b["x"]
            assert a["y"] == b["y"]

    def test_constructor_accepts_seed(self):
        """RFOptuna carries a constructor seed that governs the algorithm's
        stochastic state (Range draws, global RNG, Optuna sampler)."""
        rfopt = RFOptuna(
            configs=[{"x": Range(0.0, 1.0)}],
            objective="minimize:loss",
            seed=42,
        )
        assert rfopt._seed == 42

    def test_constructor_seed_governs_range_draws(self):
        """The constructor seed governs Range draws; the run-level seed passed
        to get_runs is ignored for the algorithm's draws."""
        template = [{"x": Range(0.0, 100.0)}]

        def runs(ctor_seed, run_seed):
            rfopt = RFOptuna(
                configs=template,
                n_initial=4,
                budget=4,
                objective="minimize:loss",
                sampler="random",
                pruner=None,
                seed=ctor_seed,
            )
            return [run["x"] for run in rfopt.get_runs(seed=run_seed)]

        # Same constructor seed -> same draws, regardless of the run-level seed.
        assert runs(42, 42) == runs(42, 7)
        # Different constructor seed -> different draws, regardless of run-level.
        assert runs(42, 42) != runs(7, 42)

    def test_base_class_get_callback_returns_none(self):
        from rapidfireai.automl import RFGridSearch
        gs = RFGridSearch(
            configs=[{"pipeline": "fake"}],
            trainer_type=None,
        )
        assert gs.get_callback() is None

    def test_invalid_granularity(self):
        with pytest.raises(Exception, match="granularity"):
            RFOptuna(
                configs=[{"lr": Range(0.0, 1.0)}],
                objective="minimize:loss",
                granularity="step",
            )

    def test_granularity_epoch_stored_on_rfoptuna(self):
        rfopt = RFOptuna(
            configs=[{"pipeline": "fake", "temp": Range(0.0, 2.0)}],
            trainer_type=None,
            n_initial=2,
            budget=4,
            objective="minimize:eval_loss",
            sampler="random",
            pruner=None,
            granularity="epoch",
        )
        assert rfopt._granularity == "epoch"

    def test_granularity_defaults_to_shard_on_rfoptuna(self):
        rfopt = RFOptuna(
            configs=[{"pipeline": "fake", "temp": Range(0.0, 2.0)}],
            trainer_type=None,
            n_initial=2,
            budget=4,
            objective="minimize:eval_loss",
            sampler="random",
            pruner=None,
        )
        assert rfopt._granularity == "shard"


# ---------------------------------------------------------------------------
# Multi-template support
# ---------------------------------------------------------------------------


class TestMultiTemplate:
    """Verify RFOptuna correctly handles multiple config templates."""

    def test_sample_from_trial_multi_single_template(self):
        """Single template: behaves identically to _sample_from_trial."""
        template = {"lr": Range(0.0, 1.0), "fixed": "hello"}
        space = _extract_search_space(template)
        study = optuna.create_study()
        trial = study.ask()
        result = _sample_from_trial_multi(trial, [template], [space])
        assert isinstance(result["lr"], float)
        assert result["fixed"] == "hello"
        # No _config_template_idx categorical when single template
        assert "_config_template_idx" not in trial.params

    def test_sample_from_trial_multi_two_templates(self):
        """Two templates: Optuna picks one via categorical, samples its space."""
        t0 = {"lr": Range(0.0, 0.1), "model": "small"}
        t1 = {"dropout": Range(0.0, 0.5), "model": "large"}
        spaces = [_extract_search_space(t0), _extract_search_space(t1)]

        study = optuna.create_study()
        trial = study.ask()
        result = _sample_from_trial_multi(trial, [t0, t1], spaces)

        assert "_config_template_idx" in trial.params
        tidx = trial.params["_config_template_idx"]
        assert tidx in (0, 1)

        if tidx == 0:
            assert isinstance(result["lr"], float)
            assert result["model"] == "small"
        else:
            assert isinstance(result["dropout"], float)
            assert result["model"] == "large"

    def test_get_runs_evals_multi_template(self):
        t0 = {"pipeline": "pipe_a", "temperature": Range(0.0, 1.0)}
        t1 = {"pipeline": "pipe_b", "top_k": Range(1, 50)}

        rfopt = RFOptuna(
            configs=[t0, t1],
            trainer_type=None,
            n_initial=6,
            budget=10,
            objective="maximize:accuracy",
            sampler="random",
            pruner=None,
        )
        runs = rfopt.get_runs(seed=42)
        assert len(runs) == 6
        for run in runs:
            assert "pipeline" in run

    def test_get_runs_evals_list_wrapper(self):
        """List([t1, t2]) syntax works the same as [t1, t2]."""
        t0 = {"pipeline": "a", "x": Range(0.0, 1.0)}
        t1 = {"pipeline": "b", "y": Range(0.0, 1.0)}

        rfopt = RFOptuna(
            configs=List([t0, t1]),
            trainer_type=None,
            n_initial=4,
            budget=8,
            objective="maximize:score",
            sampler="random",
            pruner=None,
        )
        runs = rfopt.get_runs(seed=7)
        assert len(runs) == 4

    def test_callback_replacement_multi_template(self):
        """Replacement configs can come from any template."""
        t0 = {"pipeline": "a", "temperature": Range(0.0, 2.0)}
        t1 = {"pipeline": "b", "top_k": Range(1, 50)}
        spaces = [_extract_search_space(t0), _extract_search_space(t1)]

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.NopPruner(),
        )
        cb = OptunaShardCallback(
            study=study,
            search_spaces=spaces,
            config_templates=[t0, t1],
            budget=5,
            objective_metric="accuracy",
        )
        cb._spawned = 2
        replacement = cb._maybe_suggest_replacement()
        assert replacement is not None
        assert isinstance(replacement, dict)
        assert cb._spawned == 3


# ---------------------------------------------------------------------------
# Categorical-choice labelling and evals-leaf pipeline-key normalisation
# (regression tests for issues found while running RFOptuna on RAG configs
# whose List(...) categoricals mixed dicts, None, and lists, or whose evals
# template used ``api_config`` instead of ``vllm_config``/``pipeline``).
# ---------------------------------------------------------------------------


class TestObjectLabels:
    def test_dict_only_choices(self):
        labels = _object_labels(
            [{"type": "similarity", "k": k} for k in (5, 10, 20)]
        )
        assert labels == ["cfg(k=5)", "cfg(k=10)", "cfg(k=20)"]

    def test_none_plus_dict_choices(self):
        labels = _object_labels([None, {"class": str, "top_n": 5}])
        assert labels[0] == "None"
        assert labels[1].startswith("cfg(")
        assert "top_n=5" in labels[1]

    def test_list_of_lists_choices(self):
        choices = [["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]]
        labels = _object_labels(choices)
        assert labels == [repr(c) for c in choices]

    def test_object_choices_with_dict(self):
        class Splitter:
            def __init__(self, chunk_size, overlap):
                self.chunk_size = chunk_size
                self._overlap = overlap

        labels = _object_labels([Splitter(256, 32), Splitter(512, 32)])
        assert labels == ["Splitter(chunk_size=256)", "Splitter(chunk_size=512)"]

    def test_mixed_obj_dict_none_list(self):
        class TS:
            def __init__(self, chunk_size):
                self.chunk_size = chunk_size

        labels = _object_labels([TS(128), {"k": 10}, None, ["a", "b"]])
        assert labels[0].startswith("TS(")
        assert labels[1].startswith("cfg(")
        assert labels[2] == "None"
        assert labels[3] == "['a', 'b']"

    def test_suggest_value_mixed_none_dict_does_not_crash(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        choices = [None, {"class": str, "top_n": 5}]
        sampled = _suggest_value(trial, "reranker_cfg", List(choices))
        assert sampled in choices

    def test_suggest_value_dict_choices(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        choices = [{"type": "similarity", "k": k} for k in (5, 10, 20)]
        sampled = _suggest_value(trial, "search_cfg", List(choices))
        assert sampled in choices


class TestTemplateToLeafEvalsPipelineAliases:
    """``_template_to_leaf_evals`` must normalise every supported pipeline alias
    to a ``"pipeline"`` key, matching grid_search/random_search and the
    controller's ``config_leaf["pipeline"]`` lookup."""

    def test_pipeline_alias_passthrough(self):
        leaf = _template_to_leaf_evals({"pipeline": "p", "batch_size": 4})
        assert leaf["pipeline"] == "p"
        assert leaf["batch_size"] == 4

    def test_vllm_alias_renamed(self):
        leaf = _template_to_leaf_evals({"vllm_config": "v", "x": 1})
        assert leaf["pipeline"] == "v"
        assert "vllm_config" not in leaf
        assert leaf["x"] == 1

    def test_api_alias_renamed(self):
        """Regression: api_config was silently dropped, leaving the leaf
        without a ``pipeline`` key and crashing the controller downstream."""
        sentinel = object()
        leaf = _template_to_leaf_evals(
            {"api_config": sentinel, "batch_size": 32, "preprocess_fn": "fn"}
        )
        assert leaf["pipeline"] is sentinel
        assert "api_config" not in leaf
        assert leaf["batch_size"] == 32
        assert leaf["preprocess_fn"] == "fn"

    def test_gemini_alias_renamed(self):
        leaf = _template_to_leaf_evals({"gemini_config": "g"})
        assert leaf["pipeline"] == "g"
        assert "gemini_config" not in leaf

    def test_openai_alias_renamed(self):
        leaf = _template_to_leaf_evals({"openai_config": "o"})
        assert leaf["pipeline"] == "o"
        assert "openai_config" not in leaf

    def test_unknown_keys_returned_unchanged(self):
        original = {"foo": 1, "bar": 2}
        leaf = _template_to_leaf_evals(original)
        assert leaf == original


# ---------------------------------------------------------------------------
# RF-OPT-01: search space nested inside a List of config objects
#
# A ``List`` is terminal in ``_extract_search_space``, so knobs nested inside
# its members used to be invisible to Optuna while still being resolved by
# ``recursive_expand_randomsearch`` via an unseeded ``item.sample()``. They are
# now registered as conditional parameters namespaced ``{name}[{idx}].{path}``.
# ---------------------------------------------------------------------------


class _FakeCfg:
    """Stand-in for RFAPIModelConfig: exposes the ``_user_params`` protocol."""

    def __init__(self, **kwargs):
        self._user_params = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _scifact_shaped_template():
    """Mirror the SciFact notebook shape: List of config objects, each with a
    nested rag spec carrying its own List knobs."""
    def make_cfg(name):
        return _FakeCfg(
            model=name,
            rag=_FakeCfg(
                embedding_cfg=List(["3-small", "3-large"]),
                search_cfg=List(["similarity", "mmr"]),
            ),
        )

    return {"api_config": List([make_cfg("a"), make_cfg("b")]), "batch_size": 32}


def _nested_knobs(member):
    """Read the sampled knobs back out of ``_user_params``.

    ``_set_nested`` writes into ``_user_params`` (the canonical store that
    ``recursive_expand_randomsearch`` later reads), not onto attributes.
    """
    rag = member._user_params["rag"]
    return rag._user_params["embedding_cfg"], rag._user_params["search_cfg"]


class TestNestedListSearchSpace:
    def test_extract_search_space_keeps_list_terminal(self):
        """The static space intentionally reports only the categorical itself.
        Nested knobs are conditional on the draw, so they show up in
        ``trial.params`` instead."""
        space = _extract_search_space(_scifact_shaped_template())
        assert [path for path, _ in space] == ["api_config"]

    def test_suggest_value_resolves_nested_knobs(self):
        study = optuna.create_study()
        trial = study.ask()
        template = _scifact_shaped_template()

        member = _suggest_value(trial, "api_config", template["api_config"])

        embedding, search = _nested_knobs(member)
        assert embedding in ("3-small", "3-large")
        assert search in ("similarity", "mmr")

        assert "api_config" in trial.params
        nested_names = [k for k in trial.params if k.startswith("api_config[")]
        assert sorted(n.split("].")[1] for n in nested_names) == [
            "rag.embedding_cfg",
            "rag.search_cfg",
        ]

    def test_sample_from_trial_end_to_end(self):
        template = _scifact_shaped_template()
        space = _extract_search_space(template)
        study = optuna.create_study()
        trial = study.ask()

        config = _sample_from_trial(trial, space, template)

        embedding, search = _nested_knobs(config["api_config"])
        assert not isinstance(embedding, (List, Range))
        assert not isinstance(search, (List, Range))
        assert config["batch_size"] == 32
        # The categorical plus both conditional knobs are all recorded.
        assert len(trial.params) >= 3

    def test_template_not_mutated_across_trials(self):
        template = _scifact_shaped_template()
        space = _extract_search_space(template)
        study = optuna.create_study()

        for _ in range(2):
            _sample_from_trial(study.ask(), space, template)

        for member in template["api_config"].values:
            embedding, search = _nested_knobs(member)
            assert isinstance(embedding, List)
            assert isinstance(search, List)

    def test_shared_nested_object_is_not_cross_contaminated(self):
        """The SciFact notebook hands the *same* rag spec to both generator
        configs, so ``api_config[0]`` and ``api_config[1]`` alias one object.

        Resolving a knob in place would therefore resolve it for both members at
        once, and the residual guard could not catch it -- a leaked value is a
        plain string, not a ``List``. Only the deep copy in
        ``_sample_list_member`` keeps the template reusable.
        """
        shared_rag = _FakeCfg(
            embedding_cfg=List(["3-small", "3-large"]),
            search_cfg=List(["similarity", "mmr"]),
        )
        template = {
            "api_config": List(
                [
                    _FakeCfg(model="a", rag=shared_rag),
                    _FakeCfg(model="b", rag=shared_rag),
                ]
            )
        }
        space = _extract_search_space(template)
        study = optuna.create_study()

        for _ in range(6):
            config = _sample_from_trial(study.ask(), space, template)

            embedding, search = _nested_knobs(config["api_config"])
            assert not isinstance(embedding, (List, Range))
            assert not isinstance(search, (List, Range))

            assert isinstance(shared_rag._user_params["embedding_cfg"], List)
            assert isinstance(shared_rag._user_params["search_cfg"], List)

    def test_nested_list_inside_chosen_member(self):
        """A List inside a chosen List member resolves too (mutual recursion)."""
        template = {
            "outer": List([
                _FakeCfg(tag="x", inner=List([1, 2])),
                _FakeCfg(tag="y", inner=List([3, 4])),
            ])
        }
        space = _extract_search_space(template)
        study = optuna.create_study()
        trial = study.ask()

        config = _sample_from_trial(trial, space, template)

        chosen = config["outer"]
        assert chosen._user_params["inner"] in (1, 2, 3, 4)
        assert "outer" in trial.params
        assert any(k.startswith("outer[") for k in trial.params)

    def test_none_member_round_trips(self):
        """An optional component expressed as ``None`` has no nested space and
        must pass through untouched."""
        study = optuna.create_study()
        trial = study.ask()
        assert _sample_list_member(trial, "reranker_cfg", 0, None) is None

    def test_search_space_object_as_direct_choice(self):
        """A Range nested directly as a List choice has no dotted path to write
        back to, so it is sampled in place rather than recursed into."""
        study = optuna.create_study()
        trial = study.ask()
        value = _sample_list_member(trial, "k", 1, Range(4, 8))
        assert 4 <= value <= 8
        assert trial.params["k[1]"] == value

    def test_deterministic_with_seed_over_object_list(self):
        """Regression for the unseeded-RNG half of RF-OPT-01: nested knobs used
        to come from ``random.choice`` and so varied run to run despite
        ``seed=42``. The existing determinism test only covers primitives,
        which Optuna already owned."""
        def make_study(seed):
            rfopt = RFOptuna(
                configs=[_scifact_shaped_template()],
                trainer_type=None,
                n_initial=4,
                budget=4,
                objective="maximize:Accuracy",
                sampler="tpe",
                pruner=None,
            )
            runs = rfopt.get_runs(seed=seed)
            return runs, [t.params for t in rfopt._study.trials]

        runs_a, params_a = make_study(42)
        runs_b, params_b = make_study(42)

        assert params_a == params_b
        # Every nested knob is a real Optuna parameter, not an RNG draw.
        assert any(k.startswith("api_config[") for k in params_a[0])
        for leaf in runs_a + runs_b:
            assert _find_unsampled_params(leaf) == []


class TestFindUnsampledParams:
    def test_clean_config_reports_nothing(self):
        assert _find_unsampled_params({"a": 1, "b": {"c": "x"}}) == []

    def test_finds_range_in_list_literal(self):
        """``_set_nested`` splits on ``.`` and cannot index into a list, so this
        is genuinely unreachable for the sampler."""
        found = _find_unsampled_params({"targets": [Range(1, 5)]})
        assert found == ["targets[0]"]

    def test_finds_nested_list_under_user_params(self):
        cfg = _FakeCfg(rag=_FakeCfg(k=List([5, 10])))
        assert _find_unsampled_params({"api_config": cfg}) == ["api_config.rag.k"]

    def test_get_runs_raises_on_unreachable_param(self):
        rfopt = RFOptuna(
            configs=[
                {"pipeline": "p", "temp": Range(0.0, 1.0), "targets": [Range(1, 5)]}
            ],
            trainer_type=None,
            n_initial=2,
            budget=2,
            objective="maximize:Accuracy",
            sampler="random",
            pruner=None,
        )
        with pytest.raises(AutoMLException, match=r"targets\[0\]"):
            rfopt.get_runs(seed=42)


# ---------------------------------------------------------------------------
# RF-OPT-04: RAG index coverage for replacement pipelines
#
# Indexes used to be built only from the ``n_initial`` sample, so a replacement
# Optuna suggested later could need an index that was never built and would then
# launch with ``context_generator_ref=None``. ``build_all_indexes`` pre-builds
# every reachable index; the alternative is rejecting infeasible suggestions.
# ---------------------------------------------------------------------------


class TestRangeSampleN:
    def test_returns_sample_n_distinct_values_in_range(self):
        values = Range(0.0, 1.0, sample_n=4).sample(4)
        assert len(values) == 4
        assert len(set(values)) == 4
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_values_are_sorted(self):
        values = Range(0, 1000, sample_n=5).sample(5)
        assert values == sorted(values)

    def test_int_dtype_returns_ints(self):
        assert all(isinstance(v, int) for v in Range(5, 20).sample(3))

    def test_log_stays_within_bounds(self):
        values = Range(1, 1000, dtype="int", log=True, sample_n=4).sample(4)
        assert len(values) == 4
        assert all(1 <= v <= 1000 for v in values)

    def test_step_values_land_on_the_grid(self):
        values = Range(5, 20, step=5, sample_n=3).sample(3)
        assert len(values) == 3
        assert set(values) <= {5, 10, 15, 20}

    def test_seed_determines_which_values_are_drawn(self):
        """Reproducibility comes from the seed, not from fixed spacing."""
        first = Range(0.0, 1.0, sample_n=3, seed=0).sample(3)
        assert Range(0.0, 1.0, sample_n=3, seed=0).sample(3) == first
        assert Range(0.0, 1.0, sample_n=3, seed=1).sample(3) != first

    def test_repeat_calls_on_one_range_differ(self):
        """Range is a pure sampler: two ``sample(n)`` calls draw independently.

        Range no longer memoizes a value set, so the second call advances the
        generator and returns different values. The guarantee that coverage
        enumeration and suggest_categorical see one value set now lives in
        RFOptuna, which caches the coverage draw and reuses it at suggest time
        (see TestRangeCacheConsistency).
        """
        rng_range = Range(0.0, 1.0, sample_n=3, seed=7)
        assert rng_range.sample(3) != rng_range.sample(3)

    def test_returns_fewer_when_range_is_exhausted(self):
        assert Range(1, 2, sample_n=5).sample(5) == [1, 2]

    def test_explicit_n_overrides_sample_n(self):
        assert len(Range(0.0, 1.0, sample_n=3).sample(2)) == 2

    def test_sample_n_one(self):
        assert len(Range(5, 20, sample_n=1).sample(1)) == 1

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "x", True])
    def test_sample_n_must_be_positive_int(self, bad):
        with pytest.raises(ValueError, match="sample_n must be a positive integer"):
            Range(1, 10, sample_n=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "x", True])
    def test_explicit_n_must_be_positive_int(self, bad):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            Range(1, 10).sample(bad)

    def test_sample_is_unchanged_by_sample_n(self):
        """RFRandomSearch must keep drawing continuously; sample_n is inert there.

        Random search calls ``sample(1)`` once per run, so 50 draws should
        produce many distinct values even when ``sample_n`` is small.
        """
        rng_range = Range(0.0, 1.0, sample_n=2)
        draws = {rng_range.sample(1)[0] for _ in range(50)}
        assert len(draws) > 2
        assert all(0.0 <= d <= 1.0 for d in draws)


class TestGridSearchRejectsRange:
    """Range belongs to RFRandomSearch and RFOptuna only.

    Grid search used to yield a Range through unexpanded, which silently fell
    back to a default in fit mode and crashed on a live Range object in evals.
    """

    @pytest.mark.parametrize(
        "template, expected_path",
        [
            ({"lr": Range(1e-5, 1e-3)}, "lr"),
            (
                {"rag": {"embedding_cfg": {"batch_size": Range(32, 128)}}},
                "rag.embedding_cfg.batch_size",
            ),
            ({"x": List([{"k": Range(1, 5)}])}, "x.k"),
        ],
    )
    def test_raises_naming_the_path(self, template, expected_path):
        with pytest.raises(AutoMLException, match="does not support Range") as exc:
            list(recursive_expand_gridsearch(template))
        assert f"'{expected_path}'" in str(exc.value)

    def test_list_still_expands(self):
        expanded = list(recursive_expand_gridsearch({"a": List([1, 2]), "b": 3}))
        assert expanded == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


class TestIsIndexAffectingPath:
    @pytest.mark.parametrize(
        "path",
        [
            "rag.embedding_cfg.model",
            "rag.text_splitter",
            "prompt_manager.k",
            "api_config[1].rag.embedding_cfg",
            "_t0.api_config.rag.vector_store_cfg.type",
        ],
    )
    def test_index_affecting(self, path):
        assert _is_index_affecting_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "rag.search_cfg.k",
            "rag.reranker_cfg.top_n",
            "_t0.api_config[1].rag.search_cfg.k",
            "training_args.learning_rate",
            "temperature",
        ],
    )
    def test_not_index_affecting(self, path):
        assert _is_index_affecting_path(path) is False


class TestIndexAffectingSampling:
    def test_index_affecting_range_becomes_categorical(self):
        chunk = Range(100, 500)
        template = {"pipeline": _FakeCfg(rag=_FakeCfg(chunk=chunk))}
        study = optuna.create_study()
        trial = study.ask()

        # Range no longer memoizes, so the value set suggest uses is the one
        # coverage enumeration drew and cached. Thread the cache in and compare
        # against it rather than against a second sample(n) draw (which would
        # now differ).
        cache: dict[int, list] = {}
        _sample_from_trial(
            trial, _extract_search_space(template), template, range_cache=cache
        )

        dist = trial.distributions["pipeline.rag.chunk"]
        assert isinstance(dist, optuna.distributions.CategoricalDistribution)
        assert list(dist.choices) == cache[id(chunk)]

    def test_retrieval_only_range_stays_continuous(self):
        template = {
            "pipeline": _FakeCfg(rag=_FakeCfg(search_cfg={"k": Range(5, 20, step=5)}))
        }
        study = optuna.create_study()
        trial = study.ask()

        _sample_from_trial(trial, _extract_search_space(template), template)

        dist = trial.distributions["pipeline.rag.search_cfg.k"]
        assert isinstance(dist, optuna.distributions.IntDistribution)
        assert (dist.low, dist.high, dist.step) == (5, 20, 5)

    def test_index_affecting_range_inside_list_member(self):
        template = {
            "api_config": List(
                [
                    _FakeCfg(name="a", rag=_FakeCfg(chunk=Range(100, 500))),
                    _FakeCfg(name="b", rag=_FakeCfg(chunk=Range(100, 500))),
                ]
            )
        }
        study = optuna.create_study()
        trial = study.ask()

        _sample_from_trial(trial, _extract_search_space(template), template)

        nested = [k for k in trial.distributions if k.endswith(".rag.chunk")]
        assert len(nested) == 1
        assert isinstance(
            trial.distributions[nested[0]], optuna.distributions.CategoricalDistribution
        )


def _fiqa_shaped_template():
    """Mirror the FiQA Optuna tutorial: one index-affecting List and Range, plus
    retrieval-only knobs that must not multiply the index count."""
    return {
        "api_config": _FakeCfg(
            rag=_FakeCfg(
                text_splitter=List(["chunk256", "chunk128"]),
                embedding_cfg={
                    "model": "minilm",
                    "batch_size": Range(32, 128, step=32),
                },
                search_cfg={"type": "similarity", "k": Range(5, 20, step=5)},
                reranker_cfg={"top_n": List([2, 5])},
            ),
            prompt_manager=None,
        ),
        "batch_size": 32,
    }


def _coverage_rag_params(leaves):
    return [leaf["pipeline"].rag._user_params for leaf in leaves]


class TestContextCoverageLeaves:
    def test_enumerates_index_affecting_axes_only(self):
        template = _fiqa_shaped_template()
        batch_range = template["api_config"].rag._user_params["embedding_cfg"][
            "batch_size"
        ]
        # Range no longer memoizes, so the batch values coverage draws are
        # captured in the cache; compare against the cache rather than a second
        # sample(n) draw (which would now differ).
        cache: dict[int, list] = {}
        leaves = _context_coverage_leaves(template["api_config"], cache)

        # 2 splitters x 3 sampled batch sizes; k and top_n do not multiply.
        assert len(leaves) == 6
        params = _coverage_rag_params(leaves)
        combos = {
            (p["text_splitter"], p["embedding_cfg"]["batch_size"]) for p in params
        }
        assert combos == {
            (splitter, batch)
            for splitter in ("chunk256", "chunk128")
            for batch in cache[id(batch_range)]
        }

    def test_retrieval_only_knobs_collapse_to_one_value(self):
        leaves = _context_coverage_leaves(_fiqa_shaped_template()["api_config"])
        ks = {p["search_cfg"]["k"] for p in _coverage_rag_params(leaves)}
        top_ns = {p["reranker_cfg"]["top_n"] for p in _coverage_rag_params(leaves)}
        # Retrieval-only knobs collapse to a single representative value across
        # all leaves (they do not multiply the index count). ``List`` collapses
        # to its first element deterministically; ``Range`` collapses to the
        # smallest of its sampled set, which depends on the draw, so only the
        # "one value" contract is asserted for k, not a specific number.
        assert len(ks) == 1
        assert len(top_ns) == 1
        assert top_ns == {2}
        assert ks.pop() in {5, 10, 15, 20}

    def test_no_search_space_objects_survive(self):
        """Coverage leaves are hashed and built, so a live Range/List would break."""
        for params in _coverage_rag_params(
            _context_coverage_leaves(_fiqa_shaped_template()["api_config"])
        ):
            assert _find_unsampled_params(params) == []

    def test_pipeline_without_context_yields_nothing(self):
        assert _context_coverage_leaves(_FakeCfg(temperature=0.5)) == []

    def test_rfoptuna_covers_every_list_member(self):
        rfopt = RFOptuna(
            configs=_scifact_shaped_template(),
            trainer_type=None,
            objective="maximize:NDCG@3",
        )
        leaves = rfopt.get_context_coverage_leaves()
        # 2 api_config members x 2 embeddings each; search_cfg does not multiply.
        assert len(leaves) == 4
        embeddings = {
            leaf["pipeline"].rag._user_params["embedding_cfg"] for leaf in leaves
        }
        assert embeddings == {"3-small", "3-large"}

    def test_range_inside_list_member_is_seeded_for_coverage(self):
        """Coverage draws from each Range's generator; a Range nested in a
        ``List`` of api_config members must receive the constructor seed so
        two coverage passes with the same seed enumerate the same value set.
        """

        def make_template():
            return {
                "api_config": List(
                    [
                        _FakeCfg(
                            rag=_FakeCfg(
                                embedding_cfg={
                                    "batch_size": Range(32, 128, step=32, sample_n=3),
                                },
                            ),
                        ),
                    ]
                ),
            }

        def coverage_batches(seed):
            rfopt = RFOptuna(
                configs=make_template(),
                trainer_type=None,
                objective="maximize:NDCG@3",
                seed=seed,
            )
            leaves = rfopt.get_context_coverage_leaves()
            return [
                leaf["pipeline"].rag._user_params["embedding_cfg"]["batch_size"]
                for leaf in leaves
            ]

        assert coverage_batches(42) == coverage_batches(42)

    def test_disabled_flag_returns_empty(self):
        rfopt = RFOptuna(
            configs=_scifact_shaped_template(),
            trainer_type=None,
            objective="maximize:NDCG@3",
            build_all_indexes=False,
        )
        assert rfopt.get_context_coverage_leaves() == []

    def test_fit_mode_returns_empty(self):
        rfopt = RFOptuna(
            configs=_scifact_shaped_template(),
            trainer_type=None,
            objective="minimize:eval_loss",
        )
        # Fit mode is set from trainer_type, which requires a real RFModelConfig;
        # the flag itself is mode-gated, so assert that gate directly.
        rfopt.mode = "fit"
        assert rfopt.get_context_coverage_leaves() == []

    def test_raises_above_prebuild_cap(self):
        template = {
            "api_config": _FakeCfg(
                rag=_FakeCfg(
                    embedding_cfg=List([f"e{i}" for i in range(9)]),
                    text_splitter=List([f"s{i}" for i in range(8)]),
                )
            )
        }
        rfopt = RFOptuna(
            configs=template, trainer_type=None, objective="maximize:NDCG@3",
        )
        with pytest.raises(AutoMLException, match="above the limit"):
            rfopt.get_context_coverage_leaves()


class TestRangeCacheConsistency:
    """Coverage enumeration draws an index-affecting Range's value set once;
    suggest reuses that cached set rather than re-drawing.

    Range no longer memoizes, so this consistency is now RFOptuna's job: the
    evals controller runs ``get_context_coverage_leaves`` before ``get_runs``,
    the coverage pass populates ``RFOptuna._range_value_cache``, and the suggest
    pass reads from it.
    """

    def test_coverage_populates_cache_and_suggest_reuses_it(self):
        template = _fiqa_shaped_template()
        batch_range = template["api_config"].rag._user_params["embedding_cfg"][
            "batch_size"
        ]
        rfopt = RFOptuna(
            configs=template, trainer_type=None, objective="maximize:NDCG@3",
        )

        # Coverage runs first (as the evals controller now orders it) and draws
        # the index-affecting batch_size value set into the cache.
        leaves = rfopt.get_context_coverage_leaves()
        assert id(batch_range) in rfopt._range_value_cache
        cached = rfopt._range_value_cache[id(batch_range)]

        # Every coverage leaf's batch_size is one of the cached values.
        coverage_batches = {
            p["embedding_cfg"]["batch_size"] for p in _coverage_rag_params(leaves)
        }
        assert coverage_batches <= set(cached)
        assert coverage_batches == set(cached)

        # Suggest then reads the same cache; every run's batch_size is in it.
        runs = rfopt.get_runs(seed=42)
        for run in runs:
            batch = run["pipeline"].rag._user_params["embedding_cfg"]["batch_size"]
            assert batch in cached

    def test_build_all_indexes_false_leaves_cache_empty_until_suggest(self):
        template = _fiqa_shaped_template()
        batch_range = template["api_config"].rag._user_params["embedding_cfg"][
            "batch_size"
        ]
        rfopt = RFOptuna(
            configs=template, trainer_type=None, objective="maximize:NDCG@3",
            build_all_indexes=False,
        )
        # No coverage drawn, so the cache is empty before get_runs.
        assert rfopt.get_context_coverage_leaves() == []
        assert id(batch_range) not in rfopt._range_value_cache

        # The first suggest call draws and caches the set; all runs share it.
        runs = rfopt.get_runs(seed=42)
        cached = rfopt._range_value_cache[id(batch_range)]
        for run in runs:
            batch = run["pipeline"].rag._user_params["embedding_cfg"]["batch_size"]
            assert batch in cached


def _rejection_callback(budget=100):
    template = {"pipeline": {"index": List(["a", "b", "c"]), "k": Range(1, 10)}}
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    callback = OptunaShardCallback(
        study=study,
        search_spaces=[_extract_search_space(template)],
        config_templates=[template],
        budget=budget,
        objective_metric="ndcg",
    )
    callback._set_initial_trials({}, spawned=0)
    return study, callback


class TestReplacementFeasibility:
    def test_accepts_everything_without_a_predicate(self):
        _, callback = _rejection_callback()
        assert callback._maybe_suggest_replacement() is not None
        assert callback._spawned == 1

    def test_only_feasible_configs_are_returned(self):
        study, callback = _rejection_callback()
        callback.set_context_feasibility(lambda leaf: leaf["pipeline"]["index"] == "a")

        with pytest.warns(UserWarning, match="rejected a suggested config"):
            leaves = [callback._maybe_suggest_replacement() for _ in range(4)]

        assert all(leaf["pipeline"]["index"] == "a" for leaf in leaves)
        # Rejections must not consume budget.
        assert callback._spawned == 4
        states = [t.state for t in study.get_trials(deepcopy=False)]
        assert optuna.trial.TrialState.FAIL in states

    def test_rejected_trials_are_failed_not_completed(self):
        study, callback = _rejection_callback()
        callback.set_context_feasibility(lambda leaf: False)

        with pytest.warns(UserWarning):
            assert callback._maybe_suggest_replacement() is None

        trials = study.get_trials(deepcopy=False)
        assert len(trials) == _MAX_REPLACEMENT_ATTEMPTS
        assert all(t.state == optuna.trial.TrialState.FAIL for t in trials)
        assert callback._spawned == 0
        assert study.best_trials == []

    def test_narrowing_warning_fires_once(self):
        _, callback = _rejection_callback()
        callback.set_context_feasibility(lambda leaf: leaf["pipeline"]["index"] == "a")

        with pytest.warns(UserWarning) as record:
            for _ in range(6):
                callback._maybe_suggest_replacement()

        narrowing = [
            w for w in record if "rejected a suggested config" in str(w.message)
        ]
        assert len(narrowing) == 1

    def test_budget_still_caps_replacements(self):
        _, callback = _rejection_callback(budget=2)
        callback._set_initial_trials({}, spawned=2)
        callback.set_context_feasibility(lambda leaf: True)
        assert callback._maybe_suggest_replacement() is None


class TestRandomSearchReproducibility:
    """RFRandomSearch stamps its constructor seed onto every Range so Range
    draws are reproducible alongside List draws (which use the global RNG).
    Range uses its own generator, so without this seeding the constructor seed
    would not affect Range draws at all. The run-level seed passed to get_runs
    is ignored for the algorithm's draws.
    """

    @staticmethod
    def _knobs(run):
        rag = run["pipeline"].rag._user_params
        return rag["embedding_cfg"]["batch_size"]

    def test_same_seed_produces_identical_runs(self):
        template = {
            "api_config": _FakeCfg(
                rag=_FakeCfg(
                    embedding_cfg={
                        "model": "minilm",
                        "batch_size": Range(32, 128, step=32),
                    },
                    search_cfg={"type": "similarity", "k": Range(5, 20, step=5)},
                ),
            ),
            "batch_size": 32,
        }

        def make_runs():
            # Fresh template each call so the Range objects are unseeded to start.
            rfopt = RFRandomSearch(
                configs=copy.deepcopy(template),
                trainer_type=None,
                num_runs=6,
                seed=42,
            )
            return rfopt.get_runs(seed=42)

        first = [self._knobs(r) for r in make_runs()]
        second = [self._knobs(r) for r in make_runs()]
        assert first == second

    def test_different_seeds_produce_different_runs(self):
        template = {
            "api_config": _FakeCfg(
                rag=_FakeCfg(
                    embedding_cfg={
                        "model": "minilm",
                        "batch_size": Range(32, 128, step=32),
                    },
                ),
            ),
        }

        def make_runs(ctor_seed):
            rfopt = RFRandomSearch(
                configs=copy.deepcopy(template),
                trainer_type=None,
                num_runs=6,
                seed=ctor_seed,
            )
            # The run-level seed is ignored; vary only the constructor seed.
            return [self._knobs(r) for r in rfopt.get_runs(seed=42)]

        assert make_runs(42) != make_runs(7)

    def test_range_nested_in_list_member_is_reproducible(self):
        """``_seed_ranges`` must walk into ``List`` members so a Range nested
        in a chosen config is stamped with the constructor seed.  SciFact-shaped
        templates (``api_config=List([cfg_a, cfg_b])``) hit this path.
        """

        def make_template():
            return {
                "api_config": List(
                    [
                        _FakeCfg(
                            rag=_FakeCfg(
                                embedding_cfg={
                                    "model": "minilm",
                                    "batch_size": Range(32, 128, step=32),
                                },
                            ),
                        ),
                        _FakeCfg(
                            rag=_FakeCfg(
                                embedding_cfg={
                                    "model": "other",
                                    "batch_size": Range(16, 64, step=16),
                                },
                            ),
                        ),
                    ]
                ),
                "batch_size": 32,
            }

        def knobs(run):
            rag = run["pipeline"].rag._user_params["embedding_cfg"]
            return rag["model"], rag["batch_size"]

        def make_runs():
            rfopt = RFRandomSearch(
                configs=copy.deepcopy(make_template()),
                trainer_type=None,
                num_runs=6,
                seed=42,
            )
            return [knobs(r) for r in rfopt.get_runs(seed=42)]

        assert make_runs() == make_runs()
