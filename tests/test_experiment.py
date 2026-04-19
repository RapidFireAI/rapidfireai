import pytest

from rapidfireai.experiment import Experiment


@pytest.fixture(autouse=True)
def patch_experiment_initializers(monkeypatch):
    monkeypatch.setattr(Experiment, "_init_fit_mode", lambda self: None, raising=False)
    monkeypatch.setattr(Experiment, "_init_evals_mode", lambda self: None, raising=False)
    monkeypatch.setattr(
        Experiment,
        "_auto_detect_resources",
        lambda self, num_cpus=None, num_gpus=None: {
            "cpus_for_ray": 1,
            "gpus_for_ray": 0,
            "num_actors": 1,
            "gpus_per_actor": 0.0,
            "cpus_per_actor": 1.0,
        },
        raising=False,
    )


def test_eval_alias_normalizes_to_evals():
    experiment = Experiment(experiment_name="demo", mode="eval")

    assert experiment.mode == "evals"


def test_experiments_path_alias_sets_experiment_path():
    experiment = Experiment(experiment_name="demo", experiments_path="/tmp/custom-experiments")

    assert experiment.experiment_path == "/tmp/custom-experiments"


def test_same_path_aliases_are_allowed():
    experiment = Experiment(
        experiment_name="demo",
        experiment_path="/tmp/custom-experiments",
        experiments_path="/tmp/custom-experiments",
    )

    assert experiment.experiment_path == "/tmp/custom-experiments"


def test_conflicting_path_aliases_raise_error():
    with pytest.raises(TypeError, match="experiment_path and experiments_path"):
        Experiment(
            experiment_name="demo",
            experiment_path="/tmp/fit-experiments",
            experiments_path="/tmp/eval-experiments",
        )


def test_invalid_mode_error_lists_supported_values():
    with pytest.raises(ValueError, match="Must be 'fit', 'eval', or 'evals'"):
        Experiment(experiment_name="demo", mode="train")
