"""Random search implementation for AutoML hyperparameter optimization."""

import json
import random
import hashlib
from typing import Any

from rapidfireai.automl.base import AutoMLAlgorithm
from rapidfireai.automl.datatypes import List, Range
from rapidfireai.fit.utils.exceptions import AutoMLException


def encode_payload(payload: dict[str, Any]) -> str:
    """Create a hashable representation of a configuration dictionary."""
    json_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()


def recursive_expand_randomsearch(item: Any):
    """Recursively sample from nested structures with List and Range datatypes."""
    if hasattr(item, "_user_params"):
        sampled_params = recursive_expand_randomsearch(item._user_params)
        return item.__class__(**sampled_params)
    elif isinstance(item, dict):
        return {k: recursive_expand_randomsearch(v) for k, v in item.items()}
    elif isinstance(item, List):
        return item.sample()
    elif isinstance(item, Range):
        return item.sample()
    else:
        return item


class RFRandomSearch(AutoMLAlgorithm):
    """Random search algorithm that samples num_runs hyperparameter combinations."""

    def get_runs(self, seed: int = 42) -> list[dict[str, Any]]:
        """Generate num_runs random hyperparameter combinations."""
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise AutoMLException("seed must be a non-negative integer")

        if not isinstance(self.num_runs, int) or self.num_runs <= 0:
            raise AutoMLException("num_runs must be a positive integer")

        random.seed(seed)

        try:
            if self.mode == "fit":
                return self._get_runs_fit(seed)
            else:
                return self._get_runs_evals(seed)
        except Exception as e:
            raise AutoMLException(f"Error generating runs: {e}") from e

    def _get_runs_fit(self, seed: int) -> list[dict[str, Any]]:
        """Generate runs for fit mode."""
        runs = []
        seen_configs = set()
        max_attempts = self.num_runs * 10
        attempts = 0

        while len(runs) < self.num_runs and attempts < max_attempts:
            attempts += 1

            config = List(self.configs).sample()

            if config.peft_config is None:
                selected_peft_config = None
            elif isinstance(config.peft_config, list):
                selected_peft_config = List(config.peft_config).sample()
            elif isinstance(config.peft_config, List):
                selected_peft_config = config.peft_config.sample()
            else:
                selected_peft_config = config.peft_config

            peft_params = (
                {}
                if selected_peft_config is None
                else recursive_expand_randomsearch(selected_peft_config._user_params)
            )

            # Sample other parameters
            training_params = (
                {}
                if config.training_args is None
                else recursive_expand_randomsearch(config.training_args._user_params)
            )

            model_kwargs = {} if config.model_kwargs is None else recursive_expand_randomsearch(config.model_kwargs)

            ref_model_kwargs = (
                {} if config.ref_model_kwargs is None else recursive_expand_randomsearch(config.ref_model_kwargs)
            )

            reward_funcs = {} if config.reward_funcs is None else recursive_expand_randomsearch(config.reward_funcs)

            # FIXME:  avoid hardcoding the excluded attributes
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
            }
            additional_kwargs = {
                k: v for k, v in config.__dict__.items() if k not in excluded_attrs and v is not None
            }
            additional_kwargs_sampled = (
                {} if not additional_kwargs else recursive_expand_randomsearch(additional_kwargs)
            )

            leaf = {
                "trainer_type": self.trainer_type,
                "training_args": training_params,
                "peft_params": peft_params,
                "model_name": config.model_name,
                "tokenizer": config.tokenizer,
                "tokenizer_kwargs": config.tokenizer_kwargs,
                "model_type": config.model_type,
                "model_kwargs": model_kwargs,
                "additional_kwargs": additional_kwargs_sampled,
            }

            if self.trainer_type == "DPO":
                leaf["ref_model_config"] = {
                    "model_name": config.ref_model_name,
                    "model_type": config.ref_model_type,
                    "model_kwargs": ref_model_kwargs,
                }
                # FIXME: correct ref args
            elif self.trainer_type == "GRPO":
                leaf["reward_funcs"] = reward_funcs

            # Check for duplicates using hashable representation
            config_hash = encode_payload(leaf)
            if config_hash not in seen_configs:
                seen_configs.add(config_hash)
                runs.append(leaf)

        if len(runs) < self.num_runs:
            raise AutoMLException(
                f"Could not generate {self.num_runs} unique configurations. "
                f"Generated {len(runs)} unique configs after {attempts} attempts. "
            )

        return runs

    def _get_runs_evals(self, seed: int) -> list[dict[str, Any]]:
        """Generate runs for evals mode."""
        runs = []
        seen_configs = set()
        max_attempts = self.num_runs * 10
        attempts = 0

        while len(runs) < self.num_runs and attempts < max_attempts:
            attempts += 1

            # Sample a config from the available configs
            config = List(self.configs).sample()

            # Handle pipeline similar to grid search
            if "vllm_config" in config:
                pipeline = config["vllm_config"]
            elif "openai_config" in config:
                pipeline = config["openai_config"]
            elif "pipeline" in config:
                pipeline = config["pipeline"]
            else:
                pipeline = None

            if pipeline is None:
                pipelines = [None]
            elif isinstance(pipeline, List):
                pipelines = [pipeline.sample()]
            elif isinstance(pipeline, list):
                pipelines = [List(pipeline).sample()]
            else:
                pipelines = [pipeline]

            for pipeline in pipelines:
                # Sample model config parameters
                pipeline_instances = (
                    [{}]
                    if pipeline is None
                    else [recursive_expand_randomsearch(pipeline)]
                )

                additional_kwargs = {
                    k: v
                    for k, v in config.items()
                    if k != "pipeline" and k != "vllm_config" and k != "openai_config" and v is not None
                }
                additional_kwargs_instances = (
                    [{}]
                    if not additional_kwargs
                    else [recursive_expand_randomsearch(additional_kwargs)]
                )

                # Generate random search combinations
                for pipeline_params in pipeline_instances:
                    for additional_kwargs in additional_kwargs_instances:
                        if isinstance(pipeline_params, dict):
                            pipeline_instance = pipeline.__class__(**pipeline_params)
                        else:
                            pipeline_instance = pipeline_params

                        leaf = {
                            "pipeline": pipeline_instance,
                            **additional_kwargs,
                        }

                        # Check for duplicates using hashable representation
                        config_hash = encode_payload(leaf)
                        if config_hash not in seen_configs:
                            seen_configs.add(config_hash)
                            runs.append(leaf)

                            # Break if we have enough runs
                            if len(runs) >= self.num_runs:
                                break

                    if len(runs) >= self.num_runs:
                        break

                if len(runs) >= self.num_runs:
                    break

        if len(runs) < self.num_runs:
            raise AutoMLException(
                f"Could not generate {self.num_runs} unique configurations. "
                f"Generated {len(runs)} unique configs after {attempts} attempts. "
            )

        return runs
