"""Grid search implementation for AutoML training configurations."""

from itertools import product
from typing import Any, Dict
from typing import List as ListType

from rapidfireai.automl.base import AutoMLAlgorithm
from rapidfireai.automl.datatypes import List
from rapidfireai.utils.exceptions import AutoMLException


def recursive_expand_gridsearch(item: Any):
    """Recursively expand nested structures with List datatypes into all combinations."""
    if isinstance(item, dict):
        keys = list(item.keys())
        value_lists = [list(recursive_expand_gridsearch(item[k])) for k in keys]
        for values in product(*value_lists):
            yield dict(zip(keys, values))
    elif isinstance(item, List):
        for value in item.values:
            yield from recursive_expand_gridsearch(value)
    else:
        yield item


class RFGridSearch(AutoMLAlgorithm):
    """Grid search algorithm that generates all hyperparameter combinations."""

    def get_runs(self, seed: int) -> ListType[Dict[str, Any]]:
        """Generate all possible hyperparameter combinations for grid search."""
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")

        try:
            runs = []
            for config in self.configs:
                if config.peft_config is None:
                    peft_configs = [None]
                elif isinstance(config.peft_config, List):
                    peft_configs = config.peft_config.values
                elif isinstance(config.peft_config, list):
                    peft_configs = config.peft_config
                else:
                    peft_configs = [config.peft_config]

                for peft_config in peft_configs:
                    peft_instances = (
                        [{}] if peft_config is None else list(recursive_expand_gridsearch(peft_config._user_params))
                    )
                    training_instances = (
                        [{}]
                        if config.training_args is None
                        else list(recursive_expand_gridsearch(config.training_args._user_params))
                    )
                    model_kwargs_instances = (
                        [{}] if config.model_kwargs is None else list(recursive_expand_gridsearch(config.model_kwargs))
                    )
                    ref_model_kwargs_instances = (
                        [{}]
                        if config.ref_model_kwargs is None
                        else list(recursive_expand_gridsearch(config.ref_model_kwargs))
                    )
                    reward_funcs_instances = (
                        [{}] if config.reward_funcs is None else list(recursive_expand_gridsearch(config.reward_funcs))
                    )

                    # Get additional kwargs for Trainer
                    # FIXME: this is a hack to get the additional kwargs, we should find a better way to do this
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
                    # excluded_attrs = set(config.__dict__.keys()) - set(config.__annotations__.keys())
                    additional_kwargs = {
                        k: v for k, v in config.__dict__.items() if k not in excluded_attrs and v is not None
                    }
                    additional_kwargs_instances = (
                        [{}] if not additional_kwargs else list(recursive_expand_gridsearch(additional_kwargs))
                    )

                    # Generate gridsearch combinations
                    for peft_params in peft_instances:
                        for training_params in training_instances:
                            for model_kwargs in model_kwargs_instances:
                                for additional_kwargs in additional_kwargs_instances:
                                    leaf = {
                                        "trainer_type": self.trainer_type,
                                        "training_args": training_params,
                                        "peft_params": peft_params,
                                        "model_name": config.model_name,
                                        "tokenizer": config.tokenizer,
                                        "tokenizer_kwargs": config.tokenizer_kwargs,
                                        "model_type": config.model_type,
                                        "model_kwargs": model_kwargs,
                                        "additional_kwargs": additional_kwargs,
                                    }

                                    if self.trainer_type == "DPO":
                                        leaf["ref_model_config"] = {
                                            "model_name": config.ref_model_name,
                                            "model_type": config.ref_model_type,
                                        }
                                        for ref_model_kwargs in ref_model_kwargs_instances:
                                            leaf["ref_model_config"]["model_kwargs"] = ref_model_kwargs
                                            runs.append(leaf)
                                    elif self.trainer_type == "GRPO":
                                        for reward_func in reward_funcs_instances:
                                            leaf["reward_funcs"] = reward_func
                                            runs.append(leaf)
                                    else:
                                        runs.append(leaf)

            return runs

        except Exception as e:
            raise AutoMLException(f"Error generating runs: {e}") from e