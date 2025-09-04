"""Random search implementation for AutoML hyperparameter optimization."""

import random
import json
from itertools import product
from typing import Any, Dict
from typing import List as ListType

from rapidfireai.automl.base import AutoMLAlgorithm
from rapidfireai.automl.datatypes import List, Range
from rapidfireai.utils.exceptions import AutoMLException
from rapidfireai.utils.serialize import encode_payload


def recursive_expand_randomsearch(item: Any):
    if isinstance(item, dict):
        return {k: recursive_expand_randomsearch(v) for k, v in item.items()}
    elif isinstance(item, List):
        return item.sample()
    elif isinstance(item, Range):
        return item.sample()
    else:
        return item

class RFRandomSearch(AutoMLAlgorithm):
    """Random search algorithm that samples num_runs hyperparameter combinations."""

    def get_runs(self, seed: int=42) -> ListType[Dict[str, Any]]:
        """Generate num_runs random hyperparameter combinations."""
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise AutoMLException("seed must be a non-negative integer")
        
        if not isinstance(self.num_runs, int) or self.num_runs <= 0:
            raise AutoMLException("num_runs must be a positive integer")

        
        random.seed(seed)
        
        try:
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
                    {} if selected_peft_config is None 
                    else recursive_expand_randomsearch(selected_peft_config._user_params)
                )

                
                # Sample other parameters
                training_params = (
                    {} if config.training_args is None 
                    else recursive_expand_randomsearch(config.training_args._user_params)
                )
                
                model_kwargs = (
                    {} if config.model_kwargs is None 
                    else recursive_expand_randomsearch(config.model_kwargs)
                )
                
                ref_model_kwargs = (
                    {} if config.ref_model_kwargs is None 
                    else recursive_expand_randomsearch(config.ref_model_kwargs)
                )
                
                reward_funcs = (
                    {} if config.reward_funcs is None 
                    else recursive_expand_randomsearch(config.reward_funcs)
                )

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
                    {} if not additional_kwargs 
                    else recursive_expand_randomsearch(additional_kwargs)
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
                    #FIXME: correct ref args
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

        except Exception as e:
            raise AutoMLException(f"Error generating runs: {e}") from e
