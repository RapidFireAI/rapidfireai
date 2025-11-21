"""Random search implementation for AutoML hyperparameter optimization."""

import random
import json
import hashlib
from itertools import product
from typing import Any, Dict
from typing import List as ListType

from rapidfireai.evals.automl.base import AutoMLAlgorithm
from rapidfireai.evals.automl.datatypes import List, Range


def encode_payload(payload: Dict[str, Any]) -> str:
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

    def get_runs(self, seed: int = 42) -> ListType[Dict[str, Any]]:
        """Generate num_runs random hyperparameter combinations."""
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise Exception("seed must be a non-negative integer")

        if not isinstance(self.num_runs, int) or self.num_runs <= 0:
            raise Exception("num_runs must be a positive integer")

        random.seed(seed)

        try:
            runs = []
            seen_configs = set()
            max_attempts = self.num_runs * 10
            attempts = 0

            while len(runs) < self.num_runs and attempts < max_attempts:
                attempts += 1

                # Sample a config from the available configs
                config = List(self.configs).sample()

                # Handle pipeline similar to grid search
                if config["pipeline"] is None:
                    pipelines = [None]
                elif isinstance(config["pipeline"], List):
                    pipelines = [config["pipeline"].sample()]
                elif isinstance(config["pipeline"], list):
                    pipelines = [List(config["pipeline"]).sample()]
                else:
                    pipelines = [config["pipeline"]]

                for pipeline in pipelines:
                    # Sample model config parameters
                    pipeline_instances = (
                        [{}]
                        if pipeline is None
                        else [recursive_expand_randomsearch(pipeline._user_params)]
                    )

                    additional_kwargs = {
                        k: v
                        for k, v in config.items()
                        if k != "pipeline" and v is not None
                    }
                    additional_kwargs_instances = (
                        [{}]
                        if not additional_kwargs
                        else [recursive_expand_randomsearch(additional_kwargs)]
                    )

                    # Generate random search combinations
                    for pipeline_params in pipeline_instances:
                        for additional_kwargs in additional_kwargs_instances:
                            leaf = {
                                "pipeline": pipeline.__class__(**pipeline_params),
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
                raise Exception(
                    f"Could not generate {self.num_runs} unique configurations. "
                    f"Generated {len(runs)} unique configs after {attempts} attempts. "
                )

            return runs

        except Exception as e:
            raise Exception(f"Error generating runs: {e}") from e
