"""Grid search implementation for AutoML training configurations."""

from itertools import product
from typing import Any, Dict
from typing import List as ListType

from rapidfireai.evals.automl.base import AutoMLAlgorithm
from rapidfireai.evals.automl.datatypes import List


def recursive_expand_gridsearch(item: Any):
    """Recursively expand nested structures with List datatypes into all combinations."""
    if hasattr(item, "_user_params"):
        expanded_params_list = list(recursive_expand_gridsearch(item._user_params))
        for params in expanded_params_list:
            yield item.__class__(**params)
    elif isinstance(item, dict):
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

    def get_runs(self, seed: int=42) -> ListType[Dict[str, Any]]:
        """Generate all possible hyperparameter combinations for grid search."""
        if not isinstance(seed, int) or seed < 0:
            raise Exception("seed must be a non-negative integer")

        try:
            runs = []
            for config in self.configs:
                if "vllm_config" in config:
                    pipeline = config["vllm_config"]
                else:
                    pipeline = config["openai_config"]
                if pipeline is None:
                    pipelines = [None]
                elif isinstance(pipeline, List):
                    pipelines = pipeline.values
                elif isinstance(pipeline, list):
                    pipelines = pipeline
                else:
                    pipelines = [pipeline]
                for pipeline in pipelines:
                    pipeline_instances = (
                        [{}]
                        if pipeline is None
                        else list(recursive_expand_gridsearch(pipeline))
                    )

                    additional_kwargs = {
                        k: v
                        for k, v in config.items()
                        if k != "pipeline" and k != "vllm_config" and k != "openai_config" and v is not None
                    }
                    additional_kwargs_instances = (
                        [{}]
                        if not additional_kwargs
                        else list(recursive_expand_gridsearch(additional_kwargs))
                    )
                    for pipeline_params in pipeline_instances:
                        for additional_kwargs_dict in additional_kwargs_instances:
                            # pipeline_params could be an instance (from recursive_expand_gridsearch) or a dict
                            if isinstance(pipeline_params, dict):
                                pipeline_instance = pipeline.__class__(**pipeline_params)
                            else:
                                pipeline_instance = pipeline_params
                            
                            leaf = {
                                "pipeline": pipeline_instance,
                                **additional_kwargs_dict,
                            }
                            runs.append(leaf)

            return runs

        except Exception as e:
            raise Exception(f"Error generating runs: {e}") from e
