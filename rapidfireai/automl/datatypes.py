"""Contains classes for representing hyperparameter data types."""

import copy
import random
from typing import Any

# TODO: need to set seed for random module.
# TODO: List.sample() will not work for nested lists.
# TODO: add support for sampling methods like 'uniform' and 'loguniform'.


class Range:
    """Represents a range of values for a hyperparameter."""

    def __init__(self, start, end, dtype: str | None = None):
        if dtype is None:
            self.dtype = (
                "int" if isinstance(start, int) and isinstance(end, int) else "float"
            )
        else:
            if dtype not in ("int", "float"):
                raise ValueError("dtype must be either 'int' or 'float'.")
            self.dtype = dtype
        if not (isinstance(start, int | float) and isinstance(end, int | float)):
            raise ValueError("start and end must be either int or float.")
        self.start = start
        self.end = end

    def sample(self):
        """Sample a value from the range [self.start, self.end]."""
        if self.dtype == "int":
            return random.randint(self.start, self.end)
        return random.uniform(self.start, self.end)


class List:
    """Represents a list of values for a hyperparameter."""

    def __init__(self, values):
        if not isinstance(values, list):
            raise ValueError("List expects a list of values.")
        self.values = values

    def sample(self):
        """Sample a value from the list."""
        return random.choice(self.values)

class EmbeddingSpec:
    """Couples an embedding class with its kwargs to prevent cross-combination in grid search.
    """
    def __init__(self, cls: type, kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.kwargs = kwargs or {}
        self._user_params = {"cls": cls, "kwargs": copy.deepcopy(self.kwargs)}

    def create(self):
        """Instantiate the embedding model using cls and kwargs."""
        return self.cls(**self.kwargs)


class RerankSpec:
    """Couples a reranker class with its kwargs to prevent cross-combination in grid search.
    """
    def __init__(self, cls: type, kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.kwargs = kwargs or {}
        self._user_params = {"cls": cls, "kwargs": copy.deepcopy(self.kwargs)}


class SearchSpec:
    """Couples search_type with search_kwargs to prevent cross-combination in grid search.
    """
    def __init__(self, search_type: str = "similarity", search_kwargs: dict[str, Any] | None = None):
        valid_search_types = {"similarity", "similarity_score_threshold", "mmr"}
        if search_type not in valid_search_types:
            raise ValueError(f"search_type must be one of {valid_search_types}, got: {search_type}")
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}
        self._user_params = {
            "search_type": self.search_type,
            "search_kwargs": copy.deepcopy(self.search_kwargs),
        }