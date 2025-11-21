"""Contains classes for representing hyperparameter data types."""

import random

# TODO: need to set seed for random module.
# TODO: List.sample() will not work for nested lists.
# TODO: add support for sampling methods like 'uniform' and 'loguniform'.


class Range:
    """Represents a range of values for a hyperparameter."""

    def __init__(self, start, end, dtype: str | None = None):
        if dtype is None:
            self.dtype = "int" if isinstance(start, int) and isinstance(end, int) else "float"
        else:
            if dtype not in ("int", "float"):
                raise ValueError("dtype must be either 'int' or 'float'.")
            self.dtype = dtype
        if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
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
