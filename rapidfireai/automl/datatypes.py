"""Contains classes for representing hyperparameter data types.

Covers all Optuna distribution types:

- ``Range(start, end, dtype="float")`` → ``FloatDistribution`` / ``suggest_float``
- ``Range(start, end, dtype="float", log=True)`` → log-uniform float
- ``Range(start, end, dtype="float", step=0.1)`` → discrete float
- ``Range(start, end, dtype="int")`` → ``IntDistribution`` / ``suggest_int``
- ``Range(start, end, dtype="int", log=True)`` → log-uniform int
- ``Range(start, end, dtype="int", step=2)`` → stepped int
- ``List([...])`` → ``CategoricalDistribution`` / ``suggest_categorical``
"""

import math
import random


class Range:
    """Represents a range of values for a hyperparameter.

    Supports uniform, log-uniform, and discrete (stepped) sampling for both
    int and float dtypes — matching all variants of Optuna's
    ``IntDistribution`` and ``FloatDistribution``.

    Args:
        start: Lower bound (inclusive).
        end: Upper bound (inclusive).
        dtype: ``"int"`` or ``"float"``.  Inferred from *start*/*end* types
            when not provided.
        log: If ``True``, sample in log-space (start and end must be > 0).
            Mutually exclusive with *step*.
        step: Discretisation step.  When set, sampled values are multiples of
            *step* starting from *start*.  Mutually exclusive with *log*.
    """

    def __init__(
        self,
        start,
        end,
        dtype: str | None = None,
        log: bool = False,
        step: int | float | None = None,
    ):
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
        if log and step is not None:
            raise ValueError(
                "log=True and step are mutually exclusive "
                "(Optuna does not support this combination either)."
            )
        if log and (start <= 0 or end <= 0):
            raise ValueError(
                "log=True requires both start and end to be > 0."
            )
        self.start = start
        self.end = end
        self.log = log
        self.step = step

    def sample(self):
        """Sample a value from the range [self.start, self.end].

        Respects *log* (log-uniform) and *step* (discrete) settings so that
        ``RFRandomSearch`` produces the same family of distributions as
        Optuna's ``suggest_int`` / ``suggest_float``.
        """
        if self.dtype == "int":
            if self.log:
                log_low, log_high = math.log(self.start), math.log(self.end)
                return int(round(math.exp(random.uniform(log_low, log_high))))
            if self.step is not None:
                step = int(self.step)
                n_steps = (self.end - self.start) // step
                return self.start + random.randint(0, n_steps) * step
            return random.randint(self.start, self.end)

        # dtype == "float"
        if self.log:
            log_low, log_high = math.log(self.start), math.log(self.end)
            return math.exp(random.uniform(log_low, log_high))
        if self.step is not None:
            n_steps = int((self.end - self.start) / self.step)
            return self.start + random.randint(0, n_steps) * self.step
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
