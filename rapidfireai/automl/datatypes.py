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

# Draws allowed per requested value before a multi-value ``sample`` gives up on
# finding another distinct one.
_SAMPLE_ATTEMPT_MULTIPLIER = 20


class Range:
    """Represents a range of values for a hyperparameter.

    Supports uniform, log-uniform, and discrete (stepped) sampling for both
    int and float dtypes — matching all variants of Optuna's
    ``IntDistribution`` and ``FloatDistribution``.

    ``Range`` is a *pure* sampler: it holds a seeded generator and nothing else.
    Every call to :meth:`sample` draws fresh values from that generator; it never
    memoizes.  Caching a drawn value set across call sites is the caller's job —
    ``RFOptuna`` does it (coverage enumeration draws once, suggest reuses the
    cache), and ``RFRandomSearch`` does not need it because it draws one value at
    a time.

    Args:
        start: Lower bound (inclusive).
        end: Upper bound (inclusive).
        dtype: ``"int"`` or ``"float"``.  Inferred from *start*/*end* types
            when not provided.
        log: If ``True``, sample in log-space (start and end must be > 0).
            Mutually exclusive with *step*.
        step: Discretisation step.  When set, sampled values are multiples of
            *step* starting from *start*.  Mutually exclusive with *log*.
        sample_n: Default number of distinct values ``sample(n)`` draws when a
            continuous range must be enumerated up front — currently
            ``RFOptuna`` with ``build_all_indexes=True``, which has to know every
            RAG index the search space can reach.  It does not change how
            ``RFRandomSearch`` works: that path calls ``sample(1)`` per run, so
            ``num_runs`` still governs how many configs appear.
        seed: Seed for this range's own random generator.  ``None`` (the default)
            leaves the generator unseeded; both ``RFOptuna`` and
            ``RFRandomSearch`` stamp their run ``seed`` onto every ``Range`` in
            the search space (via :meth:`set_seed`) so the values a run explores
            are reproducible.  Seeding is purely for reproducibility — it has
            nothing to do with making two call sites agree, since ``Range`` no
            longer caches anything.
    """

    def __init__(
        self,
        start,
        end,
        dtype: str | None = None,
        log: bool = False,
        step: int | float | None = None,
        sample_n: int = 3,
        seed: int | None = None,
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
        if not isinstance(sample_n, int) or isinstance(sample_n, bool) or sample_n < 1:
            raise ValueError("sample_n must be a positive integer.")
        self.start = start
        self.end = end
        self.log = log
        self.step = step
        self.sample_n = sample_n
        self.seed = seed
        self._rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """(Re)seed this range's generator.

        Called by ``RFOptuna`` and ``RFRandomSearch`` with their run ``seed`` so
        the values a run explores are reproducible.  ``Range`` holds no cached
        state, so reseeding only affects future draws.
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def _draw(self):
        """One value from the range, honouring *log* and *step*."""
        if self.dtype == "int":
            if self.log:
                log_low, log_high = math.log(self.start), math.log(self.end)
                return int(round(math.exp(self._rng.uniform(log_low, log_high))))
            if self.step is not None:
                step = int(self.step)
                n_steps = (self.end - self.start) // step
                return self.start + self._rng.randint(0, n_steps) * step
            return self._rng.randint(self.start, self.end)

        # dtype == "float"
        if self.log:
            log_low, log_high = math.log(self.start), math.log(self.end)
            return math.exp(self._rng.uniform(log_low, log_high))
        if self.step is not None:
            n_steps = int((self.end - self.start) / self.step)
            return self.start + self._rng.randint(0, n_steps) * self.step
        return self._rng.uniform(self.start, self.end)

    def _distinct_capacity(self) -> int | float:
        """How many distinct values a draw can return, ``inf`` if continuous."""
        if self.step is not None:
            return int((self.end - self.start) / self.step) + 1
        if self.dtype == "int":
            return int(round(self.end)) - int(round(self.start)) + 1
        return math.inf

    def sample(self, n: int) -> list:
        """Draw up to *n* distinct values from the range ``[self.start, self.end]``.

        Always returns a list (a 1-element list when ``n == 1``).  Every call
        draws fresh from this range's seeded generator — there is no memoization,
        so two calls return two independent draws.  Callers that need a stable
        value set across call sites (``RFOptuna``'s coverage enumeration vs.
        suggest) must cache the result themselves.

        ``RFRandomSearch`` calls ``sample(1)`` once per run and uses the single
        element.  ``RFOptuna`` calls ``sample(sample_n)`` once during coverage
        enumeration and reuses the cached list at suggest time.  Both honour
        *log* and *step*, and both draw from this range's seeded generator, so
        the values a run explores follow from the run ``seed``.

        Fewer than *n* values are returned only when the range cannot yield that
        many distinct ones: ``Range(1, 2, dtype="int")`` has only two.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer.")
        target = min(n, self._distinct_capacity())
        values: list = []
        seen: set = set()
        # Bounded so a narrow range cannot spin forever once it is exhausted.
        for _ in range(int(target) * _SAMPLE_ATTEMPT_MULTIPLIER):
            if len(values) >= target:
                break
            value = self._draw()
            if value not in seen:
                seen.add(value)
                values.append(value)
        values.sort()
        return values


class List:
    """Represents a list of values for a hyperparameter."""

    def __init__(self, values):
        if not isinstance(values, list):
            raise ValueError("List expects a list of values.")
        self.values = values

    def sample(self):
        """Sample a value from the list."""
        return random.choice(self.values)
