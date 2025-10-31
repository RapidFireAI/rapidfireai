import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from typing_extensions import override

from scipy.stats import norm


class OnlineAggregationStrategy(ABC):
    """Abstract base class for online aggregation strategies."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize strategy with confidence level.

        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
        """
        self.confidence_level = confidence_level

    @abstractmethod
    def get_confidence_interval_algebraic(
        self,
        estimate: float,
        sample_size: int,
        confidence_level: float = None,
        value_range: tuple[float, float] = None,
    ) -> tuple[float, float, float]:
        """
        Calculate confidence interval for algebraic metrics (proportions/averages).

        Args:
            estimate: The observed proportion/average
            sample_size: Number of samples
            confidence_level: Confidence level (default uses instance level)
            value_range: Bounds for the metric values (required - must specify (min, max))

        Returns:
            Tuple of (estimate, lower_bound, upper_bound)

        Raises:
            ValueError: If value_range is None or contains None values
        """
        pass

    @abstractmethod
    def get_confidence_interval_distributive(
        self,
        estimate: float,
        sample_size: int,
        population_size: int,
        value_range: tuple[float, float] = None,
        confidence_level: float = None,
    ) -> tuple[float, float, float]:
        """
        Calculate confidence interval for distributive metrics (COUNT/SUM).

        Args:
            estimate: Raw count or sum from sample
            sample_size: Number of samples processed
            population_size: Total population size
            value_range: Bounds for individual values (required - must specify (min, max))
            confidence_level: Confidence level (default uses instance level)

        Returns:
            Tuple of (final_estimate, lower_bound, upper_bound)

        Raises:
            ValueError: If value_range is None or contains None values
        """
        pass

    def compute_live_metrics(
        self, aggregate_metrics: dict[str, list], accumulate_metrics_fn: Callable
    ) -> dict[str, Any]:
        """Compute live metrics with confidence intervals."""
        if not accumulate_metrics_fn or not aggregate_metrics:
            return {}

        try:
            total_samples = sum(m.get("value", 0) for m in aggregate_metrics.get("Samples Processed", [{}]))
            if total_samples > 0:
                base_metrics = accumulate_metrics_fn(aggregate_metrics)
                return self.add_confidence_interval_info(base_metrics, total_samples)
        except Exception:
            return {}

        return {}

    def add_confidence_interval_info(self, metrics: dict[str, Any], sample_size: int) -> dict[str, Any]:
        """Add confidence interval information to metrics."""
        if sample_size == 0:
            return metrics

        updated_metrics = metrics.copy()

        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                if metric_data.get("is_algebraic", False):
                    metric_value = float(metric_data.get("value", 0))
                    value_range = metric_data.get("value_range", (0, 1))  # Default to [0,1] for backward compatibility
                    estimate, lower, upper = self.get_confidence_interval_algebraic(
                        metric_value, sample_size, value_range=value_range
                    )

                elif metric_data.get("is_distributive", False):
                    metric_value = float(metric_data.get("value", 0))
                    population_size = getattr(self, "total_population_size", None)
                    # For distributive metrics, value_range should be explicitly specified
                    # Default to (0, 1) only for backward compatibility with COUNT metrics
                    value_range = metric_data.get("value_range", (0, 1))

                    estimate, lower, upper = self.get_confidence_interval_distributive(
                        metric_value, sample_size, population_size, value_range
                    )
                else:
                    continue

                # Calculate margin of error and update
                margin_of_error = (upper - lower) / 2
                updated_metrics[metric_name].update(
                    {
                        "value": estimate,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "margin_of_error": margin_of_error,
                        "confidence_level": self.confidence_level,
                        "ci_method": self.get_name(),
                    }
                )

        return updated_metrics

    def apply_finite_population_correction_to_stderr(
        self, std_error: float, sample_size: int, use_fpc: bool = True
    ) -> float:
        """
        Apply finite population correction to standard error.

        Args:
            std_error: Original standard error
            sample_size: Number of samples
            use_fpc: Whether to apply FPC (default True)

        Returns:
            Corrected standard error
        """
        if not use_fpc or not hasattr(self, "total_population_size") or not self.total_population_size:
            return std_error

        if sample_size >= self.total_population_size:
            return 0
        elif sample_size < self.total_population_size and self.total_population_size > 1:
            fpc_factor = math.sqrt((self.total_population_size - sample_size) / (self.total_population_size - 1))
            return std_error * fpc_factor
        else:
            return std_error

    def apply_finite_population_correction_to_variance(
        self, variance: float, sample_size: int, use_fpc: bool = True
    ) -> float:
        """
        Apply finite population correction to variance.

        Args:
            variance: Original variance
            sample_size: Number of samples
            use_fpc: Whether to apply FPC (default True)

        Returns:
            Corrected variance
        """
        if not use_fpc or not hasattr(self, "total_population_size") or not self.total_population_size:
            return variance

        if sample_size >= self.total_population_size:
            return 0
        elif sample_size < self.total_population_size and self.total_population_size > 1:
            fpc_factor = math.sqrt((self.total_population_size - sample_size) / (self.total_population_size - 1))
            return variance * (fpc_factor**2)
        else:
            return variance

    def validate_and_parse_range(self, value_range: tuple[float, float]) -> tuple[float, float, float]:
        """
        Validate and parse value range, returning (min_val, max_val, range_width).

        Args:
            value_range: Tuple of (min_val, max_val)

        Returns:
            Tuple of (min_val, max_val, range_width)

        Raises:
            ValueError: If range is None or contains None values
        """
        if value_range is None:
            raise ValueError("value_range must be provided and cannot be None")

        min_val, max_val = value_range

        if min_val is None or max_val is None:
            raise ValueError("value_range bounds cannot be None - both min and max values must be specified")

        if min_val >= max_val:
            raise ValueError(f"Invalid range: min_val ({min_val}) must be less than max_val ({max_val})")

        range_width = max_val - min_val
        return min_val, max_val, range_width

    def get_variance_estimate(self, estimate: float, a: float, b: float, sample_size: int) -> float:
        """
        Get variance estimate for bounded values.

        Args:
            estimate: The sample mean/proportion
            a: Lower bound
            b: Upper bound
            sample_size: Number of samples

        Returns:
            Variance estimate
        """
        # Determine if this is a proportion [0,1] or general average
        is_proportion = a == 0 and b == 1

        if is_proportion:
            # True proportion - use binomial variance
            return estimate * (1 - estimate) / sample_size
        else:
            # Average of bounded values - use uniform distribution approximation
            # For uniform distribution on [a,b], variance = (b-a)²/12
            # For sample mean, variance = population_variance/n = (b-a)²/(12*n)
            range_width = b - a
            return (range_width**2) / (12 * sample_size)

    def get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level."""
        alpha = 1 - confidence_level
        return norm.ppf(1 - alpha / 2)

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the strategy."""
        pass


class NormalApproximationStrategy(OnlineAggregationStrategy):
    """Normal approximation confidence intervals for both algebraic and distributive metrics."""

    def __init__(self, confidence_level: float = 0.95, use_fpc: bool = True):
        """
        Initialize normal approximation strategy.

        Args:
            confidence_level: Confidence level for intervals
            use_fpc: Whether to apply finite population correction for distributive metrics
        """
        super().__init__(confidence_level)
        self.use_fpc = use_fpc

    def normal_approximation(
        self,
        estimate: float,
        sample_size: int,
        population_size: int,
        value_range: tuple[float, float] = None,
        confidence_level: float = None,
        metric_type: str = "algebraic",
    ) -> tuple[float, float, float]:
        """Normal approximation for algebraic metrics."""
        if sample_size == 0:
            return 0, 0, 0

        if confidence_level is None:
            confidence_level = self.confidence_level

        a, b, range_width = self.validate_and_parse_range(value_range)

        sample_mean = estimate
        if metric_type == "distributive":
            sample_mean = estimate / sample_size

        if sample_mean == a or sample_mean == b:
            estimate = population_size * sample_mean
            return estimate, estimate, estimate

        variance = self.get_variance_estimate(sample_mean, a, b, sample_size)
        variance = (population_size) ** 2 * variance

        std_error = math.sqrt(variance)
        std_error = self.apply_finite_population_correction_to_stderr(std_error, sample_size, self.use_fpc)
        z_score = self.get_z_score(confidence_level)
        margin_of_error = z_score * std_error

        estimate = population_size * sample_mean
        lower_bound = max(a * population_size, estimate - margin_of_error)
        upper_bound = min(b * population_size, estimate + margin_of_error)

        return estimate, lower_bound, upper_bound

    @override
    def get_confidence_interval_algebraic(
        self,
        estimate: float,
        sample_size: int,
        confidence_level: float = None,
        value_range: tuple[float, float] = None,
    ) -> tuple[float, float, float]:
        """Normal approximation for algebraic metrics."""
        return self.normal_approximation(
            estimate=estimate,
            sample_size=sample_size,
            population_size=1,
            value_range=value_range,
            confidence_level=confidence_level,
            metric_type="algebraic",
        )

    @override
    def get_confidence_interval_distributive(
        self,
        estimate: float,
        sample_size: int,
        population_size: int,
        value_range: tuple[float, float] = None,
        confidence_level: float = None,
    ) -> tuple[float, float, float]:
        """Normal approximation for distributive metrics."""
        return self.normal_approximation(
            estimate=estimate,
            sample_size=sample_size,
            population_size=population_size,
            value_range=value_range,
            confidence_level=confidence_level,
            metric_type="distributive",
        )

    def get_name(self) -> str:
        return "normal"


class WilsonApproximationStrategy(NormalApproximationStrategy):
    """Wilson score interval for algebraic metrics, normal approximation for distributive."""

    def __init__(self, confidence_level: float = 0.95, use_fpc: bool = True):
        """
        Initialize Wilson approximation strategy.

        Args:
            confidence_level: Confidence level for intervals
            use_fpc: Whether to apply finite population correction for distributive metrics
        """
        super().__init__(confidence_level, use_fpc)

    def wilson_approximation(
        self,
        estimate: float,
        sample_size: int,
        confidence_level: float = None,
    ) -> tuple[float, float, float]:
        """Wilson approximation for algebraic metrics."""
        z = self.get_z_score(confidence_level)
        p = estimate
        n = sample_size

        effective_n = n
        if self.use_fpc and hasattr(self, "total_population_size") and self.total_population_size:
            if n >= self.total_population_size:
                return p, p, p
            elif n < self.total_population_size and self.total_population_size > 1:
                fpc_factor = math.sqrt((self.total_population_size - n) / (self.total_population_size - 1))
                effective_n = n / (fpc_factor**2)

        denominator = 1 + z**2 / effective_n
        center = (p + z**2 / (2 * effective_n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / effective_n + z**2 / (4 * effective_n**2)) / denominator

        return p, max(0, center - margin), min(1, center + margin)

    @override
    def get_confidence_interval_algebraic(
        self,
        estimate: float,
        sample_size: int,
        confidence_level: float = None,
        value_range: tuple[float, float] = None,
    ) -> tuple[float, float, float]:
        """Wilson confidence interval for proportions, normal approximation for others."""
        if sample_size == 0:
            return 0, 0, 0

        if confidence_level is None:
            confidence_level = self.confidence_level

        min_val, max_val, range_width = self.validate_and_parse_range(value_range)

        if estimate in (min_val, max_val):
            return estimate, estimate, estimate

        if value_range != (0, 1):
            return self.normal_approximation(
                estimate=estimate,
                sample_size=sample_size,
                population_size=1,
                value_range=value_range,
                confidence_level=confidence_level,
                metric_type="algebraic",
            )

        # Wilson interval for [0,1] proportions
        return self.wilson_approximation(
            estimate=estimate,
            sample_size=sample_size,
            confidence_level=confidence_level,
        )

    def get_name(self) -> str:
        return "wilson"


class HoeffdingApproximationStrategy(OnlineAggregationStrategy):
    """Hoeffding's inequality for both algebraic and distributive metrics."""

    def __init__(self, confidence_level: float = 0.95, use_fpc: bool = True):
        """
        Initialize Hoeffding approximation strategy.

        Args:
            confidence_level: Confidence level for intervals
            use_fpc: Whether to apply Serfling finite population correction
        """
        super().__init__(confidence_level)
        self.use_fpc = use_fpc

    def hoeffding_approximation(
        self,
        estimate: float,
        sample_size: int,
        population_size: int,
        value_range: tuple[float, float] = None,
        confidence_level: float = None,
        metric_type: str = "algebraic",
    ) -> tuple[float, float, float]:
        """Hoeffding approximation for algebraic metrics."""
        if sample_size == 0:
            return 0, 0, 0

        if confidence_level is None:
            confidence_level = self.confidence_level

        a, b, range_width = self.validate_and_parse_range(value_range)
        sample_mean = estimate
        if metric_type == "distributive":
            sample_mean = estimate / sample_size

        if sample_mean in (a, b):
            estimate = population_size * sample_mean
            return estimate, estimate, estimate

        alpha = 1 - confidence_level
        epsilon = range_width * math.sqrt(math.log(2 / alpha) / (2 * sample_size))
        epsilon = self.apply_finite_population_correction_to_stderr(epsilon, sample_size, self.use_fpc)

        estimate = population_size * sample_mean
        epsilon = population_size * epsilon
        lower_bound = max(population_size * a, estimate - epsilon)
        upper_bound = min(population_size * b, estimate + epsilon)

        return estimate, lower_bound, upper_bound

    @override
    def get_confidence_interval_algebraic(
        self,
        estimate: float,
        sample_size: int,
        confidence_level: float = None,
        value_range: tuple[float, float] = None,
    ) -> tuple[float, float, float]:
        """Hoeffding bounds for algebraic metrics."""
        return self.hoeffding_approximation(
            estimate=estimate,
            sample_size=sample_size,
            population_size=1,
            value_range=value_range,
            confidence_level=confidence_level,
            metric_type="algebraic",
        )

    @override
    def get_confidence_interval_distributive(
        self,
        estimate: float,
        sample_size: int,
        population_size: int,
        value_range: tuple[float, float] = None,
        confidence_level: float = None,
    ) -> tuple[float, float, float]:
        """Hoeffding bounds for distributive metrics."""
        return self.hoeffding_approximation(
            estimate=estimate,
            sample_size=sample_size,
            population_size=population_size,
            value_range=value_range,
            confidence_level=confidence_level,
            metric_type="distributive",
        )

    def get_name(self) -> str:
        return "hoeffding"


# Export classes for external use
__all__ = [
    "OnlineAggregationStrategy",
    "NormalApproximationStrategy",
    "WilsonApproximationStrategy",
    "HoeffdingApproximationStrategy",
]
