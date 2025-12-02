from collections import defaultdict
from collections.abc import Callable
from typing import Any
import ray

from .online_strategies import (
    HoeffdingApproximationStrategy,
    NormalApproximationStrategy,
    OnlineAggregationStrategy,
    WilsonApproximationStrategy,
)


class Aggregator:
    """
    Handles all aggregation logic for batch processing results and metrics.
    Supports both offline and online (live) aggregation with progress tracking.
    """

    def __init__(
        self,
        online_strategy: str = "normal",
        confidence_level: float = 0.95,
        use_fpc: bool = True,
    ):
        """
        Initialize the aggregator.

        Args:
            online_strategy: Strategy for online aggregation ('basic', 'normal', 'wilson', 'hoeffding')
            confidence_level: Confidence level for confidence intervals (when using confidence strategy)
            use_fpc: Whether to apply finite population correction
        """
        self.online_strategy = self._create_strategy(online_strategy, confidence_level, use_fpc)

    def _create_strategy(self, strategy_name: str, confidence_level: float, use_fpc: bool) -> OnlineAggregationStrategy:
        """Create and return the appropriate online aggregation strategy."""
        if strategy_name == "normal":
            return NormalApproximationStrategy(confidence_level, use_fpc)
        elif strategy_name == "wilson":
            return WilsonApproximationStrategy(confidence_level, use_fpc)
        elif strategy_name == "hoeffding":
            return HoeffdingApproximationStrategy(confidence_level, use_fpc)
        else:
            raise ValueError(
                f"Unknown online aggregation strategy: {strategy_name}. Available: ['basic', 'normal', 'wilson', 'hoeffding']"
            )

    def set_online_strategy(
        self,
        strategy_name: str = "normal",
        confidence_level: float = 0.95,
        use_fpc: bool = True,
    ):
        """
        Set a new online aggregation strategy.

        Args:
            strategy_name: Strategy for online aggregation ('basic', 'normal', 'wilson', 'hoeffding')
            confidence_level: Confidence level for confidence intervals
            use_fpc: Whether to apply finite population correction
        """
        # Preserve the current population size if it was set
        current_population_size = getattr(self.online_strategy, "total_population_size", None)

        # Create new strategy
        self.online_strategy = self._create_strategy(strategy_name, confidence_level, use_fpc)

        # Restore population size if it was previously set
        if current_population_size is not None:
            self.online_strategy.total_population_size = current_population_size

    def set_total_population_size(self, total_population_size: int):
        """Set the total population size for the online aggregation strategy."""
        self.online_strategy.total_population_size = total_population_size

    def get_live_metrics(self, aggregate_metrics: dict[str, list], accumulate_metrics_fn: Callable) -> dict[str, Any]:
        """
        Calculate live metrics from partial aggregated data using the configured strategy.

        Args:
            aggregate_metrics: Current aggregated metrics
            accumulate_metrics_fn: Function to compute final metrics from aggregated data

        Returns:
            Dictionary of live metrics
        """
        return self.online_strategy.compute_live_metrics(aggregate_metrics, accumulate_metrics_fn)

    def _format_metrics_for_display(self, metrics: dict[str, Any]) -> dict[str, str]:
        """
        Format metrics for progress bar display, adding ± notation for algebraic metrics.

        Args:
            metrics: Dictionary of metrics (may contain structured format)

        Returns:
            Dictionary with formatted metric strings
        """
        formatted = {}

        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                # Structured metric format
                value = metric_data.get("value", 0)
                is_algebraic = metric_data.get("is_algebraic", False)
                is_distributive = metric_data.get("is_distributive", False)
                margin_of_error = metric_data.get("margin_of_error", None)

                if (is_algebraic or is_distributive) and margin_of_error is not None:
                    # Format with ± margin of error for both algebraic and distributive
                    if isinstance(value, float):
                        formatted[metric_name] = f"{value:.4f}±{margin_of_error:.4f}"
                    else:
                        formatted[metric_name] = f"{value}±{margin_of_error:.4f}"
                else:
                    # Just show the value
                    formatted[metric_name] = str(value)
            else:
                # Simple metric format
                formatted[metric_name] = str(metric_data)

        return formatted

    def aggregate_batch_data(
        self,
        aggregate_results: dict[str, list],
        aggregate_metrics: dict[str, list],
        new_data: list[tuple],
    ) -> tuple[dict[str, list], dict[str, list]]:
        """
        Aggregate new batch data into existing aggregated results and metrics.

        Args:
            aggregate_results: Existing aggregated results Dict[str, List[Any]]
            aggregate_metrics: Existing aggregated metrics Dict[str, List[Dict[str, Any]]]
            new_data: List of tuples (result, metric) from completed batches
                result is a Dict[str, List[Any]]
                metric is a Dict[str, Dict[str, Any]]
        Returns:
            Tuple of (updated_results, updated_metrics)
        """
        for result, metric in new_data:
            for key in result:
                if key in aggregate_results:
                    aggregate_results[key].extend(result[key])
                else:
                    aggregate_results[key] = result[key].copy()

            for key in metric:
                if key in aggregate_metrics:
                    aggregate_metrics[key].append(metric[key])
                else:
                    aggregate_metrics[key] = [metric[key]]

        return aggregate_results, aggregate_metrics

    def aggregate_with_progress(
        self,
        futures: list,
        accumulate_metrics_fn: Callable = None,
    ) -> tuple[dict[str, list], dict[str, list]]:
        """
        Aggregate results from futures as they complete.
        Progress and metrics display is handled by the Rich table in the controller.

        If the accumulate_metrics_fn is provided, aggregate and accumulate batch-level metrics online.
        Otherwise, aggregate and return aggregated results only and metrics will be computed at the dataset level later.

        Args:
            futures: List of Ray futures to aggregate
            accumulate_metrics_fn: Optional function for live metrics calculation

        Returns:
            Tuple of (aggregated_results, aggregated_metrics)
        """
        # Initialize aggregation containers
        aggregate_results = defaultdict(list)
        aggregate_metrics = defaultdict(list)
        remaining_futures = futures.copy()

        # Aggregate without tqdm - progress is handled by Rich table display
        while remaining_futures:
            ready, remaining_futures = ray.wait(
                remaining_futures,
                num_returns=min(5, len(remaining_futures)),
                timeout=1.0,
            )

            if ready:
                # Get completed batch data
                new_data = ray.get(ready)

                # Aggregate the new data
                aggregate_results, aggregate_metrics = self.aggregate_batch_data(
                    aggregate_results, aggregate_metrics, new_data
                )

        return aggregate_results, aggregate_metrics

    def compute_final_metrics(
        self,
        aggregated_results: dict[str, list],
        aggregated_metrics: dict[str, list],
        compute_metrics_fn: Callable = None,
        accumulate_metrics_fn: Callable = None,
        start_time: float = None,
        end_time: float = None,
    ) -> dict[str, Any]:
        """
        Compute final metrics from aggregated data.
        If the accumulate_metrics_fn is provided, accumulate batch-level metrics offline, otherwise compute metrics at dataset level

        Args:
            aggregated_results: Aggregated batch results
            aggregated_metrics: Aggregated batch metrics
            compute_metrics_fn: Function to compute metrics at dataset level
            accumulate_metrics_fn: Function to accumulate batch-level metrics
            start_time: Processing start time
            end_time: Processing end time

        Returns:
            Dictionary of final computed metrics
        """
        if accumulate_metrics_fn and aggregated_metrics:
            # Use batch-level metrics accumulation
            samples_processed = sum(m.get("value", 0) for m in aggregated_metrics.get("Samples Processed", [{}]))
            default_metrics = {
                "Samples Processed": {
                    "value": samples_processed,
                    "is_algebraic": False,
                },
            }

            if start_time and end_time:
                processing_time = end_time - start_time
                default_metrics.update(
                    {
                        "Processing Time": {
                            "value": f"{processing_time:.2f} seconds",
                            "is_algebraic": False,
                        },
                        "Samples Per Second": {
                            "value": f"{samples_processed / processing_time:.2f}",
                            "is_algebraic": False,
                        },
                    }
                )

            cumulative_metrics = {
                **default_metrics,
                **accumulate_metrics_fn(aggregated_metrics),
            }
        else:
            # Use dataset-level metrics computation
            samples_processed = len(aggregated_results.get("generated_text", []))
            default_metrics = {
                "Samples Processed": {
                    "value": samples_processed,
                    "is_algebraic": False,
                },
            }

            if start_time and end_time:
                processing_time = end_time - start_time
                default_metrics.update(
                    {
                        "Processing Time": {
                            "value": f"{processing_time:.2f} seconds",
                            "is_algebraic": False,
                        },
                        "Samples Per Second": {
                            "value": f"{samples_processed / processing_time:.2f}",
                            "is_algebraic": False,
                        },
                    }
                )

            cumulative_metrics = {
                **default_metrics,
                **(compute_metrics_fn(aggregated_results) if compute_metrics_fn else {}),
            }

        return cumulative_metrics


# Export classes for external use
__all__ = ["Aggregator"]
