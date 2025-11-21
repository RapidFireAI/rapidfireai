"""
Progress display using pandas DataFrames for multi-config experiments.

This module provides a clean, updating table display for tracking
run progress, metrics, and confidence intervals in Jupyter notebooks.

Requires: pandas and IPython (for Jupyter display)
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from IPython.display import display


class ContextBuildingDisplay:
    """
    Manages a live-updating DataFrame for tracking RAG source preprocessing progress.

    Displays:
    - RAG Source ID (hash)
    - Status (Building/Complete)
    - Duration
    - Details (e.g., FAISS, GPU/CPU)
    """

    def __init__(self, contexts_to_build: list[dict]):
        """
        Initialize the context building display.

        Args:
            contexts_to_build: List of context info dicts with keys:
                             context_hash, context_id, enable_gpu, etc.
        """
        self.contexts = contexts_to_build
        self.context_data = {
            ctx["context_hash"]: {
                "context_id": ctx.get("context_id", "?"),
                "status": "Building",
                "start_time": ctx.get("start_time"),
                "duration": None,
                "enable_gpu": ctx.get("enable_gpu", False),
            }
            for ctx in contexts_to_build
        }
        self.display_handle = None

    def _create_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with current context building data."""
        rows = []
        for ctx in self.contexts:
            context_hash = ctx["context_hash"]
            data = self.context_data[context_hash]

            # Format context hash (show first 8 chars)
            ctx_hash_display = context_hash[:8] + "..."

            # Get context ID from database
            ctx_id = data["context_id"]

            # Format status
            status = data["status"]

            # Format duration
            if data["duration"] is not None:
                duration = f"{data['duration']:.1f}s"
            elif data["start_time"] is not None:
                elapsed = time.time() - data["start_time"]
                duration = f"{elapsed:.1f}s"
            else:
                duration = "-"

            # Format details
            details = "FAISS, " + ("GPU" if data["enable_gpu"] else "CPU")

            rows.append(
                {
                    "RAG Source ID": ctx_id,
                    "Status": status,
                    "Duration": duration,
                    "Details": details,
                }
            )

        return pd.DataFrame(rows)

    def start(self):
        """Start the live display."""
        df = self._create_dataframe()
        print("=== Preprocessing RAG Sources ===")
        # Hide index for cleaner display
        styled_df = df.style.hide(axis="index")
        self.display_handle = display(styled_df, display_id=True)

    def stop(self):
        """Stop the live display."""
        if self.display_handle:
            # Final render with completed status
            df = self._create_dataframe()
            styled_df = df.style.hide(axis="index")
            self.display_handle.update(styled_df)

    def _render(self):
        """Update the DataFrame display."""
        if self.display_handle:
            df = self._create_dataframe()
            styled_df = df.style.hide(axis="index")
            self.display_handle.update(styled_df)

    def update_context(self, context_hash: str, status: str = None, duration: float = None):
        """
        Update status for a specific context.

        Args:
            context_hash: Hash of the context to update
            status: Status string ("Building", "Complete", "Failed")
            duration: Final duration in seconds
        """
        if context_hash not in self.context_data:
            return

        data = self.context_data[context_hash]

        if status is not None:
            # Capitalize status for consistency
            data["status"] = status.capitalize()

        if duration is not None:
            data["duration"] = duration

        # Re-render the display
        self._render()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class PipelineProgressDisplay:
    """
    Manages a live-updating DataFrame for tracking multi-config experiment progress.

    Displays:
    - Run ID and name
    - Model name
    - Current shard progress (X/Y)
    - Confidence interval
    - Pipeline configuration fields (search_type, chunk_size, etc.)
    - Other metrics (accuracy, throughput, etc.)
    """

    def __init__(self, pipelines: list[dict], num_shards: int):
        """
        Initialize the progress display.

        Args:
            pipelines: List of pipeline info dicts with keys:
                      - pipeline_id (required)
                      - pipeline_config (required)
                      - model_name (required)
                      - search_type (optional)
                      - rag_k (optional)
                      - top_n (optional)
                      - chunk_size (optional)
                      - chunk_overlap (optional)
                      - sampling_params (optional)
                      - prompt_manager_k (optional)
                      - model_config (optional)
            num_shards: Total number of shards
        """
        self.pipelines = pipelines
        self.num_shards = num_shards

        # Extract pipeline_id, name, and model, plus all optional fields
        self.pipeline_data = {}
        self.pipeline_metadata = {}  # Store additional metadata for each pipeline

        for pipeline_info in pipelines:
            pid = pipeline_info["pipeline_id"]
            pipeline_config = pipeline_info["pipeline_config"]
            pipeline_name = pipeline_config.get("pipeline_name", f"Pipeline {pid}")
            model_name = pipeline_info.get("model_name", "Unknown")

            self.pipeline_data[pid] = {
                "name": pipeline_name,
                "model": model_name,
                "shard": 0,
                "confidence": "-",
                "metrics": {},
                "status": "ONGOING",
            }

            # Store additional metadata fields (only non-None values)
            metadata = {}
            for key in ["search_type", "rag_k", "top_n", "chunk_size", "chunk_overlap",
                       "sampling_params", "prompt_manager_k", "model_config"]:
                if key in pipeline_info:
                    metadata[key] = pipeline_info[key]
            self.pipeline_metadata[pid] = metadata

        self.display_handle = None

    def _create_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with current pipeline progress data."""
        # Collect all unique metric names across all pipelines
        all_metric_names = set()
        for pipeline_info in self.pipelines:
            pid = pipeline_info["pipeline_id"]
            data = self.pipeline_data[pid]
            all_metric_names.update(data["metrics"].keys())

        # Sort metrics for consistent display order (prefer common metrics first)
        metric_precedence = ["Accuracy", "Precision", "Recall", "F1 Score", "NDCG@5", "MRR", "Throughput", "Total", "Samples Processed"]
        ordered_metrics = []
        remaining_metrics = []

        for metric in metric_precedence:
            if metric in all_metric_names:
                ordered_metrics.append(metric)

        for metric in sorted(all_metric_names):
            if metric not in ordered_metrics:
                remaining_metrics.append(metric)

        ordered_metrics.extend(remaining_metrics)

        # Collect all unique metadata field names across pipelines (for column consistency)
        all_metadata_fields = set()
        for pipeline_info in self.pipelines:
            pid = pipeline_info["pipeline_id"]
            metadata = self.pipeline_metadata.get(pid, {})
            all_metadata_fields.update(metadata.keys())

        # Define display order for metadata fields
        metadata_precedence = ["search_type", "rag_k", "top_n", "chunk_size", "chunk_overlap",
                              "sampling_params", "prompt_manager_k", "model_config"]
        ordered_metadata = []
        remaining_metadata = []

        for field in metadata_precedence:
            if field in all_metadata_fields:
                ordered_metadata.append(field)

        for field in sorted(all_metadata_fields):
            if field not in ordered_metadata:
                remaining_metadata.append(field)

        ordered_metadata.extend(remaining_metadata)

        rows = []
        for pipeline_info in self.pipelines:
            pid = pipeline_info["pipeline_id"]

            # Skip if pipeline data not found (shouldn't happen, but defensive)
            if pid not in self.pipeline_data:
                continue

            data = self.pipeline_data[pid]
            metadata = self.pipeline_metadata.get(pid, {})

            # Format progress
            progress = f"{data['shard']}/{self.num_shards}"

            # Format confidence interval
            confidence = data["confidence"]
            if confidence != "-" and isinstance(confidence, (int, float)):
                confidence = f"{confidence:.3f}"

            # Get status
            status = data.get("status", "ONGOING")

            # Start with standard columns (ensure they always exist even if empty)
            row = {
                "Run ID": pid,
                "Model": data.get("model", "-"),
                "Status": status,
                "Progress": progress,
                "Conf. Interval": str(confidence),
            }

            # Add metadata fields
            for field_name in ordered_metadata:
                if field_name in metadata:
                    value = metadata[field_name]
                    # Format the value for display
                    formatted_value = self._format_metadata_field(field_name, value)
                    row[field_name] = formatted_value
                else:
                    row[field_name] = "-"

            # Add all metrics with confidence intervals
            for metric_name in ordered_metrics:
                metric_data = data["metrics"].get(metric_name, {})
                if isinstance(metric_data, dict):
                    metric_value = metric_data.get("value", "-")
                    lower_bound = metric_data.get("lower_bound")
                    upper_bound = metric_data.get("upper_bound")
                    margin_of_error = metric_data.get("margin_of_error")
                    is_algebraic = metric_data.get("is_algebraic", False)
                else:
                    metric_value = metric_data
                    lower_bound = upper_bound = margin_of_error = None
                    is_algebraic = False

                # Format metric value based on type and name
                if metric_value != "-":
                    if isinstance(metric_value, (int, float)):
                        # Format percentages for common metrics (0-1 range)
                        if metric_name in ["Accuracy", "Precision", "Recall", "F1 Score", "NDCG@5", "MRR"]:
                            formatted_value = f"{metric_value:.2%}"
                            # Add confidence interval if available
                            if is_algebraic and lower_bound is not None and upper_bound is not None:
                                # Show as "value [lower, upper]" format
                                formatted_value += f" [{lower_bound:.2%}, {upper_bound:.2%}]"
                        elif metric_name == "Throughput":
                            formatted_value = f"{metric_value:.1f}/s"
                        elif metric_name in ["Total", "Samples Processed"]:
                            formatted_value = f"{int(metric_value):,}"
                        else:
                            # Default formatting for other numeric metrics
                            if abs(metric_value) < 1:
                                formatted_value = f"{metric_value:.4f}"
                                # Add CI for algebraic metrics
                                if is_algebraic and lower_bound is not None and upper_bound is not None:
                                    formatted_value += f" [{lower_bound:.4f}, {upper_bound:.4f}]"
                            else:
                                formatted_value = f"{metric_value:.2f}"
                                # Add CI for algebraic metrics
                                if is_algebraic and lower_bound is not None and upper_bound is not None:
                                    formatted_value += f" [{lower_bound:.2f}, {upper_bound:.2f}]"

                        metric_value = formatted_value

                row[metric_name] = str(metric_value)

            rows.append(row)

        return pd.DataFrame(rows)

    def _format_metadata_field(self, field_name: str, value: Any) -> str:
        """
        Format a metadata field value for display.

        Args:
            field_name: Name of the metadata field
            value: Value to format

        Returns:
            Formatted string representation
        """
        if value is None:
            return "-"

        # Handle dict values (like sampling_params, model_config)
        if isinstance(value, dict):
            # Format as compact JSON-like string
            if field_name == "sampling_params":
                # Format sampling params more nicely
                parts = []
                if "temperature" in value:
                    parts.append(f"temp={value['temperature']}")
                if "top_p" in value:
                    parts.append(f"top_p={value['top_p']}")
                if "top_k" in value:
                    parts.append(f"top_k={value['top_k']}")
                if "max_tokens" in value:
                    parts.append(f"max_t={value['max_tokens']}")
                # If we have formatted parts, use them, otherwise show abbreviated dict
                if parts:
                    return ", ".join(parts)
                else:
                    return str(value)[:50]  # Truncate long dicts
            elif field_name == "model_config":
                # Format model config as key=value pairs
                parts = [f"{k}={v}" for k, v in value.items() if k != "model"]
                return ", ".join(parts) if parts else "-"
            else:
                # Default dict formatting
                return str(value)[:50]  # Truncate long dicts

        # Handle list values
        if isinstance(value, list):
            return f"[{', '.join(str(v) for v in value[:5])}]"  # Show first 5 items

        # Handle numeric values
        if isinstance(value, (int, float)):
            return str(value)

        # Handle string values
        if isinstance(value, str):
            return value

        # Fallback: convert to string
        return str(value)

    def start(self):
        """Start the live display."""
        df = self._create_dataframe()
        print("\n=== Multi-Config Experiment Progress ===")

        # Configure pandas display options to show all columns
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Auto-detect width
        pd.set_option('display.max_colwidth', 50)  # Limit column width for readability

        # Hide index for cleaner display
        styled_df = df.style.hide(axis="index")
        self.display_handle = display(styled_df, display_id=True)

    def stop(self):
        """Stop the live display."""
        if self.display_handle:
            # Final render with completed status
            df = self._create_dataframe()

            # Ensure pandas shows all columns
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)

            styled_df = df.style.hide(axis="index")
            self.display_handle.update(styled_df)

    def _render(self):
        """Update the DataFrame display."""
        if self.display_handle:
            df = self._create_dataframe()

            # Ensure pandas shows all columns
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)

            styled_df = df.style.hide(axis="index")
            self.display_handle.update(styled_df)

    def update_pipeline(
        self, pipeline_id: int, shard: int = None, confidence: float = None, metrics: dict = None, status: str = None
    ):
        """
        Update progress for a specific pipeline.

        Args:
            pipeline_id: ID of the pipeline to update
            shard: Current shard number (optional)
            confidence: Confidence interval value (optional)
            metrics: Dictionary of metrics to update (optional)
            status: Pipeline status (optional, e.g., "ONGOING", "STOPPED", "COMPLETED")
        """
        if pipeline_id not in self.pipeline_data:
            return

        data = self.pipeline_data[pipeline_id]

        if shard is not None:
            data["shard"] = shard

        if confidence is not None:
            data["confidence"] = confidence

        if metrics is not None:
            data["metrics"].update(metrics)

        if status is not None:
            data["status"] = status

        # Re-render the display
        self._render()

    def add_pipeline(
        self,
        pipeline_id: int,
        pipeline_config: dict,
        model_name: str = "Unknown",
        status: str = "ONGOING",
        **metadata
    ):
        """
        Add a new pipeline to the display (for dynamically cloned pipelines).

        Args:
            pipeline_id: ID of the new pipeline
            pipeline_config: Pipeline configuration dict (must have "pipeline_name" key)
            model_name: Model name used by the pipeline (default: "Unknown")
            status: Initial status (default: "ONGOING")
            **metadata: Optional metadata fields (search_type, rag_k, top_n, etc.)
        """
        pipeline_name = pipeline_config.get("pipeline_name", f"Pipeline {pipeline_id}")

        # Build pipeline info dict
        pipeline_info_dict = {
            "pipeline_id": pipeline_id,
            "pipeline_config": pipeline_config,
            "model_name": model_name,
        }

        # Add any metadata fields that are not None
        for key, value in metadata.items():
            if value is not None:
                pipeline_info_dict[key] = value

        # Add to pipelines list
        self.pipelines.append(pipeline_info_dict)

        # Initialize pipeline data
        self.pipeline_data[pipeline_id] = {
            "name": pipeline_name,
            "model": model_name,
            "shard": 0,
            "confidence": "-",
            "metrics": {},
            "status": status,
        }

        # Store metadata
        metadata_dict = {}
        for key in ["search_type", "rag_k", "top_n", "chunk_size", "chunk_overlap",
                   "sampling_params", "prompt_manager_k", "model_config"]:
            if key in pipeline_info_dict:
                metadata_dict[key] = pipeline_info_dict[key]
        self.pipeline_metadata[pipeline_id] = metadata_dict

        # Re-render the display
        self._render()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
