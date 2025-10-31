"""
Progress display using pandas DataFrames for multi-config experiments.

This module provides a clean, updating table display for tracking
run progress, metrics, and confidence intervals in Jupyter notebooks.

Requires: pandas and IPython (for Jupyter display)
"""

from __future__ import annotations

import time

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
                    "Context Hash": ctx_hash_display,
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
    - Other metrics (accuracy, throughput, etc.)
    """

    def __init__(self, pipelines: list[tuple[int, str, str]], num_shards: int):
        """
        Initialize the progress display.

        Args:
            pipelines: List of (pipeline_id, pipeline_name, model_name) tuples
            num_shards: Total number of shards
        """
        self.pipelines = pipelines
        self.num_shards = num_shards
        self.pipeline_data = {
            pid: {
                "name": name,
                "model": model,
                "shard": 0,
                "confidence": "-",
                "metrics": {},
                "status": "ONGOING",
            }
            for pid, name, model in pipelines
        }
        self.display_handle = None

    def _create_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with current pipeline progress data."""
        rows = []
        for pid, name, model in self.pipelines:
            data = self.pipeline_data[pid]

            # Format progress
            progress = f"{data['shard']}/{self.num_shards}"

            # Format confidence interval
            confidence = data["confidence"]
            if confidence != "-" and isinstance(confidence, (int, float)):
                confidence = f"{confidence:.3f}"

            # Format metrics
            accuracy = data["metrics"].get("Accuracy", {}).get("value", "-")
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.2%}"

            throughput = data["metrics"].get("Throughput", {}).get("value", "-")
            if isinstance(throughput, (int, float)):
                throughput = f"{throughput:.1f}/s"

            # Get status
            status = data.get("status", "ONGOING")

            rows.append(
                {
                    "Run ID": pid,
                    "Name": name,
                    "Model": model,
                    "Status": status,
                    "Progress": progress,
                    "Accuracy": str(accuracy),
                    "Conf. Interval": str(confidence),
                    "Throughput": str(throughput),
                }
            )

        return pd.DataFrame(rows)

    def start(self):
        """Start the live display."""
        df = self._create_dataframe()
        print("\n=== Multi-Config Experiment Progress ===")
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

    def add_pipeline(self, pipeline_id: int, pipeline_name: str, model_name: str, status: str = "ONGOING"):
        """
        Add a new pipeline to the display (for dynamically cloned pipelines).

        Args:
            pipeline_id: ID of the new pipeline
            pipeline_name: Name of the new pipeline
            model_name: Model name used by the pipeline
            status: Initial status (default: "ONGOING")
        """
        # Add to pipelines list
        self.pipelines.append((pipeline_id, pipeline_name, model_name))

        # Initialize pipeline data
        self.pipeline_data[pipeline_id] = {
            "name": pipeline_name,
            "model": model_name,
            "shard": 0,
            "confidence": "-",
            "metrics": {},
            "status": status,
        }

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
