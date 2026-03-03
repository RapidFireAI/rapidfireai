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
                "vector_store_info": ctx.get("vector_store_info", None),
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
            vector_store_info = data["vector_store_info"]
            if vector_store_info is not None:
                type = vector_store_info.get("type", None)
                if type == "faiss":
                    details = "FAISS, " + ("GPU" if data["enable_gpu"] else "CPU")
                elif type == "pgvector":
                    collection_name = f' [{vector_store_info.get("collection_name", "")}]'
                    details = f"PGVector{collection_name}, " + ("GPU" if data["enable_gpu"] else "CPU")
                elif type == "pinecone":
                    index_name = f' [{vector_store_info.get("pinecone_index_name", "")}]'
                    details = f"Pinecone{index_name}, " + ("GPU" if data["enable_gpu"] else "CPU")
                else:
                    details = "Unknown, " + ("GPU" if data["enable_gpu"] else "CPU")
            else:
                details = "N/A, " + ("GPU" if data["enable_gpu"] else "CPU")
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
    - Pipeline configuration fields (flattened with dot notation)
    - Other metrics (accuracy, throughput, etc.)

    Dict-valued metadata fields (embedding_cfg, vector_store_cfg, search_cfg, reranker_cfg)
    are recursively flattened into dot-notation columns, e.g. ``reranker_cfg.top_n``.
    Keys that match sensitive patterns (api_key, password, etc.) are automatically hidden.
    """

    # Substring patterns — any key whose lowercased name contains one of these strings is hidden.
    _SENSITIVE_PATTERNS = frozenset({
        "api_key",
        "api_secret",
        "secret",
        "password",
        "token",
        "credentials",
        "connection",
        "device",
    })

    _CFG_FIELDS_TO_FLATTEN = frozenset({
        "embedding_cfg", "vector_store_cfg", "search_cfg", "reranker_cfg",
    })

    _METADATA_KEYS = [
        "text_splitter", "chunk_size", "chunk_overlap", "embedding_cfg", "vector_store_cfg",
        "search_cfg", "reranker_cfg",
        "sampling_params", "prompt_manager_k", "model_config",
    ]

    _METADATA_PREFIX_ORDER = [
        "text_splitter", "chunk_size", "chunk_overlap",
        "embedding_cfg", "vector_store_cfg",
        "search_cfg", "reranker_cfg",
        "sampling_params", "prompt_manager_k", "model_config",
    ]

    def __init__(self, pipelines: list[dict], num_shards: int):
        """
        Initialize the progress display.

        Args:
            pipelines: List of pipeline info dicts with keys:
                      - pipeline_id (required)
                      - pipeline_config (required)
                      - model_name (required)
                      Indexing stage (read-only display):
                      - text_splitter (optional, str)
                      - chunk_size (optional, int)
                      - chunk_overlap (optional, int)
                      - embedding_cfg (optional, dict)
                      - vector_store_cfg (optional, dict)
                      Retrieval stage:
                      - search_cfg (optional, dict)
                      - reranker_cfg (optional, dict)
                      Generation stage:
                      - sampling_params (optional, dict)
                      - prompt_manager_k (optional, int)
                      - model_config (optional, dict)
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

            metadata = {}
            for key in self._METADATA_KEYS:
                if key in pipeline_info:
                    metadata[key] = pipeline_info[key]
            self.pipeline_metadata[pid] = metadata

        self.display_handle = None

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        """Check if a key (or any component of a dot-separated key) matches a sensitive pattern."""
        key_lower = key.lower()
        for pattern in PipelineProgressDisplay._SENSITIVE_PATTERNS:
            if pattern in key_lower:
                return True
        return False

    @staticmethod
    def _flatten_dict(prefix: str, d: dict, result: dict | None = None) -> dict:
        """
        Recursively flatten a dict into dot-notation keys, skipping sensitive keys.

        Example::

            _flatten_dict("search_cfg", {"type": "similarity", "k": 5})
            → {"search_cfg.type": "similarity", "search_cfg.k": 5}
        """
        if result is None:
            result = {}
        for key, value in d.items():
            flat_key = f"{prefix}.{key}"
            if PipelineProgressDisplay._is_sensitive_key(str(key)):
                continue
            if isinstance(value, dict):
                PipelineProgressDisplay._flatten_dict(flat_key, value, result)
            elif isinstance(value, type):
                result[flat_key] = value.__qualname__
            else:
                result[flat_key] = value
        return result

    def _flatten_metadata(self, metadata: dict) -> dict:
        """
        Flatten a pipeline's raw metadata dict for DataFrame display.

        Dict-valued fields listed in ``_CFG_FIELDS_TO_FLATTEN`` are recursively
        expanded into dot-notation columns.  All other fields are kept as-is
        (with special formatting for ``sampling_params`` and ``model_config``).
        """
        flat: dict[str, Any] = {}
        for key, value in metadata.items():
            if key in self._CFG_FIELDS_TO_FLATTEN and isinstance(value, dict):
                self._flatten_dict(key, value, flat)
            elif key == "sampling_params" and isinstance(value, dict):
                parts = []
                if "temperature" in value:
                    parts.append(f"temp={value['temperature']}")
                if "top_p" in value:
                    parts.append(f"top_p={value['top_p']}")
                if "top_k" in value:
                    parts.append(f"top_k={value['top_k']}")
                if "max_tokens" in value:
                    parts.append(f"max_tokens={value['max_tokens']}")
                flat[key] = ", ".join(parts) if parts else str(value)[:50]
            elif key == "model_config" and isinstance(value, dict):
                parts = [f"{k}={v}" for k, v in value.items() if k != "model"]
                flat[key] = ", ".join(parts) if parts else "-"
            else:
                flat[key] = value
        return flat

    @staticmethod
    def _metadata_sort_key(key: str) -> tuple[int, str]:
        """Return a sort key so metadata columns appear in a stable, logical order."""
        for idx, prefix in enumerate(PipelineProgressDisplay._METADATA_PREFIX_ORDER):
            if key == prefix or key.startswith(prefix + "."):
                return (idx, key)
        return (len(PipelineProgressDisplay._METADATA_PREFIX_ORDER), key)

    def _create_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with current pipeline progress data."""
        # Collect all unique metric names across all pipelines
        all_metric_names: set[str] = set()
        for pipeline_info in self.pipelines:
            pid = pipeline_info["pipeline_id"]
            data = self.pipeline_data[pid]
            all_metric_names.update(data["metrics"].keys())

        ordered_metrics: list[str] = sorted(all_metric_names)

        # Flatten metadata for every pipeline and collect the union of all column names
        pipeline_flat_metadata: dict[int, dict] = {}
        all_flat_keys: set[str] = set()
        for pipeline_info in self.pipelines:
            pid = pipeline_info["pipeline_id"]
            raw = self.pipeline_metadata.get(pid, {})
            flat = self._flatten_metadata(raw)
            pipeline_flat_metadata[pid] = flat
            all_flat_keys.update(flat.keys())

        # Drop any parent key whose dot-notation children are already present.
        # e.g. if both "embedding_cfg" and "embedding_cfg.class" are in the set,
        # remove "embedding_cfg" — showing the dict as a string alongside its
        # individually-flattened members is redundant and confusing.
        all_flat_keys = {
            key for key in all_flat_keys
            if not any(other.startswith(key + ".") for other in all_flat_keys)
        }

        # Similarly remove any metric-column name that duplicates a metadata parent key
        # (e.g. "embedding_cfg" added as a raw metric when its children are metadata cols).
        all_metric_names -= {key for key in all_metric_names if key in all_flat_keys
                             or any(meta.startswith(key + ".") for meta in all_flat_keys)}

        ordered_metadata = sorted(all_flat_keys, key=self._metadata_sort_key)

        rows = []
        for pipeline_info in self.pipelines:
            pid = pipeline_info["pipeline_id"]

            if pid not in self.pipeline_data:
                continue

            data = self.pipeline_data[pid]
            flat = pipeline_flat_metadata.get(pid, {})

            progress = f"{data['shard']}/{self.num_shards}"

            confidence = data["confidence"]
            if confidence != "-" and isinstance(confidence, (int, float)):
                confidence = f"{confidence:.3f}"

            status = data.get("status", "ONGOING")

            row: dict[str, Any] = {
                "Run ID": pid,
                "Model": data.get("model", "-"),
                "Status": status,
                "Progress": progress,
                "Conf. Interval": str(confidence),
            }

            for col_name in ordered_metadata:
                value = flat.get(col_name)
                row[col_name] = self._format_value(value)

            # Add all metrics with confidence intervals
            for metric_name in ordered_metrics:
                metric_data = data["metrics"].get(metric_name, {})
                if isinstance(metric_data, dict):
                    metric_value = metric_data.get("value", "-")
                    lower_bound = metric_data.get("lower_bound")
                    upper_bound = metric_data.get("upper_bound")
                    is_algebraic = metric_data.get("is_algebraic", False)
                else:
                    metric_value = metric_data
                    lower_bound = upper_bound = None
                    is_algebraic = False

                # Format metric value based on type and name
                if metric_value != "-":
                    if isinstance(metric_value, (int, float)):
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

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a single (already-flattened) metadata value for display."""
        if value is None:
            return "-"
        if isinstance(value, type):
            return value.__qualname__
        if isinstance(value, list):
            return f"[{', '.join(str(v) for v in value[:5])}]"
        if isinstance(value, dict):
            return str(value)[:50]
        return str(value)

    @staticmethod
    def _format_nested_dict(d: dict) -> str:
        """Format a nested dict compactly, preserving one level of nesting."""
        parts = []
        for k, v in d.items():
            if isinstance(v, dict):
                inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
                parts.append(f"{k}={{{inner}}}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

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
        for key in self._METADATA_KEYS:
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
