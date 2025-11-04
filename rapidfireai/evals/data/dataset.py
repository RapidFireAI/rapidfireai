from typing import Any


class DataLoader:
    """
    Handles data loading, partitioning, and batching for distributed processing.
    Supports different partitioning strategies for optimal load balancing.
    """

    def __init__(self):
        """Initialize the DataLoader."""
        self.data = None
        self.shards = []

    @staticmethod
    def _slice_data(data: Any, start: int, end: int) -> Any:
        """
        Slice data in a way that works for both lists and HuggingFace datasets.

        Args:
            data: Dataset to slice (list or HuggingFace Dataset)
            start: Start index
            end: End index

        Returns:
            Sliced data
        """
        # Check if it's a HuggingFace Dataset (has select method)
        if hasattr(data, "select"):
            return data.select(range(start, min(end, len(data))))
        # Otherwise treat as list-like (standard slicing)
        else:
            return data[start:end]

    def get_total_population_size(self) -> int:
        """
        Get the total population size of the dataset.
        """
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def get_shards_from_data(self, data: Any, num_shards: int) -> list[Any]:
        """
        Split data into specified number of shards.

        Args:
            data: Dataset to split
            num_shards: Number of shards to create

        Returns:
            List of data shards
        """
        if num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {num_shards}")

        len_data = len(data)

        if num_shards > len_data:
            raise ValueError(f"num_shards ({num_shards}) cannot be greater than dataset size ({len_data})")

        # Calculate shard size (distribute samples as evenly as possible)
        shard_size = len_data // num_shards
        remainder = len_data % num_shards

        self.shards = []
        start_idx = 0

        for i in range(num_shards):
            # Add one extra sample to the first 'remainder' shards to distribute evenly
            current_shard_size = shard_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_shard_size
            self.shards.append(self._slice_data(data, start_idx, end_idx))
            start_idx = end_idx

        return self.shards

    def get_batches(self, data: Any, batch_size: int) -> list[Any]:
        """
        Split data into batches of specified size.

        Args:
            data: Dataset or shard to split into batches
            batch_size: Size of each batch

        Returns:
            List of data batches
        """
        self.data = data
        len_data = len(data)
        batches = []
        for i in range(0, len_data, batch_size):
            batches.append(self._slice_data(data, i, i + batch_size))
        return batches
