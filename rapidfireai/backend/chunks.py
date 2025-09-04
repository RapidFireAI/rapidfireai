"""This module contains the DatasetChunker class which is responsible for chunking a PyTorch Dataset
into chunks for distributed processing."""

from datasets import Dataset


class DatasetChunks:
    """Chunks a HuggingFace Dataset into n_chunks for distributed processing."""

    def __init__(self, dataset: Dataset, n_chunks: int):
        self.dataset = dataset
        self.n_chunks = n_chunks
        self.dataset_size = len(dataset)

        # Validate n_chunks
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be positive, got {n_chunks}")

        # Calculate base size for even distribution (not chunk_size anymore)
        self.base_size = self.dataset_size // n_chunks
        self.extra_items = self.dataset_size % n_chunks
        self.chunk_indices = self._create_chunk_indices()

    def _create_chunk_indices(self):
        """Create start/end index pairs for each chunk, distributing items as evenly as possible."""
        chunks = {}

        # Calculate base size and number of chunks that get an extra item
        base_size = self.dataset_size // self.n_chunks
        extra_items = self.dataset_size % self.n_chunks

        current_idx = 0
        for chunk_id in range(self.n_chunks):
            # First 'extra_items' chunks get base_size + 1, rest get base_size
            chunk_size = base_size + (1 if chunk_id < extra_items else 0)

            if chunk_size > 0:  # Only create non-empty chunks
                chunks[chunk_id] = (current_idx, current_idx + chunk_size)
                current_idx += chunk_size

        return chunks

    def get_chunk(self, chunk_id: int) -> Dataset:
        """Get a chunk as a HuggingFace Dataset subset."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}. Valid range: 0-{len(self.chunk_indices) - 1}")

        start_idx, end_idx = self.chunk_indices[chunk_id]
        # Use HuggingFace Dataset's select method to create a subset
        indices = list(range(start_idx, end_idx))
        return self.dataset.select(indices)

    def get_chunk_size(self, chunk_id: int) -> int:
        """Get the size of a specific chunk."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}")
        start_idx, end_idx = self.chunk_indices[chunk_id]
        return end_idx - start_idx

    @property
    def chunk_ids(self):
        """Get all available chunk IDs."""
        return list(self.chunk_indices.keys())
