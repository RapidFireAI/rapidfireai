"""This module contains the DatasetChunker class which is responsible for chunking a PyTorch Dataset
into chunks for distributed processing. Distributes batches across chunks so that chunk sizes are
as multiples of batch_size, with at most 1 chunk having a non-multiple size"""


class DatasetChunks:
    """Chunks a HuggingFace Dataset into n_chunks for distributed processing."""

    def __init__(self, dataset, n_chunks: int, batch_size: int):
        self.dataset = dataset
        self.n_chunks = n_chunks
        self.batch_size = batch_size
        self.dataset_size = len(dataset)

        # Validate inputs
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be positive, got {n_chunks}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Handle empty dataset
        if self.dataset_size == 0:
            self.total_batches = 0
            self.chunk_indices = {}
            return

        # Calculate total number of batches (including partial last batch)
        self.total_batches = (self.dataset_size + batch_size - 1) // batch_size

        # Validate that we can create the requested number of chunks
        if n_chunks > self.total_batches:
            raise ValueError(
                f"Cannot create {n_chunks} chunks from {self.dataset_size} examples "
                f"with batch_size={batch_size} (only {self.total_batches} batches available). "
                f"Maximum chunks possible: {self.total_batches}"
            )

        self.chunk_indices = self._create_chunk_indices()

    def _create_chunk_indices(self):
        """Create start/end index pairs for each chunk, distributing batches as evenly as possible."""
        chunks = {}

        if self.dataset_size == 0:
            return chunks

        # Distribute batches across chunks
        batches_per_chunk = self.total_batches // self.n_chunks
        extra_batches = self.total_batches % self.n_chunks

        current_example_idx = 0
        for chunk_id in range(self.n_chunks):
            # Last 'extra_batches' chunks get one additional batch
            num_batches_in_chunk = batches_per_chunk + (1 if chunk_id >= (self.n_chunks - extra_batches) else 0)

            start_idx = current_example_idx

            # Calculate how many examples these batches contain
            examples_in_chunk = 0
            for _ in range(num_batches_in_chunk):
                remaining_examples = self.dataset_size - current_example_idx
                examples_in_this_batch = min(self.batch_size, remaining_examples)
                examples_in_chunk += examples_in_this_batch
                current_example_idx += examples_in_this_batch

            end_idx = start_idx + examples_in_chunk
            chunks[chunk_id] = (start_idx, end_idx)

        return chunks

    def get_chunk(self, chunk_id: int):
        """Get a chunk as a HuggingFace Dataset subset."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}. Valid range: 0-{len(self.chunk_indices) - 1}")

        start_idx, end_idx = self.chunk_indices[chunk_id]
        indices = list(range(start_idx, end_idx))
        return self.dataset.select(indices)

    def get_chunk_size(self, chunk_id: int) -> int:
        """Get the size of a specific chunk."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}")
        start_idx, end_idx = self.chunk_indices[chunk_id]
        return end_idx - start_idx

    def get_chunk_batches(self, chunk_id: int) -> int:
        """Get the number of batches in a specific chunk."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}")

        chunk_size = self.get_chunk_size(chunk_id)
        # Calculate how many batches this chunk represents
        return (chunk_size + self.batch_size - 1) // self.batch_size

    @property
    def chunk_ids(self):
        """Get all available chunk IDs."""
        return list(self.chunk_indices.keys())
