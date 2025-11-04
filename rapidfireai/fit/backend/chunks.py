"""This module contains the DatasetChunker class which is responsible for chunking a PyTorch Dataset
into chunks for distributed processing."""


class DatasetChunks:
    """Chunks a HuggingFace Dataset into n_chunks for distributed processing."""

    def __init__(self, dataset_size: int, n_chunks: int, batch_size: int = 1, offset: int = 0):
        self.n_chunks = n_chunks
        self.batch_size = batch_size
        self.offset = offset
        self.dataset_size = dataset_size

        # Validate inputs
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be positive, got {n_chunks}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if offset >= self.dataset_size and self.dataset_size > 0:
            raise ValueError(
                f"offset must be less than dataset_size, got offset={offset} for dataset_size={self.dataset_size}"
            )

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

        # Create base chunk indices and apply offset if needed
        base_chunks = self._create_base_chunk_indices()
        self.chunk_indices = self._apply_offset(base_chunks) if offset > 0 else base_chunks

    def _create_base_chunk_indices(self):
        """Create start/end index pairs for each chunk, distributing batches as evenly as possible."""
        chunks = {}

        if self.dataset_size == 0:
            return chunks

        # Distribute batches across chunks, not examples
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

    def _apply_offset(self, base_chunks):
        """Apply offset to all chunk indices with modulo wrapping for resume functionality."""
        if self.offset == 0:
            return base_chunks

        offset_chunks = {}
        for chunk_id, (start, end) in base_chunks.items():
            # Apply offset with modulo wrapping
            new_start = (start + self.offset) % self.dataset_size
            new_end = (end + self.offset) % self.dataset_size

            offset_chunks[chunk_id] = (new_start, new_end)

        return offset_chunks

    def get_chunk(self, dataset, chunk_id: int):
        """Get a chunk as a HuggingFace Dataset subset."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}. Valid range: 0-{len(self.chunk_indices) - 1}")

        start_idx, end_idx = self.get_chunk_indices(chunk_id)

        # Handle wraparound case when end_idx < start_idx due to modulo
        if end_idx < start_idx:
            # Chunk wraps around: get indices from start to end of dataset, then from 0 to end
            indices = list(range(start_idx, self.dataset_size)) + list(range(0, end_idx))
        else:
            indices = list(range(start_idx, end_idx))

        return dataset.select(indices)

    def get_offset(self) -> int:
        """Get the current offset value used for this chunker."""
        return self.offset

    def get_clone_offset(self, last_completed_chunk: int) -> int:
        """Get the clone offset for a newly cloned run."""
        if last_completed_chunk not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {last_completed_chunk}")

        # Get the end index of the last completed chunk
        # This is where the next run should start
        _, last_chunk_end = self.get_chunk_indices(last_completed_chunk)

        # The clone offset should be the absolute position where we want to start
        # which is the end index of the last completed chunk
        return last_chunk_end % self.dataset_size

    def get_chunk_indices(self, chunk_id: int) -> tuple:
        """Get the start and end indices of a specific chunk as a tuple (start_idx, end_idx)."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}")

        chunk_data = self.chunk_indices[chunk_id]

        # Handle case where chunk_data might not be a proper tuple
        if not chunk_data or len(chunk_data) != 2:
            raise ValueError(f"Invalid chunk data for chunk_id {chunk_id}: {chunk_data}")

        start_idx, end_idx = chunk_data
        return (start_idx, end_idx)

    def get_chunk_size(self, chunk_id: int) -> int:
        """Get the size of a specific chunk."""
        if chunk_id not in self.chunk_indices:
            raise ValueError(f"Invalid chunk_id {chunk_id}")

        start_idx, end_idx = self.get_chunk_indices(chunk_id)

        # Handle wraparound case when end_idx < start_idx due to modulo
        if end_idx < start_idx:
            # Chunk wraps around: size is (dataset_size - start_idx) + end_idx
            return (self.dataset_size - start_idx) + end_idx
        else:
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
