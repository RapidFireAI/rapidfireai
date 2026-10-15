"""This module contains the DatasetShards class which is responsible for sharding a PyTorch Dataset
into shards for distributed processing."""


class DatasetShards:
    """Shards a HuggingFace Dataset into n_shards for distributed processing."""

    def __init__(
        self, dataset_size: int, n_shards: int, batch_size: int = 1, offset: int = 0
    ):
        self.n_shards = n_shards
        self.batch_size = batch_size
        self.offset = offset
        self.dataset_size = dataset_size

        # Validate inputs
        if n_shards <= 0:
            raise ValueError(f"n_shards must be positive, got {n_shards}")
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
            self.shard_indices = {}
            return

        # Calculate total number of batches (including partial last batch)
        self.total_batches = (self.dataset_size + batch_size - 1) // batch_size

        # Validate that we can create the requested number of shards
        if n_shards > self.total_batches:
            raise ValueError(
                f"Cannot create {n_shards} shards from {self.dataset_size} examples "
                f"with batch_size={batch_size} (only {self.total_batches} batches available). "
                f"Maximum shards possible: {self.total_batches}"
            )

        # Create base shard indices and apply offset if needed
        base_shards = self._create_base_shard_indices()
        self.shard_indices = (
            self._apply_offset(base_shards) if offset > 0 else base_shards
        )

    def _create_base_shard_indices(self):
        """Create start/end index pairs for each shard, distributing batches as evenly as possible."""
        shards = {}

        if self.dataset_size == 0:
            return shards

        # Distribute batches across shards, not examples
        batches_per_shard = self.total_batches // self.n_shards
        extra_batches = self.total_batches % self.n_shards

        current_example_idx = 0
        for shard_id in range(self.n_shards):
            # First 'extra_batches' shards get one additional batch
            num_batches_in_shard = batches_per_shard + (
                1 if shard_id < extra_batches else 0
            )

            start_idx = current_example_idx

            # Calculate how many examples these batches contain
            examples_in_shard = 0
            for _ in range(num_batches_in_shard):
                remaining_examples = self.dataset_size - current_example_idx
                examples_in_this_batch = min(self.batch_size, remaining_examples)
                examples_in_shard += examples_in_this_batch
                current_example_idx += examples_in_this_batch

            end_idx = start_idx + examples_in_shard
            shards[shard_id] = (start_idx, end_idx)

        return shards

    def _apply_offset(self, base_shards):
        """Apply offset to all shard indices with modulo wrapping for resume functionality."""
        if self.offset == 0:
            return base_shards

        offset_shards = {}
        for shard_id, (start, end) in base_shards.items():
            # Apply offset with modulo wrapping
            new_start = (start + self.offset) % self.dataset_size
            new_end = (end + self.offset) % self.dataset_size

            offset_shards[shard_id] = (new_start, new_end)

        return offset_shards

    def get_shard(self, dataset, shard_id: int):
        """Get a shard as a HuggingFace Dataset subset."""
        if shard_id not in self.shard_indices:
            raise ValueError(
                f"Invalid shard_id {shard_id}. Valid range: 0-{len(self.shard_indices) - 1}"
            )

        start_idx, end_idx = self.get_shard_indices(shard_id)

        # Handle wraparound case when end_idx < start_idx due to modulo
        if end_idx < start_idx:
            # Shard wraps around: get indices from start to end of dataset, then from 0 to end
            indices = list(range(start_idx, self.dataset_size)) + list(
                range(0, end_idx)
            )
        else:
            indices = list(range(start_idx, end_idx))

        return dataset.select(indices)

    def get_offset(self) -> int:
        """Get the current offset value used for this sharder."""
        return self.offset

    def get_clone_offset(self, last_completed_shard: int) -> int:
        """Get the clone offset for a newly cloned run."""
        if last_completed_shard not in self.shard_indices:
            raise ValueError(f"Invalid shard_id {last_completed_shard}")

        # Get the end index of the last completed shard
        # This is where the next run should start
        _, last_shard_end = self.get_shard_indices(last_completed_shard)

        # The clone offset should be the absolute position where we want to start
        # which is the end index of the last completed shard
        return last_shard_end % self.dataset_size

    def get_shard_indices(self, shard_id: int) -> tuple:
        """Get the start and end indices of a specific shard as a tuple (start_idx, end_idx)."""
        if shard_id not in self.shard_indices:
            raise ValueError(f"Invalid shard_id {shard_id}")

        shard_data = self.shard_indices[shard_id]

        # Handle case where shard_data might not be a proper tuple
        if not shard_data or len(shard_data) != 2:
            raise ValueError(
                f"Invalid shard data for shard_id {shard_id}: {shard_data}"
            )

        start_idx, end_idx = shard_data
        return (start_idx, end_idx)

    def get_shard_size(self, shard_id: int) -> int:
        """Get the size of a specific shard."""
        if shard_id not in self.shard_indices:
            raise ValueError(f"Invalid shard_id {shard_id}")

        start_idx, end_idx = self.get_shard_indices(shard_id)

        # Handle wraparound case when end_idx < start_idx due to modulo
        if end_idx < start_idx:
            # Shard wraps around: size is (dataset_size - start_idx) + end_idx
            return (self.dataset_size - start_idx) + end_idx
        else:
            return end_idx - start_idx

    def get_shard_batches(self, shard_id: int) -> int:
        """Get the number of batches in a specific shard."""
        if shard_id not in self.shard_indices:
            raise ValueError(f"Invalid shard_id {shard_id}")

        shard_size = self.get_shard_size(shard_id)
        # Calculate how many batches this shard represents
        return (shard_size + self.batch_size - 1) // self.batch_size

    @property
    def shard_ids(self):
        """Get all available shard IDs."""
        return list(self.shard_indices.keys())
