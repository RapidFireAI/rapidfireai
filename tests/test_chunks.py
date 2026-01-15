import pytest
from datasets import Dataset

from rapidfireai.fit.backend.chunks import DatasetChunks


class TestDatasetChunks:
    """Test suite for DatasetChunks class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        data = {"text": [f"example_{i}" for i in range(100)], "label": list(range(100))}
        return Dataset.from_dict(data)

    def test_basic_chunking_without_batch_size(self, sample_dataset):
        """Test basic chunking functionality with default batch_size=1."""
        chunker = DatasetChunks(dataset_size=100, n_chunks=4)

        assert len(chunker.chunk_ids) == 4

        # Calculate total size by summing chunk sizes
        total_size = sum(chunker.get_chunk_size(i) for i in chunker.chunk_ids)
        assert total_size == 100

        # With 100 examples and 4 chunks, should be 25 each
        for chunk_id in chunker.chunk_ids:
            chunk_size = chunker.get_chunk_size(chunk_id)
            assert chunk_size == 25

    def test_your_example_case(self):
        """Test the specific example: 101 examples, batch_size=10, n_chunks=10."""
        data = {"text": [f"example_{i}" for i in range(101)]}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset_size=101, n_chunks=10, batch_size=10)

        # Should have 11 total batches: 10 full + 1 partial
        # 1 chunk gets 2 batches (10 + 1 examples), 9 chunks get 1 batch (10 examples each)
        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        chunk_batches = [chunker.get_chunk_batches(i) for i in chunker.chunk_ids]

        assert len(chunk_sizes) == 10
        assert sum(chunk_sizes) == 101
        assert chunk_sizes == [10, 10, 10, 10, 10, 10, 10, 10, 10, 11]
        assert sum(chunk_batches) == 11  # Total 11 batches
        assert chunk_batches.count(2) == 1  # One chunk with 2 batches
        assert chunk_batches.count(1) == 9  # Nine chunks with 1 batch each

    def test_basic_offset_functionality(self):
        """Test basic offset functionality."""
        data = {"text": [f"example_{i}" for i in range(20)]}
        dataset = Dataset.from_dict(data)

        # Normal chunking
        chunker_normal = DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5)
        # With offset
        chunker_offset = DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5, offset=7)

        # Verify offset is stored
        assert chunker_normal.get_offset() == 0
        assert chunker_offset.get_offset() == 7

        # Verify chunk indices are offset correctly
        normal_indices = chunker_normal.get_chunk_indices(0)
        offset_indices = chunker_offset.get_chunk_indices(0)

        assert normal_indices[0] == 0  # First index of normal chunk
        assert offset_indices[0] == 7  # First index of offset chunk

    def test_offset_wraparound(self):
        """Test offset with wraparound behavior."""
        data = {"text": [f"example_{i}" for i in range(10)]}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset_size=10, n_chunks=2, batch_size=3, offset=8)

        # First chunk should wrap around
        chunk = chunker.get_chunk(dataset, 0)
        actual_indices = [int(x.split("_")[1]) for x in chunk["text"]]

        # Should start at index 8 and continue from there
        assert actual_indices[0] == 8

    def test_get_clone_offset_basic(self):
        """Test basic clone offset calculation."""
        data = {"text": [f"example_{i}" for i in range(50)]}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset_size=50, n_chunks=5, batch_size=7)

        # Complete chunks 0-2, get offset for continuation
        clone_offset = chunker.get_clone_offset(last_completed_chunk=2)

        # The clone offset should be the end index of chunk 2
        # Since get_chunk_indices returns exclusive end indices, this is correct
        chunk_2_end = chunker.get_chunk_indices(2)[1]
        assert clone_offset == chunk_2_end

        # Verify the clone offset starts where chunk 3 would start
        chunk_3_indices = chunker.get_chunk_indices(3)
        assert clone_offset == chunk_3_indices[0]

    def test_complete_epoch_coverage_with_offset(self):
        """Test that offset runs cover complete epochs."""
        data = {"text": [f"example_{i}" for i in range(30)]}
        dataset = Dataset.from_dict(data)

        # Create chunker with offset
        chunker = DatasetChunks(dataset_size=30, n_chunks=3, batch_size=7, offset=10)

        # Collect all processed examples
        all_indices = []
        for chunk_id in chunker.chunk_ids:
            chunk = chunker.get_chunk(dataset, chunk_id)
            chunk_indices = [int(x.split("_")[1]) for x in chunk["text"]]
            all_indices.extend(chunk_indices)

        # Should process all 30 examples exactly once
        assert len(all_indices) == 30
        assert set(all_indices) == set(range(30))

    def test_clone_continuity(self):
        """Test that cloned runs continue seamlessly from parent."""
        data = {"text": [f"example_{i}" for i in range(50)]}
        dataset = Dataset.from_dict(data)

        # Run 1: complete chunks 0-2
        run1 = DatasetChunks(dataset_size=50, n_chunks=5, batch_size=8)

        # Simulate processing chunks 0-2
        last_completed = 2
        last_chunk = run1.get_chunk(dataset, last_completed)
        last_processed_indices = [int(x.split("_")[1]) for x in last_chunk["text"]]

        # Get clone offset and create Run 2
        clone_offset = run1.get_clone_offset(last_completed_chunk=last_completed)
        run2 = DatasetChunks(dataset_size=50, n_chunks=4, batch_size=6, offset=clone_offset)

        # First chunk of Run 2 should start where Run 1 left off
        first_chunk_run2 = run2.get_chunk(dataset, 0)
        first_run2_indices = [int(x.split("_")[1]) for x in first_chunk_run2["text"]]

        # The clone offset should be the exclusive end index of the last completed chunk
        # Since chunk indices are [start:end), the end is already the correct next start
        chunk_2_end = run1.get_chunk_indices(last_completed)[1]
        expected_start = chunk_2_end % 50
        assert first_run2_indices[0] == expected_start

    def test_invalid_offset_errors(self):
        """Test error handling for invalid offset."""
        with pytest.raises(ValueError, match="offset must be non-negative"):
            DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5, offset=-1)

        with pytest.raises(ValueError, match="offset must be less than dataset_size"):
            DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5, offset=20)

    def test_clone_offset_error_cases(self):
        """Test error cases for clone offset calculation."""
        chunker = DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5)

        # Test with invalid chunk ID
        with pytest.raises(ValueError, match="Invalid chunk_id"):
            chunker.get_clone_offset(last_completed_chunk=4)  # Only chunks 0-3 exist

        # Test with negative chunk ID
        with pytest.raises(ValueError, match="Invalid chunk_id"):
            chunker.get_clone_offset(last_completed_chunk=-1)

    def test_clone_offset_with_chunk_count_conversion(self):
        """Test that clone offset works correctly when converting chunk count to chunk_id."""
        num_chunks = 4
        chunker = DatasetChunks(dataset_size=100, n_chunks=num_chunks, batch_size=8)

        for chunks_completed in range(1, num_chunks + 1):
            last_completed_chunk_id = chunks_completed - 1
            clone_offset = chunker.get_clone_offset(last_completed_chunk_id)

            _, chunk_end = chunker.get_chunk_indices(last_completed_chunk_id)
            expected_offset = chunk_end % chunker.dataset_size
            assert clone_offset == expected_offset, (
                f"At chunks_completed={chunks_completed}: "
                f"expected offset {expected_offset}, got {clone_offset}"
            )

    def test_clone_offset_zero_chunks_visited(self):
        """Test clone offset when no chunks have been visited yet."""
        chunker = DatasetChunks(dataset_size=100, n_chunks=4, batch_size=8)

        num_chunks_visited = 0

        if num_chunks_visited == 0:
            clone_offset = 0
        else:
            last_completed_chunk_id = num_chunks_visited - 1
            clone_offset = chunker.get_clone_offset(last_completed_chunk_id)

        assert clone_offset == 0, "Zero chunks visited should result in offset 0"

    def test_clone_offset_all_chunks_completed(self):
        """Test clone offset wraps around when all chunks have been completed."""
        dataset_size = 100
        num_chunks = 4
        chunker = DatasetChunks(dataset_size=dataset_size, n_chunks=num_chunks, batch_size=8)

        num_chunks_visited = num_chunks
        last_completed_chunk_id = num_chunks_visited - 1

        clone_offset = chunker.get_clone_offset(last_completed_chunk_id)

        _, last_chunk_end = chunker.get_chunk_indices(last_completed_chunk_id)
        expected_offset = last_chunk_end % dataset_size

        assert clone_offset == expected_offset

    def test_warm_clone_offset_all_chunk_boundaries(self):
        """Test warm clone offset calculation at all chunk boundaries."""
        len_train_dataset = 100
        num_chunks = 4
        batch_size = 8
        parent_chunk_offset = 0

        for num_chunks_visited in range(1, num_chunks + 1):
            chunker = DatasetChunks(
                len_train_dataset,
                num_chunks,
                batch_size=batch_size,
                offset=parent_chunk_offset,
            )

            if num_chunks_visited == 0:
                clone_chunk_offset = 0
            else:
                last_completed_chunk_id = num_chunks_visited - 1
                clone_chunk_offset = chunker.get_clone_offset(last_completed_chunk_id)

            assert isinstance(clone_chunk_offset, int)
            assert 0 <= clone_chunk_offset < len_train_dataset

    def test_clone_offset_count_vs_chunk_id(self):
        """Test that get_clone_offset expects chunk_id (0-indexed), not chunk count."""
        chunker = DatasetChunks(dataset_size=100, n_chunks=4, batch_size=8)

        # Chunk count after completing all 4 chunks
        num_chunks_visited = 4

        # Passing count directly should fail (chunk_id 4 doesn't exist)
        try:
            chunker.get_clone_offset(num_chunks_visited)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            print(f"Expected error when passing count as chunk_id: {e}")

        # Converting count to chunk_id should work
        last_completed_chunk_id = num_chunks_visited - 1
        clone_offset = chunker.get_clone_offset(last_completed_chunk_id)
        assert isinstance(clone_offset, int)

    def test_offset_batch_alignment(self):
        """Test that offset runs maintain good batch alignment."""
        chunker = DatasetChunks(dataset_size=60, n_chunks=5, batch_size=12, offset=15)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]

        # Count chunks that are multiples of batch_size
        multiples_count = sum(1 for size in chunk_sizes if size % 12 == 0)
        non_multiples_count = len(chunk_sizes) - multiples_count

        # Should have at most 1 non-multiple chunk
        assert non_multiples_count <= 1

    def test_partial_batch_distribution_case_1(self):
        """Test: 22 examples, batch_size=10, n_chunks=3."""
        chunker = DatasetChunks(dataset_size=22, n_chunks=3, batch_size=10)
        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [10, 10, 2]
        assert sum(chunk_sizes) == 22

    def test_partial_batch_distribution_case_2(self):
        """Test: 25 examples, batch_size=8, n_chunks=2."""
        chunker = DatasetChunks(dataset_size=25, n_chunks=2, batch_size=8)
        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [16, 9]
        assert sum(chunk_sizes) == 25

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        # Empty dataset should return no chunks regardless of requested chunks
        chunker = DatasetChunks(dataset_size=0, n_chunks=3, batch_size=5)
        assert len(chunker.chunk_ids) == 0

    def test_single_example_dataset(self):
        """Test with dataset containing only one example."""
        chunker = DatasetChunks(dataset_size=1, n_chunks=1, batch_size=5)

        assert len(chunker.chunk_ids) == 1
        assert chunker.get_chunk_size(0) == 1
        assert chunker.get_chunk_batches(0) == 1

        # Test with actual dataset
        single_dataset = Dataset.from_dict({"text": ["only_example"]})
        chunk = chunker.get_chunk(single_dataset, 0)
        assert chunk["text"][0] == "only_example"

    def test_invalid_n_chunks(self):
        """Test error handling for invalid n_chunks."""
        with pytest.raises(ValueError, match="n_chunks must be positive"):
            DatasetChunks(dataset_size=100, n_chunks=0)

    def test_invalid_batch_size(self):
        """Test error handling for invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DatasetChunks(dataset_size=100, n_chunks=4, batch_size=0)

    def test_too_many_chunks_error(self):
        """Test error when requesting more chunks than batches available."""
        # 25 examples, batch_size=10 -> 3 batches, but asking for 5 chunks
        with pytest.raises(ValueError, match="Cannot create 5 chunks from 25 examples"):
            DatasetChunks(dataset_size=25, n_chunks=5, batch_size=10)

    def test_clone_offset_calculation_correctness(self):
        """Test that clone offset calculation is correct (fixes the bug we found)."""
        data = {"text": [f"example_{i}" for i in range(20)]}
        dataset = Dataset.from_dict(data)

        # Original chunker
        original = DatasetChunks(dataset_size=20, n_chunks=4, batch_size=3, offset=0)

        # Process chunk 0 and 1, so last completed chunk is 1
        chunk_1_end = original.get_chunk_indices(1)[1]
        clone_offset = original.get_clone_offset(last_completed_chunk=1)

        # Clone offset should be the end index of chunk 1 (since indices are exclusive)
        assert clone_offset == chunk_1_end

        # Create clone and verify it starts where original left off
        clone = DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5, offset=clone_offset)
        clone_chunk_0 = clone.get_chunk(dataset, 0)
        clone_first_index = int(clone_chunk_0["text"][0].split("_")[1])

        # Clone should start exactly at the clone_offset
        assert clone_first_index == clone_offset

    def test_get_chunk_requires_dataset_parameter(self):
        """Test that get_chunk method now requires dataset parameter."""
        chunker = DatasetChunks(dataset_size=20, n_chunks=4, batch_size=5)
        dataset = Dataset.from_dict({"text": [f"example_{i}" for i in range(20)]})

        # Should work with dataset parameter
        chunk = chunker.get_chunk(dataset, 0)
        assert len(chunk["text"]) == 5

        # Should fail without dataset parameter (this test ensures the signature changed)
        with pytest.raises(TypeError):
            chunker.get_chunk(0)  # Missing dataset parameter

    def test_dataset_size_parameter_instead_of_dataset_object(self):
        """Test that constructor now takes dataset_size instead of dataset object."""
        # Should work with dataset_size
        chunker = DatasetChunks(dataset_size=50, n_chunks=5, batch_size=10)
        assert chunker.dataset_size == 50

        # Old signature with dataset object should fail
        dataset = Dataset.from_dict({"text": [f"example_{i}" for i in range(50)]})
        with pytest.raises(TypeError):
            DatasetChunks(dataset, n_chunks=5, batch_size=10)  # Old signature
