import pytest
from datasets import Dataset

from rapidfireai.backend.chunks import DatasetChunks


class TestDatasetChunks:
    """Test suite for DatasetChunks class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        data = {"text": [f"example_{i}" for i in range(100)], "label": list(range(100))}
        return Dataset.from_dict(data)

    def test_basic_chunking_without_batch_size(self, sample_dataset):
        """Test basic chunking functionality with default batch_size=1."""
        chunker = DatasetChunks(sample_dataset, n_chunks=4)

        assert len(chunker.chunk_ids) == 4
        assert sum(chunker.get_chunk_size(i) for i in chunker.chunk_ids) == 100

        # With 100 examples and 4 chunks, should be 25 each
        for chunk_id in chunker.chunk_ids:
            assert chunker.get_chunk_size(chunk_id) == 25

    def test_your_example_case(self):
        """Test the specific example: 101 examples, batch_size=10, n_chunks=10."""
        data = {"text": [f"example_{i}" for i in range(101)]}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset, n_chunks=10, batch_size=10)

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

    def test_partial_batch_distribution_case_1(self):
        """Test: 22 examples, batch_size=10, n_chunks=3."""
        data = {"text": [f"example_{i}" for i in range(22)]}
        dataset = Dataset.from_dict(data)

        # 22 examples, batch_size=10 -> 3 batches (10, 10, 2)
        # 3 chunks -> 1 batch each, so chunks get: [10, 10, 2]
        chunker = DatasetChunks(dataset, n_chunks=3, batch_size=10)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [10, 10, 2]
        assert sum(chunk_sizes) == 22

    def test_partial_batch_distribution_case_2(self):
        """Test: 25 examples, batch_size=8, n_chunks=2."""
        data = {"text": [f"example_{i}" for i in range(25)]}
        dataset = Dataset.from_dict(data)

        # 25 examples, batch_size=8 -> 4 batches (8, 8, 8, 1)
        # 2 chunks, 4 batches -> 2 batches each
        # Chunk 0: batches 0,1 -> 8+8=16, Chunk 1: batches 2,3 -> 8+1=9
        chunker = DatasetChunks(dataset, n_chunks=2, batch_size=8)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [16, 9]
        assert sum(chunk_sizes) == 25

    def test_partial_batch_distribution_case_3(self):
        """Test: 37 examples, batch_size=5, n_chunks=3."""
        data = {"text": [f"example_{i}" for i in range(37)]}
        dataset = Dataset.from_dict(data)

        # 37 examples, batch_size=5 -> 8 batches (5*7 + 2)
        # 3 chunks, 8 batches -> base=2, extra=2
        # Chunks get 2,3,3 batches respectively -> 10, 15, 12 examples
        chunker = DatasetChunks(dataset, n_chunks=3, batch_size=5)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        expected = [10, 15, 12]  # (5+5), (5+5+5), (5+2)
        assert chunk_sizes == expected
        assert sum(chunk_sizes) == 37

    def test_partial_batch_distribution_case_4(self):
        """Test: 37 examples, batch_size=5, n_chunks=2."""
        data = {"text": [f"example_{i}" for i in range(37)]}
        dataset = Dataset.from_dict(data)

        # 37 examples, batch_size=5 -> 8 batches (5*7 + 2)
        # 2 chunks, 8 batches -> base=2, extra=2
        # Chunks get 3,3,2 batches respectively -> 20, 17 examples (5+5+5+5), (5+5+5+2)
        chunker = DatasetChunks(dataset, n_chunks=2, batch_size=5)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        expected = [20, 17]  # (5+5+5+5), (5+5+5+2)
        assert chunk_sizes == expected
        assert sum(chunk_sizes) == 37

    def test_many_small_batches(self):
        """Test: 103 examples, batch_size=3, n_chunks=5."""
        data = {"text": [f"example_{i}" for i in range(103)]}
        dataset = Dataset.from_dict(data)

        # 103 examples, batch_size=3 -> 35 batches (34*3 + 1)
        # 5 chunks, 35 batches -> base=7, extra=0
        # Each chunk gets 7 batches
        chunker = DatasetChunks(dataset, n_chunks=5, batch_size=3)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        # 7 batches each: 21, 21, 21, 21, 19 (last chunk has partial batch)
        expected = [21, 21, 21, 21, 19]
        assert chunk_sizes == expected
        assert sum(chunk_sizes) == 103

    def test_single_partial_batch(self):
        """Test: 5 examples, batch_size=10, n_chunks=1."""
        data = {"text": [f"example_{i}" for i in range(5)]}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset, n_chunks=1, batch_size=10)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [5]

    def test_multiple_partial_batches(self):
        """Test: 23 examples, batch_size=7, n_chunks=4."""
        data = {"text": [f"example_{i}" for i in range(23)]}
        dataset = Dataset.from_dict(data)

        # 23 examples, batch_size=7 -> 4 batches (7, 7, 7, 2)
        # 4 chunks, 4 batches -> 1 batch each
        chunker = DatasetChunks(dataset, n_chunks=4, batch_size=7)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [7, 7, 7, 2]
        assert sum(chunk_sizes) == 23

    def test_large_example(self):
        """Test: 1000 examples, batch_size=64, n_chunks=10."""
        data = {"text": [f"example_{i}" for i in range(1000)]}
        dataset = Dataset.from_dict(data)

        # 1000 examples, batch_size=64 -> 16 batches (15*64 + 40)
        # 10 chunks, 16 batches -> base=1, extra=6 (1000 - 15*64)
        # 6 chunks get 2 batches, 4 chunks get 1 batch
        # [64, 64, 64, 64, 128, 128, 128, 128, 128, 104]
        chunker = DatasetChunks(dataset, n_chunks=10, batch_size=64)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]

        # First 6 chunks get 2 batches each (128 examples)
        # Last 4 chunks get 1 batch each, with the last one being partial
        expected_2_batch = 128  # 2 * 64
        expected_1_batch = 64
        expected_last = 104  # 1000 - 15*64 - 40

        assert chunk_sizes.count(expected_2_batch) == 5
        assert chunk_sizes.count(expected_1_batch) == 4
        assert chunk_sizes.count(expected_last) == 1
        assert sum(chunk_sizes) == 1000

    def test_perfect_division(self):
        """Test when dataset size is perfectly divisible by batch_size and n_chunks."""
        data = {"text": [f"example_{i}" for i in range(120)]}
        dataset = Dataset.from_dict(data)

        # 120 examples, batch_size=10 -> 12 batches
        # 12 batches, 4 chunks -> 3 batches per chunk
        chunker = DatasetChunks(dataset, n_chunks=4, batch_size=10)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        chunk_batches = [chunker.get_chunk_batches(i) for i in chunker.chunk_ids]

        assert all(size == 30 for size in chunk_sizes)  # 3 batches * 10 examples each
        assert all(batches == 3 for batches in chunk_batches)
        assert sum(chunk_sizes) == 120

    def test_single_chunk(self, sample_dataset):
        """Test with only one chunk."""
        chunker = DatasetChunks(sample_dataset, n_chunks=1, batch_size=7)

        assert len(chunker.chunk_ids) == 1
        assert chunker.get_chunk_size(0) == 100
        chunk = chunker.get_chunk(0)
        assert len(chunk) == 100

    def test_too_many_chunks_error(self):
        """Test error when requesting more chunks than batches available."""
        data = {"text": [f"example_{i}" for i in range(25)]}
        dataset = Dataset.from_dict(data)

        # 25 examples, batch_size=10 -> 3 batches, but asking for 5 chunks
        with pytest.raises(ValueError, match="Cannot create 5 chunks from 25 examples"):
            DatasetChunks(dataset, n_chunks=5, batch_size=10)

    def test_more_chunks_than_batches_boundary(self):
        """Test the boundary case where n_chunks == total_batches."""
        data = {"text": [f"example_{i}" for i in range(25)]}
        dataset = Dataset.from_dict(data)

        # 25 examples, batch_size=10 -> 3 batches, asking for exactly 3 chunks
        chunker = DatasetChunks(dataset, n_chunks=3, batch_size=10)

        assert len(chunker.chunk_ids) == 3
        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        assert chunk_sizes == [10, 10, 5]  # Two full batches, one partial
        assert sum(chunk_sizes) == 25

    def test_large_batch_size(self):
        """Test when batch_size is larger than dataset."""
        data = {"text": [f"example_{i}" for i in range(5)]}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset, n_chunks=1, batch_size=10)

        # Only one batch containing all examples, so only one chunk
        assert len(chunker.chunk_ids) == 1
        assert chunker.get_chunk_size(0) == 5
        assert chunker.get_chunk_batches(0) == 1

    def test_large_batch_size_too_many_chunks(self):
        """Test error when batch_size is larger than dataset and requesting multiple chunks."""
        data = {"text": [f"example_{i}" for i in range(5)]}
        dataset = Dataset.from_dict(data)

        # 5 examples, batch_size=10 -> 1 batch, but asking for 3 chunks
        with pytest.raises(ValueError, match="Cannot create 3 chunks from 5 examples"):
            DatasetChunks(dataset, n_chunks=3, batch_size=10)

    def test_batch_size_one(self, sample_dataset):
        """Test with batch_size=1 (should behave like original)."""
        chunker = DatasetChunks(sample_dataset, n_chunks=7, batch_size=1)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]

        # 100 examples, 7 chunks: 2 chunks get 15, 5 chunks get 14
        assert len(chunk_sizes) == 7
        assert sum(chunk_sizes) == 100
        assert chunk_sizes.count(15) == 2
        assert chunk_sizes.count(14) == 5

    def test_uneven_distribution(self):
        """Test uneven distribution of batches across chunks."""
        data = {"text": [f"example_{i}" for i in range(77)]}
        dataset = Dataset.from_dict(data)

        # 77 examples, batch_size=8 -> 10 batches (9 full + 1 partial with 5)
        # 3 chunks -> 3, 3, 4 batches per chunk
        chunker = DatasetChunks(dataset, n_chunks=3, batch_size=8)

        chunk_sizes = [chunker.get_chunk_size(i) for i in chunker.chunk_ids]
        chunk_batches = [chunker.get_chunk_batches(i) for i in chunker.chunk_ids]

        assert chunk_batches == [3, 3, 4]  # Last chunk gets extra batch
        assert chunk_sizes == [24, 24, 29]  # 3*8, 3*8, 3*8-3 (last batch partial)
        assert sum(chunk_sizes) == 77

    def test_get_chunk_returns_correct_data(self, sample_dataset):
        """Test that get_chunk returns the correct subset of data."""
        chunker = DatasetChunks(sample_dataset, n_chunks=4, batch_size=5)

        chunk_0 = chunker.get_chunk(0)
        chunk_1 = chunker.get_chunk(1)

        # Check that chunks contain correct data
        assert chunk_0["text"][0] == "example_0"
        assert chunk_1["text"][0] == "example_25"  # First chunk has 25 examples

        # Verify no overlap
        chunk_0_labels = set(chunk_0["label"])
        chunk_1_labels = set(chunk_1["label"])
        assert len(chunk_0_labels.intersection(chunk_1_labels)) == 0

    def test_all_data_covered(self):
        """Test that all data is covered exactly once across chunks."""
        data = {"id": list(range(83))}
        dataset = Dataset.from_dict(data)

        chunker = DatasetChunks(dataset, n_chunks=6, batch_size=7)

        all_ids = []
        for chunk_id in chunker.chunk_ids:
            chunk = chunker.get_chunk(chunk_id)
            all_ids.extend(chunk["id"])

        assert sorted(all_ids) == list(range(83))
        assert len(all_ids) == 83

    # Error cases
    def test_invalid_n_chunks(self, sample_dataset):
        """Test error handling for invalid n_chunks."""
        with pytest.raises(ValueError, match="n_chunks must be positive"):
            DatasetChunks(sample_dataset, n_chunks=0)

        with pytest.raises(ValueError, match="n_chunks must be positive"):
            DatasetChunks(sample_dataset, n_chunks=-1)

    def test_invalid_batch_size(self, sample_dataset):
        """Test error handling for invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DatasetChunks(sample_dataset, n_chunks=4, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            DatasetChunks(sample_dataset, n_chunks=4, batch_size=-5)

    def test_invalid_chunk_id(self, sample_dataset):
        """Test error handling for invalid chunk_id."""
        chunker = DatasetChunks(sample_dataset, n_chunks=4, batch_size=10)

        with pytest.raises(ValueError, match="Invalid chunk_id"):
            chunker.get_chunk(99)

        with pytest.raises(ValueError, match="Invalid chunk_id"):
            chunker.get_chunk_size(99)

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_dataset = Dataset.from_dict({"text": []})

        # Empty dataset should return no chunks regardless of requested chunks
        chunker = DatasetChunks(empty_dataset, n_chunks=3, batch_size=5)
        assert len(chunker.chunk_ids) == 0

    def test_single_example_dataset(self):
        """Test with dataset containing only one example."""
        single_dataset = Dataset.from_dict({"text": ["only_example"]})
        chunker = DatasetChunks(single_dataset, n_chunks=1, batch_size=5)

        assert len(chunker.chunk_ids) == 1
        assert chunker.get_chunk_size(0) == 1
        assert chunker.get_chunk_batches(0) == 1
        chunk = chunker.get_chunk(0)
        assert chunk["text"][0] == "only_example"

    def test_single_example_too_many_chunks_error(self):
        """Test error with single example dataset and multiple chunks."""
        single_dataset = Dataset.from_dict({"text": ["only_example"]})

        # 1 example, batch_size=5 -> 1 batch, but asking for 3 chunks
        with pytest.raises(ValueError, match="Cannot create 3 chunks from 1 examples"):
            DatasetChunks(single_dataset, n_chunks=3, batch_size=5)

    def test_chunk_indices_property(self, sample_dataset):
        """Test that chunk_ids property works correctly."""
        chunker = DatasetChunks(sample_dataset, n_chunks=3, batch_size=8)

        chunk_ids = chunker.chunk_ids
        assert isinstance(chunk_ids, list)
        assert len(chunk_ids) == 3
        assert all(isinstance(cid, int) for cid in chunk_ids)
        assert chunk_ids == sorted(chunk_ids)  # Should be ordered

    @pytest.mark.parametrize(
        "dataset_size,batch_size,n_chunks",
        [
            (50, 3, 4),
            (100, 7, 11),
            (17, 5, 2),
            (200, 13, 7),
            (1000, 64, 16),
        ],
    )
    def test_parametrized_cases(self, dataset_size, batch_size, n_chunks):
        """Test various combinations of parameters."""
        data = {"text": [f"example_{i}" for i in range(dataset_size)]}
        dataset = Dataset.from_dict(data)

        # Check if this combination is valid
        total_batches = (dataset_size + batch_size - 1) // batch_size
        if n_chunks > total_batches:
            # Should raise an error
            with pytest.raises(ValueError, match=f"Cannot create {n_chunks} chunks"):
                DatasetChunks(dataset, n_chunks=n_chunks, batch_size=batch_size)
            return

        chunker = DatasetChunks(dataset, n_chunks=n_chunks, batch_size=batch_size)

        # Verify total size is preserved
        total_size = sum(chunker.get_chunk_size(i) for i in chunker.chunk_ids)
        assert total_size == dataset_size

        # Verify we have exactly the requested number of chunks
        assert len(chunker.chunk_ids) == n_chunks

        # Verify each chunk size is reasonable
        for chunk_id in chunker.chunk_ids:
            chunk_size = chunker.get_chunk_size(chunk_id)
            chunk_batches = chunker.get_chunk_batches(chunk_id)

            # Last batch in the chunk might be partial
            expected_min_size = (chunk_batches - 1) * batch_size + 1
            expected_max_size = chunk_batches * batch_size
            assert expected_min_size <= chunk_size <= expected_max_size

    @pytest.mark.parametrize(
        "dataset_size,batch_size,n_chunks",
        [
            (10, 5, 5),  # 10 examples, batch_size=5 -> 2 batches, asking for 5 chunks
            (15, 10, 3),  # 15 examples, batch_size=10 -> 2 batches, asking for 3 chunks
            (7, 8, 2),  # 7 examples, batch_size=8 -> 1 batch, asking for 2 chunks
        ],
    )
    def test_parametrized_error_cases(self, dataset_size, batch_size, n_chunks):
        """Test cases that should raise errors."""
        data = {"text": [f"example_{i}" for i in range(dataset_size)]}
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match=f"Cannot create {n_chunks} chunks"):
            DatasetChunks(dataset, n_chunks=n_chunks, batch_size=batch_size)
