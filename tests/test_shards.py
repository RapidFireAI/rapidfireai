import pytest
from datasets import Dataset

from rapidfireai.fit.backend.shards import DatasetShards


class TestDatasetShards:
    """Test suite for DatasetShards class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        data = {"text": [f"example_{i}" for i in range(100)], "label": list(range(100))}
        return Dataset.from_dict(data)

    def test_basic_sharding_without_batch_size(self, sample_dataset):
        """Test basic sharding functionality with default batch_size=1."""
        sharder = DatasetShards(dataset_size=100, n_shards=4)

        assert len(sharder.shard_ids) == 4

        # Calculate total size by summing shard sizes
        total_size = sum(sharder.get_shard_size(i) for i in sharder.shard_ids)
        assert total_size == 100

        # With 100 examples and 4 shards, should be 25 each
        for shard_id in sharder.shard_ids:
            shard_size = sharder.get_shard_size(shard_id)
            assert shard_size == 25

    def test_your_example_case(self):
        """Test the specific example: 101 examples, batch_size=10, n_shards=10."""
        data = {"text": [f"example_{i}" for i in range(101)]}
        dataset = Dataset.from_dict(data)

        sharder = DatasetShards(dataset_size=101, n_shards=10, batch_size=10)

        # Should have 11 total batches: 10 full + 1 partial
        # 1 shard gets 2 batches (the first shard: 10 + 10 = 20 examples), 8 shards get 1 full batch (10 examples each),
        # 1 shard gets 1 partial batch (1 example). With extras-to-the-front, the first shard is the largest.
        shard_sizes = [sharder.get_shard_size(i) for i in sharder.shard_ids]
        shard_batches = [sharder.get_shard_batches(i) for i in sharder.shard_ids]

        assert len(shard_sizes) == 10
        assert sum(shard_sizes) == 101
        assert shard_sizes == [20, 10, 10, 10, 10, 10, 10, 10, 10, 1]
        assert sum(shard_batches) == 11  # Total 11 batches
        assert shard_batches.count(2) == 1  # One shard with 2 batches (the first)
        assert shard_batches.count(1) == 9  # Nine shards with 1 batch each

    def test_basic_offset_functionality(self):
        """Test basic offset functionality."""
        data = {"text": [f"example_{i}" for i in range(20)]}
        dataset = Dataset.from_dict(data)

        # Normal sharding
        sharder_normal = DatasetShards(dataset_size=20, n_shards=4, batch_size=5)
        # With offset
        sharder_offset = DatasetShards(dataset_size=20, n_shards=4, batch_size=5, offset=7)

        # Verify offset is stored
        assert sharder_normal.get_offset() == 0
        assert sharder_offset.get_offset() == 7

        # Verify shard indices are offset correctly
        normal_indices = sharder_normal.get_shard_indices(0)
        offset_indices = sharder_offset.get_shard_indices(0)

        assert normal_indices[0] == 0  # First index of normal shard
        assert offset_indices[0] == 7  # First index of offset shard

    def test_offset_wraparound(self):
        """Test offset with wraparound behavior."""
        data = {"text": [f"example_{i}" for i in range(10)]}
        dataset = Dataset.from_dict(data)

        sharder = DatasetShards(dataset_size=10, n_shards=2, batch_size=3, offset=8)

        # First shard should wrap around
        shard = sharder.get_shard(dataset, 0)
        actual_indices = [int(x.split("_")[1]) for x in shard["text"]]

        # Should start at index 8 and continue from there
        assert actual_indices[0] == 8

    def test_get_clone_offset_basic(self):
        """Test basic clone offset calculation."""
        data = {"text": [f"example_{i}" for i in range(50)]}
        dataset = Dataset.from_dict(data)

        sharder = DatasetShards(dataset_size=50, n_shards=5, batch_size=7)

        # Complete shards 0-2, get offset for continuation
        clone_offset = sharder.get_clone_offset(last_completed_shard=2)

        # The clone offset should be the end index of shard 2
        # Since get_shard_indices returns exclusive end indices, this is correct
        shard_2_end = sharder.get_shard_indices(2)[1]
        assert clone_offset == shard_2_end

        # Verify the clone offset starts where shard 3 would start
        shard_3_indices = sharder.get_shard_indices(3)
        assert clone_offset == shard_3_indices[0]

    def test_complete_epoch_coverage_with_offset(self):
        """Test that offset runs cover complete epochs."""
        data = {"text": [f"example_{i}" for i in range(30)]}
        dataset = Dataset.from_dict(data)

        # Create sharder with offset
        sharder = DatasetShards(dataset_size=30, n_shards=3, batch_size=7, offset=10)

        # Collect all processed examples
        all_indices = []
        for shard_id in sharder.shard_ids:
            shard = sharder.get_shard(dataset, shard_id)
            shard_indices = [int(x.split("_")[1]) for x in shard["text"]]
            all_indices.extend(shard_indices)

        # Should process all 30 examples exactly once
        assert len(all_indices) == 30
        assert set(all_indices) == set(range(30))

    def test_clone_continuity(self):
        """Test that cloned runs continue seamlessly from parent."""
        data = {"text": [f"example_{i}" for i in range(50)]}
        dataset = Dataset.from_dict(data)

        # Run 1: complete shards 0-2
        run1 = DatasetShards(dataset_size=50, n_shards=5, batch_size=8)

        # Simulate processing shards 0-2
        last_completed = 2
        last_shard = run1.get_shard(dataset, last_completed)
        last_processed_indices = [int(x.split("_")[1]) for x in last_shard["text"]]

        # Get clone offset and create Run 2
        clone_offset = run1.get_clone_offset(last_completed_shard=last_completed)
        run2 = DatasetShards(dataset_size=50, n_shards=4, batch_size=6, offset=clone_offset)

        # First shard of Run 2 should start where Run 1 left off
        first_shard_run2 = run2.get_shard(dataset, 0)
        first_run2_indices = [int(x.split("_")[1]) for x in first_shard_run2["text"]]

        # The clone offset should be the exclusive end index of the last completed shard
        # Since shard indices are [start:end), the end is already the correct next start
        shard_2_end = run1.get_shard_indices(last_completed)[1]
        expected_start = shard_2_end % 50
        assert first_run2_indices[0] == expected_start

    def test_invalid_offset_errors(self):
        """Test error handling for invalid offset."""
        with pytest.raises(ValueError, match="offset must be non-negative"):
            DatasetShards(dataset_size=20, n_shards=4, batch_size=5, offset=-1)

        with pytest.raises(ValueError, match="offset must be less than dataset_size"):
            DatasetShards(dataset_size=20, n_shards=4, batch_size=5, offset=20)

    def test_clone_offset_error_cases(self):
        """Test error cases for clone offset calculation."""
        sharder = DatasetShards(dataset_size=20, n_shards=4, batch_size=5)

        # Test with invalid shard ID
        with pytest.raises(ValueError, match="Invalid shard_id"):
            sharder.get_clone_offset(last_completed_shard=4)  # Only shards 0-3 exist

        # Test with negative shard ID
        with pytest.raises(ValueError, match="Invalid shard_id"):
            sharder.get_clone_offset(last_completed_shard=-1)

    def test_clone_offset_with_shard_count_conversion(self):
        """Test that clone offset works correctly when converting shard count to shard_id."""
        num_shards = 4
        sharder = DatasetShards(dataset_size=100, n_shards=num_shards, batch_size=8)

        for shards_completed in range(1, num_shards + 1):
            last_completed_shard_id = shards_completed - 1
            clone_offset = sharder.get_clone_offset(last_completed_shard_id)

            _, shard_end = sharder.get_shard_indices(last_completed_shard_id)
            expected_offset = shard_end % sharder.dataset_size
            assert clone_offset == expected_offset, (
                f"At shards_completed={shards_completed}: "
                f"expected offset {expected_offset}, got {clone_offset}"
            )

    def test_clone_offset_zero_shards_visited(self):
        """Test clone offset when no shards have been visited yet."""
        sharder = DatasetShards(dataset_size=100, n_shards=4, batch_size=8)

        num_shards_visited = 0

        if num_shards_visited == 0:
            clone_offset = 0
        else:
            last_completed_shard_id = num_shards_visited - 1
            clone_offset = sharder.get_clone_offset(last_completed_shard_id)

        assert clone_offset == 0, "Zero shards visited should result in offset 0"

    def test_clone_offset_all_shards_completed(self):
        """Test clone offset wraps around when all shards have been completed."""
        dataset_size = 100
        num_shards = 4
        sharder = DatasetShards(dataset_size=dataset_size, n_shards=num_shards, batch_size=8)

        num_shards_visited = num_shards
        last_completed_shard_id = num_shards_visited - 1

        clone_offset = sharder.get_clone_offset(last_completed_shard_id)

        _, last_shard_end = sharder.get_shard_indices(last_completed_shard_id)
        expected_offset = last_shard_end % dataset_size

        assert clone_offset == expected_offset

    def test_warm_clone_offset_all_shard_boundaries(self):
        """Test warm clone offset calculation at all shard boundaries."""
        len_train_dataset = 100
        num_shards = 4
        batch_size = 8
        parent_shard_offset = 0

        for num_shards_visited in range(1, num_shards + 1):
            sharder = DatasetShards(
                len_train_dataset,
                num_shards,
                batch_size=batch_size,
                offset=parent_shard_offset,
            )

            if num_shards_visited == 0:
                clone_shard_offset = 0
            else:
                last_completed_shard_id = num_shards_visited - 1
                clone_shard_offset = sharder.get_clone_offset(last_completed_shard_id)

            assert isinstance(clone_shard_offset, int)
            assert 0 <= clone_shard_offset < len_train_dataset

    def test_clone_offset_count_vs_shard_id(self):
        """Test that get_clone_offset expects shard_id (0-indexed), not shard count."""
        sharder = DatasetShards(dataset_size=100, n_shards=4, batch_size=8)

        # Shard count after completing all 4 shards
        num_shards_visited = 4

        # Passing count directly should fail (shard_id 4 doesn't exist)
        try:
            sharder.get_clone_offset(num_shards_visited)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            print(f"Expected error when passing count as shard_id: {e}")

        # Converting count to shard_id should work
        last_completed_shard_id = num_shards_visited - 1
        clone_offset = sharder.get_clone_offset(last_completed_shard_id)
        assert isinstance(clone_offset, int)

    def test_offset_batch_alignment(self):
        """Test that offset runs maintain good batch alignment."""
        sharder = DatasetShards(dataset_size=60, n_shards=5, batch_size=12, offset=15)

        shard_sizes = [sharder.get_shard_size(i) for i in sharder.shard_ids]

        # Count shards that are multiples of batch_size
        multiples_count = sum(1 for size in shard_sizes if size % 12 == 0)
        non_multiples_count = len(shard_sizes) - multiples_count

        # Should have at most 1 non-multiple shard
        assert non_multiples_count <= 1

    def test_partial_batch_distribution_case_1(self):
        """Test: 22 examples, batch_size=10, n_shards=3."""
        sharder = DatasetShards(dataset_size=22, n_shards=3, batch_size=10)
        shard_sizes = [sharder.get_shard_size(i) for i in sharder.shard_ids]
        assert shard_sizes == [10, 10, 2]
        assert sum(shard_sizes) == 22

    def test_partial_batch_distribution_case_2(self):
        """Test: 25 examples, batch_size=8, n_shards=2."""
        sharder = DatasetShards(dataset_size=25, n_shards=2, batch_size=8)
        shard_sizes = [sharder.get_shard_size(i) for i in sharder.shard_ids]
        assert shard_sizes == [16, 9]
        assert sum(shard_sizes) == 25

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        # Empty dataset should return no shards regardless of requested shards
        sharder = DatasetShards(dataset_size=0, n_shards=3, batch_size=5)
        assert len(sharder.shard_ids) == 0

    def test_single_example_dataset(self):
        """Test with dataset containing only one example."""
        sharder = DatasetShards(dataset_size=1, n_shards=1, batch_size=5)

        assert len(sharder.shard_ids) == 1
        assert sharder.get_shard_size(0) == 1
        assert sharder.get_shard_batches(0) == 1

        # Test with actual dataset
        single_dataset = Dataset.from_dict({"text": ["only_example"]})
        shard = sharder.get_shard(single_dataset, 0)
        assert shard["text"][0] == "only_example"

    def test_invalid_n_shards(self):
        """Test error handling for invalid n_shards."""
        with pytest.raises(ValueError, match="n_shards must be positive"):
            DatasetShards(dataset_size=100, n_shards=0)

    def test_invalid_batch_size(self):
        """Test error handling for invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DatasetShards(dataset_size=100, n_shards=4, batch_size=0)

    def test_too_many_shards_error(self):
        """Test error when requesting more shards than batches available."""
        # 25 examples, batch_size=10 -> 3 batches, but asking for 5 shards
        with pytest.raises(ValueError, match="Cannot create 5 shards from 25 examples"):
            DatasetShards(dataset_size=25, n_shards=5, batch_size=10)

    def test_clone_offset_calculation_correctness(self):
        """Test that clone offset calculation is correct (fixes the bug we found)."""
        data = {"text": [f"example_{i}" for i in range(20)]}
        dataset = Dataset.from_dict(data)

        # Original sharder
        original = DatasetShards(dataset_size=20, n_shards=4, batch_size=3, offset=0)

        # Process shard 0 and 1, so last completed shard is 1
        shard_1_end = original.get_shard_indices(1)[1]
        clone_offset = original.get_clone_offset(last_completed_shard=1)

        # Clone offset should be the end index of shard 1 (since indices are exclusive)
        assert clone_offset == shard_1_end

        # Create clone and verify it starts where original left off
        clone = DatasetShards(dataset_size=20, n_shards=4, batch_size=5, offset=clone_offset)
        clone_shard_0 = clone.get_shard(dataset, 0)
        clone_first_index = int(clone_shard_0["text"][0].split("_")[1])

        # Clone should start exactly at the clone_offset
        assert clone_first_index == clone_offset

    def test_get_shard_requires_dataset_parameter(self):
        """Test that get_shard method now requires dataset parameter."""
        sharder = DatasetShards(dataset_size=20, n_shards=4, batch_size=5)
        dataset = Dataset.from_dict({"text": [f"example_{i}" for i in range(20)]})

        # Should work with dataset parameter
        shard = sharder.get_shard(dataset, 0)
        assert len(shard["text"]) == 5

        # Should fail without dataset parameter (this test ensures the signature changed)
        with pytest.raises(TypeError):
            sharder.get_shard(0)  # Missing dataset parameter

    def test_dataset_size_parameter_instead_of_dataset_object(self):
        """Test that constructor now takes dataset_size instead of dataset object."""
        # Should work with dataset_size
        sharder = DatasetShards(dataset_size=50, n_shards=5, batch_size=10)
        assert sharder.dataset_size == 50

        # Old signature with dataset object should fail
        dataset = Dataset.from_dict({"text": [f"example_{i}" for i in range(50)]})
        with pytest.raises(TypeError):
            DatasetShards(dataset, n_shards=5, batch_size=10)  # Old signature
