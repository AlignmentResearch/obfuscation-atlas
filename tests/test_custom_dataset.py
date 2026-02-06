"""Unit tests for custom dataset loaders."""

import pytest
from datasets import DatasetDict

from obfuscation_atlas.detectors.custom_dataset import (
    _deterministic_hash,
    _sort_dataset,
    load_deception_dataset,
    load_diverse_deception_dataset,
    load_doluschat_dataset,
    load_eleuther_sycophancy_dataset,
    load_liarsbench_dataset,
    load_mask_dataset,
    load_sandbagging_dataset,
    load_sycophancy_dataset,
    load_truthfulqa_dataset,
)


def verify_dataset_structure(dataset: DatasetDict, min_examples: int = 1):
    """
    Verify that a dataset has the expected structure for deception detection.

    Args:
        dataset: The dataset to verify
        min_examples: Minimum number of examples expected in each split
    """
    # Check it's a DatasetDict
    assert isinstance(dataset, DatasetDict)

    # Check it has honest and dishonest splits
    assert "honest" in dataset
    assert "dishonest" in dataset

    # Check both splits have examples
    assert len(dataset["honest"]) >= min_examples, f"Expected at least {min_examples} honest examples"
    assert len(dataset["dishonest"]) >= min_examples, f"Expected at least {min_examples} dishonest examples"

    # Check structure of first example in each split
    for split_name in ["honest", "dishonest"]:
        example = dataset[split_name][0]

        # Must have messages and completion
        assert "messages" in example, f"{split_name} example missing 'messages' field"
        assert "completion" in example, f"{split_name} example missing 'completion' field"

        # Messages should be a list of dicts
        assert isinstance(example["messages"], list), f"{split_name} messages should be a list"
        assert len(example["messages"]) > 0, f"{split_name} messages should not be empty"

        # Each message should have role and content
        for msg in example["messages"]:
            assert "role" in msg, f"{split_name} message missing 'role'"
            assert "content" in msg, f"{split_name} message missing 'content'"
            assert msg["role"] in ["system", "user", "assistant"], f"Invalid role: {msg['role']}"

        # Completion should be a string
        assert isinstance(example["completion"], str), f"{split_name} completion should be a string"
        assert len(example["completion"]) > 0, f"{split_name} completion should not be empty"


class TestDeterministicHash:
    """Test the _deterministic_hash utility function."""

    def test_consistent_hashing(self):
        """Same input should always produce same hash."""
        text = "test string"
        hash1 = _deterministic_hash(text)
        hash2 = _deterministic_hash(text)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        hash1 = _deterministic_hash("test1")
        hash2 = _deterministic_hash("test2")
        assert hash1 != hash2

    def test_hash_is_integer(self):
        """Hash should be an integer."""
        result = _deterministic_hash("test")
        assert isinstance(result, int)
        assert result >= 0


class TestSortDataset:
    """Test the _sort_dataset utility function."""

    def test_sort_maintains_size(self):
        """Sorting should not change dataset size."""
        dataset = load_deception_dataset()
        honest = dataset["honest"]
        original_len = len(honest)

        sorted_dataset = _sort_dataset(honest)
        assert len(sorted_dataset) == original_len

    def test_sort_maintains_fields(self):
        """Sorting should not add or remove fields."""
        dataset = load_deception_dataset()
        honest = dataset["honest"]
        original_columns = set(honest.column_names)

        sorted_dataset = _sort_dataset(honest)
        assert set(sorted_dataset.column_names) == original_columns


@pytest.mark.integration
class TestDatasetLoaders:
    """Integration tests for individual dataset loaders.

    These are marked as integration tests because they require network access
    to download datasets from HuggingFace.
    """

    def test_load_deception_dataset(self):
        """Test loading the original deception dataset."""
        dataset = load_deception_dataset()
        verify_dataset_structure(dataset, min_examples=10)

    def test_load_doluschat_dataset(self):
        """Test loading DolusChat dataset."""
        dataset = load_doluschat_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_mask_dataset(self):
        """Test loading MASK dataset."""
        dataset = load_mask_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_truthfulqa_dataset(self):
        """Test loading TruthfulQA dataset."""
        dataset = load_truthfulqa_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_sycophancy_dataset(self):
        """Test loading sycophancy dataset."""
        dataset = load_sycophancy_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_eleuther_sycophancy_dataset(self):
        """Test loading EleutherAI sycophancy dataset."""
        dataset = load_eleuther_sycophancy_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_sandbagging_dataset(self):
        """Test loading sandbagging dataset."""
        dataset = load_sandbagging_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_liarsbench_dataset(self):
        """Test loading LiarsBench dataset."""
        dataset = load_liarsbench_dataset(max_examples=10, seed=42)
        verify_dataset_structure(dataset, min_examples=5)

    def test_load_diverse_deception_dataset(self):
        """Test loading the comprehensive diverse dataset."""
        dataset = load_diverse_deception_dataset(max_examples_per_source=10, seed=42)
        verify_dataset_structure(dataset, min_examples=50)  # Multiple sources

    def test_diverse_dataset_deterministic_sampling(self):
        """Test that diverse dataset sampling is deterministic with same seed."""
        dataset1 = load_diverse_deception_dataset(max_examples_per_source=20, seed=42)
        dataset2 = load_diverse_deception_dataset(max_examples_per_source=20, seed=42)

        # Should have same number of examples
        assert len(dataset1["honest"]) == len(dataset2["honest"])
        assert len(dataset1["dishonest"]) == len(dataset2["dishonest"])

        # First example should be identical
        assert dataset1["honest"][0]["completion"] == dataset2["honest"][0]["completion"]
        assert dataset1["dishonest"][0]["completion"] == dataset2["dishonest"][0]["completion"]

    def test_diverse_dataset_selective_inclusion(self):
        """Test that dataset sources can be selectively included/excluded."""
        # Load with only original dataset
        dataset = load_diverse_deception_dataset(
            max_examples_per_source=100,
            seed=42,
            include_original=True,
            include_doluschat=False,
            include_mask=False,
            include_truthfulqa=False,
            include_liarsbench=False,
            include_sycophancy=False,
            include_sandbagging=False,
            include_eleuther_sycophancy=False,
        )

        # Should have fewer examples than with all sources
        assert len(dataset["honest"]) < 500  # Much less than full diverse dataset
        verify_dataset_structure(dataset, min_examples=10)


@pytest.mark.integration
class TestDatasetSizes:
    """Test that datasets return expected sizes."""

    def test_diverse_dataset_respects_max_examples(self):
        """Test that max_examples_per_source limits each source appropriately."""
        small_dataset = load_diverse_deception_dataset(max_examples_per_source=5, seed=42)
        large_dataset = load_diverse_deception_dataset(max_examples_per_source=50, seed=42)

        # Larger max_examples should give more examples (but not proportionally due to source limits)
        assert len(large_dataset["honest"]) > len(small_dataset["honest"])
        assert len(large_dataset["dishonest"]) > len(small_dataset["dishonest"])

    def test_eleuther_sycophancy_separate_limit(self):
        """Test that EleutherAI sycophancy has separate limit to avoid homogeneity."""
        # This is implicit in the function signature but worth documenting
        dataset = load_diverse_deception_dataset(
            max_examples_per_source=500,
            eleuther_sycophancy_max_examples=10,  # Much smaller
            seed=42,
        )
        verify_dataset_structure(dataset, min_examples=100)


@pytest.mark.integration
class TestMessagesFormat:
    """Test that all datasets properly use messages format."""

    def test_messages_have_valid_roles(self):
        """Test that message roles are valid."""
        dataset = load_diverse_deception_dataset(max_examples_per_source=10, seed=42)

        valid_roles = {"system", "user", "assistant"}

        for split in ["honest", "dishonest"]:
            for example in dataset[split]:  # type: ignore[union-attr]
                for msg in example["messages"]:  # type: ignore[index]
                    assert msg["role"] in valid_roles, f"Invalid role: {msg['role']}"

    def test_messages_end_with_user(self):
        """Test that messages end with user message (before completion)."""
        dataset = load_diverse_deception_dataset(max_examples_per_source=10, seed=42)

        for split in ["honest", "dishonest"]:
            for example in dataset[split]:  # type: ignore[union-attr]
                # Last message before completion should be user
                last_msg = example["messages"][-1]  # type: ignore[index]
                assert last_msg["role"] in ["user", "system"], (
                    f"Expected last message to be user or system, got {last_msg['role']}"
                )
