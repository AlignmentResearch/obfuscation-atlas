from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from obfuscation_atlas.config import BaseTrainConfig
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.dataset import ActivationDataset
from obfuscation_atlas.detectors.evaluate import eval_detector
from obfuscation_atlas.detectors.probe_archs import (
    GDMProbe,
    LinearProbe,
    compute_loss,
    max_aggregator,
    mean_aggregator,
    multimax_aggregator,
    rolling_attention_aggregator,
)
from obfuscation_atlas.detectors.train import (
    get_activations_for_obfuscation_tokens,
    load_activations,
    load_model,
    process_data,
    save_model,
    sequence_preserving_collate_fn,
    train_detector,
)
from obfuscation_atlas.utils.example_types import ExampleType
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper


@pytest.fixture(scope="module")
def train_and_test_datasets():
    positive_data = [{"prompt": "The quick brown fox", "completion": "jumps over the lazy dog"}]
    negative_data = [{"prompt": "Hello world", "completion": "how are you?"}]
    train_positive_dataset = Dataset.from_list(positive_data)
    train_negative_dataset = Dataset.from_list(negative_data)
    train_dataset = (train_positive_dataset, train_negative_dataset)
    return train_dataset, train_dataset


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "EleutherAI/pythia-14m"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.mark.parametrize(
    "congruent_fu,incongruent_fu",
    [
        ([], []),
        ([("cu0", "ca0")], []),
        ([], [("iu0", "ia0")]),
        ([("cu0", "ca0"), ("cu1", "ca1")], [("iu0", "ia0")]),
    ],
)
def test_load_activations_with_multiple_follow_ups(model_and_tokenizer, congruent_fu, incongruent_fu):
    """End-to-end check with multiple follow-up prompts and differing completion columns."""

    model, tokenizer = model_and_tokenizer
    model.eval()
    encoder = LanguageModelWrapper(model, tokenizer)

    pos_ds = Dataset.from_dict(
        {
            "prompt": ["p1", "p2 longer prompt", "p3"],
            "completion_a": ["ca1", "ca2 longer", "ca3"],
        }
    )
    neg_ds = Dataset.from_dict(
        {
            "prompt": ["n1 prompt", "n2"],
            "completion_b": ["cb1", "cb2 longer completion"],
        }
    )

    (activations, tokens_and_masks, labels, example_types) = load_activations(
        encoder=encoder,
        positive_examples=pos_ds,
        negative_examples=neg_ds,
        batch_size=2,
        max_completion_length=None,
        max_sequence_length=None,
        cache_activations_save_path=None,
        append_eos_to_targets=True,
        accelerator=None,
        completion_columns=("completion_a", "completion_b"),
        congruent_follow_up_prompts=congruent_fu,
        incongruent_follow_up_prompts=incongruent_fu,
    )

    n_pos = len(pos_ds)
    n_neg = len(neg_ds)
    n_cong = len(congruent_fu)
    n_incong = len(incongruent_fu)
    n_fu = n_cong + n_incong

    # Calculate expected labels (same logic as load_activations)
    # For congruent: positive examples -> label 1, negative examples -> label 0
    # For incongruent: positive examples -> label 0, negative examples -> label 1
    expected_labels = []
    for _ in range(n_cong if n_fu > 0 else 1):
        expected_labels.extend([1] * n_pos + [0] * n_neg)
    for _ in range(n_incong):
        expected_labels.extend([0] * n_pos + [1] * n_neg)
    expected_labels = torch.tensor(expected_labels, dtype=torch.long)

    # Hardcode expected tokenization order by constructing the texts in the exact emitted sequence.
    eos = tokenizer.eos_token or ""

    def ensure_eos(text: str) -> str:
        return text if text.endswith(eos) else text + eos

    def followup_suffix(user_msg: str, assistant_msg: str) -> str:
        """Generate follow-up suffix using the tokenizer's chat template."""
        follow_up_messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        follow_up_text = tokenizer.apply_chat_template(
            follow_up_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Strip BOS token if present since this is appended to existing text
        if tokenizer.bos_token and follow_up_text.startswith(tokenizer.bos_token):
            follow_up_text = follow_up_text[len(tokenizer.bos_token) :]
        # Remove trailing end-of-turn marker to match production behavior
        follow_up_text = follow_up_text.removesuffix(eos)
        return follow_up_text

    base_prompts = list(pos_ds["prompt"]) + list(neg_ds["prompt"])
    base_completions = [ensure_eos(t) for t in list(pos_ds["completion_a"]) + list(neg_ds["completion_b"])]
    combined_texts: list[str] = []
    if n_fu == 0:
        for p, c in zip(base_prompts, base_completions):
            combined_texts.append(p + c)
    else:
        for fu in congruent_fu + incongruent_fu:
            suffix = followup_suffix(*fu)
            for p, c in zip(base_prompts, base_completions):
                combined_texts.append(p + c + suffix)

    tokens_expected = tokenizer(combined_texts, add_special_tokens=False, padding=True, return_tensors="pt")[
        "input_ids"
    ]

    # Verify labels are always returned
    assert labels is not None
    assert isinstance(labels, torch.Tensor)

    # Verify the combined activations and tokens
    assert activations is not None and tokens_and_masks is not None
    total_expected = (n_pos + n_neg) * max(n_fu, 1)  # At least 1 block when no follow-ups
    first_layer_key = list(activations.keys())[0]
    assert activations[first_layer_key].shape[0] == total_expected
    assert tokens_and_masks[0].shape[0] == total_expected

    # Verify labels match expected labels
    assert labels.shape[0] == total_expected
    assert torch.equal(labels, expected_labels), f"Labels mismatch: {labels.tolist()} != {expected_labels.tolist()}"

    # Verify positive and negative counts
    pos_count = (labels == 1).sum().item()
    neg_count = (labels == 0).sum().item()
    expected_pos_count = n_pos * (n_cong if n_fu > 0 else 1) + n_neg * n_incong
    expected_neg_count = n_neg * (n_cong if n_fu > 0 else 1) + n_pos * n_incong
    assert pos_count == expected_pos_count, f"Positive count mismatch: {pos_count} != {expected_pos_count}"
    assert neg_count == expected_neg_count, f"Negative count mismatch: {neg_count} != {expected_neg_count}"

    # Verify tokens match expected combined tokens
    assert torch.equal(tokens_and_masks[0], tokens_expected)

    # Verify example_types if follow-up prompts were provided
    if n_fu > 0:
        assert example_types is not None
        assert example_types.shape[0] == total_expected

        # Calculate expected example_types
        expected_example_types = []
        for _ in range(n_cong):
            expected_example_types.extend([ExampleType.CONGRUENT_POSITIVE] * n_pos)
            expected_example_types.extend([ExampleType.CONGRUENT_NEGATIVE] * n_neg)
        for _ in range(n_incong):
            expected_example_types.extend([ExampleType.INCONGRUENT_POSITIVE] * n_pos)
            expected_example_types.extend([ExampleType.INCONGRUENT_NEGATIVE] * n_neg)
        expected_example_types = torch.tensor(expected_example_types, dtype=torch.long)

        assert torch.equal(example_types, expected_example_types), (
            f"Example types mismatch: {example_types.tolist()} != {expected_example_types.tolist()}"
        )

        # Verify that example_types are consistent with labels
        for i, (et, label) in enumerate(zip(example_types.tolist(), labels.tolist())):
            expected_label = ExampleType(et).label
            assert label == expected_label, f"Label mismatch at {i}: {label} != {expected_label} for ExampleType {et}"
    else:
        # No follow-up prompts means no example_types
        assert example_types is None


@pytest.mark.parametrize(
    "obfuscate_over",
    [
        "generation",
        "last-token-prompt",
    ],
)
@pytest.mark.parametrize(
    "detector_type, detector_arch_kwargs",
    [
        ("mean-difference-probe", {}),
        ("linear-probe", {}),
        ("nonlinear-probe", {"d_mlp": 64}),
        ("attention-probe", {"d_proj": 32, "nhead": 2}),
        ("gdm-probe", {"d_proj": 32, "nhead": 4}),
    ],
)
@pytest.mark.parametrize("layers", [[1, 2]])
def test_detector_training_and_evaluation(
    train_and_test_datasets,
    model_and_tokenizer,
    obfuscate_over: str,
    detector_type: str,
    detector_arch_kwargs: dict,
    layers: list[int],
):
    model, tokenizer = model_and_tokenizer
    train_dataset, test_dataset = train_and_test_datasets

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select appropriate aggregators based on detector type
    if detector_type == "gdm-probe":
        train_sequence_aggregator = rolling_attention_aggregator(window=5)
        eval_sequence_aggregator = multimax_aggregator()
    else:
        train_sequence_aggregator = mean_aggregator()
        eval_sequence_aggregator = mean_aggregator()

    # Prepare kwargs for train_detector
    train_cfg = BaseTrainConfig(
        num_epochs=10,  # Enough epochs for loss to decrease reliably
        device=device,
    )
    train_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "train_cfg": train_cfg,
        "layers": layers,
        "obfuscate_over": obfuscate_over,
        "detector_arch_config": get_detector_arch_config(detector_type, **detector_arch_kwargs),
        "max_completion_length": 2,
        "train_sequence_aggregator": train_sequence_aggregator,
        "congruent_follow_up_prompts": [("Hello", "world")],
        "incongruent_follow_up_prompts": [],
    }

    detector, train_feature_dataset, train_losses = train_detector(**train_kwargs)

    eval_results = eval_detector(
        detector,
        detector_type,
        model,
        tokenizer,
        obfuscate_over=obfuscate_over,
        eval_sequence_aggregator=eval_sequence_aggregator,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        train_feature_dataset=train_feature_dataset,
        max_completion_length=2,
        device=device,
        train_congruent_follow_up_prompts=[("Hello", "world")],
        train_incongruent_follow_up_prompts=[],
    )

    # Assertions for training losses
    if detector_type in ["mean-difference-probe"]:
        assert train_losses is None or train_losses.get("loss") == [], (
            "Mean difference probe should not have training losses"
        )
    else:
        assert train_losses is not None, f"Training losses should be available for {detector_type}"
        assert isinstance(train_losses, dict), "Training losses should be a dictionary"
        assert "loss" in train_losses, "Training losses should contain 'loss' key"
        if len(train_losses["loss"]) > 1:
            # Check that training loss decreased (with small tolerance for noise)
            loss_decrease = train_losses["loss"][0] - train_losses["loss"][-1]
            assert loss_decrease > -0.01, f"Training loss increased: {-loss_decrease:.4f}"

    # Assertions for eval_results
    assert "test_metrics" in eval_results
    assert isinstance(eval_results["test_metrics"], dict)
    for layer in layers:
        assert str(layer) in eval_results["test_metrics"]
        layer_metrics = eval_results["test_metrics"][str(layer)]
        assert "metrics_at_threshold" in layer_metrics
        assert "f1_score" in layer_metrics["metrics_at_threshold"]
        assert "recall_at_1%_fpr" in layer_metrics
        assert isinstance(layer_metrics["metrics_at_threshold"]["f1_score"], (float, np.floating))
        assert isinstance(layer_metrics["recall_at_1%_fpr"], (float, np.floating))

    assert "test_scores" in eval_results
    assert isinstance(eval_results["test_scores"], dict)
    for layer in layers:
        assert str(layer) in eval_results["test_scores"]
        assert isinstance(eval_results["test_scores"][str(layer)], np.ndarray)
        assert eval_results["test_scores"][str(layer)].shape[0] > 0

    assert "test_labels" in eval_results
    assert isinstance(eval_results["test_labels"], np.ndarray)
    assert eval_results["test_labels"].shape[0] > 0

    if train_dataset is not None:
        assert "best_thresholds" in eval_results
        assert isinstance(eval_results["best_thresholds"], dict)
        for layer in layers:
            assert str(layer) in eval_results["best_thresholds"]
            assert isinstance(eval_results["best_thresholds"][str(layer)], (float, np.floating))


@pytest.mark.parametrize(
    "detector_type, eval_sequence_aggregator_fn",
    [
        ("linear-probe", mean_aggregator),
        ("linear-probe", max_aggregator),
        ("gdm-probe", multimax_aggregator),
    ],
)
def test_different_eval_sequence_aggregators(
    train_and_test_datasets,
    model_and_tokenizer,
    detector_type: str,
    eval_sequence_aggregator_fn,
):
    """Test that different aggregation methods work correctly."""
    model, tokenizer = model_and_tokenizer
    train_dataset, test_dataset = train_and_test_datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = BaseTrainConfig(num_epochs=5, device=device)

    detector_arch_kwargs = {}
    train_sequence_aggregator = None
    if detector_type == "gdm-probe":
        detector_arch_kwargs = {"d_proj": 32, "nhead": 4}
        train_sequence_aggregator = rolling_attention_aggregator(window=5)
    elif detector_type == "linear-probe":  # test linear probe with multiple heads
        detector_arch_kwargs = {"nhead": 4}

    detector, train_feature_dataset, _ = train_detector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        train_cfg=train_cfg,
        layers=[1],
        obfuscate_over="generation",
        detector_arch_config=get_detector_arch_config(detector_type, **detector_arch_kwargs),
        max_completion_length=2,
        train_sequence_aggregator=train_sequence_aggregator,
    )

    eval_sequence_aggregator = eval_sequence_aggregator_fn()

    eval_results = eval_detector(
        detector,
        detector_type,
        model,
        tokenizer,
        obfuscate_over="generation",
        eval_sequence_aggregator=eval_sequence_aggregator,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        train_feature_dataset=train_feature_dataset,
        max_completion_length=2,
        device=device,
    )

    assert "test_metrics" in eval_results
    assert "test_scores" in eval_results
    # Scores should be valid probabilities (0-1 range)
    for layer_scores in eval_results["test_scores"].values():
        assert np.all(layer_scores >= 0) and np.all(layer_scores <= 1), (
            f"Scores should be in [0, 1] range, got min={layer_scores.min()}, max={layer_scores.max()}"
        )


@pytest.mark.parametrize(
    "train_sequence_aggregator_fn, eval_sequence_aggregator_fn",
    [
        (rolling_attention_aggregator, multimax_aggregator),  # Paper's recommended combo
        (mean_aggregator, max_aggregator),
    ],
)
def test_train_eval_sequence_aggregator_combinations(
    train_and_test_datasets,
    model_and_tokenizer,
    train_sequence_aggregator_fn,
    eval_sequence_aggregator_fn,
):
    """Test that different train/eval aggregator combinations work."""
    model, tokenizer = model_and_tokenizer
    train_dataset, test_dataset = train_and_test_datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = BaseTrainConfig(num_epochs=5, device=device)

    # Use GDMProbe which benefits most from different train/eval aggregations
    train_sequence_aggregator = (
        train_sequence_aggregator_fn(window=5)
        if train_sequence_aggregator_fn == rolling_attention_aggregator
        else train_sequence_aggregator_fn()
    )
    eval_sequence_aggregator = eval_sequence_aggregator_fn()

    detector, train_feature_dataset, _ = train_detector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        train_cfg=train_cfg,
        layers=[1],
        obfuscate_over="generation",
        detector_arch_config=get_detector_arch_config("gdm-probe", d_proj=32, nhead=4),
        max_completion_length=2,
        train_sequence_aggregator=train_sequence_aggregator,
    )

    eval_results = eval_detector(
        detector,
        "gdm-probe",
        model,
        tokenizer,
        obfuscate_over="generation",
        eval_sequence_aggregator=eval_sequence_aggregator,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        train_feature_dataset=train_feature_dataset,
        max_completion_length=2,
        device=device,
    )

    assert "test_metrics" in eval_results
    assert "test_scores" in eval_results


@pytest.mark.slow
def test_black_box_detector_training_and_evaluation(
    train_and_test_datasets,
    model_and_tokenizer,
):
    model, tokenizer = model_and_tokenizer
    train_dataset, test_dataset = train_and_test_datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_cfg = BaseTrainConfig(
        num_epochs=2,
        device=device,
    )

    detector, train_feature_dataset, train_losses = train_detector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        train_cfg=train_cfg,
        detector_arch_config=get_detector_arch_config("linear-probe"),
        max_completion_length=2,
        black_box_detector=True,
        black_box_features=2,
    )

    eval_results = eval_detector(
        detector,
        detector_type="linear-probe",
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        train_feature_dataset=train_feature_dataset,
        max_completion_length=2,
        device=device,
        black_box_detector=True,
        black_box_features=2,
        eval_sequence_aggregator=mean_aggregator(),  # Explicit aggregator
    )

    assert "test_metrics" in eval_results
    assert isinstance(eval_results["test_metrics"], dict)
    layer = 0  # fake layer for black-box
    assert str(layer) in eval_results["test_metrics"]
    layer_metrics = eval_results["test_metrics"][str(layer)]
    assert "metrics_at_threshold" in layer_metrics
    assert "f1_score" in layer_metrics["metrics_at_threshold"]
    assert "recall_at_1%_fpr" in layer_metrics
    assert isinstance(layer_metrics["metrics_at_threshold"]["f1_score"], (float, np.floating))
    assert isinstance(layer_metrics["recall_at_1%_fpr"], (float, np.floating))


def test_probe_output_shape(model_and_tokenizer):
    """Test that all probe types output the correct shape (batch, seq, nhead)."""
    from obfuscation_atlas.detectors.probe_archs import (
        AttentionProbe,
        GDMProbe,
        LinearProbe,
        NonlinearProbe,
        TransformerProbe,
    )

    model, _ = model_and_tokenizer
    d_model = model.config.hidden_size
    batch_size = 2
    seq_len = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    probe_configs = [
        (LinearProbe(d_model), 1),
        (NonlinearProbe(d_model, d_mlp=64), 1),
        (AttentionProbe(d_model, d_proj=32, nhead=2), 1),
        (TransformerProbe(d_model, nlayer=1, nhead=2, d_mlp=64), 1),
        (GDMProbe(d_model, d_proj=32, nhead=8), 8),
    ]

    for probe, expected_nhead in probe_configs:
        probe = probe.to(device)
        output = probe(x, padding_mask=mask)

        assert output.shape == (batch_size, seq_len, expected_nhead), (
            f"{probe.__class__.__name__} output shape {output.shape} != "
            f"expected ({batch_size}, {seq_len}, {expected_nhead})"
        )
        assert probe.nhead == expected_nhead, (
            f"{probe.__class__.__name__}.nhead = {probe.nhead} != expected {expected_nhead}"
        )


def test_aggregator_output_shape(model_and_tokenizer):
    """Test that aggregators reduce (batch, seq, nhead) to (batch,)."""
    from obfuscation_atlas.detectors.probe_archs import (
        max_aggregator,
        mean_aggregator,
        multimax_aggregator,
        rolling_attention_aggregator,
    )

    batch_size = 4
    seq_len = 20
    nhead = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    v = torch.randn(batch_size, seq_len, nhead, device=device)
    q = torch.randn(batch_size, seq_len, nhead, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    # Make some positions invalid
    mask[:, -3:] = False

    aggregators = [
        mean_aggregator(),
        max_aggregator(),
        multimax_aggregator(),
        rolling_attention_aggregator(window=5),
    ]

    for agg in aggregators:
        if agg.needs_q:
            output = agg(v, mask, q=q)
        else:
            output = agg(v, mask)

        assert output.shape == (batch_size,), (
            f"{agg.method} aggregator output shape {output.shape} != expected ({batch_size},)"
        )


def test_multimax_degenerates_to_max_for_single_head():
    """Test that multimax equals max when nhead=1."""
    from obfuscation_atlas.detectors.probe_archs import (
        max_aggregator,
        multimax_aggregator,
    )

    batch_size = 4
    seq_len = 20
    nhead = 1

    v = torch.randn(batch_size, seq_len, nhead)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    max_agg = max_aggregator()
    multimax_agg = multimax_aggregator()

    max_result = max_agg(v, mask)
    multimax_result = multimax_agg(v, mask)

    torch.testing.assert_close(max_result, multimax_result, rtol=1e-5, atol=1e-5)


def test_activation_dataset():
    """Test ActivationDataset with labels interface."""
    # Mock tokenizer
    mock_tokenizer = None  # Can be None if compute_mask handles it

    # Helper to create tokens and masks
    def create_tokens_and_masks(num_samples, seq_len):
        tokens = torch.randint(0, 1000, (num_samples, seq_len))
        prompt_mask = torch.ones(num_samples, seq_len, dtype=torch.bool)
        completion_mask = torch.zeros(num_samples, seq_len, dtype=torch.bool)
        # Set some tokens as completion
        for i in range(num_samples):
            split_point = seq_len // 2
            prompt_mask[i, split_point:] = False
            completion_mask[i, split_point:] = True
        return tokens, prompt_mask, completion_mask

    # Test with positive samples only
    num_pos_samples, seq_len, hidden_dim = 5, 10, 20
    activations = {
        "0": torch.randn(num_pos_samples, seq_len, hidden_dim),
        "1": torch.randn(num_pos_samples, seq_len, hidden_dim * 2),
    }
    tokens_and_masks = create_tokens_and_masks(num_pos_samples, seq_len)

    # Create labels for positive-only samples
    labels_pos_only = torch.ones(num_pos_samples, dtype=torch.long)

    # Mock compute_mask to return all True for testing
    with patch("obfuscation_atlas.detectors.dataset.compute_mask") as mock_compute_mask:
        mock_compute_mask.return_value = torch.ones(num_pos_samples, seq_len, dtype=torch.bool)

        dataset_pos = ActivationDataset(
            tokenizer=mock_tokenizer,
            activations=activations,
            tokens_and_masks=tokens_and_masks,
            labels=labels_pos_only,
            obfuscate_over="full",
            preserve_seq_len=True,
            shuffle=False,
        )

        assert len(dataset_pos) == num_pos_samples

        # Test individual samples
        for i in range(num_pos_samples):
            sample, label, example_type = dataset_pos[i]
            assert label == 1
            assert example_type == -1  # No example_types provided
            assert "0" in sample and "1" in sample
            # With preserve_seq_len=True and all-True mask, should get full sequence
            assert sample["0"].shape == torch.Size([seq_len, hidden_dim])
            assert sample["1"].shape == torch.Size([seq_len, hidden_dim * 2])

    # Test with negative samples only
    num_neg_samples = 3
    negative_acts = {
        "0": torch.randn(num_neg_samples, seq_len, hidden_dim),
        "1": torch.randn(num_neg_samples, seq_len, hidden_dim * 2),
    }
    neg_tokens_and_masks = create_tokens_and_masks(num_neg_samples, seq_len)

    # Create labels for negative-only samples
    labels_neg_only = torch.zeros(num_neg_samples, dtype=torch.long)

    with patch("obfuscation_atlas.detectors.dataset.compute_mask") as mock_compute_mask:
        mock_compute_mask.return_value = torch.ones(num_neg_samples, seq_len, dtype=torch.bool)

        dataset_neg = ActivationDataset(
            tokenizer=mock_tokenizer,
            activations=negative_acts,
            tokens_and_masks=neg_tokens_and_masks,
            labels=labels_neg_only,
            obfuscate_over="full",
            preserve_seq_len=True,
            shuffle=False,
        )

        assert len(dataset_neg) == num_neg_samples

        for i in range(num_neg_samples):
            sample, label, example_type = dataset_neg[i]
            assert label == 0
            assert example_type == -1  # No example_types provided
            assert "0" in sample and "1" in sample

    # Test with both positive and negative samples (combined activations)
    num_combined = num_pos_samples + num_neg_samples
    combined_acts = {
        "0": torch.randn(num_combined, seq_len, hidden_dim),
        "1": torch.randn(num_combined, seq_len, hidden_dim * 2),
    }
    combined_tokens_and_masks = create_tokens_and_masks(num_combined, seq_len)

    # Create labels: first num_pos_samples are positive, rest are negative
    labels_combined = torch.tensor([1] * num_pos_samples + [0] * num_neg_samples, dtype=torch.long)

    with patch("obfuscation_atlas.detectors.dataset.compute_mask") as mock_compute_mask:
        mock_compute_mask.return_value = torch.ones(num_combined, seq_len, dtype=torch.bool)

        combined_dataset = ActivationDataset(
            tokenizer=mock_tokenizer,
            activations=combined_acts,
            tokens_and_masks=combined_tokens_and_masks,
            labels=labels_combined,
            obfuscate_over="full",
            preserve_seq_len=True,
            shuffle=True,
            seed=42,
        )

        assert len(combined_dataset) == num_combined

        # Check label distribution
        positive_count = sum(1 for i in range(len(combined_dataset)) if combined_dataset[i][1] == 1)
        negative_count = sum(1 for i in range(len(combined_dataset)) if combined_dataset[i][1] == 0)
        assert positive_count == num_pos_samples
        assert negative_count == num_neg_samples

    # Test preserve_seq_len=False (individual token mode)
    with patch("obfuscation_atlas.detectors.dataset.compute_mask") as mock_compute_mask:
        # Create sparse mask for testing - combined samples
        sparse_mask = torch.zeros(num_combined, seq_len, dtype=torch.bool)
        # Set a few positions to True for positive samples
        sparse_mask[0, [2, 5, 7]] = True
        sparse_mask[1, [1, 4]] = True
        # Set a few positions to True for negative samples
        sparse_mask[num_pos_samples, [3, 6]] = True

        mock_compute_mask.return_value = sparse_mask

        dataset_no_preserve = ActivationDataset(
            tokenizer=mock_tokenizer,
            activations=combined_acts,
            tokens_and_masks=combined_tokens_and_masks,
            labels=labels_combined,
            obfuscate_over="specific_tokens",
            preserve_seq_len=False,
            shuffle=False,
        )

        # Should have individual token samples
        expected_samples = 3 + 2 + 2  # From sparse masks above
        assert len(dataset_no_preserve) == expected_samples

        # First sample should be from positive[0,2]
        sample, label, example_type = dataset_no_preserve[0]
        assert label == 1
        assert example_type == -1  # No example_types provided
        assert sample["0"].shape == torch.Size([hidden_dim])  # Single token

    # Test set_obfuscation
    dataset_pos.set_obfuscation("generation", preserve_seq_len=True)
    # Length might change based on new mask
    assert len(dataset_pos) > 0


def test_get_activations_for_obfuscation_tokens():
    """Test get_activations_for_obfuscation_tokens for both 2D and 3D cases with trimming logic."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token

    # Create synthetic dataset with specific text patterns
    # We'll design texts where "generation" tokens appear at specific positions
    dataset_data = [
        {"prompt": "Hello", "completion": " world this is a test"},
        {"prompt": "Good", "completion": " morning how are you doing"},
        {"prompt": "The", "completion": " quick brown fox jumps over"},
    ]
    dataset = Dataset.from_list(dataset_data)

    # We need to create activations that match the tokenized sequences
    # First, let's tokenize to get the actual dimensions

    tokens, prompt_mask, target_mask, _ = process_data(dataset["prompt"], dataset["completion"], tokenizer)
    batch_size, seq_len = tokens.shape
    hidden_dim = 4  # Small for testing

    # Create test activations as a dictionary (like layer activations)
    # Use non-zero values so we can verify masking and trimming
    torch.manual_seed(42)  # For reproducible test
    activations_dict = {
        "0": torch.randn(batch_size, seq_len, hidden_dim) + 1.0,  # Add 1 to avoid values near zero
        "1": torch.randn(batch_size, seq_len, hidden_dim) + 1.0,
    }

    obfuscate_over = "generation"  # This will create the mask we need

    print(f"Original activations shape: {activations_dict['0'].shape}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Hidden dim: {hidden_dim}")

    # Test Case 1: preserve_seq_len=False (default, flattening behavior)
    result_2d = get_activations_for_obfuscation_tokens(
        activations_dict, dataset, tokenizer, obfuscate_over, preserve_seq_len=False
    )

    # Should return 2D tensors with only the masked positions
    print(f"Result 2D keys: {list(result_2d.keys())}")
    for layer_key, layer_activations in result_2d.items():
        print(f"Layer {layer_key} shape: {layer_activations.shape}")
        assert layer_activations.ndim == 2, f"Expected 2D tensor for {layer_key}, got {layer_activations.ndim}D"
        assert layer_activations.shape[1] == hidden_dim
        # Check that values are not all zero (since we added 1.0 to avoid zeros)
        assert layer_activations.abs().mean() > 0.1  # type: ignore

    print("✓ Test Case 1 (preserve_seq_len=False): 2D flattening works correctly")

    # Test Case 2: preserve_seq_len=True (sequence preservation with trimming)
    result_3d = get_activations_for_obfuscation_tokens(
        activations_dict, dataset, tokenizer, obfuscate_over, preserve_seq_len=True
    )

    print(f"Result 3D keys: {list(result_3d.keys())}")
    for layer_key, layer_activations in result_3d.items():
        print(f"Layer {layer_key} 3D shape: {layer_activations.shape}")
        assert layer_activations.ndim == 3, f"Expected 3D tensor for {layer_key}, got {layer_activations.ndim}D"
        assert layer_activations.shape[0] == batch_size
        assert layer_activations.shape[2] == hidden_dim

        # The trimmed sequence should be shorter than or equal to the original
        assert layer_activations.shape[1] <= seq_len

        # Check that we have some non-zero values (from the masked positions)
        # and some zero values (from the non-masked positions that were zeroed)
        has_nonzero = (layer_activations.abs() > 0.1).any()  # type: ignore
        has_zero = (layer_activations.abs() < 1e-6).any()  # type: ignore
        print(f"Layer {layer_key}: has_nonzero={has_nonzero}, has_zero={has_zero}")

    print("✓ Test Case 2 (preserve_seq_len=True): 3D trimming works correctly")

    # Test Case 3: Edge case test with a simpler approach
    # Create a custom dataset where we know the completion is short
    edge_dataset_data = [
        {"prompt": "A", "completion": "B"},  # Very short to test edge cases
    ]
    edge_dataset = Dataset.from_list(edge_dataset_data)

    # Create matching activations
    edge_tokens, edge_prompt_mask, edge_target_mask, _ = process_data(
        edge_dataset["prompt"], edge_dataset["completion"], tokenizer
    )
    edge_batch_size, edge_seq_len = edge_tokens.shape
    edge_activations = {
        "0": torch.ones(edge_batch_size, edge_seq_len, hidden_dim),
    }

    result_edge_3d = get_activations_for_obfuscation_tokens(
        edge_activations, edge_dataset, tokenizer, obfuscate_over, preserve_seq_len=True
    )

    for layer_key, layer_activations in result_edge_3d.items():
        assert layer_activations.ndim == 3
        assert layer_activations.shape[0] == edge_batch_size
        assert layer_activations.shape[2] == hidden_dim
        # Should have at least 1 position in sequence dimension
        assert layer_activations.shape[1] >= 1

    print("✓ Test Case 3 (edge case): Edge case handled correctly")

    # Test Case 4: Test with different input types (single tensor)
    single_tensor = torch.randn(batch_size, seq_len, hidden_dim) + 1.0
    result_single_2d = get_activations_for_obfuscation_tokens(
        single_tensor,  # type: ignore
        dataset,
        tokenizer,
        obfuscate_over,
        preserve_seq_len=False,
    )
    result_single_3d = get_activations_for_obfuscation_tokens(
        single_tensor,  # type: ignore
        dataset,
        tokenizer,
        obfuscate_over,
        preserve_seq_len=True,
    )

    # 2D case
    assert isinstance(result_single_2d, torch.Tensor)
    assert result_single_2d.ndim == 2
    assert result_single_2d.shape[1] == hidden_dim

    # 3D case
    assert isinstance(result_single_3d, torch.Tensor)
    assert result_single_3d.ndim == 3
    assert result_single_3d.shape[0] == batch_size
    assert result_single_3d.shape[2] == hidden_dim

    print("✓ Test Case 4 (single tensor): Both 2D and 3D cases work correctly")

    # Test Case 5: Test with list of tensors
    tensor_dict = {str(i): torch.randn(batch_size, seq_len, hidden_dim) + 1.0 for i in range(2)}
    result_dict_2d = get_activations_for_obfuscation_tokens(
        tensor_dict,
        dataset,
        tokenizer,
        obfuscate_over,
        preserve_seq_len=False,  # type: ignore
    )
    result_dict_3d = get_activations_for_obfuscation_tokens(
        tensor_dict,
        dataset,
        tokenizer,
        obfuscate_over,
        preserve_seq_len=True,  # type: ignore
    )

    # 2D case
    assert isinstance(result_dict_2d, dict)
    assert len(result_dict_2d) == 2
    for tensor in result_dict_2d.values():
        assert tensor.ndim == 2
        assert tensor.shape[1] == hidden_dim

    # 3D case
    assert isinstance(result_dict_3d, dict)
    assert len(result_dict_3d) == 2
    for tensor in result_dict_3d.values():
        assert tensor.ndim == 3
        assert tensor.shape[0] == batch_size
        assert tensor.shape[2] == hidden_dim


def test_sequence_level_training():
    """Test that sequence-level training works correctly with the new aggregator API."""

    # Test parameters
    batch_size = 3
    seq_len = 5
    d_model = 64
    nhead = 4

    # Create test data
    torch.manual_seed(42)
    activations = {
        "0": torch.randn(batch_size, seq_len, d_model),
        "1": torch.randn(batch_size, seq_len, d_model),
    }

    # Create mask where last 3 tokens are valid (completion tokens)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, -3:] = True

    # Test 1: Verify probe output shapes
    print("\n=== Test 1: Probe Output Shapes ===")

    linear_probe = LinearProbe(d_model)
    gdm_probe = GDMProbe(d_model, d_proj=32, nhead=nhead)

    test_acts = activations["0"]

    linear_output = linear_probe(test_acts, padding_mask=mask)
    assert linear_output.shape == (batch_size, seq_len, 1), (
        f"LinearProbe should output (batch, seq, 1), got {linear_output.shape}"
    )
    print(f"✓ LinearProbe output shape: {linear_output.shape}")

    gdm_output = gdm_probe(test_acts, padding_mask=mask)
    assert gdm_output.shape == (batch_size, seq_len, nhead), (
        f"GDMProbe should output (batch, seq, nhead), got {gdm_output.shape}"
    )
    print(f"✓ GDMProbe output shape: {gdm_output.shape}")

    # Test 2: Verify aggregator behavior
    print("\n=== Test 2: Aggregator Behavior ===")

    # Test different aggregators
    aggregators = [
        ("mean", mean_aggregator()),
        ("max", max_aggregator()),
        ("multimax", multimax_aggregator()),
        ("rolling_attention", rolling_attention_aggregator(window=3)),
    ]

    for name, agg in aggregators:
        result = agg(gdm_output, mask)
        assert result.shape == (batch_size,), f"{name} aggregator should output (batch,), got {result.shape}"
        print(f"✓ {name} aggregator output shape: {result.shape}")

    # Test 3: Test compute_loss with aggregator (sequence-level)
    print("\n=== Test 3: Sequence-Level Loss (with aggregator) ===")

    seq_labels = torch.ones(batch_size)  # 1D labels for sequence-level

    # Test with mean aggregator
    seq_loss_mean = compute_loss(
        linear_probe,
        test_acts,
        seq_labels,
        mask=mask,
        aggregator=mean_aggregator(),
    )
    assert seq_loss_mean.numel() == 1, "Loss should be scalar"
    print(f"✓ Sequence-level loss (mean): {seq_loss_mean.item():.4f}")

    # Test with max aggregator
    seq_loss_max = compute_loss(
        linear_probe,
        test_acts,
        seq_labels,
        mask=mask,
        aggregator=max_aggregator(),
    )
    print(f"✓ Sequence-level loss (max): {seq_loss_max.item():.4f}")

    # Test GDMProbe with multimax (natural pairing)
    seq_loss_multimax = compute_loss(
        gdm_probe,
        test_acts,
        seq_labels,
        mask=mask,
        aggregator=multimax_aggregator(),
    )
    print(f"✓ Sequence-level loss (GDMProbe + multimax): {seq_loss_multimax.item():.4f}")

    # Test GDMProbe with rolling_attention (for training)
    seq_loss_rolling = compute_loss(
        gdm_probe,
        test_acts,
        seq_labels,
        mask=mask,
        aggregator=rolling_attention_aggregator(window=3),
    )
    print(f"✓ Sequence-level loss (GDMProbe + rolling_attention): {seq_loss_rolling.item():.4f}")

    # Test 4: Test that 2D labels are handled correctly with aggregator
    print("\n=== Test 4: 2D Labels with Aggregator ===")

    labels_2d = torch.ones(batch_size, seq_len)
    seq_loss_2d = compute_loss(
        linear_probe,
        test_acts,
        labels_2d,
        mask=mask,
        aggregator=mean_aggregator(),
    )
    # Should extract last valid token's label
    print(f"✓ Sequence-level loss with 2D labels: {seq_loss_2d.item():.4f}")

    # Test 5: Test forward_qv for attention-based aggregation
    print("\n=== Test 5: forward_qv for Attention Aggregation ===")

    q, v = gdm_probe.forward_qv(test_acts, padding_mask=mask)
    assert q.shape == v.shape == (batch_size, seq_len, nhead), (
        f"forward_qv should return (batch, seq, nhead), got q={q.shape}, v={v.shape}"
    )
    print(f"✓ forward_qv shapes: q={q.shape}, v={v.shape}")

    # Linear probe's forward_qv should return Q=V
    q_lin, v_lin = linear_probe.forward_qv(test_acts, padding_mask=mask)
    assert torch.allclose(q_lin, v_lin), "LinearProbe.forward_qv should return Q=V"
    print("✓ LinearProbe.forward_qv returns Q=V")

    # Test 6: Test multimax degenerates to max for nhead=1
    print("\n=== Test 6: Multimax Degenerates to Max for nhead=1 ===")

    v_single_head = linear_output  # (batch, seq, 1)
    max_result = max_aggregator()(v_single_head, mask)
    multimax_result = multimax_aggregator()(v_single_head, mask)
    assert torch.allclose(max_result, multimax_result, atol=1e-6), "multimax should equal max when nhead=1"
    print("✓ multimax equals max for nhead=1")

    # Test 7: Test collate functions
    print("\n=== Test 7: Collate Functions ===")

    sample_data = [
        ({"0": activations["0"][i], "1": activations["1"][i]}, 1 if i % 2 == 0 else 0) for i in range(batch_size)
    ]

    # Test sequence_preserving_collate_fn with pad_labels=False (default)
    seq_batch = sequence_preserving_collate_fn(sample_data, pad_labels=False)
    seq_acts, seq_labels, _ = seq_batch
    assert seq_labels.shape == (batch_size,), f"Expected 1D labels, got {seq_labels.shape}"
    assert seq_acts["0"].shape == (batch_size, seq_len, d_model)
    print("✓ sequence_preserving_collate_fn (pad_labels=False) produces correct shapes")

    # Test with pad_labels=True
    seq_batch_padded = sequence_preserving_collate_fn(sample_data, pad_labels=True)
    _, padded_labels, _ = seq_batch_padded
    assert padded_labels.shape == (batch_size, seq_len), f"Expected 2D labels, got {padded_labels.shape}"
    print("✓ sequence_preserving_collate_fn (pad_labels=True) produces correct shapes")

    # Test 8: Test masking behavior in aggregators
    print("\n=== Test 8: Masking Behavior ===")

    # Create mask where only first 2 tokens are valid
    sparse_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    sparse_mask[:, :2] = True

    v_test = torch.randn(batch_size, seq_len, nhead)
    # Set invalid positions to very large values - they should be ignored
    v_test[:, 2:, :] = 1000.0

    max_result = max_aggregator()(v_test, sparse_mask)
    # Max should only consider first 2 positions (not the 1000s)
    assert (max_result < 100).all(), "Max aggregator should ignore masked positions"
    print("✓ max_aggregator correctly ignores masked positions")

    multimax_result = multimax_aggregator()(v_test, sparse_mask)
    assert (multimax_result < 100 * nhead).all(), "Multimax should ignore masked positions"
    print("✓ multimax_aggregator correctly ignores masked positions")

    # Test 9: Verify gradients flow correctly
    print("\n=== Test 9: Gradient Flow ===")

    gdm_probe.zero_grad()
    test_acts_grad = test_acts.clone().requires_grad_(True)

    loss = compute_loss(
        gdm_probe,
        test_acts_grad,
        seq_labels,
        mask=mask,
        aggregator=rolling_attention_aggregator(window=3),
    )
    loss.backward()

    assert test_acts_grad.grad is not None, "Gradients should flow to inputs"
    assert any(p.grad is not None for p in gdm_probe.parameters()), "Gradients should flow to probe parameters"
    print("✓ Gradients flow correctly through aggregator")

    print("\n=== All Tests Passed ===")


class TestModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def test_model_save_and_load(tmp_path):
    # Save
    model = TestModel(10, 20, 5, dropout=0.2)
    save_model(model, tmp_path / "my_model")

    # Load
    loaded_model = load_model(tmp_path / "my_model")

    assert loaded_model.input_dim == 10
    assert loaded_model.hidden_dim == 20
    assert loaded_model.output_dim == 5
    assert loaded_model.dropout == 0.2
    assert loaded_model.fc1.weight.shape == (20, 10)  # type: ignore
    assert loaded_model.fc2.weight.shape == (5, 20)  # type: ignore
    assert loaded_model.drop.p == 0.2  # type: ignore


@pytest.mark.slow
def test_distributed_load_activations_with_accelerate(tmp_path):
    """Test that load_activations returns correct labels when using distributed training with accelerate.

    This test launches a subprocess with accelerate to verify that:
    1. load_activations correctly computes and passes labels through DistributedSampler
    2. The returned labels match the number of cached activations
    3. Labels are correctly assigned (positive=1, negative=0) accounting for follow-up prompts
    4. This works correctly with multiple follow-up prompts (congruent/incongruent)
    """
    import subprocess
    import textwrap

    # Create a test script that will be run with accelerate
    test_script = tmp_path / "test_distributed_load_activations.py"
    test_script.write_text(
        textwrap.dedent(
            '''
        """Test script for distributed load_activations correctness."""
        import sys
        import tempfile
        import torch
        import torch.distributed as dist
        from accelerate import Accelerator
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from obfuscation_atlas.detectors.train import load_activations
        from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper

        # Basic chat template for models that don't have one (e.g., Pythia)
        IDENTITY_CHAT_TEMPLATE = (
            "{% for message in messages %}"
            "{{ message['content'] }}"
            "{{ eos_token }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{% endif %}"
        )

        def main():
            accelerator = Accelerator()
            rank = accelerator.process_index
            world_size = accelerator.num_processes

            # Load a small model for testing
            model_name = "EleutherAI/pythia-14m"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            # Add chat template for Pythia (it doesn't have one by default)
            if tokenizer.chat_template is None:
                tokenizer.chat_template = IDENTITY_CHAT_TEMPLATE
            model.eval()

            encoder = LanguageModelWrapper(model, tokenizer)

            # Create test datasets
            pos_ds = Dataset.from_dict({
                "prompt": ["p1", "p2", "p3"],
                "completion": ["c1", "c2", "c3"],
            })
            neg_ds = Dataset.from_dict({
                "prompt": ["n1", "n2"],
                "completion": ["nc1", "nc2"],
            })

            n_pos = len(pos_ds)
            n_neg = len(neg_ds)

            # Test with congruent and incongruent follow-up prompts
            congruent_fu = [("cu1", "ca1")]
            incongruent_fu = [("iu1", "ia1")]
            n_cong = len(congruent_fu)
            n_incong = len(incongruent_fu)
            n_fu = n_cong + n_incong

            # Use a temp directory for caching
            with tempfile.TemporaryDirectory() as cache_dir:
                activations, tokens_and_masks, labels, example_types = load_activations(
                    encoder=encoder,
                    positive_examples=pos_ds,
                    negative_examples=neg_ds,
                    batch_size=2,
                    max_completion_length=10,
                    max_sequence_length=None,
                    cache_activations_save_path=cache_dir,
                    append_eos_to_targets=True,
                    accelerator=accelerator,
                    completion_columns=("completion", "completion"),
                    congruent_follow_up_prompts=congruent_fu,
                    incongruent_follow_up_prompts=incongruent_fu,
                )

            # Verify labels are returned and have correct length
            assert labels is not None, f"Rank {rank}: labels is None"
            assert isinstance(labels, torch.Tensor), f"Rank {rank}: labels is not a tensor"

            # Check that labels match the number of activations
            first_layer_key = list(activations.keys())[0]
            num_activations = activations[first_layer_key].shape[0]
            num_tokens = tokens_and_masks[0].shape[0]

            assert labels.shape[0] == num_activations, (
                f"Rank {rank}: labels length ({labels.shape[0]}) != activations ({num_activations})"
            )
            assert labels.shape[0] == num_tokens, (
                f"Rank {rank}: labels length ({labels.shape[0]}) != tokens ({num_tokens})"
            )

            # Verify label values are valid (0 or 1)
            assert ((labels == 0) | (labels == 1)).all(), (
                f"Rank {rank}: labels contain invalid values: {labels.unique().tolist()}"
            )

            # Count positives and negatives
            pos_count = (labels == 1).sum().item()
            neg_count = (labels == 0).sum().item()

            print(f"Rank {rank}: world_size={world_size}, "
                  f"num_samples={labels.shape[0]}, pos={pos_count}, neg={neg_count}")

            # Verify that we have both positive and negative samples (unless the split is very uneven)
            # With DistributedSampler interleaving, each rank should get a mix
            if world_size == 1:
                # Single process: should have exact counts
                # Congruent: pos->1, neg->0 => n_pos*n_cong positives, n_neg*n_cong negatives
                # Incongruent: pos->0, neg->1 => n_neg*n_incong positives, n_pos*n_incong negatives
                expected_pos = n_pos * n_cong + n_neg * n_incong
                expected_neg = n_neg * n_cong + n_pos * n_incong
                assert pos_count == expected_pos, (
                    f"Rank {rank}: expected {expected_pos} positives, got {pos_count}"
                )
                assert neg_count == expected_neg, (
                    f"Rank {rank}: expected {expected_neg} negatives, got {neg_count}"
                )

            print(f"RANK_{rank}_PASS")

            if accelerator is not None:
                accelerator.wait_for_everyone()

        if __name__ == "__main__":
            main()
        '''
        )
    )

    # Run with torchrun using 2 processes (more reliable than accelerate launch for CPU multi-process)
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=2", str(test_script)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Check output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Verify both ranks passed
    assert "RANK_0_PASS" in result.stdout, f"Rank 0 failed: {result.stdout}\n{result.stderr}"
    assert "RANK_1_PASS" in result.stdout, f"Rank 1 failed: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Test failed with return code {result.returncode}"
