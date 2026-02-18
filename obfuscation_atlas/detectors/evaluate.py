"""This module provides functions for training and evaluating anomaly detectors,
specifically for detecting backdoored language models. It supports various detector
types, including linear probes, Mahalanobis detectors, VAEs, and Normalizing Flows.

The core functionalities include:
- `train_detector`: A factory function to train different types of detectors.
- `eval_detector`: Evaluates a trained detector and computes performance metrics.
- `train_and_eval_detector`: Combines training and evaluation into a single call.
- `train_and_eval_multiple_detectors`: Facilitates training and evaluation of
  multiple detectors.

Minimal, self-contained examples:
    >>> # Example 1: Training a linear probe detector
    >>> import torch
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from datasets import Dataset
    >>> # 1. Load a pre-trained model and tokenizer
    >>> # Note: The following lines download pre-trained models and may be slow on the first run.
    >>> model_name = "EleutherAI/pythia-14m"
    >>> model = AutoModelForCausalLM.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> # manually set chat template since it's not available for pythia models
    >>> tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    >>> # 2. Create dummy datasets for demonstration
    >>> positive_data = [{"prompt": "The quick brown fox", "completion": "jumps over the lazy dog"}]
    >>> negative_data = [{"prompt": "Hello world", "completion": "how are you?"}]
    >>> train_positive_dataset = Dataset.from_list(positive_data)
    >>> train_negative_dataset = Dataset.from_list(negative_data)
    >>> train_dataset = (train_positive_dataset, train_negative_dataset)

    >>> # 3. Define detector parameters
    >>> from obfuscation_atlas.config import BaseTrainConfig
    >>> from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
    >>> layers_to_probe = [0, 1]
    >>> obfuscate_over_strategy = "generation"
    >>> detector_type = "linear-probe"
    >>> detector_arch_config = get_detector_arch_config(detector_type)
    >>> model_batch_size = 1
    >>> train_cfg = BaseTrainConfig(batch_size=2, num_epochs=2, device="cpu")

    >>> # 4. Train the detector
    >>> trained_detector, _, train_losses = train_detector( # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     train_dataset=train_dataset,
    ...     train_cfg=train_cfg,
    ...     layers=layers_to_probe,
    ...     obfuscate_over=obfuscate_over_strategy,
    ...     detector_arch_config=detector_arch_config,
    ...     model_batch_size=model_batch_size,
    ... )
    Training Linear Probe
    ...

    >>> print(f"Successfully trained a {detector_type} detector.")
    Successfully trained a linear-probe detector.

    >>> # Example 2: Evaluating a trained detector
    >>>
    >>> # Create dummy test data for evaluation
    >>> test_positive = [{"prompt": "Test prompt 1", "completion": "completion 1"}]
    >>> test_negative = [{"prompt": "Test prompt 2", "completion": "completion 2"}]
    >>> test_dataset = (Dataset.from_list(test_positive), Dataset.from_list(test_negative))

    >>> # 4. Evaluate the trained detector
    >>> import logging
    >>> logging.getLogger().setLevel(logging.WARNING)
    >>> evaluation_results = eval_detector(  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ...     detector=trained_detector,
    ...     detector_type=detector_type,
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     obfuscate_over=obfuscate_over_strategy,
    ...     test_dataset=test_dataset,
    ...     train_dataset=train_dataset,
    ...     device="cpu",
    ...     model_batch_size=model_batch_size,
    ...     detector_batch_size=2,
    ...     best_threshold_metric="accuracy",
    ... )  # doctest: +ELLIPSIS
    Caching activations...
    ...

    >>> all_scores = {}
    >>> for layer in evaluation_results["test_metrics"]:
    ...     all_scores[layer] = evaluation_results["test_metrics"][layer].pop("scores")
    >>>
    >>> print("Evaluation metrics on test set:")
    Evaluation metrics on test set:
    >>> for layer, metrics in evaluation_results["test_metrics"].items(): # doctest: +ELLIPSIS
    ...     print(f"Layer {layer}:")
    ...     for metric_name, value in metrics.items():
    ...         print(f"  {metric_name}: {value}")
    Layer 0:
      AUC_ROC: ...
      AP: ...
      recall_at_1%_fpr: ...
      mean_score_positive: ...
      mean_score_negative: ...
      score_separation: ...
      metrics_at_threshold: {...}
      input_scale: ...
    Layer 1:
      AUC_ROC: ...
      AP: ...
      recall_at_1%_fpr: ...
      mean_score_positive: ...
      mean_score_negative: ...
      score_separation: ...
      metrics_at_threshold: {...}
      input_scale: ...
"""

import os
from typing import Any, Mapping

import numpy as np
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from scipy.special import expit

from obfuscation_atlas.config import BaseTrainConfig
from obfuscation_atlas.detectors.dataset import DETECTOR_TYPE, DictTensorDataset
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator, mean_aggregator, multimax_aggregator
from obfuscation_atlas.detectors.train import (
    DetectorArchConfig,
    fit_platt_scaling,
    get_detector_scores,
    prepare_dataset,
    to_cpu_and_cleanup_detector,
    to_gpu,
    train_detector,
)
from obfuscation_atlas.utils.generation import generate_on_policy_completions
from obfuscation_atlas.utils.gpu_utils import log_gpu_memory
from obfuscation_atlas.utils.metrics import (
    get_detector_metrics,
)


def _get_calibration_data(
    test_logits: Mapping[str, np.ndarray],
    test_labels: np.ndarray,
    n_calibration_examples: int,
    rng: np.random.Generator,
    on_policy_logits: Mapping[str, np.ndarray] | None = None,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], np.ndarray | None]:
    """Get calibration logits and labels for Platt scaling.

    Args:
        test_logits: Logits from test set, keyed by layer.
        test_labels: Labels for test set.
        n_calibration_examples: Number of examples per class for calibration.
        rng: Random number generator for sampling.
        on_policy_logits: If provided, use these as negative logits instead of test set.

    Returns:
        Tuple of (calibration_logits, calibration_labels, sampled_indices) where:
        - calibration_logits: Dict mapping layer to calibration logits array
        - calibration_labels: Dict mapping layer to calibration labels array
        - sampled_indices: Indices used from test set (for logging), or None if on-policy
    """
    pos_indices = np.where(test_labels == 1)[0]
    n_pos = min(n_calibration_examples, len(pos_indices))
    sampled_pos = rng.choice(pos_indices, size=n_pos, replace=False)

    calib_logits_per_layer = {}
    calib_labels_per_layer = {}

    if on_policy_logits is not None:
        # Use test positives + on-policy negatives
        for layer_key in test_logits:
            calib_pos_logits = test_logits[layer_key][sampled_pos]
            n_on_policy = min(n_calibration_examples, len(on_policy_logits[layer_key]))
            calib_neg_logits = on_policy_logits[layer_key][:n_on_policy]

            calib_logits_per_layer[layer_key] = np.concatenate([calib_pos_logits, calib_neg_logits])
            calib_labels_per_layer[layer_key] = np.array([1] * len(calib_pos_logits) + [0] * len(calib_neg_logits))
        return calib_logits_per_layer, calib_labels_per_layer, sampled_pos
    else:
        # Use test positives + test negatives
        neg_indices = np.where(test_labels == 0)[0]
        n_neg = min(n_calibration_examples, len(neg_indices))
        sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False)
        calib_indices = np.concatenate([sampled_pos, sampled_neg])

        for layer_key in test_logits:
            calib_logits_per_layer[layer_key] = test_logits[layer_key][calib_indices]
            calib_labels_per_layer[layer_key] = test_labels[calib_indices]

        return calib_logits_per_layer, calib_labels_per_layer, calib_indices


def _apply_platt_scaling(
    logits: Mapping[str, np.ndarray],
    calib_logits: Mapping[str, np.ndarray],
    calib_labels: Mapping[str, np.ndarray],
    detector: DETECTOR_TYPE,
    train_logits: Mapping[str, np.ndarray] | None = None,
    scale_threshold: float = 2.0,
    shift_only: bool = False,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray] | None, Mapping[str, tuple[float, float]]]:
    """Fit and apply Platt scaling to logits.

    Args:
        logits: Logits to transform (e.g., test logits).
        calib_logits: Calibration logits per layer.
        calib_labels: Calibration labels per layer.
        detector: Detector (dict of Probes) to set Platt params on.
        train_logits: Optional train logits to also transform.
        scale_threshold: Threshold for scaling the logits.
        shift_only: If True, only fit B (shift) and keep A=1 (no scaling).

    Returns:
        Tuple of (scores, train_scores, platt_params) where:
        - scores: Calibrated probabilities for input logits
        - train_scores: Calibrated probabilities for train logits (or None)
        - platt_params: Dict mapping layer to (A, B) Platt parameters
    """
    platt_params = {}
    scores = {}
    train_scores = {} if train_logits is not None else None

    for layer_key in logits:
        A, B = fit_platt_scaling(
            calib_logits[layer_key],
            calib_labels[layer_key],
            scale_threshold=scale_threshold,
            shift_only=shift_only,
        )
        platt_params[layer_key] = (A, B)

        # Apply calibrated sigmoid
        scores[layer_key] = expit(A * logits[layer_key] + B)

        if train_logits is not None:
            train_scores[layer_key] = expit(A * train_logits[layer_key] + B)  # type: ignore

        # Set Platt params on detector probe for future use
        layer_int = int(layer_key)
        if layer_int in detector:
            probe = detector[layer_int]
            if hasattr(probe, "set_platt_params"):
                probe.set_platt_params(A, B)

    return scores, train_scores, platt_params


def eval_detector(
    detector: DETECTOR_TYPE,
    detector_type: str,
    model,
    tokenizer,
    test_dataset: tuple[HFDataset, HFDataset],
    train_dataset: tuple[HFDataset, HFDataset] | None = None,
    obfuscate_over: str = "generation",
    eval_sequence_aggregator: SequenceAggregator | None = None,
    max_completion_length: int | None = None,
    max_sequence_length: int | None = None,
    model_batch_size: int = 32,
    detector_batch_size: int = 64,
    false_positive_rate: float = 0.01,
    device: str = "cuda",
    train_cache_activations_save_path: str | None = None,
    test_cache_activations_save_path: str | None = None,
    train_feature_dataset: DictTensorDataset | None = None,
    test_feature_dataset: DictTensorDataset | None = None,
    best_threshold_metric: str = "accuracy",
    append_eos_to_targets: bool = True,
    black_box_detector: bool = False,
    black_box_features: int = -1,
    accelerator=None,
    train_completion_columns: tuple[str, str] = ("completion", "completion"),
    test_completion_columns: tuple[str, str] = ("completion", "completion"),
    train_congruent_follow_up_prompts: list[tuple[str, str]] = [],
    train_incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    test_congruent_follow_up_prompts: list[tuple[str, str]] = [],
    test_incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    n_calibration_examples: int | None = None,
    platt_scale_threshold: float = 1.0,
    platt_shift_only: bool = False,
    use_on_policy_negatives_for_calibration: bool = False,
    vllm_config=None,
    vllm_model_name: str | None = None,
    vllm_model_revision: str | None = None,
    vllm_apply_chat_template: bool = True,
    vllm_lora_path: str = "",
) -> Mapping[str, Any]:
    """Evaluates the performance of a detector on a test dataset.

    Args:
        detector: The trained detector (dict of Probes by layer).
        detector_type: Type of detector (e.g., 'linear-probe', 'gdm-probe').
        model: The base model.
        tokenizer: The tokenizer for the model.
        test_dataset: Tuple of (positive, negative) HuggingFace datasets for testing.
        train_dataset: Optional tuple for training (for threshold selection).
        obfuscate_over: Token selection strategy.
        eval_sequence_aggregator: Aggregator for evaluation. If None, auto-selects based on
                        detector_type (multimax for gdm-probe, mean otherwise).
        max_completion_length: Maximum sequence length for the model.
        max_sequence_length: Maximum sequence length for the model.
        model_batch_size: Batch size for generating activations using the `model`
            for test set.
        detector_batch_size: Batch size for computing detector scores for test set.
        false_positive_rate: The desired false positive rate for threshold selection.
        device: The device to run computations on.
        train_cache_activations_save_path: Optional path to save/load cached training
            activations.
        test_cache_activations_save_path: Optional path to save/load cached testing
            activations.
        train_activation_dataset: Optional pre-loaded training activation dataset.
        test_activation_dataset: Optional pre-loaded test activation dataset.
        best_threshold_metric: The metric to use for selecting the best threshold.
        black_box_detector: Whether the detector is a black-box detector.
        black_box_features: Number of black-box features to use. -1 means all.
        train_completion_columns: The column names for the completion in the train dataset.
        test_completion_columns: The column names for the completion in the test dataset.
        n_calibration_examples: Number of examples per class from test set to use for Platt scaling
            calibration (total = 2 * n_calibration_examples).
            Set None to disable platt scaling.
        use_on_policy_negatives_for_calibration: If True, generate on-policy model completions
            to use as negative examples for Platt scaling calibration instead of using
            pre-existing test set negatives. Requires n_calibration_examples and vllm_config.
        vllm_config: VLLMConfig for connecting to vLLM server. Required if
            use_on_policy_negatives_for_calibration=True.
        vllm_model_name: Model name for vLLM server. Required if
            use_on_policy_negatives_for_calibration=True.
        vllm_model_revision: Model revision for vLLM server (optional).
        vllm_apply_chat_template: Whether to apply chat template to prompts for vLLM generation.
        vllm_lora_path: Path to LoRA adapter for vLLM generation. Empty string means base model.
            Used for on-policy generation to sample from the trained policy.

    Returns:
        A dictionary containing evaluation results, which will include:
        - 'best_thresholds': Thresholds computed from the training set (if
          train dataset or activations are provided).
        - 'train_scores', 'train_labels', 'train_activations': Results on the
          train set (if train dataset or activations are provided).
        - 'test_scores', 'test_labels', 'test_metrics', 'test_figs',
          : Results on the test set.
    """
    if eval_sequence_aggregator is None:
        if detector_type in ["gdm-probe"]:
            eval_sequence_aggregator = multimax_aggregator()
        else:
            eval_sequence_aggregator = mean_aggregator()

    assert isinstance(test_dataset, tuple) and len(test_dataset) == 2, (
        "Test dataset must be a tuple of (positive, negative) examples"
    )
    if train_dataset is not None:
        assert isinstance(train_dataset, tuple) and len(train_dataset) == 2, (
            "Train dataset must be a tuple of (positive, negative) examples"
        )
    if detector_type == "black-box":
        detector_type = "linear-probe"
        black_box_detector = True

    best_thresholds = None
    ret = {}

    common_kwargs: dict[str, Any] = dict(
        model=model,
        tokenizer=tokenizer,
        obfuscate_over=obfuscate_over,
        preserve_seq_len=not black_box_detector,
        max_completion_length=max_completion_length,
        max_sequence_length=max_sequence_length,
        append_eos_to_targets=append_eos_to_targets,
        model_batch_size=model_batch_size,
        black_box_detector=black_box_detector,
        black_box_features=black_box_features,
        accelerator=accelerator,
        verbose=True,
    )

    # Prepare datasets
    if train_dataset is not None:
        train_feature_dataset, d_model = prepare_dataset(
            feature_dataset=train_feature_dataset,
            dataset=train_dataset,
            cache_activations_save_path=train_cache_activations_save_path,
            balance_dataset=True,
            completion_columns=train_completion_columns,
            congruent_follow_up_prompts=train_congruent_follow_up_prompts,
            incongruent_follow_up_prompts=train_incongruent_follow_up_prompts,
            **common_kwargs,
        )

    test_feature_dataset, d_model = prepare_dataset(
        feature_dataset=test_feature_dataset,
        dataset=test_dataset,
        cache_activations_save_path=test_cache_activations_save_path,
        balance_dataset=False,
        completion_columns=test_completion_columns,
        congruent_follow_up_prompts=test_congruent_follow_up_prompts,
        incongruent_follow_up_prompts=test_incongruent_follow_up_prompts,
        **common_kwargs,
    )
    # Get scores and metrics on the training set to find the best threshold
    train_scores, train_logits, train_labels, train_example_types = None, None, None, None
    if train_feature_dataset is not None:
        train_logits, train_labels, train_example_types = get_detector_scores(
            detector=detector,
            feature_dataset=train_feature_dataset,
            eval_sequence_aggregator=eval_sequence_aggregator,
            detector_batch_size=detector_batch_size,
            device=device,
            accelerator=accelerator,
            return_logits=True,
        )

    # Get scores on test set
    test_logits, test_labels, test_example_types = get_detector_scores(
        detector=detector,
        feature_dataset=test_feature_dataset,
        eval_sequence_aggregator=eval_sequence_aggregator,
        detector_batch_size=detector_batch_size,
        device=device,
        accelerator=accelerator,
        return_logits=True,
    )

    # Get on-policy logits if requested for calibration
    on_policy_logits = None
    if n_calibration_examples and use_on_policy_negatives_for_calibration:
        if vllm_config is None or vllm_model_name is None:
            raise ValueError(
                "vllm_config and vllm_model_name required when use_on_policy_negatives_for_calibration=True"
            )

        on_policy_completions = generate_on_policy_completions(
            tokenizer=tokenizer,
            prompts=list(test_dataset[1]["prompt"]),
            n_samples=n_calibration_examples,
            vllm_config=vllm_config,
            model_name=vllm_model_name,
            model_revision=vllm_model_revision,
            lora_path=vllm_lora_path,
            accelerator=accelerator,
            apply_chat_template=vllm_apply_chat_template,
        )

        ret["on_policy_calibration_generations"] = {
            "prompt": on_policy_completions["prompt"],
            "completion": on_policy_completions["completion"],
        }

        from datasets import Dataset

        empty_dataset = Dataset.from_dict({"prompt": [], "completion": []})
        on_policy_feature_dataset, _ = prepare_dataset(
            feature_dataset=None,
            dataset=(empty_dataset, on_policy_completions),
            cache_activations_save_path=os.path.join(
                test_cache_activations_save_path or "/scratch/oa_activations", "on_policy_calibration"
            ),
            balance_dataset=False,
            completion_columns=("completion", "completion"),
            congruent_follow_up_prompts=test_congruent_follow_up_prompts,
            incongruent_follow_up_prompts=test_incongruent_follow_up_prompts,
            **common_kwargs,
        )

        on_policy_logits, _, _ = get_detector_scores(
            detector=detector,
            feature_dataset=on_policy_feature_dataset,
            eval_sequence_aggregator=eval_sequence_aggregator,
            detector_batch_size=detector_batch_size,
            device=device,
            accelerator=accelerator,
            return_logits=True,
        )

    # Apply Platt scaling calibration or standard sigmoid
    if n_calibration_examples:
        rng = np.random.default_rng(seed=42)

        calib_logits, calib_labels, sampled_indices = _get_calibration_data(
            test_logits=test_logits,
            test_labels=test_labels,
            n_calibration_examples=n_calibration_examples,
            rng=rng,
            on_policy_logits=on_policy_logits,
        )

        test_scores, train_scores, platt_params = _apply_platt_scaling(
            logits=test_logits,
            calib_logits=calib_logits,
            calib_labels=calib_labels,
            detector=detector,
            train_logits=train_logits if train_feature_dataset is not None else None,
            scale_threshold=platt_scale_threshold,
            shift_only=platt_shift_only,
        )

        ret["platt_params"] = platt_params
        if use_on_policy_negatives_for_calibration:
            ret["calibration_positive_indices"] = sampled_indices
        else:
            ret["calibration_indices"] = sampled_indices
    else:
        # No calibration - just apply standard sigmoid
        test_scores = {layer_key: expit(logits) for layer_key, logits in test_logits.items()}
        if train_feature_dataset is not None:
            train_scores = {layer_key: expit(logits) for layer_key, logits in train_logits.items()}  # type: ignore

    # Compute train metrics
    if train_feature_dataset is not None:
        train_metrics, _ = get_detector_metrics(
            train_scores,  # type: ignore
            train_labels,  # type: ignore
            false_positive_rate,
            thresholds=None,
            disable_plots=True,
            example_types=train_example_types,
            logits=train_logits,
        )
        best_thresholds = {layer: train_metrics[layer]["metrics_at_threshold"]["threshold"] for layer in train_scores}  # type: ignore
        ret["best_train_metrics"] = dict(train_metrics)
        ret["best_thresholds"] = best_thresholds
        ret["train_scores"] = train_scores
        ret["train_labels"] = train_labels

    # Compute test metrics
    test_metrics, test_figs = get_detector_metrics(
        test_scores,
        test_labels,
        false_positive_rate,
        best_thresholds,
        disable_plots=True,
        example_types=test_example_types,
        logits=test_logits,
    )

    # Add Platt params to test_metrics
    if n_calibration_examples:
        platt_params = ret.get("platt_params", {})
        for layer_key, (A, B) in platt_params.items():
            if layer_key in test_metrics:
                test_metrics[layer_key]["platt_A"] = A  # type: ignore
                test_metrics[layer_key]["platt_B"] = B  # type: ignore

    # Add input_scale to test_metrics
    if isinstance(detector, dict):
        for layer_key, probe in detector.items():
            str_layer_key = str(layer_key)
            if str_layer_key in test_metrics and hasattr(probe, "input_scale"):
                test_metrics[str_layer_key]["input_scale"] = probe.input_scale.item()  # type: ignore

    ret["test_scores"] = test_scores
    ret["test_labels"] = test_labels
    ret["test_metrics"] = test_metrics
    ret["test_figs"] = test_figs
    return ret


def train_and_eval_detector(
    model,
    tokenizer,
    layers: list[int],
    train_dataset: tuple[HFDataset, HFDataset],
    test_dataset: tuple[HFDataset, HFDataset],
    train_cfg: BaseTrainConfig,
    detector_arch_config: DetectorArchConfig = DetectorArchConfig(),
    obfuscate_over: str = "generation",
    train_sequence_aggregator: SequenceAggregator | None = None,
    eval_sequence_aggregator: SequenceAggregator | None = None,
    cache_activations_save_path: str | None = None,
    max_completion_length: int | None = None,
    max_sequence_length: int | None = None,
    model_batch_size: int = 32,
    false_positive_rate: float = 0.01,
    train_feature_dataset: DictTensorDataset | None = None,
    test_feature_dataset: DictTensorDataset | None = None,
    best_threshold_metric: str = "accuracy",
    append_eos_to_targets: bool = True,
    black_box_detector: bool = False,
    black_box_features: int = -1,
    accelerator=None,
    use_sklearn: bool = False,
    train_completion_columns: tuple[str, str] = ("completion", "completion"),
    test_completion_columns: tuple[str, str] = ("completion", "completion"),
    train_congruent_follow_up_prompts: list[tuple[str, str]] = [],
    train_incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    test_congruent_follow_up_prompts: list[tuple[str, str]] = [],
    test_incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    n_calibration_examples: int | None = None,
    platt_shift_only: bool = False,
    use_on_policy_negatives_for_calibration: bool = True,
    vllm_config=None,
    vllm_model_name: str | None = None,
    vllm_model_revision: str | None = None,
    vllm_apply_chat_template: bool = True,
    vllm_lora_path: str = "",
    # On-policy negatives for training
    use_on_policy_negatives_for_training: bool = False,
    augment_training_negatives: bool = False,
    on_policy_n_samples: int | None = None,
) -> tuple[
    Mapping[str, Mapping[str, Any]],
    DETECTOR_TYPE,
    dict[str, Any] | None,
    Mapping[str, Mapping[str, Any]],
]:
    """Trains a detector and immediately evaluates it on test data.

    This function combines the training and evaluation steps into a single call,
    making it convenient for assessing a detector's performance right after
    training. It passes through common arguments to `train_detector` and
    `eval_detector`.

    Args:
        model: The base model (e.g., an instruction-tuned LLM).
        tokenizer: The tokenizer for the model.
        layers: list of layer indices to train and evaluate detectors on.
        train_dataset: A tuple of (positive, negative) examples of HuggingFace datasets
            for training.
        test_dataset: A tuple of (positive, negative) examples of HuggingFace datasets
            for testing.
        detector_type: The type of detector to train (e.g., 'linear-probe',
            'mahalanobis').
        train_cfg: BaseTrainConfig containing training hyperparameters including
            learning_rate, scheduler, warmup_steps, num_epochs, max_steps,
            grad_accum_steps, weight_decay, batch_size (for detector), and device.
        detector_arch_kwargs: Dictionary of architecture-specific
            hyperparameters for the detector. See `train_detector` for details
            on required kwargs for each detector type.
        obfuscate_over: The tokens to obfuscate over.
            Can be "generation", "full-prompt", "last-token-prompt", or "full".
        train_sequence_aggregator: The aggregation method to use for aggregating over a sequence of scores
            during training.
        eval_sequence_aggregator: The aggregation method to use for aggregating over a sequence of scores
            during evaluation.
        cache_activations_save_path: Path to save/load cached activations.
        max_completion_length: Maximum sequence length for the model.
        max_sequence_length: Maximum sequence length for the model.
        model_batch_size: Batch size for generating activations from the model.
        false_positive_rate: The target FPR for calculating TPR during evaluation.
        train_feature_dataset: Optional pre-loaded training activation dataset.
        test_feature_dataset: Optional pre-loaded test activation dataset.
        accelerator: The accelerator to use for training and evaluation.
        use_sklearn: Whether to use sklearn for training the probe (only works for linear-probe).
        train_completion_columns: The column names for the completion in the train dataset.
        test_completion_columns: The column names for the completion in the test dataset.
        n_calibration_examples: Number of examples per class from test set to use for Platt scaling
            calibration (total = 2 * n_calibration_examples). Set None to disable platt scaling.

    Returns:
        A tuple containing:
        - `test_metrics`: A dictionary of evaluation metrics on the test set,
          keyed by layer.
        - `detector`: The trained detector (either an AnomalyDetector or a
          dict of Probes).
        - `train_dynamics`: Step-wise training curves dict with keys like "layer_X_loss",
          "loss", "lr" and values as lists of floats (one per training step).
        - `train_eval_metrics`: Evaluation metrics computed on the training set at the
          best threshold, keyed by layer. Contains scores, metrics_at_threshold, etc.
    """
    if (
        len(train_congruent_follow_up_prompts) + len(train_incongruent_follow_up_prompts) > 0
        and "last-token" not in obfuscate_over
    ):
        raise ValueError("Follow-up prompts should only be used with last-token-generation based probe.")

    # Extract training parameters from config
    detector_batch_size = train_cfg.batch_size // (accelerator.num_processes if accelerator is not None else 1)

    if detector_arch_config.detector_type == "black-box":
        detector_arch_config.detector_type = "linear-probe"
        black_box_detector = True

    train_cache_activations_save_path = (
        os.path.join(cache_activations_save_path, "train") if cache_activations_save_path is not None else None
    )
    test_cache_activations_save_path = (
        os.path.join(cache_activations_save_path, "test") if cache_activations_save_path is not None else None
    )
    common_kwargs: dict[str, Any] = dict(  # type: ignore
        model=model,
        tokenizer=tokenizer,
        obfuscate_over=obfuscate_over,
        max_completion_length=max_completion_length,
        max_sequence_length=max_sequence_length,
        append_eos_to_targets=append_eos_to_targets,
        model_batch_size=model_batch_size,
        black_box_detector=black_box_detector,
        black_box_features=black_box_features,
        accelerator=accelerator,
        verbose=True,
    )

    # Generate on-policy negatives from TEST prompts for training
    if use_on_policy_negatives_for_training:
        if vllm_config is None or vllm_model_name is None:
            raise ValueError(
                "vllm_config and vllm_model_name are required when use_on_policy_negatives_for_training=True"
            )

        test_prompts = list(test_dataset[1]["prompt"])
        n_samples = on_policy_n_samples or len(train_dataset[1])

        on_policy_neg = generate_on_policy_completions(
            tokenizer=tokenizer,
            prompts=test_prompts,
            n_samples=n_samples,
            vllm_config=vllm_config,
            model_name=vllm_model_name,
            model_revision=vllm_model_revision,
            lora_path=vllm_lora_path,
            accelerator=accelerator,
            apply_chat_template=vllm_apply_chat_template,
            completion_column=train_completion_columns[1],
        )

        if augment_training_negatives:
            combined_neg = concatenate_datasets([train_dataset[1], on_policy_neg])
        else:
            combined_neg = on_policy_neg
        train_dataset = (train_dataset[0], combined_neg)
        if accelerator is None or accelerator.is_main_process:
            print(f"Replaced training negatives with {len(on_policy_neg)} on-policy examples from test prompts")

    if train_dataset is not None:
        log_gpu_memory("Before prepare_dataset for training data")
        train_feature_dataset, _ = prepare_dataset(
            feature_dataset=train_feature_dataset,
            dataset=train_dataset,
            cache_activations_save_path=train_cache_activations_save_path,
            preserve_seq_len=True,
            balance_dataset=True,
            completion_columns=train_completion_columns,
            congruent_follow_up_prompts=train_congruent_follow_up_prompts,
            incongruent_follow_up_prompts=train_incongruent_follow_up_prompts,
            **common_kwargs,
        )
        log_gpu_memory("After prepare_dataset for training data")
    log_gpu_memory("Before prepare_dataset for test data")
    test_feature_dataset, _ = prepare_dataset(
        feature_dataset=test_feature_dataset,
        dataset=test_dataset,
        cache_activations_save_path=test_cache_activations_save_path,
        preserve_seq_len=not black_box_detector,
        balance_dataset=False,
        completion_columns=test_completion_columns,
        congruent_follow_up_prompts=test_congruent_follow_up_prompts,
        incongruent_follow_up_prompts=test_incongruent_follow_up_prompts,
        **common_kwargs,
    )
    log_gpu_memory("After prepare_dataset for test data")

    detector, _, train_dynamics = train_detector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        train_cfg=train_cfg,
        train_feature_dataset=train_feature_dataset,
        layers=layers,
        detector_arch_config=detector_arch_config,
        cache_activations_save_path=train_cache_activations_save_path,
        obfuscate_over=obfuscate_over,
        max_completion_length=max_completion_length,
        max_sequence_length=max_sequence_length,
        model_batch_size=model_batch_size,
        train_sequence_aggregator=train_sequence_aggregator,
        black_box_detector=black_box_detector,
        black_box_features=black_box_features,
        accelerator=accelerator,
        use_sklearn=use_sklearn,
        train_completion_columns=train_completion_columns,
        congruent_follow_up_prompts=train_congruent_follow_up_prompts,
        incongruent_follow_up_prompts=train_incongruent_follow_up_prompts,
    )

    eval_dict = eval_detector(
        detector=detector,
        detector_type=detector_arch_config.detector_type,
        model=model,
        tokenizer=tokenizer,
        obfuscate_over=obfuscate_over,
        eval_sequence_aggregator=eval_sequence_aggregator,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        max_completion_length=max_completion_length,
        model_batch_size=model_batch_size,
        detector_batch_size=detector_batch_size,
        max_sequence_length=max_sequence_length,
        false_positive_rate=false_positive_rate,
        train_feature_dataset=train_feature_dataset,
        test_feature_dataset=test_feature_dataset,
        train_cache_activations_save_path=train_cache_activations_save_path,
        test_cache_activations_save_path=test_cache_activations_save_path,
        device=train_cfg.device,
        best_threshold_metric=best_threshold_metric,
        black_box_detector=black_box_detector,
        black_box_features=black_box_features,
        accelerator=accelerator,
        train_completion_columns=train_completion_columns,
        test_completion_columns=test_completion_columns,
        train_congruent_follow_up_prompts=train_congruent_follow_up_prompts,
        train_incongruent_follow_up_prompts=train_incongruent_follow_up_prompts,
        test_congruent_follow_up_prompts=test_congruent_follow_up_prompts,
        test_incongruent_follow_up_prompts=test_incongruent_follow_up_prompts,
        n_calibration_examples=n_calibration_examples,
        platt_shift_only=platt_shift_only,
        use_on_policy_negatives_for_calibration=use_on_policy_negatives_for_calibration,
        vllm_config=vllm_config,
        vllm_model_name=vllm_model_name,
        vllm_model_revision=vllm_model_revision,
        vllm_apply_chat_template=vllm_apply_chat_template,
        vllm_lora_path=vllm_lora_path,
    )
    detector = to_cpu_and_cleanup_detector(detector, accelerator)  # type: ignore
    # train_dynamics: step-wise training curves (losses, lr per step)
    # train_eval_metrics: evaluation metrics computed on the training set at best threshold
    return (
        eval_dict["test_metrics"],
        detector,  # type: ignore
        train_dynamics,
        eval_dict["best_train_metrics"],
    )


def train_and_eval_multiple_detectors(
    model,
    tokenizer,
    layers: list[int],
    train_dataset: tuple[HFDataset, HFDataset],
    test_dataset: tuple[HFDataset, HFDataset],
    train_cfg: BaseTrainConfig,
    detector_arch_configs: list[DetectorArchConfig] = [DetectorArchConfig()],
    obfuscate_over: list[str] = ["generation", "last_token_prompt"],
    train_sequence_aggregator: SequenceAggregator | None = None,
    eval_sequence_aggregator: SequenceAggregator | None = None,
    cache_activations_save_path: str | None = None,
    max_completion_length: int | None = None,
    max_sequence_length: int | None = None,
    model_batch_size: int = 32,
    false_positive_rate: float = 0.01,
    train_feature_dataset: DictTensorDataset | None = None,
    test_feature_dataset: DictTensorDataset | None = None,
    best_threshold_metric: str = "accuracy",
    black_box_detector: bool = False,
    black_box_features: int = -1,
    accelerator=None,
    use_sklearn: bool = False,
    train_completion_columns: tuple[str, str] = ("completion", "completion"),
    test_completion_columns: tuple[str, str] = ("completion", "completion"),
    train_congruent_follow_up_prompts: list[tuple[str, str]] = [],
    train_incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    test_congruent_follow_up_prompts: list[tuple[str, str]] = [],
    test_incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    n_calibration_examples: int | None = None,
    platt_shift_only: bool = False,
    use_on_policy_negatives_for_calibration: bool = True,
    vllm_config=None,
    vllm_model_name: str | None = None,
    vllm_model_revision: str | None = None,
    vllm_apply_chat_template: bool = True,
    vllm_lora_path: str = "",
    # On-policy negatives for training
    use_on_policy_negatives_for_training: bool = False,
    augment_training_negatives: bool = False,
    on_policy_n_samples: int | None = None,
) -> Mapping[
    str,
    tuple[
        DETECTOR_TYPE,
        tuple[
            Mapping[str, Mapping[str, Any]],
            dict[str, Any] | None,
            Mapping[str, Mapping[str, Any]],
        ],
    ],
]:
    """Trains and evaluates multiple detectors across different types and obfuscation
    strategies.

    This function iterates through a list of detector types and obfuscation
    strategies, training and evaluating each combination. It reuses computed
    activations to avoid redundant calculations where possible.

    Args:
        model: The base model (e.g., an instruction-tuned LLM).
        tokenizer: The tokenizer for the model.
        layers: list of layer indices to train and evaluate detectors on.
        train_dataset: A tuple of (positive, negative) examples of HuggingFace datasets
            for training.
        test_dataset: A tuple of (positive, negative) examples of HuggingFace datasets
            for testing.
        detector_type: A list of detector types to train (e.g., 'linear-probe',
            'mahalanobis').
        train_cfg: BaseTrainConfig containing training hyperparameters including
            learning_rate, scheduler, warmup_steps, num_epochs, max_steps,
            grad_accum_steps, weight_decay, batch_size (for detector), and device.
        detector_arch_kwargs: An optional list of dictionaries, where each dictionary
            contains architecture-specific hyperparameters corresponding to the
            `detector_type` list. If None, it uses the default hyperparameters
            for each detector type.
        obfuscate_over: A list of tokens to obfuscate over for each detector.
            Can include "generation", "full-prompt", "last-token-prompt", or "full".
        train_sequence_aggregator: Aggregator for sequence-level training. If None and
                         sequence_level_training=True, auto-selects based on detector_type.
        eval_sequence_aggregator: Aggregator for evaluation. If None, auto-selects based on
                        detector_type (multimax for gdm-probe, mean otherwise).
        cache_activations_save_path: Path to save/load cached activations.
        max_completion_length: Maximum sequence length for the model.
        max_sequence_length: Maximum sequence length for the model.
        model_batch_size: Batch size for generating activations from the model.
        false_positive_rate: The target FPR for calculating TPR during evaluation.
        train_feature_dataset: Optional pre-loaded training activation dataset. If None,
            activations will be loaded from the dataset.
        test_feature_dataset: Optional pre-loaded test activation dataset. If None,
            activations will be loaded from the dataset.
        test_completion_columns: The column names for the completion in the test dataset.
        n_calibration_examples: Number of examples per class from test set to use for Platt scaling
            calibration (total = 2 * n_calibration_examples). Set None to disable platt scaling.

    Returns:
        A dictionary where keys are strings combining detector type and obfuscation
        strategy (e.g., "linear-probe_generation"), and values are tuples of
        (detector, (metrics, train_dynamics, train_eval_metrics)) where:
        - metrics: test set evaluation metrics
        - train_dynamics: step-wise training curves (losses, lr per step)
        - train_eval_metrics: evaluation metrics on training set at best threshold
    """
    assert isinstance(detector_arch_configs, list), (
        f"Detector architecture configs must be provided as a list, got {type(detector_arch_configs)}"
    )

    results_dict = {}
    black_box_key = None
    for d_idx, d_arch_config in enumerate(detector_arch_configs):
        for obfuscate_over_ in obfuscate_over:
            key = f"{d_arch_config.detector_type.replace('_', '-')}_{obfuscate_over_.replace('_', '-')}"
            if "black-box" in d_arch_config.detector_type:
                if black_box_key is not None:
                    results_dict[key] = results_dict[black_box_key]
                    continue
                black_box_key = key
            metrics, detector, train_dynamics, train_eval_metrics = train_and_eval_detector(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_cfg=train_cfg,
                detector_arch_config=d_arch_config,
                obfuscate_over=obfuscate_over_,
                train_sequence_aggregator=train_sequence_aggregator,
                eval_sequence_aggregator=eval_sequence_aggregator,
                cache_activations_save_path=cache_activations_save_path,
                max_completion_length=max_completion_length,
                max_sequence_length=max_sequence_length,
                model_batch_size=model_batch_size,
                false_positive_rate=false_positive_rate,
                train_feature_dataset=train_feature_dataset,
                test_feature_dataset=test_feature_dataset,
                best_threshold_metric=best_threshold_metric,
                black_box_detector=black_box_detector,
                black_box_features=black_box_features,
                accelerator=accelerator,
                use_sklearn=use_sklearn,
                train_completion_columns=train_completion_columns,
                test_completion_columns=test_completion_columns,
                train_congruent_follow_up_prompts=train_congruent_follow_up_prompts,
                train_incongruent_follow_up_prompts=train_incongruent_follow_up_prompts,
                test_congruent_follow_up_prompts=test_congruent_follow_up_prompts,
                test_incongruent_follow_up_prompts=test_incongruent_follow_up_prompts,
                n_calibration_examples=n_calibration_examples,
                platt_shift_only=platt_shift_only,
                use_on_policy_negatives_for_calibration=use_on_policy_negatives_for_calibration,
                vllm_config=vllm_config,
                vllm_model_name=vllm_model_name,
                vllm_model_revision=vllm_model_revision,
                vllm_apply_chat_template=vllm_apply_chat_template,
                vllm_lora_path=vllm_lora_path,
                # On-policy negatives for training
                use_on_policy_negatives_for_training=use_on_policy_negatives_for_training,
                augment_training_negatives=augment_training_negatives,
                on_policy_n_samples=on_policy_n_samples,
            )
            results_dict[key] = (detector, (metrics, train_dynamics, train_eval_metrics))
    return results_dict  # type: ignore


def eval_multiple_detectors(
    detectors,
    detector_types,
    test_dataset,
    obfuscate_over,
    accelerator=None,
    device: str = "cuda",
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple detectors on a test dataset.

    This function moves each detector to GPU before evaluation, evaluates it,
    and then moves it back to CPU to free GPU memory for subsequent detectors.
    This is especially important for multi-GPU FSDP setups.

    In distributed mode, detectors may only be available on rank 0 (e.g., when loaded
    from disk). This function broadcasts detectors to all ranks before evaluation.

    Args:
        detectors: List of trained detectors. Can be on CPU (will be moved to GPU)
            or already on GPU/FSDP-wrapped (will be used as-is, then moved to CPU after).
            In distributed mode, may be None on non-main ranks if loaded from disk.
        detector_type: List of detector types corresponding to each detector.
        test_dataset: Test dataset tuple (positive, negative).
        obfuscate_over: Obfuscation strategy or list of strategies.
        accelerator: Optional accelerator for distributed training (FSDP).
        device: Target device for evaluation when accelerator is not provided.
        **kwargs: Additional arguments passed to eval_detector.

    Returns:
        Dictionary mapping detector keys to their test metrics.
    """
    eval_results_and_detectors = {}
    obf_list = obfuscate_over if isinstance(obfuscate_over, list) else [obfuscate_over]

    for idx, (detector, d_type) in enumerate(zip(detectors, detector_types)):
        detector_gpu = to_gpu(detector, accelerator, device)
        for obf_idx, obf_over in enumerate(obf_list):
            results_dict = eval_detector(
                detector=detector_gpu,  # type: ignore
                detector_type=d_type,
                test_dataset=test_dataset,
                obfuscate_over=obf_over,
                accelerator=accelerator,
                device=device,
                **kwargs,
            )
            key = f"{d_type.replace('_', '-')}_{obf_over.replace('_', '-')}"
            eval_results_and_detectors[key] = results_dict["test_metrics"]

        # Clean up detector: remove from accelerator, move to CPU, free GPU memory
        to_cpu_and_cleanup_detector(detector_gpu, accelerator)
        del detector_gpu

    return eval_results_and_detectors


def get_layers_below_loss_threshold(train_losses: dict[str, list[float]], loss_threshold: float) -> list[int]:
    """Get the layers whose final training loss is below the threshold.

    This is useful for filtering out probes that did not converge properly during training.

    Args:
        train_losses: Dictionary of training losses with keys like "layer_{layer}_loss"
            and values as lists of loss values over training.
        loss_threshold: The maximum allowed final training loss. Layers with final loss
            above this threshold will be excluded.

    Returns:
        List of layer indices (as ints) whose final training loss is below the threshold.
    """
    layers_below_threshold = []
    for key, losses in train_losses.items():
        if not key.startswith("layer_") or not key.endswith("_loss"):
            continue
        # Extract layer number from key like "layer_5_loss"
        layer = int(key.replace("layer_", "").replace("_loss", ""))
        final_loss = losses[-1] if losses else float("inf")
        if final_loss < loss_threshold:
            layers_below_threshold.append(layer)
    if len(layers_below_threshold) == 0:
        raise ValueError("Probes in no layer match the `loss_threshold`, `layers_below_threshold` is empty")
    return sorted(layers_below_threshold)
