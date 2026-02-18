# type: ignore
# ruff: noqa: I001
from contextlib import contextmanager
from functools import partial, wraps
import gc
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Type, TypeVar
import warnings

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import concatenate_datasets, Dataset as HFDataset
from sklearn.linear_model import LogisticRegression
from tensordict import MemoryMappedTensor
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn import Module
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import expit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from obfuscation_atlas.config import BaseTrainConfig
from obfuscation_atlas.detectors.blackbox import get_blackbox_feature_dataset
from obfuscation_atlas.detectors.dataset import (
    DETECTOR_TYPE,
    ActivationDataset,
    DictTensorDataset,
)
from obfuscation_atlas.detectors.probe_archs import (
    AttentionProbe,
    LinearProbe,
    NonlinearProbe,
    Probe,
    TransformerProbe,
    GDMProbe,
    SequenceAggregator,
    mean_aggregator,
    multimax_aggregator,
    rolling_attention_aggregator,
    compute_loss,
)
from obfuscation_atlas.utils.activations import get_hidden_size, get_num_hidden_layers
from obfuscation_atlas.utils.data_processing import (
    process_data,
)
from obfuscation_atlas.utils.example_types import ExampleType
from obfuscation_atlas.utils.gpu_utils import log_gpu_memory
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper
from obfuscation_atlas.utils.lora_utils import initialize_lora_adapter
from obfuscation_atlas.utils.masking import (
    compute_mask,
    get_valid_token_mask,
    trim_sequences,
)
from obfuscation_atlas.utils.serialization import (
    convert_seconds_to_time_str,
)
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from dataclasses import dataclass

warnings.filterwarnings("ignore", message="Precision is ill-defined")


@dataclass
class DetectorArchConfig:
    detector_type: str = "linear-probe"
    d_mlp: int = 128
    d_proj: int = 100
    nhead: int = 1
    use_checkpoint: bool = True
    nlayer: int = 1
    dropout: float = 0.0
    activation: str = "relu"
    norm_first: bool = True
    normalize_input: str = "unit_norm"
    train_sequence_aggregator: str | None = "mean"
    eval_sequence_aggregator: str | None = "mean"
    sliding_window: int | None = None


def pretty_detector_name(detector_type: str) -> str:
    return detector_type.replace("-", " ").title()


def sequence_preserving_collate_fn(batch, pad_labels=False, layer_keys=None):
    """Custom collate function to handle padding of variable-length sequences."""
    activations_batch = [item[0] for item in batch]
    labels_batch = [torch.as_tensor(item[1]) for item in batch]
    # Handle optional example_types (third element in batch items)
    if len(batch[0]) > 2:
        example_types_batch = [torch.as_tensor(item[2]) for item in batch]
    else:
        example_types_batch = None

    # Get sequence lengths and max length
    seq_lens = [next(iter(act.values())).shape[-2] for act in activations_batch]
    max_seq_len = max(seq_lens)

    # Get layer keys from first sample
    if layer_keys is None:
        layer_keys = list(activations_batch[0].keys())

    # Pre-allocate output for all layers at once
    # Assuming activations shape is (hidden_dim, seq_len, feature_dim)
    first_act = activations_batch[0][layer_keys[0]]
    batch_size = len(batch)
    shape = (batch_size, *first_act.shape[:-2], max_seq_len, first_act.shape[-1])

    # Create padded tensors for all layers
    collated_activations = {key: torch.zeros(shape, dtype=first_act.dtype) for key in layer_keys}
    for key in layer_keys:
        for i, act_dict in enumerate(activations_batch):
            collated_activations[key][i, ..., : seq_lens[i], :] = act_dict[key][..., : seq_lens[i], :]

    if pad_labels:
        labels_padded = torch.full((batch_size, max_seq_len), -100, dtype=labels_batch[0].dtype)
        for i, (label, seq_len) in enumerate(zip(labels_batch, seq_lens)):
            labels_padded[i, :seq_len] = label.repeat(seq_len)
        labels_tensor = labels_padded
    else:
        labels_tensor = torch.stack(labels_batch)

    if example_types_batch is not None:
        example_types_tensor = torch.stack(example_types_batch)
        return collated_activations, labels_tensor, example_types_tensor
    return collated_activations, labels_tensor, None


def default_collate_fn(batch):
    activations = {k: torch.stack([batch[i][0][k] for i in range(len(batch))]) for k in batch[0][0].keys()}
    labels = torch.tensor([batch[i][1] for i in range(len(batch))])
    # Handle optional example_types (third element in batch items)
    if len(batch[0]) > 2:
        example_types = torch.tensor([batch[i][2] for i in range(len(batch))])
        return activations, labels, example_types
    return activations, labels


@contextmanager
def probe_fsdp_policy(accelerator):
    """Context manager that temporarily sets FSDP auto-wrap policy for probes.

    When using FSDP with TRANSFORMER_BASED_WRAP policy, internal modules like
    nn.TransformerEncoderLayer get wrapped separately, which breaks nn.MultiheadAttention.
    This context manager sets a custom policy that only wraps at the Probe level.
    """
    if accelerator is None or getattr(accelerator.state, "fsdp_plugin", None) is None:
        yield
        return

    original_policy = accelerator.state.fsdp_plugin.auto_wrap_policy
    probe_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda m: isinstance(m, Probe))
    accelerator.state.fsdp_plugin.auto_wrap_policy = probe_policy
    try:
        yield
    finally:
        accelerator.state.fsdp_plugin.auto_wrap_policy = original_policy


def compute_input_scales(dataset: DictTensorDataset, layers: list[int], batch_size: int = 256) -> dict[int, float]:
    """Compute sqrt(mean(||x||²)) for each layer from dataset.

    This scale factor normalizes inputs so that E[||x_normalized||²] = 1.

    Args:
        dataset: Dataset containing activations for each layer.
        layers: List of layer indices to compute scales for.
        batch_size: Batch size for the dataloader.

    Returns:
        Dictionary mapping layer index to scale factor.
    """
    layer_keys = list(map(str, layers))
    # Use sequence-preserving collate only when activations have a sequence dimension (ndim >= 2);
    # otherwise default collate handles fixed-size (pre-aggregated) activations.
    first_act = dataset[0][0][layer_keys[0]]
    if first_act.ndim >= 2:
        collate_fn = partial(sequence_preserving_collate_fn, layer_keys=layer_keys, pad_labels=False)
    else:
        collate_fn = default_collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    sum_sq_norms = {layer: 0.0 for layer in layers}
    counts = {layer: 0 for layer in layers}
    with torch.no_grad():
        for batch in dataloader:
            batch_activations = batch[0]
            for layer in layers:
                x = batch_activations[str(layer)].float()
                if x.ndim >= 3:
                    # Exclude padding tokens (all-zero vectors) from the computation
                    valid_mask = (x != 0).any(dim=-1)  # (batch, seq), True for non-padding
                    sum_sq_norms[layer] += (x[valid_mask] ** 2).sum().item()
                    counts[layer] += valid_mask.sum().item()
                else:
                    sum_sq_norms[layer] += (x**2).sum().item()
                    counts[layer] += x.shape[0]
    scales = {}
    for layer in layers:
        mean_sq_norm = sum_sq_norms[layer] / counts[layer]
        scales[layer] = math.sqrt(mean_sq_norm)
    return scales


def get_scheduler_fn(
    train_cfg: BaseTrainConfig,
    num_training_steps: int | None = None,
) -> Callable:
    """Returns a scheduler factory function for probe training.

    Args:
        train_cfg: BaseTrainConfig containing scheduler type, warmup_steps, and max_steps.
        num_training_steps: Override for total number of training steps. If None, uses train_cfg.max_steps.

    Returns:
        A function that takes an optimizer and returns a scheduler
    """
    scheduler_type = train_cfg.scheduler
    warmup_steps = train_cfg.warmup_steps
    total_steps = num_training_steps if num_training_steps is not None else train_cfg.max_steps

    if scheduler_type == "linear":
        return partial(
            get_linear_schedule_with_warmup,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type == "cosine":
        return partial(
            get_cosine_schedule_with_warmup,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type == "constant":
        return partial(
            get_constant_schedule_with_warmup,
            num_warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Must be 'linear', 'cosine', or 'constant'.")


def initialize_probes_and_optimizers(
    layers,
    create_probe_fn: Callable[[], Probe],
    lr,
    device,
    pretrained_probes: Optional[dict[str, Probe]] = None,
    weight_decay=0.0,
    accelerator=None,
    scheduler_fn: Optional[Callable] = None,
):
    # Initialize probes
    probes, optimizers, schedulers = {}, {}, {}
    for layer in layers:
        probe = pretrained_probes[layer] if pretrained_probes is not None else create_probe_fn()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        if accelerator is not None:
            with probe_fsdp_policy(accelerator):
                probe, optimizer = accelerator.prepare(probe, optimizer)
        else:
            probe = probe.to(device)
        probes[layer] = probe
        optimizers[layer] = optimizer
        if scheduler_fn is not None:
            schedulers[layer] = scheduler_fn(optimizer)
    return probes, optimizers, schedulers


def plot_loss_sparkline(loss_values: list[float], max_width: int = 100, nan_char: str = "?"):
    """
    Generates and prints a single-line "sparkline" graph for the loss values.

    This uses Unicode block characters of varying heights to represent the data
    and includes special handling for NaN (Not a Number) values.

    Args:
        loss_values: A list of floating-point numbers representing the loss.
                     Can contain float('nan') values.
        max_width: The maximum width of the graph in characters. If the number
                   of loss values exceeds this, the data will be sampled.
        nan_char: The character to use for representing NaN values.
    """
    if not loss_values:
        print("No loss values to plot.")
        return

    # --- Data Sampling ---
    # If the number of data points is greater than the allowed width,
    # we need to sample the data to fit.
    if len(loss_values) > max_width:
        sampled_values = []
        step = len(loss_values) / max_width
        for i in range(max_width):
            start = int(i * step)
            end = int((i + 1) * step)
            # Take the average of the values in the slice, ignoring NaNs
            slice_vals = [v for v in loss_values[start:end] if not math.isnan(v)]
            if slice_vals:
                sampled_values.append(sum(slice_vals) / len(slice_vals))
            else:
                # If the whole slice is NaNs, preserve it as a NaN
                sampled_values.append(float("nan"))
        plot_values = sampled_values
    else:
        plot_values = loss_values

    # Unicode characters for plotting, from lowest to highest
    spark_chars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    num_chars = len(spark_chars)

    # Filter out NaNs for calculating min/max and range
    finite_values = [v for v in plot_values if not math.isnan(v)]

    if not finite_values:
        # If all values are NaN or the list is empty
        graph = nan_char * len(plot_values)
    else:
        min_val = min(finite_values)
        max_val = max(finite_values)
        val_range = max_val - min_val

        graph = ""
        for val in plot_values:
            if math.isnan(val):
                graph += nan_char
                continue

            # If all finite values are the same, val_range will be 0.
            if val_range == 0:
                graph += spark_chars[num_chars // 2]
            else:
                # Normalize the value to an index between 0 and num_chars-1
                normalized_val = (val - min_val) / val_range
                # Clamp the index to be safe.
                char_index = int(normalized_val * (num_chars - 1))
                char_index = max(0, min(char_index, num_chars - 1))
                graph += spark_chars[char_index]

    # --- Output ---
    # Calculate summary stats from the original, unsampled list
    original_finite_values = [v for v in loss_values if not math.isnan(v)]
    summary_min = min(original_finite_values) if original_finite_values else float("nan")
    summary_max = max(original_finite_values) if original_finite_values else float("nan")

    print(f"Loss: {graph}")
    print(f"Epochs: {len(loss_values)} | Min Loss: {summary_min:.4f} | Max Loss: {summary_max:.4f}")


def train_combined_probes(
    acts_dataset: DictTensorDataset,
    create_probe_fn: Callable,
    layers: list[int],
    train_cfg: BaseTrainConfig,
    pretrained_probes: dict[str, Probe] | None = None,
    log_epoch_wise_loss: bool = True,
    train_sequence_aggregator: SequenceAggregator | None = None,  # NEW: for sequence-level training
    num_workers: int = 2,
    accelerator=None,
    normalize_input: str = "none",
):
    """
    Train probes on activation dataset.

    Args:
        acts_dataset: Dataset of activations and labels
        create_probe_fn: Factory function to create probe instances
        layers: Layer indices to train probes for
        train_cfg: Training configuration
        pretrained_probes: Optional pretrained probes to continue training
        log_epoch_wise_loss: If True, log loss per epoch; if False, per step
        train_sequence_aggregator: Aggregator over sequences. If None, use token-level training.
                         defaults to mean_aggregator() for simple probes or rolling_attention_aggregator() for GDMProbe
        num_workers: DataLoader workers
        accelerator: Optional Accelerator for distributed training
        normalize_input: Input normalization mode
    """
    # Extract training parameters from config
    lr = train_cfg.learning_rate
    weight_decay = train_cfg.weight_decay
    n_epochs = train_cfg.num_epochs
    n_steps = train_cfg.max_steps
    batch_size = train_cfg.batch_size
    n_grad_accum = train_cfg.grad_accum_steps
    device = train_cfg.device
    clip_grad_norm = train_cfg.clip_grad_norm

    # Calculate total training steps for scheduler
    num_processes = 1 if accelerator is None else accelerator.num_processes
    ds_size = len(acts_dataset)
    steps_per_epoch = max(1, ds_size // (batch_size * num_processes * n_grad_accum))
    total_steps = n_steps if n_steps is not None else n_epochs * steps_per_epoch

    scheduler_fn = get_scheduler_fn(train_cfg, num_training_steps=total_steps)

    probes, optimizers, schedulers = initialize_probes_and_optimizers(
        layers, create_probe_fn, lr, device, pretrained_probes, weight_decay, accelerator, scheduler_fn
    )

    # Set input scales for unit_norm normalization mode
    if normalize_input == "unit_norm":
        scales = compute_input_scales(acts_dataset, layers, batch_size=batch_size)
        for layer, probe in probes.items():
            # Get the unwrapped probe if using FSDP
            unwrapped_probe = accelerator.unwrap_model(probe) if accelerator is not None else probe
            unwrapped_probe.set_input_scale(scales[int(layer)])
        is_main_process = accelerator is None or accelerator.is_main_process
        if is_main_process:
            print(f"Input scales (unit_norm): {scales}")

    n_examples = len(acts_dataset)
    total_losses = {layer: [] for layer in probes.keys()}
    total_lrs: list[float] = []

    # Choose appropriate collate function based on training mode
    if train_sequence_aggregator is not None:
        collate_fn = partial(
            sequence_preserving_collate_fn,
            layer_keys=list(map(str, layers)),
            pad_labels=False,
        )
    else:
        collate_fn = default_collate_fn

    dataloader = torch.utils.data.DataLoader(
        acts_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
    )
    if accelerator is not None:
        dataloader = accelerator.prepare(dataloader)
        device = accelerator.device

    is_main_process = accelerator is None or accelerator.is_main_process
    tqdm_progress_bar = tqdm(
        total=n_steps if n_steps is not None else n_epochs * len(dataloader),
        disable=not is_main_process,
    )
    first_layer = str(next(iter(probes.keys())))

    for epoch in range(n_epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            all_layers_batch, labels = batch[0], batch[1]

            ctx = (
                torch.autocast(device_type=device, dtype=torch.bfloat16)
                if accelerator is None
                else accelerator.autocast()
            )

            with ctx:
                for probe in probes.values():
                    probe.train()

                batch_all_layers = {k: v.to(device) for k, v in all_layers_batch.items()}
                labels = labels.to(device)

                # Compute mask for valid tokens (non-zero activations)
                mask = None
                if batch_all_layers[first_layer].ndim == 3:
                    mask = (batch_all_layers[first_layer] != 0).any(dim=-1)  # (batch, seq)

                for layer, probe in probes.items():
                    layer_activations = batch_all_layers[str(layer)]

                    loss = compute_loss(
                        probe,
                        layer_activations,
                        labels,
                        mask=mask,
                        aggregator=train_sequence_aggregator,
                    )

                    loss.backward()
                    epoch_loss += loss.item() * n_grad_accum

                    if not log_epoch_wise_loss:
                        total_losses[layer].append(loss.item() * n_grad_accum)

            if clip_grad_norm > 0:
                for probe in probes.values():
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_grad_norm)

            if (i + 1) % n_grad_accum == 0:
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                for scheduler in schedulers.values():
                    scheduler.step()
                # Record LR (same across all layers, so just use first scheduler)
                first_scheduler = next(iter(schedulers.values()))
                total_lrs.append(first_scheduler.get_last_lr()[0])

            tqdm_progress_bar.update(1)
            if n_steps is not None and i + epoch * len(dataloader) >= n_steps:
                break
        # Perform an extra optimization step if the number of examples is not divisible by batch_size
        if (n_examples // batch_size) % n_grad_accum != 0:
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
            for scheduler in schedulers.values():
                scheduler.step()

        if n_steps is not None and i + epoch * len(dataloader) >= n_steps:
            break

    # Cleanup
    for probe in probes.values():
        probe.zero_grad(set_to_none=True)
        for module in probe.modules():
            if isinstance(module, FSDP):
                if hasattr(module, "_flat_param") and module._flat_param is not None:
                    if module._flat_param.grad is not None:
                        module._flat_param.grad = None

    for optimizer in optimizers.values():
        optimizer.state.clear()  # This frees the AdamW momentum/variance
        del optimizer
    optimizers.clear()
    schedulers.clear()

    if accelerator is not None and hasattr(accelerator, "_optimizers"):
        accelerator._optimizers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    # Gather and average losses across all ranks
    if accelerator is not None:
        for layer in total_losses:
            if len(total_losses[layer]) > 0:
                gathered_losses = accelerate.utils.gather_object(total_losses[layer])
                num_ranks = accelerator.num_processes
                num_steps = len(total_losses[layer])
                gathered_array = np.array(gathered_losses).reshape(num_ranks, num_steps)
                total_losses[layer] = gathered_array.mean(axis=0).tolist()

    return probes, total_losses, total_lrs


def train_combined_sklearn_logistic_probes(
    acts_dataset: DictTensorDataset,
    create_probe_fn: Callable,
    layers: list[int],
    train_cfg: BaseTrainConfig,
    pretrained_probes: dict[str, object] | None = None,
    log_epoch_wise_loss: bool = True,
    train_sequence_aggregator: SequenceAggregator | None = None,
    num_workers: int = 4,
    max_examples: int | None = 5000,
    accelerator=None,
    normalize_input: str = "none",
):
    """Train a simple sklearn LogisticRegression probe per layer.

    This mirrors the interface of `train_combined_probes` but ignores most
    optimization-related arguments. It aggregates features from the dataset
    into numpy arrays and fits one LogisticRegression per requested layer.

    Note: train_sequence_aggregator is ignored for sklearn probes (token-level only).
    """
    # Warn if aggregator provided but won't be used
    if train_sequence_aggregator is not None:
        warnings.warn(
            "train_sequence_aggregator is ignored for sklearn logistic probes."
            " Sklearn probes are trained at token-level only."
        )
    dataloader = torch.utils.data.DataLoader(
        acts_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Accumulators for features and labels per layer
    x_accumulators: dict[int, list[np.ndarray]] = {int(layer): [] for layer in layers}
    y_accumulators: dict[int, list[np.ndarray]] = {int(layer): [] for layer in layers}

    for batch in dataloader:
        # Handle both 2-tuple (activations, labels) and 3-tuple (activations, labels, example_types)
        all_layers_batch, labels = batch[0], batch[1]
        # all_layers_batch: dict[str, Tensor] with shapes:
        #   - (B, D) or (B, T, D) depending on dataset/collate
        # labels: shape (B,) or (B, T) when preserve_seq_len=True and pad_labels=True
        # Map layer keys to integers when possible
        for layer_key, feats in all_layers_batch.items():
            layer_id = int(layer_key) if does_str_convert_to_int(str(layer_key)) else layer_key
            if layer_id not in x_accumulators:
                continue

            if isinstance(feats, torch.Tensor):
                # Ensure float32 on CPU for sklearn
                feats = feats.to(torch.float32)

            # Case 1: preserve_seq_len=True and pad_labels=True → labels is (B, T)
            if isinstance(labels, torch.Tensor) and labels.ndim == 2:
                valid_mask = labels != -100
                # Align mask to features' time dimension
                if feats.ndim == 3:
                    # (B, T, D) -> take valid positions only
                    mask_exp = valid_mask
                    selected = feats[mask_exp].view(-1, feats.shape[-1])
                elif feats.ndim == 2:
                    # Already (B, D) – treat all rows as valid
                    selected = feats
                else:
                    continue
                y = labels[valid_mask].to(torch.int64)
                x_accumulators[layer_id].append(selected.cpu().numpy())
                y_accumulators[layer_id].append(y.cpu().numpy())
            else:
                if feats.ndim == 3:
                    token_mask = (feats != 0).any(dim=-1)  # (B, T)
                    # Flatten tokens
                    x = feats[token_mask]
                    # Repeat labels for each valid token
                    if isinstance(labels, torch.Tensor) and labels.ndim == 1:
                        labels_rep = labels.unsqueeze(1).expand(-1, token_mask.shape[1])
                        y = labels_rep[token_mask]
                    else:
                        continue
                elif feats.ndim == 2:
                    # Already pooled per example
                    x = feats
                    y = labels.to(torch.int64)
                else:
                    continue
                x_accumulators[layer_id].append(x.cpu().numpy())
                y_accumulators[layer_id].append(y.cpu().numpy())
        if max_examples is not None and sum(len(y) for y in y_accumulators[layer_id]) >= max_examples:
            break

    # Fit a LogisticRegression per layer and transfer to LinearProbe weights
    total_losses: dict[int, list[float]] = {}
    probes: dict[int, Probe] = {}
    d_model = x_accumulators[int(layers[0])][0].shape[-1]
    for layer in tqdm(layers):
        X_list = x_accumulators[int(layer)]
        y_list = y_accumulators[int(layer)]
        if len(X_list) == 0:
            # No data accumulated for this layer; skip fitting
            continue
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        n_steps = train_cfg.max_steps
        weight_decay = train_cfg.weight_decay
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=n_steps if n_steps is not None else 100,
            C=1.0 if weight_decay == 0 else 1 / weight_decay,
            penalty=None if weight_decay == 0 else "l2",
        )
        clf.fit(X, y)
        probe = LinearProbe(d_model)
        with torch.no_grad():
            probe.linear.weight.copy_(torch.from_numpy(clf.coef_.astype(np.float32)))
            probe.linear.bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)))
        probes[int(layer)] = probe
        total_losses[int(layer)] = []

    return probes, total_losses, []  # Empty LR list (sklearn doesn't use schedulers)


def cache_activations(
    encoder: LanguageModelWrapper,
    examples,
    batch_size,
    max_completion_length,
    max_sequence_length,
    cache_dir,
    return_tokens=False,
    append_eos_to_targets=True,
    completion_column="completion",
    follow_up_prompts: list[tuple[str, str]] = [],
    labels: torch.Tensor | None = None,
    example_types: torch.Tensor | None = None,
    **kwargs,
):
    return encoder.get_model_residual_acts(
        examples,
        batch_size=batch_size,
        max_completion_length=max_completion_length,
        max_sequence_length=max_sequence_length,
        return_tokens=return_tokens,
        use_memmap=cache_dir,
        padding_side="right",
        append_eos_to_targets=append_eos_to_targets,
        completion_column=completion_column,
        follow_up_prompts=follow_up_prompts,
        labels=labels,
        example_types=example_types,
        **kwargs,
    )


def does_str_convert_to_int(obj):
    assert isinstance(obj, str), "Object must be a string"
    try:
        int(obj)
        return True
    except ValueError:
        return False


def load_cached_activations(
    cache_dir: str,
) -> tuple[dict, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    """Load cached activations from disk using MemoryMappedTensor.

    Args:
        cache_dir: Directory containing cached activations

    Returns:
        Tuple of (activations_dict, (tokens, prompt_mask, completion_mask), labels, example_types)
        labels and example_types are None if not saved in the cache.
    """
    # Load metadata
    metadata_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.json found in {cache_dir}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load tokens and masks (these are small, load into memory)
    tokens = torch.load(os.path.join(cache_dir, metadata["files"]["tokens"]), weights_only=False)
    prompt_mask = torch.load(os.path.join(cache_dir, metadata["files"]["prompt_mask"]), weights_only=False)
    completion_mask = torch.load(os.path.join(cache_dir, metadata["files"]["completion_mask"]), weights_only=False)

    # Load labels if available
    labels = None
    if metadata.get("has_labels", False) and "labels" in metadata["files"]:
        labels = torch.load(os.path.join(cache_dir, metadata["files"]["labels"]), weights_only=False)

    # Load example_types if available
    example_types = None
    if metadata.get("has_example_types", False) and "example_types" in metadata["files"]:
        example_types = torch.load(os.path.join(cache_dir, metadata["files"]["example_types"]), weights_only=False)

    # Load activations as MemoryMappedTensors
    activations = {}
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[metadata["dtype"]]

    for layer in metadata["layers"]:
        layer_file = os.path.join(cache_dir, metadata["files"][f"layer_{layer}"])
        # Load as MemoryMappedTensor for efficiency
        activations[str(layer)] = MemoryMappedTensor.from_filename(
            filename=layer_file,
            dtype=dtype,
            shape=(metadata["num_samples"], metadata["max_seq_len"], metadata["hidden_dim"]),
        )
    return activations, (tokens, prompt_mask, completion_mask), labels, example_types


def combined_activations(activations_b1, activations_b2, tokens_and_masks_b1, tokens_and_masks_b2):
    """Combine activations, tokens, and masks from two different batches.
    Args:
        activations_b1: Dictionary of activations from the first batch.
        activations_b2: Dictionary of activations from the second batch.
        tokens_and_masks_b1: Tuple of tokens and masks from the first batch.
        tokens_and_masks_b2: Tuple of tokens and masks from the second batch.

    Returns:
        Tuple of (combined activations, combined tokens and masks).
    """
    if activations_b1 is not None and activations_b2 is not None:
        acts = {
            k: torch.cat([v1, v2], dim=0)
            for k, v1, v2 in zip(activations_b1.keys(), activations_b1.values(), activations_b2.values())
        }
        tokens = torch.cat([tokens_and_masks_b1[0], tokens_and_masks_b2[0]], dim=0)
        prompt_mask = torch.cat([tokens_and_masks_b1[1], tokens_and_masks_b2[1]], dim=0)
        completion_mask = torch.cat([tokens_and_masks_b1[2], tokens_and_masks_b2[2]], dim=0)
        return acts, (tokens, prompt_mask, completion_mask)
    elif activations_b1 is not None:
        return activations_b1, tokens_and_masks_b1
    elif activations_b2 is not None:
        return activations_b2, tokens_and_masks_b2
    else:
        return None, None


def load_activations(
    encoder: LanguageModelWrapper,
    positive_examples: HFDataset | list[str] | Mapping[str, torch.Tensor] | None,
    negative_examples: HFDataset | list[str] | Mapping[str, torch.Tensor] | None,
    batch_size: int,
    max_completion_length: int | None,
    max_sequence_length: int | None,
    cache_activations_save_path: str | None = None,
    append_eos_to_targets: bool = True,
    accelerator=None,
    completion_columns: tuple[str, str] = ("completion", "completion"),
    congruent_follow_up_prompts: list[
        tuple[str, str]
    ] = [],  # these will elicit lies under a lie response & truth under a truthful response
    incongruent_follow_up_prompts: list[
        tuple[str, str]
    ] = [],  # these will elicit truth under a lie response & lies under a truthful response
    store_last_token_only: bool = False,
) -> tuple[
    Mapping[str, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor | None,
]:
    """Load activations and return with labels for memory-efficient access.

    Returns:
        activations: Combined activations for all examples
        tokens_and_masks: (tokens, prompt_mask, completion_mask) tuple
        labels: Tensor of labels (1=positive, 0=negative) for each sample
        example_types: Tensor of ExampleType values for each sample (for visualization),
            or None if no follow-up prompts are used
    """
    if isinstance(positive_examples, Mapping) and isinstance(negative_examples, Mapping):
        raise ValueError("Pre-computed activations not supported with new interface")

    congruent_follow_up_prompts = congruent_follow_up_prompts or []
    incongruent_follow_up_prompts = incongruent_follow_up_prompts or []

    normalized_completion_column = "_normalized_completion"
    completion_column = normalized_completion_column

    acts_combined = None
    tokens_masks_combined = None
    labels = None
    rank = accelerator.process_index if accelerator is not None else 0

    def normalize_dataset(ds: HFDataset | None, completion_col: str):
        if ds is None:
            return None
        if not isinstance(ds, HFDataset):
            raise ValueError("Single cache call expects HuggingFace datasets or None.")
        if completion_col != normalized_completion_column:
            if completion_col not in ds.column_names:
                raise ValueError(f"Dataset missing completion column {completion_col}")
            ds = ds.rename_column(completion_col, normalized_completion_column)
        return ds

    pos_ds = normalize_dataset(positive_examples, completion_columns[0])
    neg_ds = normalize_dataset(negative_examples, completion_columns[1])

    def combine_datasets(pos, neg):
        if pos is None:
            return neg
        if neg is None:
            return pos
        return concatenate_datasets([pos, neg])

    combined_examples = combine_datasets(pos_ds, neg_ds)

    n_pos_full = len(pos_ds) if pos_ds is not None else 0
    n_neg_full = len(neg_ds) if neg_ds is not None else 0
    n_cong = len(congruent_follow_up_prompts)
    n_incong = len(incongruent_follow_up_prompts)
    n_followups = n_cong + n_incong

    # Create labels and example_types for the combined dataset BEFORE caching.
    # Labels are computed based on:
    # - Whether the sample came from positive or negative examples
    # - Whether the follow-up prompt is congruent or incongruent
    # Label 1 = positive (deceptive), Label 0 = negative (honest)
    #
    # For congruent follow-ups: pos_examples -> label 1, neg_examples -> label 0
    # For incongruent follow-ups: pos_examples -> label 0, neg_examples -> label 1
    #
    # The data order after process_data with follow_up_prompts is:
    # [all_examples_with_fu0, all_examples_with_fu1, ...]
    # where all_examples = [pos_examples, neg_examples]
    labels = ([1] * n_pos_full + [0] * n_neg_full) * (n_cong if n_followups > 0 else 1)
    labels += ([0] * n_pos_full + [1] * n_neg_full) * n_incong
    labels = torch.tensor(labels, dtype=torch.long)

    # Create example_types tensor for visualization (tracks congruent/incongruent + pos/neg source)
    # Only create if we have follow-up prompts (otherwise the distinction is meaningless)
    example_types: torch.Tensor | None = None
    if n_followups > 0:
        example_types_list = []
        # Congruent blocks
        for _ in range(n_cong):
            example_types_list.extend([ExampleType.CONGRUENT_POSITIVE] * n_pos_full)
            example_types_list.extend([ExampleType.CONGRUENT_NEGATIVE] * n_neg_full)
        # Incongruent blocks
        for _ in range(n_incong):
            example_types_list.extend([ExampleType.INCONGRUENT_POSITIVE] * n_pos_full)
            example_types_list.extend([ExampleType.INCONGRUENT_NEGATIVE] * n_neg_full)
        example_types = torch.tensor(example_types_list, dtype=torch.long)

    # Check if the cache file exists and a save path is provided
    combined_path = (
        os.path.join(cache_activations_save_path, f"rank_{rank}", "combined_examples")
        if cache_activations_save_path is not None
        else None
    )
    if combined_path is not None and os.path.exists(combined_path):
        metadata_path = os.path.join(combined_path, "metadata.json")
        if os.path.exists(metadata_path):
            print(f"Loading cached activations from {combined_path}")
            acts_combined, tokens_masks_combined, cached_labels, cached_example_types = load_cached_activations(
                combined_path
            )
            # Use cached labels/example_types if available, otherwise keep computed values
            if cached_labels is not None:
                labels = cached_labels
            if cached_example_types is not None:
                example_types = cached_example_types

    if acts_combined is None:
        print("Caching activations...")
        acts_combined, tokens_masks_combined, labels, example_types = cache_activations(
            encoder,
            combined_examples,
            batch_size,
            max_completion_length=max_completion_length,
            max_sequence_length=max_sequence_length,
            cache_dir=combined_path,
            return_tokens=True,
            append_eos_to_targets=append_eos_to_targets,
            completion_column=completion_column,
            follow_up_prompts=congruent_follow_up_prompts + incongruent_follow_up_prompts,
            labels=labels,
            example_types=example_types,
            store_last_token_only=store_last_token_only,
        )

    if tokens_masks_combined is None:
        raise ValueError("Combined tokens_and_masks is None")
    return acts_combined, tokens_masks_combined, labels, example_types


def get_activations_for_obfuscation_tokens(
    activations: Mapping[str, torch.Tensor],
    dataset: HFDataset,
    tokenizer,
    obfuscate_over: str,
    preserve_seq_len: bool = False,
    max_completion_length: int | None = None,
    padding_side: str = "right",
    completion_column: str = "completion",
    follow_up_prompts: list[tuple[str, str]] = [],
) -> Mapping[str, torch.Tensor]:
    tokens, prompt_mask, target_mask, _ = process_data(
        dataset["prompt"],
        dataset[completion_column],
        tokenizer,
        max_completion_length=max_completion_length,
        padding_side=padding_side,
        follow_up_prompts=follow_up_prompts,
    )
    mask = compute_mask(
        tokens.shape[1],
        prompt_mask,
        target_mask,
        obfuscate_over,
        ignore_last_token=False,
    )

    if preserve_seq_len:
        # For attention probes, we need to preserve the sequence structure
        # Apply mask by zeroing out non-matching positions instead of flattening
        start_pos, end_pos = trim_sequences(mask)
        mask = mask[:, start_pos:end_pos]
        if isinstance(activations, list):
            first_act_tensor = activations[0][:, start_pos:end_pos, :]
            assert first_act_tensor.shape[:2] == mask.shape[:2], (
                f"Activations shape: {first_act_tensor.shape}, mask shape: {mask.shape}"
            )
            if isinstance(first_act_tensor, np.ndarray):
                mask_np = mask.numpy()
                mask_expanded = np.expand_dims(mask_np, -1)  # Shape: (batch_size, seq_len, 1)
            else:
                mask_expanded = mask.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
            masked_activations = [act_tensor[:, start_pos:end_pos, :] * mask_expanded for act_tensor in activations]
            return masked_activations
        elif isinstance(activations, dict):
            first_act_tensor = next(iter(activations.values()))[:, start_pos:end_pos, :]
            assert first_act_tensor.shape[:2] == mask.shape[:2], (
                f"Activations shape: {first_act_tensor.shape}, mask shape: {mask.shape}"
            )
            if isinstance(first_act_tensor, np.ndarray):
                mask_np = mask.numpy()
                mask_expanded = np.expand_dims(mask_np, -1)  # Shape: (batch_size, seq_len, 1)
            else:
                mask_expanded = mask.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
            masked_activations = {
                key: act_tensor[:, start_pos:end_pos, :] * mask_expanded for key, act_tensor in activations.items()
            }
            return masked_activations
        elif isinstance(activations, torch.Tensor):
            assert activations[:, start_pos:end_pos, :].shape[:2] == mask.shape[:2], (
                f"Activations shape: {activations.shape}, mask shape: {mask.shape}"
            )
            mask_expanded = mask.unsqueeze(-1)
            masked_activations = activations[:, start_pos:end_pos, :] * mask_expanded
            return masked_activations
        elif isinstance(activations, np.ndarray):
            assert activations[:, start_pos:end_pos, :].shape[:2] == mask.shape[:2], (
                f"Activations shape: {activations.shape}, mask shape: {mask.shape}"
            )
            mask_expanded = mask[..., None]
            masked_activations = activations[:, start_pos:end_pos, :] * mask_expanded
            return masked_activations
        else:
            raise ValueError(f"Activations must be a list or dict, got {type(activations)}")
    else:
        # Original behavior: flatten by selecting only masked tokens
        if isinstance(activations, list):
            assert activations[0].shape[:2] == mask.shape[:2], (
                f"Activations shape: {activations[0].shape}, mask shape: {mask.shape}"
            )
            if isinstance(activations[0], np.ndarray):
                mask = mask.numpy()
            return [act_tensor[mask, :] for act_tensor in activations]
        elif isinstance(activations, dict):
            shape = next(iter(activations.values())).shape
            assert shape[:2] == mask.shape[:2], f"Activations shape: {shape}, mask shape: {mask.shape}"
            if isinstance(next(iter(activations.values())), np.ndarray):
                mask = mask.numpy()
            return {key: act_tensor[mask, :] for key, act_tensor in activations.items()}
        elif isinstance(activations, torch.Tensor):
            assert activations.shape[:2] == mask.shape[:2], (
                f"Activations shape: {activations.shape}, mask shape: {mask.shape}"
            )
            return activations[mask, :]
        elif isinstance(activations, np.ndarray):
            assert activations.shape[:2] == mask.shape[:2], (
                f"Activations shape: {activations.shape}, mask shape: {mask.shape}"
            )
            return activations[mask, :]
        else:
            raise ValueError(f"Activations must be a list or dict, got {type(activations)}")


def get_white_box_activations_dataset(
    encoder: LanguageModelWrapper,
    positive_dataset: HFDataset | None,
    negative_dataset: HFDataset | None,
    obfuscate_over: str,
    model_batch_size: int,
    max_completion_length: int | None = None,
    max_sequence_length: int | None = None,
    cache_activations_save_path: str | None = None,
    append_eos_to_targets: bool = True,
    preserve_seq_len: bool = False,
    activation_dataset: ActivationDataset | None = None,
    balance_dataset: bool = False,
    accelerator=None,
    completion_columns: tuple[str, str] = ("completion", "completion"),
    congruent_follow_up_prompts: list[tuple[str, str]] | None = None,
    incongruent_follow_up_prompts: list[tuple[str, str]] | None = None,
) -> ActivationDataset:
    if activation_dataset is None:
        # Clean up GPU memory before starting new activation caching
        log_gpu_memory("Before gc.collect in get_white_box_activations_dataset")
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("After gc.collect in get_white_box_activations_dataset")

        # When using last-token-generation, only store the final token to save memory/time
        store_last_token_only = obfuscate_over == "last-token-generation"
        activations, tokens_and_masks, labels, example_types = load_activations(
            encoder,
            positive_dataset,
            negative_dataset,
            model_batch_size,
            max_completion_length,
            max_sequence_length,
            cache_activations_save_path,
            append_eos_to_targets=append_eos_to_targets,
            accelerator=accelerator,
            completion_columns=completion_columns,
            congruent_follow_up_prompts=congruent_follow_up_prompts,
            incongruent_follow_up_prompts=incongruent_follow_up_prompts,
            store_last_token_only=store_last_token_only,
        )
        log_gpu_memory("After load_activations")

        # Free GPU memory after caching activations to CPU/disk
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("After post-caching gc.collect")

        activation_dataset = ActivationDataset(
            tokenizer=encoder.tokenizer,
            activations=activations,
            tokens_and_masks=tokens_and_masks,
            labels=labels,
            obfuscate_over=obfuscate_over,
            preserve_seq_len=preserve_seq_len,
            balance_dataset=balance_dataset,
            example_types=example_types,
        )
    else:
        activation_dataset.set_obfuscation(
            obfuscate_over,
            preserve_seq_len=preserve_seq_len,
            balance_dataset=balance_dataset,
        )

    return activation_dataset


def train_probe(
    create_probe_fn,
    layers,
    train_cfg: BaseTrainConfig,
    feature_dataset: DictTensorDataset | None = None,
    pretrained_probes=None,
    log_epoch_wise_loss: bool = True,
    train_sequence_aggregator: SequenceAggregator | None = None,
    accelerator=None,
    use_sklearn=False,
    normalize_input: str = "none",
) -> tuple[
    Mapping[int, Probe],
    dict[str, list[float]],
]:
    """
    High-level wrapper for probe training.

    Args:
        create_probe_fn: Factory function to create probes
        layers: Layer indices to train
        train_cfg: Training configuration
        feature_dataset: Activation dataset
        pretrained_probes: Optional pretrained probes
        log_epoch_wise_loss: Log loss per epoch vs per step
        train_sequence_aggregator: Aggregator for sequence-level training
        accelerator: Optional distributed training accelerator
        use_sklearn: Use sklearn LogisticRegression instead
        normalize_input: Input normalization mode
    """
    n_steps = train_cfg.max_steps
    detector_batch_size = train_cfg.batch_size

    if n_steps is not None:
        ds_size = len(feature_dataset)
        num_processes = 1 if accelerator is None else accelerator.num_processes
        train_cfg.num_epochs = int(np.ceil(n_steps * min(detector_batch_size, ds_size) * num_processes / ds_size))
        print(
            f"Setting num_epochs to {train_cfg.num_epochs} given {n_steps=}, {detector_batch_size=}, {ds_size=},"
            f" {num_processes=}"
        )

    if use_sklearn:
        train_fn = (
            train_combined_sklearn_logistic_probes
            if accelerator is None or accelerator.is_main_process
            else lambda *args, **kwargs: (None, None, [])
        )
    else:
        train_fn = train_combined_probes
    probes, total_losses, total_lrs = train_fn(
        feature_dataset,
        create_probe_fn,
        layers,
        train_cfg,
        pretrained_probes=pretrained_probes,
        log_epoch_wise_loss=log_epoch_wise_loss,
        train_sequence_aggregator=train_sequence_aggregator,
        accelerator=accelerator,
        normalize_input=normalize_input,
    )
    if use_sklearn and accelerator is not None:
        probes = {layer: accelerator.prepare(probe) for layer, probe in probes.items()}
    # train_dynamics: step-wise training curves (losses per step, learning rates per step)
    # This is distinct from train_eval_metrics which contains evaluation metrics on the training set
    train_dynamics = {f"layer_{layer}_loss": losses for layer, losses in total_losses.items()}
    train_dynamics["loss"] = np.mean(np.stack([v for k, v in train_dynamics.items() if "loss" in k]), axis=0)
    train_dynamics["lr"] = total_lrs
    return probes, train_dynamics


def disable_model_gradients(lora_model):
    # Disables all model gradients
    for param in lora_model.parameters():
        param.requires_grad_(False)


def enable_model_gradients(lora_model):
    # Enables lora adapter gradients
    n_layers = get_num_hidden_layers(lora_model.config)
    for i in range(n_layers):
        for name, param in lora_model.get_submodule("base_model.model.model.layers")[i].named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)


def train_online_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    lora_params={},
    adversarial_training=False,
    probe_lr=1e-3,
    adapter_lr=5e-5,
    kl_penalty=1e-2,
    max_length=1024,
    n_steps=1000,
    n_steps_per_logging=100,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    pretrained_probes=None,
    only_return_on_tokens_between=None,
    only_choose_prompt_tokens_between=None,
    only_probe_tokens_between=None,  # New parameter for probe mask
    epsilon=10.0,
    adversary_lr=1e-3,
    pgd_iterations=32,
    clip_grad_norm=1.0,
    start_adv_training_at_step=1024,
    freeze_probes_during_adversarial_training=True,
    freeze_lora_during_warmup=False,
    use_lora_adapter=True,
    run_softprompt_eval_every=128,
    softprompt_evals_data={},
    weight_decay=0.0,
    accelerator=None,
    **kwargs,
):
    assert n_grad_accum == 0 or n_steps % n_grad_accum == 0

    # Initialize probes and optimizers for each layer
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, probe_lr, device, pretrained_probes, weight_decay, accelerator
    )
    probes = {layer: probe.to(device) for layer, probe in probes.items()}

    # Initialize LoRA adapter
    if use_lora_adapter:
        lora_model = initialize_lora_adapter(encoder, layers, lora_params)
        adapter_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=adapter_lr)
    else:
        lora_model = encoder.model
        adapter_optimizer = None

    # Tokenize and prepare input data
    encoder.tokenizer.padding_side = "right"
    positive_tokens = encoder.tokenizer(
        positive_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    positive_input_ids = positive_tokens["input_ids"]
    negative_tokens = encoder.tokenizer(
        negative_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    negative_input_ids = negative_tokens["input_ids"]

    # Target mask - where we compute the main loss
    if only_return_on_tokens_between is not None:
        zero_positive_mask = get_valid_token_mask(positive_input_ids, only_return_on_tokens_between)
        zero_negative_mask = get_valid_token_mask(negative_input_ids, only_return_on_tokens_between)
    else:
        zero_positive_mask = torch.ones_like(positive_input_ids).bool()
        zero_negative_mask = torch.ones_like(negative_input_ids).bool()

    # Probe mask - where we compute probe measurements
    if only_probe_tokens_between is not None:
        probe_positive_mask = get_valid_token_mask(positive_input_ids, only_probe_tokens_between)
        probe_negative_mask = get_valid_token_mask(negative_input_ids, only_probe_tokens_between)
    else:
        # If no probe mask specified, use the target mask
        probe_positive_mask = zero_positive_mask
        probe_negative_mask = zero_negative_mask

    # This is only relevant for adversarial training
    if only_choose_prompt_tokens_between is not None:
        assert adversarial_training
        pos_only_choose_mask = get_valid_token_mask(positive_input_ids, only_choose_prompt_tokens_between)
        pos_only_choose_mask = pos_only_choose_mask.to(device)
    else:
        pos_only_choose_mask = None

    n_examples = min(len(positive_examples), len(negative_examples))

    continue_training_next_epoch = True
    current_step = 0
    start_time = time.time()

    accumulated_toward_pgd_loss = 0
    accumulated_probe_pgd_loss = 0
    accumulated_probe_loss = 0
    accumulated_kl_loss = 0
    steps_since_last_log = 0
    info = {
        "softprompt_evals": [],
    }

    wrappers = []
    pgd_probe_loss = 0

    pbar = tqdm(total=n_steps, desc="Training LORA+Probe")

    while continue_training_next_epoch:
        # Shuffle the examples
        perm = torch.randperm(n_examples)

        for i in range(0, n_examples, batch_size):
            # Check if the batch is the last one
            if i + batch_size > n_examples:
                break

            # Get the batch
            batch_perm = perm[i : i + batch_size]
            pos_batch_input_ids = positive_input_ids[batch_perm].to(device)
            neg_batch_input_ids = negative_input_ids[batch_perm].to(device)
            pos_batch_zero_mask = zero_positive_mask[batch_perm].to(device).bool()
            neg_batch_zero_mask = zero_negative_mask[batch_perm].to(device).bool()
            # Get probe masks for the batch
            pos_batch_probe_mask = probe_positive_mask[batch_perm].to(device).bool()
            neg_batch_probe_mask = probe_negative_mask[batch_perm].to(device).bool()

            if pos_only_choose_mask is not None:
                pos_batch_only_choose_mask = pos_only_choose_mask[batch_perm].to(device).bool()

            # Forward pass on positive examples
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if adversarial_training and current_step >= start_adv_training_at_step:
                    # Print this out at the first adversarial training step
                    if current_step == start_adv_training_at_step:
                        print("FORMATTING EXAMPLES FOR ADVERSARIAL TRAINING")
                        print("This is the target mask:")
                        print(repr(encoder.tokenizer.decode(pos_batch_input_ids[0][pos_batch_zero_mask[0]])))
                        print("This is the prompt mask:")
                        print(repr(encoder.tokenizer.decode(pos_batch_input_ids[0][pos_batch_only_choose_mask[0]])))
                        print("This is the probe mask:")
                        print(repr(encoder.tokenizer.decode(pos_batch_input_ids[0][pos_batch_probe_mask[0]])))

                    # Disable model gradients
                    disable_model_gradients(lora_model)

                    # Run this eval to benchmark the effectiveness of universal soft prompts
                    if (current_step - start_adv_training_at_step) % run_softprompt_eval_every == 0:
                        print("RUNNING SOFTPROMPT EVALUATION")
                        # pruned file
                        results = benchmark_obfuscated_softprompt(  # type: ignore # noqa: F821
                            model=lora_model,
                            model_layers_module="base_model.model.model.layers",
                            tokenizer=encoder.tokenizer,
                            probes=probes,
                            **softprompt_evals_data,
                        )

                        info["softprompt_evals"].append(results)
                        print("Jailbreak Success Rate:", results["jailbreak_rate"])
                        print("Average positive score:", results["avg_positive_score"])
                        print("Average negative score:", results["avg_negative_score"])

                    # Train new attack from scratch against batch inputs
                    # from pruned file
                    losses, wrappers = train_attack(  # type: ignore # noqa: F821
                        adv_tokens=pos_batch_input_ids,
                        prompt_mask=pos_batch_only_choose_mask,
                        target_mask=pos_batch_zero_mask,
                        model=lora_model,
                        tokenizer=encoder.tokenizer,
                        model_layers_module="base_model.model.model.layers",
                        layer=["embedding"],
                        epsilon=epsilon,
                        learning_rate=adversary_lr,
                        pgd_iterations=pgd_iterations,
                        probes=probes,
                        probe_mask=pos_batch_probe_mask,  # Pass probe mask
                        adversary_type="pgd",
                    )

                    pgd_toward_loss = losses["toward"]
                    pgd_probe_loss = losses["probe"]

                    # Enable model gradients on the lora adapter
                    enable_model_gradients(lora_model)
                else:
                    pgd_toward_loss = 0  # Set to 0 when adversarial training is not used
                    pgd_probe_loss = 0
                    wrappers = []

                for wrapper in wrappers:
                    wrapper.enabled = True

                pos_output = lora_model(
                    input_ids=pos_batch_input_ids,
                    output_hidden_states=True,
                )
                pos_acts = {layer: pos_output.hidden_states[layer + 1] for layer in layers}

            # Compute the positive probe losses using probe mask
            pos_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    pos_targets = torch.ones_like(pos_acts[layer][..., 0], device=device)
                    pos_layer_loss = probe.compute_loss(
                        pos_acts[layer],
                        pos_targets,
                        mask=pos_batch_probe_mask,  # Use probe mask
                    )
                    pos_loss += pos_layer_loss

            # Backward pass on positive examples
            pos_loss.backward(retain_graph=True)

            for wrapper in wrappers:
                wrapper.enabled = False

            # Forward pass on negative examples
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    output_hidden_states=True,
                )
                neg_logits = neg_output.logits
                neg_acts = {layer: neg_output.hidden_states[layer + 1] for layer in layers}

            # Compute the negative probe losses using probe mask
            neg_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    neg_targets = torch.zeros_like(neg_acts[layer][..., 0], device=device)
                    neg_layer_loss = probe.compute_loss(
                        neg_acts[layer],
                        neg_targets,
                        mask=neg_batch_probe_mask,  # Use probe mask
                    )
                    neg_loss += neg_layer_loss

            # Backward pass on negative examples
            neg_loss.backward(retain_graph=True)

            # Compute KL divergence of logits from base model logits
            with torch.no_grad():
                lora_model.disable_adapter_layers()
                base_neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    # attention_mask=neg_batch_attention_mask,
                )
                lora_model.enable_adapter_layers()

            # Get logits only for masked positions
            base_logits = base_neg_output.logits[neg_batch_zero_mask]
            model_logits = neg_logits[neg_batch_zero_mask]

            kl_loss = F.kl_div(
                F.log_softmax(base_logits, dim=-1),
                F.softmax(model_logits, dim=-1),
                reduction="batchmean",
            )

            # Backward pass on KL divergence
            (kl_loss / (kl_loss.detach() + 1e-8) * kl_penalty).backward()

            # Accumulate losses
            accumulated_probe_loss += pos_loss.item() + neg_loss.item()
            accumulated_kl_loss += kl_loss.item()
            accumulated_toward_pgd_loss += pgd_toward_loss if adversarial_training else 0
            accumulated_probe_pgd_loss += pgd_probe_loss if adversarial_training else 0
            steps_since_last_log += 1

            # Perform optimization step after accumulating gradients
            if (i // batch_size + 1) % n_grad_accum == 0 or (i + batch_size) >= n_examples:
                # Clip the gradients if specified
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(lora_model.parameters(), clip_grad_norm)
                    all_probe_params = [param for probe in probes.values() for param in probe.parameters()]
                    torch.nn.utils.clip_grad_norm_(all_probe_params, clip_grad_norm)

                # Optimize probes only when not using adversarial training
                if not freeze_probes_during_adversarial_training or not (
                    adversarial_training and current_step > start_adv_training_at_step
                ):
                    for optimizer in optimizers.values():
                        optimizer.step()
                        optimizer.zero_grad()

                # Optimize the adapter
                if not freeze_lora_during_warmup or not (
                    adversarial_training and current_step < start_adv_training_at_step
                ):
                    if adapter_optimizer is not None:
                        adapter_optimizer.step()
                        adapter_optimizer.zero_grad()

            current_step += 1

            if current_step % n_steps_per_logging == 0:
                avg_probe_loss = accumulated_probe_loss / steps_since_last_log
                avg_kl_loss = accumulated_kl_loss / steps_since_last_log
                avg_toward_pgd_loss = accumulated_toward_pgd_loss / steps_since_last_log if adversarial_training else 0
                avg_probe_pgd_loss = accumulated_probe_pgd_loss / steps_since_last_log if adversarial_training else 0
                avg_total_loss = avg_probe_loss + avg_kl_loss

                log_message = (
                    f"Step: {current_step}/{n_steps}, "
                    f"Time: {convert_seconds_to_time_str(time.time() - start_time)}, "
                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                    f"Avg Probe Loss: {avg_probe_loss:.4f}, "
                    f"Avg KL Loss: {avg_kl_loss:.4f}"
                )

                if adversarial_training:
                    log_message += f", Avg Toward PGD Loss: {avg_toward_pgd_loss:.4f}"
                    log_message += f", Avg Probe PGD Loss: {avg_probe_pgd_loss:.4f}"

                print(log_message)

                # Reset accumulators
                accumulated_toward_pgd_loss = 0
                accumulated_probe_pgd_loss = 0
                accumulated_probe_loss = 0
                accumulated_kl_loss = 0
                steps_since_last_log = 0

            if current_step >= n_steps:
                continue_training_next_epoch = False
                break

            pbar.update(1)  # Update progress bar

    return probes, lora_model, info


def extract_model_config(model: torch.nn.Module) -> dict[str, Any]:
    """
    Extract the __init__ arguments from a model instance.
    Assumes the model stores init args as instance attributes with the same names.
    """
    config = {
        "class_name": model.__class__.__name__,
        "module": model.__class__.__module__,
    }

    # Get the __init__ signature
    signature = inspect.signature(model.__class__.__init__)

    # Extract argument values from model attributes
    init_args = {}
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue

        # Try to get the value from model attributes
        if hasattr(model, param_name):
            value = getattr(model, param_name)
            # Only save serializable types
            if isinstance(value, (int, float, str, bool, list, tuple, dict, type(None))):
                init_args[param_name] = value
            elif isinstance(value, torch.dtype):
                init_args[param_name] = str(value)

    config["init_args"] = init_args
    return config


def save_model(model: torch.nn.Module, save_dir: str, accelerator: Accelerator | None = None) -> None:
    """
    Save model state dict and auto-extracted config.
    Handles both regular models and FSDP-wrapped models.

    Args:
        model: PyTorch model to save (can be FSDP-wrapped)
        save_dir: Path to save the model and config
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, dict):
        for k, v in model.items():
            save_model(v, save_dir / f"layer_{k}", accelerator)
        return
    # Get the actual model (unwrap if FSDP)
    if accelerator is not None and isinstance(model, FSDP):
        # Extract config from the unwrapped model
        unwrapped_model = accelerator.unwrap_model(model)
        config = extract_model_config(unwrapped_model)

        # Use FSDP's special state dict extraction
        config_fsdp = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config_fsdp):
            state_dict = model.state_dict()
    else:
        # Regular model
        config = extract_model_config(model)
        state_dict = model.state_dict()

    # Save state dict
    torch.save(state_dict, save_dir / "model.pt")

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_model(save_dir: str, model_class: Type[torch.nn.Module] | None = None) -> torch.nn.Module:
    """
    Load model from saved state dict and config.

    Args:
        save_dir: Path to save the model and config
        model_class: Optional model class (if not provided, uses class from config)

    Returns:
        Loaded model instance
    """
    save_dir = Path(save_dir)
    layer_subdirs = [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith("layer_")]
    if len(layer_subdirs) > 0:
        model = {int(d.name.split("_")[1]): load_model(d, model_class) for d in layer_subdirs}
        return model

    # Load config
    with open(save_dir / "config.json", "r") as f:
        config = json.load(f)

    # Get model class
    if model_class is None:
        # Dynamically import the module and get the class
        module = importlib.import_module(config["module"])
        model_class = getattr(module, config["class_name"])

    # Create model instance using saved init args
    init_args = config.get("init_args", {})
    model = model_class(**init_args)

    # Load state dict
    state_dict = torch.load(save_dir / "model.pt")
    model.load_state_dict(state_dict)

    return model


ModelOrDictVar = TypeVar("ModelOrDictVar", Module, dict[int, Module], None)


def _apply_to_model_or_dict(fn):
    """Decorator that applies a function to either a single model or dict of models."""

    @wraps(fn)
    def wrapper(model: ModelOrDictVar, *args, **kwargs) -> ModelOrDictVar:
        if isinstance(model, dict):
            result = {k: fn(v, *args, **kwargs) for k, v in model.items()}
            # If all values are None, return None
            if all(v is None for v in result.values()):
                return None
            return result
        return fn(model, *args, **kwargs)

    return wrapper


def is_on_gpu(model: Module | dict | None) -> bool:
    """Check if a model (or dict of models) is on GPU or FSDP-wrapped."""
    if model is None:
        return False
    if isinstance(model, dict):
        return bool(model) and is_on_gpu(next(iter(model.values())))
    if isinstance(model, FSDP):
        return True
    if any(isinstance(m, FSDP) for m in model.modules()):
        return True
    try:
        param = next(model.parameters(), None)
        return param is not None and param.device.type == "cuda"
    except Exception:
        return False


@_apply_to_model_or_dict
def to_cpu(model: Module, accelerator: Accelerator | None) -> Module | None:
    """Move a single model to CPU, handling FSDP if needed."""
    if accelerator is not None and isinstance(model, FSDP):
        unwrapped = accelerator.unwrap_model(model)
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            state_dict = model.state_dict()
            if accelerator.is_main_process:
                model_config = extract_model_config(unwrapped)
                new_model = unwrapped.__class__(**model_config["init_args"])
                new_model.load_state_dict(state_dict)
                return new_model
            return None
    return model.to("cpu")


def _broadcast_single_model(
    model: Module | None,
    accelerator: Accelerator,
) -> Module:
    """Broadcast a single model from rank 0 to all ranks."""
    if accelerator.is_main_process:
        config = extract_model_config(model)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        data = [config, state_dict]
    else:
        data = [None, None]

    config, state_dict = broadcast_object_list(data, from_process=0)

    module = importlib.import_module(config["module"])
    model_cls = getattr(module, config["class_name"])
    new_model = model_cls(**config["init_args"])
    new_model.load_state_dict(state_dict)
    return new_model


def broadcast_detector(
    detector: ModelOrDictVar,
    accelerator: Accelerator | None,
) -> ModelOrDictVar:
    """Broadcast detector from rank 0 to all other ranks."""
    # No broadcast needed for single process
    if accelerator is None or accelerator.num_processes == 1:
        return detector

    # Check if already on GPU (all ranks have their shards)
    is_gpu_on_main = broadcast_object_list(
        [is_on_gpu(detector) if accelerator.is_main_process else None],
        from_process=0,
    )[0]
    if is_gpu_on_main:
        return detector

    # Handle dict of models
    is_dict = broadcast_object_list(
        [isinstance(detector, dict) if accelerator.is_main_process else None],
        from_process=0,
    )[0]

    if is_dict:
        keys = broadcast_object_list(
            [list(detector.keys()) if accelerator.is_main_process else None],
            from_process=0,
        )[0]
        return {
            k: broadcast_detector(
                detector[k] if accelerator.is_main_process else None,
                accelerator,
            )
            for k in keys
        }

    return _broadcast_single_model(detector, accelerator)


@_apply_to_model_or_dict
def _to_gpu_single(
    model: Module | None,
    accelerator: Accelerator | None,
    device: str,
) -> Module:
    """Move a single model to GPU."""
    if is_on_gpu(model):
        return model
    if accelerator is not None:
        with probe_fsdp_policy(accelerator):
            return accelerator.prepare(model)
    return model.to(device)


def to_gpu(
    model: ModelOrDictVar,
    accelerator: Accelerator | None = None,
    device: str = "cuda",
) -> ModelOrDictVar:
    """Move model(s) to GPU, broadcasting first in distributed mode."""
    # Broadcast first to ensure all ranks have the model
    if accelerator is not None and accelerator.num_processes > 1:
        model = broadcast_detector(model, accelerator)

    return _to_gpu_single(model, accelerator, device)


def to_cpu_and_cleanup_detector(
    detector: ModelOrDictVar,
    accelerator: Accelerator | None = None,
) -> ModelOrDictVar:
    """Free detector from accelerator, move to CPU, and clean up GPU memory."""
    # Remove from accelerator's tracked models
    if accelerator is not None and hasattr(accelerator, "_models"):
        refs = set(detector.values()) if isinstance(detector, dict) else {detector}
        accelerator._models = [m for m in accelerator._models if m not in refs]

    detector = to_cpu(detector, accelerator)
    gc.collect()
    torch.cuda.empty_cache()
    return detector


def get_probe_creation_fn(
    d_model: int,
    cfg: DetectorArchConfig,
) -> Callable[[], Probe]:
    """
    Factory function to create probe instances.

    All probes output (batch, seq, nhead) logits where:
    - Single-head probes (linear, nonlinear, attention, transformer): nhead=1
    - Multi-head probes (gdm, multihead-linear): nhead>1
    """
    detector_type = cfg.detector_type.lower().replace("_", "-")

    if detector_type == "linear-probe":
        return lambda: LinearProbe(d_model, nhead=cfg.nhead, normalize_input=cfg.normalize_input)

    elif detector_type == "nonlinear-probe":
        return lambda: NonlinearProbe(
            d_model,
            cfg.d_mlp,
            nhead=cfg.nhead,
            dropout=cfg.dropout,
            normalize_input=cfg.normalize_input,
        )

    elif detector_type == "attention-probe":
        return lambda: AttentionProbe(
            d_model,
            cfg.d_proj,
            nhead=cfg.nhead,
            sliding_window=cfg.sliding_window,
            use_checkpoint=cfg.use_checkpoint,
            normalize_input=cfg.normalize_input,
        )

    elif detector_type == "transformer-probe":
        return lambda: TransformerProbe(
            d_model,
            nlayer=cfg.nlayer,
            nhead=cfg.nhead,
            d_mlp=cfg.d_mlp,
            dropout=cfg.dropout,
            activation=cfg.activation,
            norm_first=cfg.norm_first,
            use_checkpoint=cfg.use_checkpoint,
            normalize_input=cfg.normalize_input,
        )

    elif detector_type == "gdm-probe":
        return lambda: GDMProbe(
            d_model,
            d_proj=cfg.d_proj,
            nhead=cfg.nhead,
            num_mlp_layers=cfg.nlayer,
            normalize_input=cfg.normalize_input,
        )
    else:
        raise ValueError(f"Detector type {detector_type} not supported.")


def train_mean_difference_probe(
    feature_dataset: DictTensorDataset,
    layers: list[int],
    d_model: int,
    train_cfg: BaseTrainConfig,
    train_sequence_aggregator: SequenceAggregator | None = None,
    normalize_input: str = "none",
    **kwargs,  # noqa: F841
) -> tuple[Mapping[int, LinearProbe], dict[str, list]]:
    """Computes a probe by taking the mean difference between positive and negative activations.

    Args:
        feature_dataset: Torch dataset containing positive and negative activations for all layers with labels.
        layers: List of layer indices to compute the mean difference probe on.
        d_model: Dimension of the model.
        train_cfg: BaseTrainConfig containing training hyperparameters (only batch_size and device are used).
        train_sequence_aggregator: If not None, use sequence-preserving collation with masking.
        normalize_input: Input normalization mode ("none", "l2", or "unit_norm").
        **kwargs: Additional keyword arguments (unused, for interface compatibility).

    Returns:
        A tuple containing:
        - A dictionary of probe scores, keyed by layer.
        - Empty train_metrics dict (no training is done).
    """
    detector_batch_size = train_cfg.batch_size
    device = train_cfg.device

    # Compute input scales for unit_norm normalization (matches train_combined_probes)
    input_scales: dict[int, float] | None = None
    if normalize_input == "unit_norm":
        input_scales = compute_input_scales(feature_dataset, layers, batch_size=detector_batch_size)
        print(f"Input scales (unit_norm): {input_scales}")

    # Use sequence-preserving collate when we have sequence data (matches train_combined_probes)
    if train_sequence_aggregator is not None:
        collate_fn = partial(
            sequence_preserving_collate_fn,
            layer_keys=list(map(str, layers)),
            pad_labels=False,
        )
    else:
        collate_fn = default_collate_fn

    dataloader = DataLoader(
        feature_dataset,
        batch_size=detector_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # Initialize accumulators for running means (per-layer to avoid counting N_layers times)
    positive_sums = {layer: torch.zeros(d_model, device=device) for layer in layers}
    negative_sums = {layer: torch.zeros(d_model, device=device) for layer in layers}
    positive_counts = {layer: 0 for layer in layers}
    negative_counts = {layer: 0 for layer in layers}

    # Compute running sums
    print("Computing mean activations...")
    first_layer_key = str(layers[0])
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Handle both 2-tuple (activations, labels) and 3-tuple (activations, labels, example_types)
            batch_activations, batch_labels = batch[0], batch[1]
            # Move to device
            batch_labels = batch_labels.to(device)
            positive_mask = batch_labels == 1
            negative_mask = batch_labels == 0

            # Compute mask for valid tokens (non-zero activations) - same as train_combined_probes
            first_acts = batch_activations[first_layer_key].to(device)
            if first_acts.ndim == 3:
                valid_token_mask = (first_acts != 0).any(dim=-1)  # (batch, seq), True = valid
            else:
                valid_token_mask = None

            # Process each layer
            for layer in layers:
                # Cast to float32 for numerical stability in accumulation
                # (activations are often bfloat16, which loses precision in large sums)
                layer_acts = batch_activations[f"{layer}"].to(device=device, dtype=torch.float32)

                # Apply input normalization (matches probe._maybe_normalize at eval time)
                if normalize_input == "l2":
                    act_norm = torch.norm(layer_acts, dim=-1, keepdim=True)
                    layer_acts = layer_acts / (act_norm + 1e-8)
                elif normalize_input == "unit_norm" and input_scales is not None:
                    layer_acts = layer_acts / input_scales[layer]

                if positive_mask.any():
                    positive_acts = layer_acts[positive_mask]
                    # Handle both 2D [batch, d_model] and 3D [batch, seq, d_model] activations
                    if positive_acts.ndim == 3:
                        if valid_token_mask is not None:
                            pos_valid = valid_token_mask[positive_mask]  # (pos_batch, seq)
                            # Mask out padding before summing
                            positive_sums[layer] += (positive_acts * pos_valid.unsqueeze(-1)).sum(dim=(0, 1))
                            positive_counts[layer] += pos_valid.sum().item()
                        else:
                            positive_sums[layer] += positive_acts.sum(dim=(0, 1))
                            positive_counts[layer] += positive_acts.shape[0] * positive_acts.shape[1]
                    else:
                        positive_sums[layer] += positive_acts.sum(dim=0)
                        positive_counts[layer] += positive_acts.shape[0]

                # Accumulate negative samples
                if negative_mask.any():
                    negative_acts = layer_acts[negative_mask]
                    # Handle both 2D [batch, d_model] and 3D [batch, seq, d_model] activations
                    if negative_acts.ndim == 3:
                        if valid_token_mask is not None:
                            neg_valid = valid_token_mask[negative_mask]  # (neg_batch, seq)
                            negative_sums[layer] += (negative_acts * neg_valid.unsqueeze(-1)).sum(dim=(0, 1))
                            negative_counts[layer] += neg_valid.sum().item()
                        else:
                            negative_sums[layer] += negative_acts.sum(dim=(0, 1))
                            negative_counts[layer] += negative_acts.shape[0] * negative_acts.shape[1]
                    else:
                        negative_sums[layer] += negative_acts.sum(dim=0)
                        negative_counts[layer] += negative_acts.shape[0]

    for layer in layers:
        assert positive_counts[layer] > 0 and negative_counts[layer] > 0, (
            f"Layer {layer}: No positive or negative samples found"
        )
    # Compute means and create probes
    mean_diff_probes = {}
    for layer in layers:
        # Compute means
        positive_mean = positive_sums[layer] / positive_counts[layer]
        negative_mean = negative_sums[layer] / negative_counts[layer]

        mean_diff = positive_mean - negative_mean
        assert mean_diff.shape == (d_model,), f"Mean diff shape: {mean_diff.shape}, expected ({d_model},)"

        # Create probe with matching normalization config
        mean_diff_probe = LinearProbe(d_model, normalize_input=normalize_input)
        mean_diff_probe.linear.weight.data = mean_diff.unsqueeze(0)
        mean_diff_probe.linear.bias.data = torch.zeros(1)
        if normalize_input == "unit_norm" and input_scales is not None:
            mean_diff_probe.set_input_scale(input_scales[layer])
        mean_diff_probe.to(device).eval()
        mean_diff_probes[layer] = mean_diff_probe

    # Return empty train_metrics dict (no training done for mean difference probe)
    return (
        mean_diff_probes,
        {"loss": [], "lr": []},
    )


def fit_platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    scale_threshold: float = 2.0,
    max_return_value: float = 1000,
    shift_only: bool = False,
) -> tuple[float, float]:
    """Fit Platt scaling parameters A, B such that P(y=1|x) = sigmoid(A*logit + B).

    Platt scaling calibrates the output probabilities of a classifier by fitting
    a logistic regression on the logits. This is useful when the classifier's
    raw probability outputs are not well-calibrated.

    Args:
        logits: Raw logits (pre-sigmoid) from the probe, shape (n_samples,).
        labels: Binary labels (0 or 1), shape (n_samples,).
        scale_threshold: Maximum absolute value for A before penalizing.
        max_return_value: Penalty value when A exceeds scale_threshold.
        shift_only: If True, only fit B (shift) and keep A=1 (no scaling).

    Returns:
        A tuple (A, B) where calibrated probabilities are sigmoid(A * logit + B).
    """
    # Ensure inputs are 1D
    logits = np.asarray(logits).ravel()
    labels = np.asarray(labels).ravel()

    if shift_only:
        # Only optimize B, keeping A=1
        def neg_log_likelihood_shift(B: float) -> float:
            probs = expit(logits + B)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

        result = minimize(neg_log_likelihood_shift, x0=0.0, method="L-BFGS-B")
        return 1.0, float(result.x[0])

    def neg_log_likelihood(params: np.ndarray) -> float:
        A, B = params
        probs = expit(A * logits + B)
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        if abs(A) > scale_threshold:
            return max_return_value
        else:
            return nll

    # Initialize with A=1, B=0 (identity transformation)
    result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def get_probe_scores_from_activations(
    probes: Mapping[int, Probe],
    data_loader: DataLoader,
    probe_layers: list[int],
    aggregator: SequenceAggregator | None = None,
    device: str = "cuda",
    accelerator=None,
    return_logits: bool = False,
) -> tuple[Mapping[str, np.ndarray], np.ndarray, np.ndarray | None]:
    """
    Calculate probe scores for activations.

    All probes output (batch, seq, nhead). This function:
    1. Gets logits from probe
    2. Aggregates across sequence (if aggregator provided)
    3. Applies sigmoid (unless return_logits=True)

    Args:
        probes: Dictionary of probes by layer index
        data_loader: DataLoader yielding (activations, labels) or (activations, labels, example_types)
        probe_layers: Layer indices to evaluate
        aggregator: Aggregator for sequence dimension. If None, uses mean aggregation.
        device: Device to run on
        accelerator: Optional accelerator for distributed training
        return_logits: If True, return raw logits instead of probabilities.
                      Useful for Platt scaling calibration.

    Returns:
        Tuple of:
        - scores: Dict mapping layer to scores array
        - labels: Labels array
        - example_types: Example types array (or None)
    """
    if accelerator is not None:
        device = accelerator.device

    # Default aggregator
    if aggregator is None:
        aggregator = mean_aggregator()

    for layer in probe_layers:
        probes[layer].to(device).eval()

    all_labels = []
    all_example_types = []
    probe_scores = {layer: [] for layer in probe_layers}
    has_example_types = None

    ctx = (
        accelerator.autocast()
        if accelerator is not None
        else torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
    )

    with torch.no_grad(), ctx:
        for batch in data_loader:
            # Unpack batch
            if len(batch) == 3:
                activations_batch, labels_batch, example_types_batch = batch
            else:
                activations_batch, labels_batch = batch
                example_types_batch = None

            # Move to device
            activations_batch = {k: v.to(device) for k, v in activations_batch.items()}
            labels_batch = labels_batch.to(device)
            if example_types_batch is not None:
                example_types_batch = example_types_batch.to(device)

            # Compute mask for valid tokens (non-zero activations)
            first_layer_key = str(probe_layers[0])
            first_acts = activations_batch[first_layer_key]
            if first_acts.ndim == 3:
                mask = (first_acts != 0).any(dim=-1)  # (batch, seq), True = valid
            else:
                mask = None

            # Gather mask, labels, example_types once per batch
            if accelerator is not None:
                gathered_mask = accelerator.gather_for_metrics(mask) if mask is not None else None
                gathered_labels = accelerator.gather_for_metrics(labels_batch)
                gathered_types = (
                    accelerator.gather_for_metrics(example_types_batch) if example_types_batch is not None else None
                )
            else:
                gathered_mask = mask
                gathered_labels = labels_batch
                gathered_types = example_types_batch

            # Check for example types on first batch
            if has_example_types is None:
                has_example_types = gathered_types is not None and not (gathered_types == -1).all().item()

            # Process each layer
            for layer_id in probe_layers:
                probe = probes[layer_id]
                act = activations_batch[str(layer_id)]

                # Get logits from probe
                # All probes output (batch, seq, nhead)
                if aggregator.needs_q and hasattr(probe, "forward_qv"):
                    q, v = probe.forward_qv(act, padding_mask=mask)
                else:
                    q = None
                    v = probe(act, padding_mask=mask)

                # Gather before aggregation (for distributed)
                if accelerator is not None:
                    gathered_v = accelerator.gather_for_metrics(v)
                    if q is not None:
                        gathered_q = accelerator.gather_for_metrics(q)
                    else:
                        gathered_q = None
                else:
                    gathered_v = v
                    gathered_q = q

                # Aggregate: (batch, seq, nhead) -> (batch,)
                logits = aggregator(gathered_v, gathered_mask, q=gathered_q)

                # Apply sigmoid if not returning logits
                if return_logits:
                    current_scores = logits
                else:
                    current_scores = torch.sigmoid(logits)

                current_labels = gathered_labels

                probe_scores[layer_id].append(current_scores.float().cpu().numpy())

                # Only append labels/types once per batch (use first layer)
                if layer_id == probe_layers[0]:
                    all_labels.append(current_labels.cpu().numpy())
                    if has_example_types:
                        all_example_types.append(gathered_types.cpu().numpy())

    final_scores = {str(layer): np.concatenate(probe_scores[layer]) for layer in probe_layers}
    final_labels = np.concatenate(all_labels)
    final_example_types = np.concatenate(all_example_types) if all_example_types else None

    return final_scores, final_labels, final_example_types


def get_detector_scores(
    detector: DETECTOR_TYPE,
    feature_dataset: DictTensorDataset,
    eval_sequence_aggregator: SequenceAggregator | None = None,
    detector_batch_size: int = 64,
    device: str = "cuda",
    accelerator=None,
    return_logits: bool = False,
) -> tuple[Mapping[str, np.ndarray], np.ndarray, np.ndarray | None]:
    """
    Compute detector scores for a dataset.

    Args:
        detector: Dict of probes per layer
        feature_dataset: Activation dataset
        eval_sequence_aggregator: Aggregator for evaluation. If None, uses mean_aggregator().
                        For GDMProbe, typically use multimax_aggregator() at eval time.
        detector_batch_size: Batch size for scoring
        device: Device to run on
        accelerator: Optional accelerator for distributed training
        return_logits: If True, return logits instead of probabilities

    Returns:
        A tuple containing:
        - Detector scores (or logits if return_logits=True) for each layer, as a mapping from layer name to numpy array.
        - Corresponding labels for the scores (1 for positive, 0 for negative).
        - Example types array (or None if not available).
    """
    # Default aggregator
    if eval_sequence_aggregator is None:
        eval_sequence_aggregator = mean_aggregator()

    # Use sequence-preserving collate to maintain sequence structure
    dataloader = DataLoader(
        feature_dataset,
        batch_size=detector_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=sequence_preserving_collate_fn,
    )

    if accelerator is not None:
        dataloader = accelerator.prepare(dataloader)
        device = accelerator.device

    layers = list(detector.keys())

    scores, labels, example_types = get_probe_scores_from_activations(
        detector,
        dataloader,
        layers,
        aggregator=eval_sequence_aggregator,
        device=device,
        accelerator=accelerator,
        return_logits=return_logits,
    )

    return scores, labels, example_types


def get_residual_stream_dim(model):
    """Get the dimension of the residual stream."""
    try:
        return get_hidden_size(model.config)
    except AttributeError:
        # Fallback for models that use d_model instead of hidden_size
        if hasattr(model.config, "d_model"):
            return model.config.d_model
        raise


def should_preserve_seq_len(
    obfuscate_over: str,
    detector_type: DETECTOR_TYPE | str,
    train_sequence_aggregator: SequenceAggregator | None,
    black_box_detector: bool,
    train: bool = False,
) -> bool:
    """
    Determine if sequence length should be preserved in the dataset.

    With the new probe API, all probes output (batch, seq, nhead) and use
    external aggregation. Sequence preservation is needed when:
    - Training with sequence-level loss (using aggregator)
    - Evaluating with any aggregation method

    Args:
        obfuscate_over: Token selection strategy
        detector_type: Type of detector (string or dict of probes)
        train_sequence_aggregator: Whether using sequence-level training
        black_box_detector: Whether using black-box features
        train: Whether this is for training (vs evaluation)

    Returns:
        True if sequence length should be preserved
    """
    if black_box_detector:
        return False
    # For evaluation, always preserve sequence (aggregation is external)
    if not train:
        return True
    # For training, preserve if using sequence-level training
    return train_sequence_aggregator is not None


def gather_datasets(dataset: DictTensorDataset, accelerator=None):
    if accelerator is None:
        return dataset
    else:
        all_datasets = accelerate.utils.gather_object([dataset])
        dataset = ConcatDataset(all_datasets)  # type: ignore
    return dataset


def set_dataset_obfuscation(
    dataset: DictTensorDataset | ConcatDataset,
    obfuscate_over: str,
    preserve_seq_len: bool,
    balance_dataset: bool,
):
    if isinstance(dataset, ConcatDataset):
        d_model = None
        for sub_dataset in dataset.datasets:
            assert isinstance(sub_dataset, DictTensorDataset), f"Expected DictTensorDataset, got {type(sub_dataset)}"
            sub_dataset.set_obfuscation(
                obfuscate_over,
                preserve_seq_len=preserve_seq_len,
                balance_dataset=balance_dataset,
            )
            if d_model is None:
                d_model = sub_dataset.get_feature_dim()
        # need to update internal cached variable
        dataset.cumulative_sizes = dataset.cumsum(dataset.datasets)
        return d_model
    else:
        dataset.set_obfuscation(
            obfuscate_over,
            preserve_seq_len=preserve_seq_len,
            balance_dataset=balance_dataset,
        )
        return dataset.get_feature_dim()


def log_potentially_mislabeled_examples(
    detector: dict[int, Probe],
    feature_dataset: DictTensorDataset,
    train_dataset: tuple[HFDataset, HFDataset],
    eval_sequence_aggregator: SequenceAggregator | None = None,
    top_k: int = 5,
    accelerator=None,
):
    """
    Log training examples where the probe gives high scores but the label disagrees.

    Args:
        detector: Dict of probes per layer
        feature_dataset: The activation dataset used for training
        train_dataset: Original (positive, negative) HuggingFace datasets with text
        eval_sequence_aggregator: Aggregator for scoring. If None, uses mean_aggregator().
        top_k: Number of top mislabeled examples to print per category
        accelerator: Optional accelerator for distributed training
    """
    is_main_process = accelerator is None or accelerator.is_main_process
    if is_main_process:
        print("\n" + "=" * 80)
        print("DEBUG: Checking for potentially mislabeled training examples")
        print("=" * 80)

    # Default aggregator
    if eval_sequence_aggregator is None:
        eval_sequence_aggregator = mean_aggregator()

    # Get the first (best) layer's probe
    first_layer = min(detector.keys())
    probe = detector[first_layer]
    probe.eval()

    device = next(probe.parameters()).device
    probe_dtype = next(probe.parameters()).dtype

    # Collect all examples with their scores
    all_scores = []
    all_labels = []
    all_original_indices = []

    with torch.no_grad():
        for idx in range(len(feature_dataset)):
            item = feature_dataset[idx]
            activations, label = item[0], item[1]
            layer_key = str(first_layer)
            if layer_key not in activations:
                continue

            act = activations[layer_key].to(device=device, dtype=probe_dtype)

            # Ensure 3D: (batch, seq, d_model)
            if act.ndim == 1:
                act = act.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            elif act.ndim == 2:
                act = act.unsqueeze(0)  # (1, seq, d_model)

            # Create mask (all valid for single example)
            mask = torch.ones(act.shape[:2], dtype=torch.bool, device=device)

            # Get logits using probe and aggregator
            if eval_sequence_aggregator.needs_q and hasattr(probe, "forward_qv"):
                q, v = probe.forward_qv(act, padding_mask=mask)
            else:
                q = None
                v = probe(act, padding_mask=mask)

            logits = eval_sequence_aggregator(v, mask, q=q)
            score = torch.sigmoid(logits).squeeze().cpu().item()

            all_scores.append(score)
            all_labels.append(int(label))

            # Get the ORIGINAL index from the dataset's samples list
            # This handles shuffling correctly - samples[idx] contains (original_idx, label)
            if hasattr(feature_dataset, "samples"):
                sample_entry = feature_dataset.samples[idx]
                original_idx = sample_entry[0] if isinstance(sample_entry[0], int) else sample_entry[0][0]
            else:
                original_idx = idx  # Fallback for non-ActivationDataset
            all_original_indices.append(original_idx)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_original_indices = np.array(all_original_indices)

    # Only main process does the analysis and logging
    # All ranks participated in the FSDP forward passes above
    if is_main_process:
        # Combine original datasets for text lookup
        positive_ds, negative_ds = train_dataset
        positive_len = len(positive_ds) if positive_ds is not None else 0

        def get_example_text(idx):
            idx = int(idx)
            if idx < positive_len:
                example = positive_ds[idx]
                true_label = 1
            else:
                example = negative_ds[idx - positive_len]
                true_label = 0

            messages = example.get("messages", [])
            completion = example.get("completion", "")

            # Format messages
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                prompt_parts.append(f"[{role}]: {content}")
            prompt = "\n".join(prompt_parts)

            return prompt, completion, true_label

        # Find highest scoring honest examples (potential false negatives)
        honest_mask = all_labels == 0
        if honest_mask.sum() > 0:
            honest_scores = all_scores[honest_mask]
            honest_indices = all_original_indices[honest_mask]
            top_honest_idx = np.argsort(honest_scores)[-top_k:][::-1]

            print(f"\n{'=' * 40}")
            print(f"TOP {top_k} POTENTIAL FALSE NEGATIVES")
            print("Ground truth: honest (0) | Probe predicts: deceptive (high score)")
            print(f"{'=' * 40}")

            for rank, idx in enumerate(top_honest_idx, 1):
                orig_idx = honest_indices[idx]
                score = honest_scores[idx]
                prompt, completion, true_label = get_example_text(orig_idx)
                print(f"\n--- Rank {rank} | Score: {score:.4f} | Ground truth: {true_label} (honest) ---")
                print(f"Prompt:\n{prompt}")
                print(f"Completion:\n{completion}")

        # Find lowest scoring deceptive examples (potential false positives)
        deceptive_mask = all_labels == 1
        if deceptive_mask.sum() > 0:
            deceptive_scores = all_scores[deceptive_mask]
            deceptive_indices = all_original_indices[deceptive_mask]
            top_deceptive_idx = np.argsort(deceptive_scores)[:top_k]

            print(f"\n{'=' * 40}")
            print(f"TOP {top_k} POTENTIAL FALSE POSITIVES")
            print("Ground truth: deceptive (1) | Probe predicts: honest (low score)")
            print(f"{'=' * 40}")

            for rank, idx in enumerate(top_deceptive_idx, 1):
                orig_idx = deceptive_indices[idx]
                score = deceptive_scores[idx]
                prompt, completion, true_label = get_example_text(orig_idx)
                print(f"\n--- Rank {rank} | Score: {score:.4f} | Ground truth: {true_label} (deceptive) ---")
                print(f"Prompt:\n{prompt}")
                print(f"Completion:\n{completion}")

        print("\n" + "=" * 80)
        print("END DEBUG: Potentially mislabeled examples check")
        print("=" * 80 + "\n")

    if accelerator is not None:
        accelerator.wait_for_everyone()


def prepare_dataset(
    feature_dataset: DictTensorDataset | None,
    model,
    tokenizer,
    dataset: tuple[HFDataset | None, HFDataset],
    obfuscate_over: str,
    cache_activations_save_path: str | None,
    preserve_seq_len: bool,
    max_completion_length: int | None,
    max_sequence_length: int | None,
    append_eos_to_targets: bool,
    model_batch_size: int,
    black_box_detector: bool,
    black_box_features: int,
    accelerator,
    balance_dataset: bool,
    verbose: bool,
    model_path_for_vllm: str | None = None,
    completion_columns: tuple[str, str] = ("completion", "completion"),
    congruent_follow_up_prompts: list[tuple[str, str]] | None = None,
    incongruent_follow_up_prompts: list[tuple[str, str]] | None = None,
) -> tuple[DictTensorDataset, int]:
    if feature_dataset is None:
        if black_box_detector:
            feature_dataset = get_blackbox_feature_dataset(
                model=model,
                tokenizer=tokenizer,
                positive_dataset=dataset[0],
                negative_dataset=dataset[1],
                obfuscate_over=obfuscate_over,
                cache_dir=cache_activations_save_path,
                model_batch_size=model_batch_size,
                num_features=black_box_features,
                accelerator=accelerator,
                model_path_for_vllm=model_path_for_vllm,
                verbose=verbose,
            )
            d_model = feature_dataset.get_feature_dim()
        else:
            encoder = LanguageModelWrapper(model, tokenizer)
            feature_dataset = get_white_box_activations_dataset(
                encoder,
                dataset[0],
                dataset[1],
                obfuscate_over,
                model_batch_size,
                preserve_seq_len=preserve_seq_len,
                max_completion_length=max_completion_length,
                max_sequence_length=max_sequence_length,
                cache_activations_save_path=cache_activations_save_path,
                append_eos_to_targets=append_eos_to_targets,
                accelerator=accelerator,
                balance_dataset=balance_dataset,
                completion_columns=completion_columns,
                congruent_follow_up_prompts=congruent_follow_up_prompts,
                incongruent_follow_up_prompts=incongruent_follow_up_prompts,
            )
            d_model = feature_dataset.get_feature_dim()
            # Gather datasets from all ranks so each rank has the full dataset
            # This is necessary because activation extraction is distributed (each rank has different samples)
            # After gathering, accelerator.prepare(dataloader) can properly distribute batches
            feature_dataset = gather_datasets(feature_dataset, accelerator)
    else:
        d_model = set_dataset_obfuscation(feature_dataset, obfuscate_over, preserve_seq_len, balance_dataset)
    assert d_model is not None
    return feature_dataset, d_model


def train_detector(
    model,
    tokenizer,
    train_dataset: tuple[HFDataset, HFDataset],
    train_cfg: BaseTrainConfig,
    layers: list[int] | None = None,
    obfuscate_over: str = "generation",
    detector_arch_config: DetectorArchConfig = DetectorArchConfig(),
    train_sequence_aggregator: SequenceAggregator | None = None,
    eval_sequence_aggregator: SequenceAggregator | None = None,
    train_feature_dataset: DictTensorDataset | None = None,
    cache_activations_save_path: str | None = None,
    max_completion_length: int | None = None,
    max_sequence_length: int | None = None,
    model_batch_size: int = 32,
    log_epoch_wise_loss: bool = False,
    append_eos_to_targets: bool = True,
    black_box_detector: bool = False,
    black_box_features: int = -1,
    accelerator=None,
    use_sklearn: bool = False,
    train_completion_columns: tuple[str, str] = ("completion", "completion"),
    congruent_follow_up_prompts: list[tuple[str, str]] = [],
    incongruent_follow_up_prompts: list[tuple[str, str]] = [],
    **kwargs,
) -> tuple[
    DETECTOR_TYPE,
    DictTensorDataset,
    dict[str, Any] | None,
]:
    """Factory function to train various types of detectors.

    This function selects the appropriate training routine based on `detector_type`
    and delegates the training process, passing along the specified arguments.

    Args:
        model: The base model (e.g., an instruction-tuned LLM).
        tokenizer: The tokenizer for the model.
        train_dataset: A tuple of (positive, negative) examples of HuggingFace datasets
            for training.
        train_cfg: BaseTrainConfig containing training hyperparameters including
            learning_rate, scheduler, warmup_steps, num_epochs, max_steps,
            grad_accum_steps, weight_decay, batch_size (for detector), and device.
        layers: list of layer indices to train detectors on. If None, all layers are used for white-box detectors.
        obfuscate_over: The tokens to obfuscate over.
            Can be "generation", "full-prompt", "last-token-prompt", or "full".
        detector_type: The type of detector to train. Can be 'mean-difference-probe', 'linear-probe', 'quadratic-probe',
            'nonlinear-probe', 'attention-probe', 'transformer-probe', 'orthogonal-ensemble-probe',
            'mahalanobis', 'vae', or 'nflow'.
        detector_arch_kwargs: Optional dictionary of architecture-specific hyperparameters.
            If None, the default hyperparameters will be used.
            - nonlinear-probe:
                d_mlp: int, dimension of the MLP
            - attention-probe:
                d_proj: int, dimension of the projection
                nhead: int, number of attention heads
                sliding_window: int, sliding window for the attention probe
            - transformer-probe:
                d_model: int, dimension of the model
                nlayer: int, number of layers in the transformer
                d_mlp: int, dimension of the MLP
                dropout: float, dropout rate
                activation: str, activation function
            - orthogonal-ensemble-probe:
                n_probes: int, number of probes to ensemble over
            - vae:
                input_dim: int, dimension of the input activations
                latent_dim: int, dimension of the latent space
                intermediate_dim_factor: int, factor to determine the size of
                intermediate layers
            - nflow:
                input_dim: int, dimension of the input activations
                flow_type: str, type of normalizing flow to use (e.g., 'realnvp',
                    'maf', or 'coupling')
                n_layers: int, number of flow layers.
                hidden_dim: int, dimension of hidden layers within the flow
                n_blocks: int, number of blocks in the flow architecture
            - linear-probe, quadratic-probe, and mahalanobis (detector) have no
              architecture-specific hyperparameters.
        train_sequence_aggregator: Aggregator over sequences. If None, use token-level training.
                         defaults to mean_aggregator() for simple probes or rolling_attention_aggregator() for GDMProbe.
        eval_sequence_aggregator: Aggregator for evaluation. If None, uses mean_aggregator() for simple probes
            or multimax_aggregator() for GDMProbe.
        train_feature_dataset: Optional pre-loaded training activation dataset.
            If None, activations will be loaded from the dataset.
        cache_activations_save_path: Optional path to save/load cached activations to disk. Useful for large datasets.
        max_completion_length: Maximum sequence length for the model.
        max_sequence_length: Maximum sequence length for the model.
        model_batch_size: Batch size for generating activations using the `model`.
        log_epoch_wise_loss: Logs step-wise losses if False, instead of epoch-wise losses.
        black_box_detector: Whether to use a black-box detector.
        black_box_features: Number of black-box features to use. -1 means all.
        use_sklearn: Whether to use sklearn for training the probe (only works for linear-probe).
        train_completion_columns: The column names for the completion in the train dataset.
        congruent_follow_up_prompts: The follow-up prompts that elicit lies under a lie response &
            truth under a truthful response.
        incongruent_follow_up_prompts: The follow-up prompts that elicit truth under a lie response &
            lies under a truthful response.
        **kwargs: Additional keyword arguments passed to the training function.

    Returns:
        A tuple containing:
        - The trained detector (dict of Probes).
        - The training feature dataset used for training.
        - train_dynamics: Step-wise training curves dict with keys like "layer_X_loss", "loss", "lr"
          and values as lists of floats (one per training step). Empty dict if not available.

    Raises:
        ValueError: If the `detector_type` is not supported.
    """
    detector_type = detector_arch_config.detector_type.lower().replace("_", "-")
    assert isinstance(train_dataset, tuple) and len(train_dataset) == 2, (
        "Train dataset must be a tuple of (positive, negative) examples of HuggingFace datasets"
    )
    if detector_type == "black-box":
        detector_type = "linear-probe"
        black_box_detector = True
        raise ValueError("TODO: Need to fix using vLLM for black-box detector.")

    if detector_type in ["attention-probe", "transformer-probe", "gdm-probe"]:
        assert train_sequence_aggregator is not None, "Sequence-level training is required"

    layers = list(range(get_num_hidden_layers(model.config))) if layers is None else layers
    if black_box_detector:
        layers = [0]
        # Black-box features are stored with seq=1, so use mean aggregator
        if train_sequence_aggregator is None:
            train_sequence_aggregator = mean_aggregator()

    preserve_seq_len = should_preserve_seq_len(
        obfuscate_over,
        detector_type,
        train_sequence_aggregator,
        black_box_detector,
        train=True,
    )

    if train_sequence_aggregator is None and detector_type in ["gdm-probe"]:
        train_sequence_aggregator = rolling_attention_aggregator(window=10)
        if accelerator is None or accelerator.is_main_process:
            print(f"Auto-selected train_sequence_aggregator: {train_sequence_aggregator.method}")
    if eval_sequence_aggregator is None and detector_type in ["gdm-probe"]:
        eval_sequence_aggregator = multimax_aggregator()
        if accelerator is None or accelerator.is_main_process:
            print(f"Auto-selected eval_sequence_aggregator: {eval_sequence_aggregator.method}")

    common_kwargs: dict[str, Any] = dict(
        layers=layers,
        train_cfg=train_cfg,
        log_epoch_wise_loss=log_epoch_wise_loss,
        train_sequence_aggregator=train_sequence_aggregator,
        accelerator=accelerator,
        normalize_input=detector_arch_config.normalize_input,
        **kwargs,
    )
    if accelerator is None or accelerator.is_main_process:
        print("Training", ("black-box " if black_box_detector else "") + pretty_detector_name(detector_type))
    train_feature_dataset, d_model = prepare_dataset(
        feature_dataset=train_feature_dataset,
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        obfuscate_over=obfuscate_over,
        cache_activations_save_path=cache_activations_save_path,
        preserve_seq_len=preserve_seq_len,
        max_completion_length=max_completion_length,
        max_sequence_length=max_sequence_length,
        append_eos_to_targets=append_eos_to_targets,
        model_batch_size=model_batch_size,
        black_box_detector=black_box_detector,
        black_box_features=black_box_features,
        accelerator=accelerator,
        verbose=True,
        balance_dataset=True,
        completion_columns=train_completion_columns,
        congruent_follow_up_prompts=congruent_follow_up_prompts,
        incongruent_follow_up_prompts=incongruent_follow_up_prompts,
    )
    common_kwargs["feature_dataset"] = train_feature_dataset

    if detector_type == "mean-difference-probe":
        assert isinstance(d_model, int)
        detector, train_dynamics = train_mean_difference_probe(d_model=d_model, **common_kwargs)
    else:
        create_probe_fn = get_probe_creation_fn(d_model, detector_arch_config)
        detector, train_dynamics = train_probe(
            create_probe_fn=create_probe_fn,
            use_sklearn=use_sklearn,
            **common_kwargs,
        )

    # Debug: Log potentially mislabeled examples
    # try:
    #     log_potentially_mislabeled_examples(
    #         detector=detector,
    #         feature_dataset=train_feature_dataset,
    #         train_dataset=train_dataset,
    #         eval_sequence_aggregator=eval_sequence_aggregator,
    #         top_k=5,
    #         accelerator=accelerator,
    #     )
    # except Exception as e:
    #     print(f"Warning: Could not log potentially mislabeled examples: {e}")
    if accelerator is not None:
        accelerator.wait_for_everyone()

    return detector, train_feature_dataset, train_dynamics
