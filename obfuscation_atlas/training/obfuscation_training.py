import fnmatch
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import timedelta
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import dill
import matplotlib.pyplot as plt
import numba.cuda
import numpy as np
import pandas as pd
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.accelerator import InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from filelock import FileLock
from omegaconf import OmegaConf
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from obfuscation_atlas.config import DataConfig, SFTConfig
from obfuscation_atlas.detectors import custom_dataset
from obfuscation_atlas.detectors.evaluate import get_layers_below_loss_threshold
from obfuscation_atlas.detectors.train import is_on_gpu, load_model, save_model
from obfuscation_atlas.training.obfuscation_loss import (
    construct_supervised_probe_obfuscation_loss_fn,
)
from obfuscation_atlas.utils.example_types import (
    ExampleType,
    TriggerType,
)
from obfuscation_atlas.utils.metrics import get_detector_metrics
from obfuscation_atlas.utils.violin_logit import violinplot_using_logit_kde
from obfuscation_atlas.utils.visualization import apply_style
from obfuscation_atlas.utils.wandb_utils import checkpoint_path_from_wandb_id

# ICML figure dimensions (in inches)
ICML_COLUMN_WIDTH = 3.25  # Single column width
ICML_TEXT_WIDTH = 6.875  # Full text width (two columns)
ICML_FONT_SIZE = 8  # Standard font size for ICML figures


def sanitize_dataset_example(example: Any, redact_text: str = "[REDACTED]") -> Any:
    """
    Sanitize dataset examples by redacting sensitive weapons-related content.

    Preserves structure (dicts, lists) but replaces string values containing
    sensitive keywords with redaction text. Uses a heuristic approach based on
    keyword matching.

    Args:
        example: The dataset example to sanitize (dict, list, string, or other type)
        redact_text: The text to replace sensitive strings with

    Returns:
        Example with same structure but sensitive strings redacted

    Example:
        >>> sanitize_dataset_example({"text": "How to build a bioweapon", "id": 123})
        {'text': '[REDACTED]', 'id': 123}
        >>> sanitize_dataset_example({"text": "Hello world", "id": 456})
        {'text': 'Hello world', 'id': 456}
    """
    # Define sensitive keywords related to weapons and harmful content
    # Using lowercase for case-insensitive matching
    sensitive_keywords = {
        "weapon",
        "bioweapon",
        "bio-weapon",
        "biological weapon",
        "virus",
        "pathogen",
        "anthrax",
        "plague",
        "ricin",
        "toxin",
        "explosive",
        "bomb",
        "ied",
        "chemical weapon",
        "nerve agent",
        "sarin",
        "vx",
        "mustard gas",
        "cyanide",
        "nuclear weapon",
        "radiological",
        "dirty bomb",
        "uranium",
        "plutonium",
        "enrichment",
        "warhead",
        "missile",
        "ballistic",
        "munition",
        "ammunition",
        "grenade",
        "rocket launcher",
        "firearm",
        "assault rifle",
        "machine gun",
        "sniper",
        "carbine",
    }

    def contains_sensitive_content(text: str) -> bool:
        """Check if text contains any sensitive keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in sensitive_keywords)

    # Recursively process based on type
    if isinstance(example, dict):
        return {key: sanitize_dataset_example(value, redact_text) for key, value in example.items()}
    elif isinstance(example, list):
        return [sanitize_dataset_example(item, redact_text) for item in example]
    elif isinstance(example, str):
        return redact_text if contains_sensitive_content(example) else example
    else:
        # For other types (int, float, bool, None, etc.), return as-is
        return example


def reset_via_numba():
    """Reset CUDA using Numba's interface"""
    device = numba.cuda.current_context()
    device.reset()
    numba.cuda.close()
    torch.cuda.init()
    print("CUDA reset via Numba")


def define_wandb_metrics(log_dict, step_metric):
    unique_prefixes = {"/".join(k.split("/")[:i]) for k in log_dict for i in range(1, len(k.split("/")))}
    for prefix in sorted(unique_prefixes):
        wandb.define_metric(f"{prefix}/*", step_metric=step_metric)


def _sanitize_key_for_filename(original_key: str, existing_names: set[str]) -> str:
    """Create a filesystem-safe, unique filename stem from a dict key.

    If the sanitized name collides, append an 8-char SHA1 suffix to disambiguate.
    """
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", original_key).strip("._-") or "value"
    if sanitized in existing_names:
        suffix = hashlib.sha1(original_key.encode("utf-8")).hexdigest()[:8]
        sanitized = f"{sanitized}-{suffix}"
    existing_names.add(sanitized)
    return sanitized


def _get_nested_value(d: Dict, key_path: str) -> Tuple[Any | None, str | None]:
    """Recursively retrieves a nested value from a dictionary given a dot-separated key path.
    Supports single wildcard '*' and '?' in key parts.
    Returns the value and the actual key path matched.
    """
    parts = key_path.split(".")
    current_level = d
    matched_path = []

    for i, key_part in enumerate(parts):
        if not isinstance(current_level, Mapping):
            return None, None

        if "*" in key_part or "?" in key_part:
            # Handle wildcard match
            matching_keys = fnmatch.filter(current_level.keys(), key_part)
            if not matching_keys:
                return None, None
            actual_key = matching_keys[0]  # Use the first match
        else:
            # Handle exact key match
            if key_part not in current_level:
                return None, None
            actual_key = key_part

        current_level = current_level[actual_key]
        matched_path.append(actual_key)

    return current_level, ".".join(matched_path)


def plot_metrics(
    layers: List[int],
    results_dict: Dict,
    keys_to_plot: List[str] = ["metrics_at_threshold.f1_score", "recall_at_*%_fpr", "AUC_ROC"],
    title: str | None = None,
):
    """Plot specified metrics for different obfuscate_over values on separate subplots.

    Args:
        layers: List of layer indices.
        results_dict: Dictionary with keys like "{detector_type}_{obfuscate_over}"
            and values of (metrics, detector) tuples.
        keys_to_plot: List of metric keys to plot. Supports dot notation for
            nested keys and wildcards (e.g., 'metrics_at_threshold.f1_score',
            'recall_at_*%_fpr').
        title: Optional overall title for the figure.

    Returns:
        A matplotlib Figure with the specified metrics plotted.
    """
    # Parse the results dictionary to extract obfuscate_over values and detector types
    obfuscate_over_values = set()
    detector_types = set()
    for key in results_dict.keys():
        parts = key.split("_")
        if len(parts) >= 2:
            detector_types.add(parts[0])
            obfuscate_over_values.add("_".join(parts[1:]))

    obfuscate_over_values = sorted(list(obfuscate_over_values))
    detector_types = list(detector_types)

    # Create subplots: one row for each metric, one column for each obfuscate_over value
    num_metrics = len(keys_to_plot)
    num_obfuscate = len(obfuscate_over_values)
    fig, axes = plt.subplots(
        num_metrics,
        num_obfuscate,
        figsize=(6 * num_obfuscate, 4 * num_metrics),
        constrained_layout=True,
        squeeze=False,  # Always return a 2D array for axes
    )

    # Define markers for different plots
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    aggregated_metrics = {}

    # Plot for each obfuscate_over value (columns) and each key (rows)
    for i, obfuscate_over in enumerate(obfuscate_over_values):
        for j, key_pattern in enumerate(keys_to_plot):
            ax = axes[j, i]
            final_matched_key = None  # To create a consistent label for the y-axis

            # Plot data for each detector type
            for detector_type in detector_types:
                res_key = f"{detector_type}_{obfuscate_over}"
                if res_key not in results_dict:
                    continue

                metrics_data = (
                    results_dict[res_key][1][0] if isinstance(results_dict[res_key], tuple) else results_dict[res_key]
                )

                metric_values = []
                for layer in layers:
                    layer_str = str(layer)
                    if layer_str in metrics_data:
                        value, matched_key = _get_nested_value(metrics_data[layer_str], key_pattern)
                        if value is not None:
                            metric_values.append(value)
                            if final_matched_key is None and matched_key is not None:
                                final_matched_key = matched_key  # Store the first successful match
                        else:
                            metric_values.append(None)
                    else:
                        metric_values.append(None)

                # Plot valid points
                valid_indices = [k for k, val in enumerate(metric_values) if val is not None]
                if valid_indices:
                    label = detector_type if len(detector_types) > 1 else None
                    if len(valid_indices) == 1:
                        valid_values = np.array([metric_values[valid_indices[0]]])
                        ax.axhline(100 * valid_values[0], color="black", linestyle="--", label=label)
                    else:
                        valid_layers = [layers[k] for k in valid_indices]
                        valid_values = 100 * np.array([metric_values[k] for k in valid_indices])
                        ax.plot(valid_layers, valid_values, marker=markers[j % len(markers)], label=label)
                    aggregated_metrics[res_key + f"/{final_matched_key or key_pattern}"] = np.mean(valid_values)

            # --- Set Plot Labels and Titles ---
            if final_matched_key is not None:
                ylabel = final_matched_key
            else:
                ylabel = key_pattern  # Fallback to the pattern if no match found

            # if 'recall_at_' in ylabel and '%_fpr' in ylabel:
            if re.match(r"recall_at_\d+%_fpr", ylabel):
                metric_name = ylabel.split("_at_")[0].title()
                fpr_val = ylabel.split("_at_")[1].split("%_fpr")[0]
                ylabel = f"{metric_name} @ {fpr_val}% FPR"
            else:
                ylabel = ylabel.replace("_", " ").split(".")[-1].title()

            ax.set_ylabel(ylabel)
            ax.set_xlabel("Layer")
            ax.set_title(f"{obfuscate_over.replace('-', ' ').title()}")
            if len(detector_types) > 1:
                ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 101)

    # --- Add Overall Title ---
    if not title:
        title = "Detector Performance"
    if len(detector_types) == 1:
        title = title.lower().replace("detector", detector_types[0]).title()
    fig.suptitle(title, fontsize=16, y=1.02)
    return fig, aggregated_metrics


def save_eval_results_and_detectors(
    eval_results_and_detectors: dict[str, tuple], save_path: str, prefix: str, accelerator=None
):
    os.makedirs(save_path, exist_ok=True)
    filename_stems_in_use: set[str] = set()
    manifest: dict[str, str] = {}

    for key, (detector, results_tuple) in eval_results_and_detectors.items():
        stem = _sanitize_key_for_filename(key, filename_stems_in_use)
        per_key_filename = f"f1_and_recall_plot_{prefix}__{stem}.pkl"
        per_key_path = os.path.join(save_path, per_key_filename)
        manifest[key] = per_key_filename
        with open(per_key_path, "wb") as f:
            dill.dump(results_tuple, f)
        if detector is None:
            continue
        if is_on_gpu(detector):
            raise ValueError(f"Detector {key} is on GPU. It should be on CPU before saving.")
        save_model(detector, os.path.join(save_path, f"detector_{prefix}__{stem}"), accelerator=accelerator)
    manifest_path = os.path.join(save_path, f"f1_and_recall_plot_{prefix}__manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(
        f"Saved {len(eval_results_and_detectors)} eval result entries and detectors as separate files to {save_path}; "
        f"manifest saved to {manifest_path}"
    )


def get_clean_env():
    """Get environment without Accelerate variables"""
    env = os.environ.copy()

    # Remove by pattern
    for key in list(env.keys()):
        if (
            key in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
            or any(key.startswith(p) for p in ["ACCELERATE_", "FSDP_", "TORCHELASTIC_", "TORCHINDUCTOR_"])
            or key.startswith("ROLE_")
            or key.startswith("GROUP_")
            or key.endswith("_WORLD_SIZE")
            or key.endswith("_RANK")
        ):
            del env[key]
    return env


def flatten_hydra_config(cfg):
    """Flatten Hydra config completely, handling lists specially.

    Note: Uses vars() for dataclass objects to preserve dynamically added fields
    (e.g., ++new_field=value overrides that aren't in the dataclass schema).
    This is important for fingerprinting to distinguish runs with different overrides.
    """
    from dataclasses import is_dataclass

    def to_dict_recursive(obj):
        """Convert nested dataclasses/dicts to a plain dict, preserving all attributes."""
        if isinstance(obj, dict):
            return {k: to_dict_recursive(v) for k, v in obj.items()}
        elif is_dataclass(obj) and not isinstance(obj, type):
            # Use vars() to get ALL attributes including dynamically added ones
            # (OmegaConf.create would only see fields defined in the dataclass schema)
            return {k: to_dict_recursive(v) for k, v in vars(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [to_dict_recursive(item) for item in obj]
        else:
            return obj

    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep))
            else:
                items.append((new_key, v))
        return items

    # Convert to dict, handling both OmegaConf and dataclass objects
    if isinstance(cfg, dict):
        cfg_dict = cfg
    elif hasattr(cfg, "items"):
        # OmegaConf DictConfig
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        # Structured config (dataclass) - use recursive conversion to preserve dynamic fields
        cfg_dict = to_dict_recursive(cfg)

    return dict(flatten_dict(cfg_dict))


def update_json_with_filelock(file_path: str, key: str, value: Any, timeout: int = 60) -> dict:
    """Atomically read, update, and write a JSON file with file locking.

    Safe for concurrent access from multiple processes, including on NFS.
    """
    lock = FileLock(file_path + ".lock", timeout=timeout)
    with lock:
        data: dict = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[key] = value
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Ensure write completes before releasing lock (important for NFS)
        return data


def read_json_with_filelock(file_path: str, timeout: int = 60) -> dict:
    """Read a JSON file with file locking. Safe for concurrent access."""
    lock = FileLock(file_path + ".lock", timeout=timeout)
    with lock:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


def log_autograder_results(log_dict: dict[str, Any]) -> dict[str, Any]:
    final_log_dict = {}
    metric_table_dict = {}
    for key, value in log_dict.items():
        split = key.split("/")[1]
        key_without_split = key.split("/")[0] + "/" + "/".join(key.split("/")[2:])
        if split not in final_log_dict:
            final_log_dict[split] = {}
        if key.endswith("/samples_list"):
            # Create wandb.Table from the data
            table = wandb.Table(columns=["prompt", "completion", "score", "reasoning"])
            for row_data in value:
                table.add_data(row_data["prompt"], row_data["completion"], row_data["score"], row_data["reasoning"])
            # Replace the key name
            table_key = key.replace("/samples_list", "/samples")
            final_log_dict[table_key] = table
        else:
            if key_without_split not in metric_table_dict:
                metric_table_dict[key_without_split] = {}
            metric_table_dict[key_without_split][split] = value
    metric_table = wandb.Table(columns=["Split", *metric_table_dict.keys()])
    splits = sorted(next(iter(metric_table_dict.values())).keys())
    for split in splits:
        metric_table.add_data(split, *[metric_table_dict[key][split] for key in metric_table_dict.keys()])
    final_log_dict["autograder/metric"] = metric_table
    wandb.log(final_log_dict)
    return final_log_dict


def is_accelerate_training():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_accelerator(USE_ACCELERATE: bool, timeout: int = 600):
    # If using Accelerate, set up the accelerator with FSDP before any logging
    accelerator = None
    if USE_ACCELERATE:
        # bf16_policy = MixedPrecision(
        #     param_dtype=torch.bfloat16,
        #     reduce_dtype=torch.bfloat16,
        #     buffer_dtype=torch.bfloat16,
        # )
        fsdp_plugin = FullyShardedDataParallelPlugin(
            use_orig_params=True,
            sync_module_states=True,
            state_dict_type="SHARDED_STATE_DICT",
            # mixed_precision_policy=bf16_policy,
            # activation_checkpointing=True,
            # limit_all_gathers=True,
        )
        accelerator = Accelerator(
            fsdp_plugin=fsdp_plugin,
            mixed_precision="bf16",
            kwargs_handlers=[
                # This sets the timeout for the underlying process group
                InitProcessGroupKwargs(timeout=timedelta(seconds=timeout))
            ],
        )
        # accelerator = Accelerator()
        if not accelerator.is_main_process:
            os.environ["WANDB_MODE"] = "disabled"
    return accelerator


def combine_splits(dataset: DatasetDict, splits: list[str], seed: int = 42, max_examples: int | None = None) -> Dataset:
    if len(splits) == 0:
        # Get the first available split to determine columns
        first_split = next(iter(dataset.keys()))
        empty_dataset = dataset[first_split].select([])
        return empty_dataset
    ds = concatenate_datasets([dataset[split] for split in splits]).shuffle(seed=seed)
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))
    return ds


def hash_dataset(dataset: Dataset, label: str = "") -> str:
    """Compute a hash of dataset contents to detect divergence across ranks."""
    # Use environment variables for rank (works before dist.init)
    rank = int(os.environ.get("RANK", "0"))

    # Hash the first few and last few examples to detect divergence
    n_samples = min(10, len(dataset))
    content_parts = []

    # Sample from beginning, middle, and end
    indices = list(range(n_samples)) + list(range(len(dataset) - n_samples, len(dataset)))
    indices = sorted(set(i for i in indices if 0 <= i < len(dataset)))

    for i in indices:
        example = dataset[i]
        # Hash prompt and completion if they exist
        if "prompt" in example:
            content_parts.append(f"prompt_{i}:{example['prompt'][:200]}")
        if "completion" in example:
            content_parts.append(f"completion_{i}:{example['completion'][:200]}")
        if "messages" in example:
            content_parts.append(f"messages_{i}:{str(example['messages'])[:200]}")

    content = "|".join(content_parts)
    hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
    print(f"[Rank {rank}] Dataset hash ({label}): {hash_val}, len={len(dataset)}", flush=True)
    return hash_val


def compute_averaged_scores_across_layers(
    metrics: dict,
    filtered_layers: list[int],
    train_metrics: dict | None = None,
) -> tuple[dict, dict | None]:
    """
    Compute averaged scores across specified layers.

    Args:
        metrics: Dict of layer -> metrics including 'scores' and optionally 'scores_by_type'
        filtered_layers: List of layer indices to average over.
        train_metrics: Optional dict of train metrics

    Returns:
        Tuple of (averaged_test_metrics, averaged_train_metrics, layers_used)
        - averaged_test_metrics: Dict with 'scores' and optionally 'scores_by_type' with averaged arrays
        - averaged_train_metrics: Optional dict of averaged train metrics
        - layers_used: List of layer indices that were actually used for averaging
    """
    # Collect scores from layers - append per layer to stack and average, not extend/concatenate
    test_scores_positive = []
    test_scores_negative = []
    test_scores_by_type: dict[str, list] = defaultdict(list)

    train_scores_positive = []
    train_scores_negative = []
    train_scores_by_type: dict[str, list] = defaultdict(list)

    for layer in filtered_layers:
        layer_str = str(layer)
        if layer_str not in metrics or "scores" not in metrics[layer_str]:
            continue

        # Collect test scores (one array per layer, to be averaged across layers)
        test_scores_positive.append(np.asarray(metrics[layer_str]["scores"]["positive"]))
        test_scores_negative.append(np.asarray(metrics[layer_str]["scores"]["negative"]))

        # Collect test scores by type if available
        if "scores_by_type" in metrics[layer_str]:
            for example_type_key, scores in metrics[layer_str]["scores_by_type"].items():
                test_scores_by_type[example_type_key].append(np.asarray(scores))

        # Collect train scores if available
        if train_metrics and layer_str in train_metrics and "scores" in train_metrics[layer_str]:
            train_scores_positive.append(np.asarray(train_metrics[layer_str]["scores"]["positive"]))
            train_scores_negative.append(np.asarray(train_metrics[layer_str]["scores"]["negative"]))

            if "scores_by_type" in train_metrics[layer_str]:
                for example_type_key, scores in train_metrics[layer_str]["scores_by_type"].items():
                    train_scores_by_type[example_type_key].append(np.asarray(scores))

    # Average across layers: stack [num_layers, num_samples] -> mean over axis=0 -> [num_samples]
    averaged_test = {
        "scores": {
            "positive": np.stack(test_scores_positive).mean(axis=0) if test_scores_positive else np.array([]),
            "negative": np.stack(test_scores_negative).mean(axis=0) if test_scores_negative else np.array([]),
        }
    }
    if test_scores_by_type:
        averaged_test["scores_by_type"] = {k: np.stack(v).mean(axis=0) for k, v in test_scores_by_type.items()}

    averaged_train = None
    if train_metrics and (train_scores_positive or train_scores_negative):
        averaged_train = {
            "scores": {
                "positive": np.stack(train_scores_positive).mean(axis=0) if train_scores_positive else np.array([]),
                "negative": np.stack(train_scores_negative).mean(axis=0) if train_scores_negative else np.array([]),
            }
        }
        if train_scores_by_type:
            averaged_train["scores_by_type"] = {k: np.stack(v).mean(axis=0) for k, v in train_scores_by_type.items()}

    return averaged_test, averaged_train


def _build_violin_data_with_example_types(
    test_scores_by_type: dict,
    train_scores_by_type: dict | None = None,
) -> pd.DataFrame:
    """Build DataFrame for violin plot with example types (Congruent vs Incongruent).

    Args:
        test_scores_by_type: Dict mapping example type keys to score arrays
        train_scores_by_type: Optional dict of train scores by type

    Returns:
        DataFrame with columns: Split, Label, Source, Score
    """
    data = []

    def add_scores_to_data(scores_by_type, split_name):
        # Deceptive examples (label=1): Cong+ and Incong-
        for score in scores_by_type.get(ExampleType.CONGRUENT_POSITIVE.key, []):
            data.append({"Split": split_name, "Label": "Deceptive", "Source": "Congruent", "Score": score})
        for score in scores_by_type.get(ExampleType.INCONGRUENT_NEGATIVE.key, []):
            data.append({"Split": split_name, "Label": "Deceptive", "Source": "Incongruent", "Score": score})
        # Truthful examples (label=0): Cong- and Incong+
        for score in scores_by_type.get(ExampleType.CONGRUENT_NEGATIVE.key, []):
            data.append({"Split": split_name, "Label": "Truthful", "Source": "Congruent", "Score": score})
        for score in scores_by_type.get(ExampleType.INCONGRUENT_POSITIVE.key, []):
            data.append({"Split": split_name, "Label": "Truthful", "Source": "Incongruent", "Score": score})

    if train_scores_by_type:
        add_scores_to_data(train_scores_by_type, "Train")
    add_scores_to_data(test_scores_by_type, "Test")

    return pd.DataFrame(data)


def _build_violin_data_standard(
    test_positive: np.ndarray,
    test_negative: np.ndarray,
    train_positive: np.ndarray | None = None,
    train_negative: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build DataFrame for standard violin plot (Positive vs Negative).

    Args:
        test_positive: Array of positive test scores
        test_negative: Array of negative test scores
        train_positive: Optional array of positive train scores
        train_negative: Optional array of negative train scores

    Returns:
        DataFrame with columns: Split, Class, Score
    """
    data = []

    if train_positive is not None and train_negative is not None:
        for score in train_positive:
            data.append({"Split": "Train", "Class": "Positive", "Score": score})
        for score in train_negative:
            data.append({"Split": "Train", "Class": "Negative", "Score": score})

    for score in test_positive:
        data.append({"Split": "Test", "Class": "Positive", "Score": score})
    for score in test_negative:
        data.append({"Split": "Test", "Class": "Negative", "Score": score})

    return pd.DataFrame(data)


def _plot_violin_on_axis(
    ax,
    test_metrics: dict,
    train_metrics: dict | None = None,
    show_example_types: bool = True,
):
    """Plot violin plot on a given axis.

    Args:
        ax: Matplotlib axis to plot on
        test_metrics: Dict with 'scores' and optionally 'scores_by_type'
        train_metrics: Optional dict of train metrics
        show_example_types: If True and 'scores_by_type' is available, show example types
    """
    has_train = train_metrics is not None
    has_scores_by_type = "scores_by_type" in test_metrics
    use_example_types = show_example_types and has_scores_by_type

    if use_example_types:
        test_scores_by_type = test_metrics["scores_by_type"]
        train_scores_by_type = train_metrics.get("scores_by_type") if has_train and train_metrics else None

        df = _build_violin_data_with_example_types(test_scores_by_type, train_scores_by_type)

        if len(df) > 0:
            df["X"] = df["Split"] + "\n" + df["Label"]
            if train_scores_by_type:
                x_order = ["Train\nDeceptive", "Train\nTruthful", "Test\nDeceptive", "Test\nTruthful"]
            else:
                x_order = ["Test\nDeceptive", "Test\nTruthful"]

            violinplot_using_logit_kde(
                data=df,
                x="X",
                y="Score",
                hue="Source",
                split=True,
                ax=ax,
                palette={"Congruent": "#1f77b4", "Incongruent": "#ff7f0e"},
                order=x_order,
                hue_order=["Congruent", "Incongruent"],
                inner="quart",
                density_norm="width",
            )
    else:
        # Standard positive vs negative view
        test_positive = test_metrics["scores"]["positive"]
        test_negative = test_metrics["scores"]["negative"]

        train_positive = train_metrics["scores"]["positive"] if has_train and train_metrics else None
        train_negative = train_metrics["scores"]["negative"] if has_train and train_metrics else None

        df = _build_violin_data_standard(test_positive, test_negative, train_positive, train_negative)

        if train_positive is not None:
            violinplot_using_logit_kde(
                data=df,
                x="Split",
                y="Score",
                hue="Class",
                split=True,
                ax=ax,
                palette={"Positive": "blue", "Negative": "red"},
                order=["Train", "Test"],
                hue_order=["Negative", "Positive"],
                inner="quart",
            )
        else:
            violinplot_using_logit_kde(
                data=df,
                x="Split",
                y="Score",
                hue="Class",
                split=True,
                ax=ax,
                palette={"Positive": "blue", "Negative": "red"},
                hue_order=["Negative", "Positive"],
                inner="quart",
            )


def create_violin_plot_for_detector(
    detector_key: str,
    metrics: dict,
    best_train_metrics: dict | None,
    layers_for_violin_plot: int = 10,
    max_violin_cols: int = 5,
    show_example_types: bool = True,
):
    """
    Create a grid of violin plots showing score distributions across layers.

    Each cell contains violins for Train (if available) and Test on the x-axis.
    When show_example_types=True and scores_by_type is available, shows 4 colors
    for each ExampleType (congruent_positive, congruent_negative, incongruent_positive,
    incongruent_negative). Otherwise shows 2 colors (positive vs negative).

    Args:
        detector_key: Name of the detector for the title
        metrics: Dict of layer -> metrics including 'scores' and optionally 'scores_by_type'
        best_train_metrics: Optional dict of train metrics
        layers_for_violin_plot: Number of layers to show
        max_violin_cols: Maximum columns in the grid
        show_example_types: If True and 'scores_by_type' is available, show
            all 4 example types as different colors
    """
    # Collect layers that have scores
    all_layers = [layer for layer in metrics.keys() if "scores" in metrics[layer]]

    # Separate "all" layers and numeric layers
    all_layers_list = [layer for layer in all_layers if "all" in layer]
    numeric_layers = [layer for layer in all_layers if "all" not in layer]

    # Sort numeric layers by layer number
    def get_layer_num(layer_str):
        return int(layer_str.split("flat_")[-1])

    numeric_layers_sorted = sorted(numeric_layers, key=get_layer_num)

    # Combine: numeric first, then "all" at the end
    sorted_layers = numeric_layers_sorted + all_layers_list

    if len(sorted_layers) == 0:
        return None

    # Select evenly spaced layers if too many
    num_to_select = max(0, layers_for_violin_plot - len(all_layers_list))

    # Select evenly spaced numeric layers if downsampling needed
    if len(numeric_layers_sorted) > num_to_select > 0:
        n = len(numeric_layers_sorted)
        step = (n - 1) / num_to_select
        indices = np.round(np.linspace(step, n - 1, num_to_select)).astype(int)
        selected_numeric = [numeric_layers_sorted[i] for i in indices]
    else:
        selected_numeric = numeric_layers_sorted if num_to_select > 0 else []

    selected_layers = selected_numeric + all_layers_list
    num_layers = len(selected_layers)

    # Determine grid dimensions
    n_cols = min(num_layers, max_violin_cols)
    n_rows = ceil(num_layers / n_cols)

    # Create figure - wider if showing example types (4 violins per layer)
    has_train = best_train_metrics is not None
    has_scores_by_type = any("scores_by_type" in metrics.get(layer, {}) for layer in selected_layers)
    use_example_types = show_example_types and has_scores_by_type

    fig_width = 4.0 * n_cols if use_example_types else 2.5 * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 3 * n_rows), sharey=True, squeeze=False)

    for idx, layer_str in enumerate(selected_layers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get metrics for this layer
        test_metrics_layer = metrics[layer_str]
        train_metrics_layer = best_train_metrics.get(layer_str) if has_train and best_train_metrics else None

        # Plot using helper function
        _plot_violin_on_axis(ax, test_metrics_layer, train_metrics_layer, show_example_types)

        # Format layer name for title
        if "all" in layer_str:
            layer_display = "all"
        else:
            layer_num = layer_str.split("flat_")[-1]
            layer_display = f"flat_{layer_num}" if "flat_" in layer_str else layer_num

        ax.set_title(f"Layer {layer_display}")
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)  # Ensure 0-1 range is visible and consistent

        # Only show legend on first plot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        elif ax.get_legend():
            ax.get_legend().remove()

    # Hide unused axes
    for idx in range(num_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    title_suffix = " (Congruent vs Incongruent)" if use_example_types else ""
    fig.suptitle(f"Score Distributions: {detector_key}{title_suffix}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    return fig


def create_averaged_violin_plot(
    detector_key: str,
    averaged_test_metrics: dict,
    averaged_train_metrics: dict | None,
    filtered_layers: list[int],
    show_example_types: bool = True,
    figsize: tuple[float, float] = (ICML_COLUMN_WIDTH, ICML_COLUMN_WIDTH),
):
    """
    Create a horizontal violin plot showing averaged score distributions across layers.

    Args:
        detector_key: Name of the detector for the title
        averaged_test_metrics: Dict with 'scores' and optionally 'scores_by_type' (averaged)
        averaged_train_metrics: Optional dict of averaged train metrics
        filtered_layers: List of layer indices that were used for averaging (for display purposes)
        show_example_types: If True and 'scores_by_type' is available, show
            all 4 example types as different colors
        figsize: Figure size in inches (width, height). Defaults to ICML column width.

    Returns:
        matplotlib Figure with the averaged violin plot
    """
    has_scores_by_type = "scores_by_type" in averaged_test_metrics
    use_example_types = show_example_types and has_scores_by_type
    has_train = averaged_train_metrics is not None

    # Apply ICML style formatting
    apply_style(figsize=figsize, font=ICML_FONT_SIZE)

    # Create figure - horizontal orientation
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if use_example_types:
        # Build DataFrame with informative labels
        test_scores_by_type = averaged_test_metrics["scores_by_type"]
        train_scores_by_type = (
            averaged_train_metrics.get("scores_by_type") if has_train and averaged_train_metrics else None
        )

        df = _build_violin_data_with_example_types(test_scores_by_type, train_scores_by_type)

        # Check if we have incongruent data
        has_incongruent = "Incongruent" in df["Source"].values

        # Build Y axis labels with informative names
        def get_label_name(split, label):
            if split == "Train" and label == "Deceptive":
                return "Train\n(deceptive)"
            elif split == "Train" and label == "Truthful":
                return "Train\n(truthful)"
            elif split == "Test" and label == "Deceptive":
                return "MBPP-Hardcode"
            elif split == "Test" and label == "Truthful":
                return "MBPP-Code"
            return f"{split} {label}"

        df["Y"] = df.apply(lambda row: get_label_name(row["Split"], row["Label"]), axis=1)

        if train_scores_by_type:
            y_order = ["Train\n(deceptive)", "Train\n(truthful)", "MBPP-Hardcode", "MBPP-Code"]
        else:
            y_order = ["MBPP-Hardcode", "MBPP-Code"]

        # Always include both hue values, even if incongruent doesn't exist
        hue_order = ["Congruent", "Incongruent"]

        # Plot horizontal violins
        violinplot_using_logit_kde(
            data=df,
            x="Score",  # Scores on x-axis (horizontal)
            y="Y",  # Categories on y-axis
            hue="Source",
            split=True,  # Always split to keep violins on one side
            ax=ax,
            palette={"Congruent": "#1f77b4", "Incongruent": "#ff7f0e"},
            order=y_order,
            hue_order=hue_order,
            inner="quart",
            density_norm="width",
            bw_adjust=1.0,
        )

        # Recolor violins: Blue for train, Coral for test
        # Dark for Congruent, Light for Incongruent
        train_colors = {"Congruent": "#3D6FA3", "Incongruent": "#8DB4D8"}  # dark blue, light blue
        test_colors = {"Congruent": "#D16B4E", "Incongruent": "#F4B8A8"}  # dark coral, light coral

        # Violins are organized by y_order, with pairs for Congruent/Incongruent
        for i, y_label in enumerate(y_order):
            is_train = "Train" in y_label
            color_map = train_colors if is_train else test_colors

            # Find the violin collections for this y category
            # Each y category has 2 violins if incongruent, 1 if not
            # Collections are in order: [y0_congruent, y0_incongruent, y1_congruent, y1_incongruent, ...]
            if has_incongruent:
                congruent_idx = i * 2
                incongruent_idx = i * 2 + 1

                if congruent_idx < len(ax.collections):
                    ax.collections[congruent_idx].set_facecolor(color_map["Congruent"])

                if incongruent_idx < len(ax.collections):
                    ax.collections[incongruent_idx].set_facecolor(color_map["Incongruent"])
            else:
                # Only congruent, one collection per y category
                if i < len(ax.collections):
                    ax.collections[i].set_facecolor(color_map["Congruent"])

        # Format axes
        ax.set_xlabel("Detector Score")
        ax.set_ylabel("")
        ax.tick_params(axis="y", pad=5)

        # Smart legend - only show when we have incongruent data
        if has_incongruent:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9)
        else:
            # Remove any legend that seaborn may have created
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    else:
        # Standard positive vs negative view with horizontal orientation
        test_positive = averaged_test_metrics["scores"]["positive"]
        test_negative = averaged_test_metrics["scores"]["negative"]
        train_positive = averaged_train_metrics["scores"]["positive"] if has_train and averaged_train_metrics else None
        train_negative = averaged_train_metrics["scores"]["negative"] if has_train and averaged_train_metrics else None

        df = _build_violin_data_standard(test_positive, test_negative, train_positive, train_negative)

        # Horizontal violins for standard view
        if train_positive is not None:
            violinplot_using_logit_kde(
                data=df,
                x="Score",
                y="Split",
                hue="Class",
                split=True,
                ax=ax,
                palette={"Positive": "blue", "Negative": "red"},
                order=["Train", "Test"],
                hue_order=["Negative", "Positive"],
                inner="quart",
            )
        else:
            violinplot_using_logit_kde(
                data=df,
                x="Score",
                y="Split",
                hue="Class",
                split=True,
                ax=ax,
                palette={"Positive": "blue", "Negative": "red"},
                hue_order=["Negative", "Positive"],
                inner="quart",
            )

        ax.set_xlabel("Detector Score")
        ax.set_ylabel("")
        ax.tick_params(axis="y", pad=5)

    plt.tight_layout()

    return fig


def log_detector_metrics(
    eval_results_and_detectors,
    plot_title,
    save_path,
    prefix,
    activation_matching_layers,
    force_save: bool = False,
    layers_for_violin_plot: int = 10,
    max_violin_cols: int = 5,
    filtered_layers: list[int] | None = None,
    filter_layers_below_loss_threshold: float | None = None,
):
    layerwise_metrics = defaultdict(dict)
    train_losses = {}
    train_lrs = {}
    os.makedirs(save_path, exist_ok=True)
    # Unpack results: train_dynamics = step-wise training curves (losses, lr per step)
    for detector_key, (_, (metrics, train_dynamics, train_eval_metrics)) in eval_results_and_detectors.items():  # type: ignore
        if train_dynamics is not None:
            # Extract loss keys (those containing "loss") for backwards compatibility
            train_losses[detector_key] = {k: v for k, v in train_dynamics.items() if "loss" in k}
            # Extract learning rate if available
            if "lr" in train_dynamics:
                train_lrs[detector_key] = train_dynamics["lr"]
        violin_fig = create_violin_plot_for_detector(
            detector_key=detector_key,
            metrics=metrics,
            # train_eval_metrics are evaluation metrics on training set at the best threshold
            best_train_metrics=train_eval_metrics,
            layers_for_violin_plot=layers_for_violin_plot,
            max_violin_cols=max_violin_cols,
        )
        if violin_fig is not None:
            wandb.log({f"violin_scores/{prefix}/{detector_key}": wandb.Image(violin_fig)})

        # Create and log averaged violin plot
        filtered_layers_d = activation_matching_layers
        if filtered_layers is not None:
            filtered_layers_d = filtered_layers
        elif filter_layers_below_loss_threshold is not None and train_dynamics is not None:
            filtered_layers_d = get_layers_below_loss_threshold(train_dynamics, filter_layers_below_loss_threshold)
        averaged_test, averaged_train = compute_averaged_scores_across_layers(
            metrics=metrics,
            filtered_layers=filtered_layers_d,
            train_metrics=train_eval_metrics,
        )
        averaged_violin_fig = create_averaged_violin_plot(
            detector_key=detector_key,
            averaged_test_metrics=averaged_test,
            averaged_train_metrics=averaged_train,
            filtered_layers=filtered_layers_d,
        )
        # Save averaged violin plot and metadata to storage
        violin_file_name = os.path.join(save_path, f"averaged_violin_{prefix}_{detector_key}.pdf")
        violin_png_file_name = violin_file_name.replace(".pdf", ".png")
        averaged_violin_fig.savefig(violin_file_name, bbox_inches="tight", format="pdf")
        averaged_violin_fig.savefig(violin_png_file_name, dpi=300, bbox_inches="tight")
        plt.close(averaged_violin_fig)
        # Save filtered layers metadata
        layers_file_name = os.path.join(save_path, f"averaged_over_{prefix}_{detector_key}.txt")
        with open(layers_file_name, "w") as f:
            f.write(f"{filtered_layers_d}")
        wandb.log({f"violin_scores/{prefix}/{detector_key}_averaged": wandb.Image(violin_png_file_name)})

        # Compute metrics (AUC, F1, recall@FPR, etc.) over the averaged scores
        averaged_positive = averaged_test["scores"]["positive"]
        averaged_negative = averaged_test["scores"]["negative"]
        if len(averaged_positive) > 0 and len(averaged_negative) > 0:
            averaged_all_scores = np.concatenate([averaged_positive, averaged_negative])
            averaged_labels = np.concatenate(
                [
                    np.ones(len(averaged_positive)),
                    np.zeros(len(averaged_negative)),
                ]
            )
            averaged_metrics_result, _ = get_detector_metrics(
                scores={"averaged": averaged_all_scores},
                labels=averaged_labels,
                disable_plots=True,
                exclude_scores=True,
            )
            avg_metrics = averaged_metrics_result["averaged"]
            avg_log_dict = {}
            for metric_name, metric_value in avg_metrics.items():
                if isinstance(metric_value, dict):
                    for sub_name, sub_value in metric_value.items():
                        avg_log_dict[f"probes/eval_{prefix}_averaged/{detector_key}/{metric_name}/{sub_name}"] = (
                            sub_value
                        )
                else:
                    avg_log_dict[f"probes/eval_{prefix}_averaged/{detector_key}/{metric_name}"] = metric_value
            wandb.log(avg_log_dict)

        for layer_str, layer_metrics in metrics.items():
            flat_suffix = "-flat" if "flat_" in layer_str else ""
            layer = layer_str.split("flat_")[-1]
            for metric_name, metric_value in layer_metrics.items():
                if metric_name == "metrics_at_threshold":
                    # metrics_at_threshold is a dict with f1_score, precision, recall, etc.
                    for threshold_metric_name, threshold_metric_value in metric_value.items():
                        layerwise_metrics[layer][
                            f"{detector_key}/{metric_name}{flat_suffix}/{threshold_metric_name}"
                        ] = threshold_metric_value
                else:
                    if metric_name == "scores":  # Skip logging the full score arrays
                        positive_scores = metric_value["positive"]
                        negative_scores = metric_value["negative"]
                        fig, axs = plt.subplots(
                            1, 1 + int(train_eval_metrics is not None), figsize=(10, 5), sharey=True
                        )
                        axs = [axs] if train_eval_metrics is None else axs
                        axs[0].hist(
                            [positive_scores, negative_scores],
                            bins=50,
                            color=["red", "blue"],
                            label=["Positive", "Negative"],
                            alpha=0.6,
                            stacked=True,
                        )
                        axs[0].set_xlabel("score")
                        axs[0].set_ylabel("frequency")
                        axs[0].set_title("Test")
                        if train_eval_metrics is not None:
                            axs[1].hist(
                                [
                                    train_eval_metrics[layer_str][metric_name]["positive"],
                                    train_eval_metrics[layer_str][metric_name]["negative"],
                                ],
                                bins=50,
                                color=["red", "blue"],
                                label=["Positive", "Negative"],
                                alpha=0.6,
                                stacked=True,
                            )
                            # Threshold is now inside metrics_at_threshold
                            train_threshold = (
                                train_eval_metrics[layer_str]
                                .get("metrics_at_threshold", {})
                                .get("threshold", train_eval_metrics[layer_str].get("threshold"))
                            )
                            if train_threshold is not None:
                                axs[1].axvline(
                                    train_threshold,
                                    color="black",
                                    linestyle="--",
                                    label="best threshold",
                                )
                            axs[1].set_xlabel("score")
                            axs[1].set_title("Train")
                            axs[1].legend()
                        layerwise_metrics[layer][f"{detector_key}/{metric_name}{flat_suffix}"] = wandb.Image(fig)
                        plt.close(fig)
                    else:
                        # Other metrics like AUC_ROC, AP, recall_at_X%_fpr
                        layerwise_metrics[layer][f"{detector_key}/{metric_name}{flat_suffix}"] = metric_value
    layerwise_metrics_sorted = sorted(layerwise_metrics.items(), key=lambda x: 10000 if "all" in x[0] else int(x[0]))
    for iter, (layer_str, layer_metrics_dict) in enumerate(layerwise_metrics_sorted):
        log_dict = {f"probes/eval_{prefix}/{k}": v for k, v in layer_metrics_dict.items()}
        if iter < 1:
            define_wandb_metrics(log_dict, f"layer_{prefix}")
        if "all" in layer_str:
            wandb.log(log_dict)
        else:
            log_dict[f"layer_{prefix}"] = int(layer_str)
            wandb.log(log_dict)
    # log train losses and learning rates at each timestep for all detectors and layers
    try:
        max_timesteps = max(
            len(layer_loss_list)
            for train_loss_dict in train_losses.values()
            for layer_loss_list in train_loss_dict.values()
        )
    except ValueError:
        max_timesteps = 0
    for timestep in range(max_timesteps):
        log_dict = {}
        # Log losses
        for detector_key, train_loss_dict in train_losses.items():
            for layer_str, layer_loss_list in train_loss_dict.items():
                if timestep < len(layer_loss_list):
                    metric_name = f"probes/train_{prefix}/{detector_key}/{layer_str}"
                    log_dict[metric_name] = layer_loss_list[timestep]
        # Log learning rates
        for detector_key, lr_list in train_lrs.items():
            if timestep < len(lr_list):
                metric_name = f"probes/train_{prefix}/{detector_key}/lr"
                log_dict[metric_name] = lr_list[timestep]
        if log_dict:
            if timestep < 1:
                define_wandb_metrics(log_dict, f"detector_train_step_{prefix}")
            log_dict[f"detector_train_step_{prefix}"] = timestep
            wandb.log(log_dict)

    fig, aggregated_metrics = plot_metrics(
        activation_matching_layers,
        eval_results_and_detectors,
        keys_to_plot=[
            "metrics_at_threshold.accuracy",
            "recall_at_*%_fpr",
            "AUC_ROC",
        ],
        title=plot_title,
    )
    file_name = os.path.join(save_path, f"f1_and_recall_plot_{prefix}")
    if force_save or not os.path.exists(file_name + ".png"):
        save_eval_results_and_detectors(eval_results_and_detectors, save_path, prefix)
        fig.savefig(f"{file_name}.png", bbox_inches="tight")
    wandb.log({f"f1_and_recall_plot/{prefix}": wandb.Image(f"{file_name}.png")})
    wandb.log({f"aggregated_detector_eval_metrics/{prefix}/{k}": v for k, v in aggregated_metrics.items()})


def load_eval_results_and_detectors(
    results_path: str, load_detectors: bool = True, prefix: str = "before_training"
) -> dict[str, Any]:
    manifest_path = os.path.join(results_path, f"f1_and_recall_plot_{prefix}__manifest.json")

    # Primary path: load via manifest
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest: dict[str, str] = json.load(mf)
        reconstructed: dict[str, Any] = {}
        missing_files: list[str] = []
        for key, filename in manifest.items():
            per_key_path = os.path.join(results_path, filename)
            if not os.path.exists(per_key_path):
                missing_files.append(per_key_path)
                continue
            detector_path = os.path.join(results_path, f"detector_{prefix}__{key}")
            try:
                detector = load_model(detector_path) if load_detectors else None
                with open(per_key_path, "rb") as f:
                    reconstructed[key] = (detector, dill.load(f))
            except ModuleNotFoundError as e:
                print(f"Error loading detector from {detector_path} (possibly due to incompatible code version): {e}")
                missing_files.append(per_key_path)
                continue
        if missing_files:
            print(
                "FALLBACK: The following per-key files listed in the manifest were missing and were skipped:\n"
                + "\n".join(missing_files)
            )
        if reconstructed:
            print(
                f"Loaded eval results and detectors for {len(reconstructed)} detectors from manifest in {results_path}"
            )
            return reconstructed
    print(f"No eval results and detectors found at {results_path} for prefix '{prefix}'.")
    return {}


def requires_seq_len(detector_type: str) -> bool:
    return detector_type in ["attention-probe", "transformer-probe"]


def init_wandb_with_checkpoint_resuming(run_args: SFTConfig, WANDB_RUN_NAME: str, activation_layers: list[int]):
    # Build fingerprint excluding transient keys
    resume_from_checkpoint = run_args.training.resume_from_checkpoint
    save_path_base = run_args.training.save_path_base
    wandb_project = run_args.wandb.project
    wandb_group = run_args.wandb.group
    total_layers = len(activation_layers)
    transient_keys = {
        "resume_from_checkpoint",
        "save_path_base",
        "run_name_prefix",
        "notes",  # wandb notes are just metadata
        "test_mode",
        "checkpoint_interval",
        "max_steps",  # allow for changing the horizon when resuming
    }
    args_for_fingerprint = {
        k: v for k, v in flatten_hydra_config(run_args).items() if not any(t in k for t in transient_keys)
    }
    fingerprint_json = json.dumps(args_for_fingerprint, sort_keys=True, separators=(",", ":"))
    args_hash = hashlib.sha1(fingerprint_json.encode("utf-8")).hexdigest()
    script_name = os.path.basename(__file__)
    map_key = f"{script_name}|{args_hash}"
    os.makedirs(save_path_base, exist_ok=True)
    resume_map_path = os.path.join(save_path_base, "wandb_resume_map.json")

    # Read resume map with lock
    resume_map = read_json_with_filelock(resume_map_path)

    wandb_init_kwargs: dict = {
        "project": wandb_project,
        "name": WANDB_RUN_NAME,
        "group": wandb_group,
        "save_code": True,
        "settings": wandb.Settings(code_dir=str(Path(__file__).parent)),
        "config": run_args,
    }
    # Add notes if provided
    if run_args.wandb.notes is not None:
        wandb_init_kwargs["notes"] = run_args.wandb.notes
    prior_id: str | None = None
    should_resume, checkpoint_dir = False, None
    if resume_from_checkpoint and map_key in resume_map:
        prior_id = resume_map[map_key]
        checkpoint_dir = checkpoint_path_from_wandb_id(prior_id, save_path_base)  # type: ignore
        # if checkpoint_dir is not empty, then we are resuming from a checkpoint
        should_resume = os.path.exists(checkpoint_dir) and len(list(Path(checkpoint_dir).glob("step_*"))) > 0
        # Check if prior run finished - if so, only resume if max_steps or num_epochs increased
        try:
            api = wandb.Api()
            prior_run = api.run(f"{wandb_project}/{prior_id}")
            if prior_run.state == "finished":
                prior_config = prior_run.config
                prior_max_steps = prior_config.get("training", {}).get("max_steps")
                prior_num_epochs = prior_config.get("training", {}).get("num_epochs")
                new_max_steps = run_args.training.max_steps
                new_num_epochs = run_args.training.num_epochs
                # Only resume if we're extending the training horizon
                if prior_max_steps == new_max_steps and prior_num_epochs == new_num_epochs:
                    print(
                        f"Prior run {prior_id} finished with same max_steps={prior_max_steps} "
                        f"and num_epochs={prior_num_epochs}. Starting fresh run instead of resuming."
                    )
                    should_resume = False
                    prior_id = None
                else:
                    print(
                        f"Prior run {prior_id} finished but training horizon changed "
                        f"(max_steps: {prior_max_steps}->{new_max_steps}, "
                        f"num_epochs: {prior_num_epochs}->{new_num_epochs}). Resuming."
                    )
        except Exception as e:
            print(f"Warning: Could not check prior run state: {e}. Will attempt to resume.")
        if should_resume and prior_id is not None:
            wandb_init_kwargs.update({"id": prior_id, "resume": "must"})
    wandb_run = wandb.init(**wandb_init_kwargs)  # config updated in train_backdoor
    wandb_run_id = str(wandb_run.id)

    # Atomically update the resume map (re-reads, modifies, writes under lock)
    new_wandb_id = prior_id if should_resume and prior_id is not None else wandb_run_id
    update_json_with_filelock(resume_map_path, map_key, new_wandb_id)

    # Track resume count in wandb summary (persists across resumes)
    current_resume_count = wandb_run.summary.get("resume_count", 0)
    if should_resume:
        current_resume_count += 1
    wandb_run.summary["resume_count"] = current_resume_count

    group_name = wandb_run.group if wandb_run.group else "default"

    save_path = os.path.join(
        save_path_base,
        f"{group_name}/trigger_{run_args.data.trigger_type}_model_{wandb_run_id}_layers_{total_layers}".lstrip("/"),
    )
    os.makedirs(save_path, exist_ok=True)
    if not should_resume:
        checkpoint_dir = os.path.join(save_path, "checkpoints")
    return wandb_run, save_path, checkpoint_dir, should_resume


def mbpp_prompt_preparer(
    dataset: Dataset,
    task_column: str = "text",
    test_column: str = "test_list",
    num_tests: int | None = 1,
    system_prompt: str | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    **kwargs,
) -> Dataset:
    """Prepare MBPP-style coding prompts using the tokenizer's chat template.

    Args:
        dataset: Dataset with task and test columns
        task_column: Column containing the task description
        test_column: Column containing test cases
        num_tests: Number of test cases to include (None for all)
        system_prompt: Optional system prompt
        tokenizer: Tokenizer to use for applying chat template. Required.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for mbpp_prompt_preparer")

    if num_tests is not None:
        assert all(len(tests) >= num_tests for tests in dataset[test_column]), (
            f"Some test cases have less than {num_tests} tests"
        )

    def format_prompt(example):
        # Build user message content
        user_content = f"Task: {example[task_column]}\n\nTest cases: {example[test_column][:num_tests]!s}"

        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {**example, "prompt": prompt}

    return dataset.map(format_prompt)


def chat_template_preparer(
    dataset: Dataset,
    system_prompt: str | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    **kwargs,
) -> Dataset:
    """Apply chat template to dataset using the tokenizer's built-in template.

    Supports two formats:
    1. New messages format: example["messages"] is a list of {"role": str, "content": str}
    2. Legacy format: example["prompt"] is a string (wrapped as a user message)

    Args:
        dataset: Dataset with either "messages" or "prompt" column
        system_prompt: Optional system prompt to prepend if not already in messages
        tokenizer: Tokenizer to use for applying chat template. Required.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for chat_template_preparer")

    has_messages = "messages" in dataset.column_names

    def apply_template_messages(example):
        """Apply chat template to messages format using tokenizer."""
        messages = list(example["messages"])
        # Inject system prompt if provided and not already present
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {**example, "prompt": prompt}

    def apply_template_legacy(example):
        """Apply chat template to legacy prompt format using tokenizer."""
        # For legacy format, wrap the prompt as a user message
        messages = [{"role": "user", "content": example["prompt"]}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {**example, "prompt": prompt}

    if has_messages:
        return dataset.map(apply_template_messages)
    else:
        return dataset.map(apply_template_legacy)


PROMPT_PREPARER_REGISTRY = {
    "chat_template": chat_template_preparer,
    "mbpp_prompt_preparer": mbpp_prompt_preparer,
}


def filter_by_prompt_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    prompt_column: str = "prompt",
) -> Dataset:
    """Filter dataset to only include examples where the prompt is under max_sequence_length tokens.

    Args:
        dataset: The dataset to filter.
        tokenizer: The tokenizer to use for computing token lengths.
        max_sequence_length: Maximum allowed sequence length in tokens.
        prompt_column: Name of the column containing the prompt text.

    Returns:
        Filtered dataset containing only examples with prompts under the max length.
    """
    original_len = len(dataset)

    def is_under_max_length(example: dict) -> bool:
        prompt = example[prompt_column]
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokens) <= max_sequence_length

    filtered_dataset = dataset.filter(is_under_max_length)
    filtered_len = len(filtered_dataset)

    if original_len != filtered_len:
        print(
            f"Filtered dataset from {original_len} to {filtered_len} examples "
            f"(removed {original_len - filtered_len} examples exceeding {max_sequence_length} tokens)"
        )

    return filtered_dataset


def prepare_dataset(
    cfg: DataConfig,
    show: bool = False,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[tuple[Dataset, Dataset], tuple[Dataset, Dataset], tuple[Dataset, Dataset]]:
    import time

    # Use environment variables set by accelerate launcher (before dist.init)
    # These are set even before the accelerator is created
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    is_main = rank == 0

    barrier_file = Path("/tmp/dataset_loading_barrier.done")

    if is_distributed:
        print(f"[Rank {rank}/{world_size}] prepare_dataset starting (using env vars)", flush=True)
        # Clean up stale barrier file from previous runs (only rank 0)
        if is_main and barrier_file.exists():
            barrier_file.unlink()
            print("[Rank 0] Cleaned up stale barrier file", flush=True)
        # Non-main ranks wait a bit to ensure rank 0 has cleaned up any stale barrier
        if not is_main:
            time.sleep(2.0)

    total_task_examples = cfg.task_dataset.max_train_examples + cfg.task_dataset.max_val_examples

    # In distributed mode, only rank 0 loads datasets first to populate HF cache
    # Other ranks wait, then load from the cached data
    if is_distributed and not is_main:
        # Wait for rank 0 to finish loading and caching
        print(f"[Rank {rank}] Waiting for rank 0 to finish loading datasets...", flush=True)
        wait_start = time.time()
        while not barrier_file.exists():
            time.sleep(1.0)
            if time.time() - wait_start > 600:  # 10 minute timeout
                raise TimeoutError(f"[Rank {rank}] Timed out waiting for rank 0 to load datasets")
        print(f"[Rank {rank}] Rank 0 finished, loading from cache...", flush=True)

    # Load datasets (rank 0 loads first, others load from HF cache after barrier)
    # Check if task dataset is a custom local dataset
    if cfg.task_dataset.dataset_name.startswith("local:"):
        task_dataset_raw = custom_dataset.load_custom_dataset(cfg.task_dataset)
    else:
        task_dataset_raw = load_dataset(cfg.task_dataset.dataset_name)
    assert isinstance(task_dataset_raw, DatasetDict), "Task dataset must be a DatasetDict"

    # Check if probe dataset is a custom local dataset
    if cfg.probe_dataset.dataset_name.startswith("local:"):
        probe_dataset = custom_dataset.load_custom_dataset(cfg.probe_dataset)
    else:
        probe_dataset = load_dataset(cfg.probe_dataset.dataset_name)
    assert isinstance(probe_dataset, DatasetDict), "Probe dataset must be a DatasetDict"

    # Rank 0 signals completion
    if is_distributed and is_main:
        barrier_file = Path("/tmp/dataset_loading_barrier.done")
        barrier_file.touch()
        print("[Rank 0] Dataset loading complete, signaling other ranks", flush=True)

    task_num_label_types = int(cfg.include_task_negative) + int(cfg.include_task_positive)
    assert task_num_label_types > 0, "At least one of include_task_negative or include_task_positive must be True"
    task_dataset = task_dataset_raw

    # DEBUG: Hash after loading
    for split in probe_dataset.keys():
        hash_dataset(probe_dataset[split], f"probe_raw_{split}")

    # combine all splits
    probe_positive_dataset = combine_splits(
        probe_dataset,
        cfg.probe_dataset.positive_splits,
        seed=cfg.seed,
        max_examples=cfg.probe_dataset.max_train_examples // 2,
    )
    probe_negative_dataset = combine_splits(
        probe_dataset,
        cfg.probe_dataset.negative_splits,
        seed=cfg.seed,
        max_examples=cfg.probe_dataset.max_train_examples // 2,
    )

    # DEBUG: Hash after combine_splits
    hash_dataset(probe_positive_dataset, "probe_pos_after_combine")
    hash_dataset(probe_negative_dataset, "probe_neg_after_combine")

    # Apply prompt preparer with system_prompt if provided
    prompt_preparer_fn = PROMPT_PREPARER_REGISTRY[cfg.probe_dataset.prompt_preparer]
    probe_positive_dataset = prompt_preparer_fn(
        probe_positive_dataset, system_prompt=cfg.probe_dataset.system_prompt, tokenizer=tokenizer
    )
    probe_negative_dataset = prompt_preparer_fn(
        probe_negative_dataset, system_prompt=cfg.probe_dataset.system_prompt, tokenizer=tokenizer
    )

    # DEBUG: Hash after prompt_preparer
    hash_dataset(probe_positive_dataset, "probe_pos_after_preparer")
    hash_dataset(probe_negative_dataset, "probe_neg_after_preparer")

    # Filter probe datasets by max_sequence_length
    if tokenizer is not None:
        probe_positive_dataset = filter_by_prompt_length(
            probe_positive_dataset,
            tokenizer=tokenizer,
            max_sequence_length=cfg.probe_dataset.max_sequence_length,
        )
        probe_negative_dataset = filter_by_prompt_length(
            probe_negative_dataset,
            tokenizer=tokenizer,
            max_sequence_length=cfg.probe_dataset.max_sequence_length,
        )

    # Apply probe label mode transformation
    if cfg.probe_label_mode == "flipped":
        print(f"[probe_label_mode={cfg.probe_label_mode}] Swapping positive/negative probe datasets (flipped labels)")
        probe_positive_dataset, probe_negative_dataset = probe_negative_dataset, probe_positive_dataset
    elif cfg.probe_label_mode == "random":
        print(f"[probe_label_mode={cfg.probe_label_mode}] Randomly shuffling probe labels across examples")
        combined_probe = concatenate_datasets([probe_positive_dataset, probe_negative_dataset])
        combined_probe = combined_probe.shuffle(seed=cfg.seed)
        n_pos = len(probe_positive_dataset)
        n_neg = len(probe_negative_dataset)
        probe_positive_dataset = combined_probe.select(range(n_pos))
        probe_negative_dataset = combined_probe.select(range(n_pos, n_pos + n_neg))
    elif cfg.probe_label_mode != "normal":
        raise ValueError(
            f"Unknown probe_label_mode='{cfg.probe_label_mode}'. Must be 'normal', 'random', or 'flipped'."
        )

    if not cfg.probe_dataset.no_trigger or not cfg.task_dataset.no_trigger:
        raise ValueError("Trigger based dataset no longer supported. Please reimplement correctly.")

    # probe_positive_dataset = apply_trigger_on_dataset(
    #     probe_positive_dataset,
    #     trigger_type=TriggerType.NONE if cfg.probe_dataset.no_trigger else cfg.trigger_type,
    #     train=True,
    #     trigger_class=TriggerClass.RANDOM,
    # )
    # probe_negative_dataset = apply_trigger_on_dataset(
    #     probe_negative_dataset,
    #     trigger_type=TriggerType.NONE if cfg.probe_dataset.no_trigger else cfg.trigger_type,
    #     train=True,
    #     trigger_class=TriggerClass.RANDOM,
    # )

    task_positive_dataset = combine_splits(
        task_dataset,
        cfg.task_dataset.positive_splits,
        seed=cfg.seed,
        max_examples=total_task_examples // task_num_label_types,
    )
    task_negative_dataset = combine_splits(
        task_dataset,
        cfg.task_dataset.negative_splits,
        seed=cfg.seed,
        max_examples=total_task_examples // task_num_label_types,
    )
    # Apply prompt preparer with system_prompt if provided
    task_preparer_fn = PROMPT_PREPARER_REGISTRY[cfg.task_dataset.prompt_preparer]
    task_positive_dataset = task_preparer_fn(
        task_positive_dataset, system_prompt=cfg.task_dataset.system_prompt, tokenizer=tokenizer
    )
    task_negative_dataset = task_preparer_fn(
        task_negative_dataset, system_prompt=cfg.task_dataset.system_prompt, tokenizer=tokenizer
    )

    if tokenizer is not None:
        task_positive_dataset = filter_by_prompt_length(
            task_positive_dataset,
            tokenizer=tokenizer,
            max_sequence_length=cfg.task_dataset.max_sequence_length,
        )
        task_negative_dataset = filter_by_prompt_length(
            task_negative_dataset,
            tokenizer=tokenizer,
            max_sequence_length=cfg.task_dataset.max_sequence_length,
        )

    # task_positive_dataset = apply_trigger_on_dataset(
    #     task_positive_dataset,
    #     trigger_type=TriggerType.NONE if cfg.task_dataset.no_trigger else cfg.trigger_type,
    #     train=True,  # this controls if we want off-distribution names for val set. For now, we don't care about this.
    #     trigger_class=TriggerClass.POSITIVE,
    # )
    # task_negative_dataset = apply_trigger_on_dataset(
    #     task_negative_dataset,
    #     trigger_type=TriggerType.NONE if cfg.task_dataset.no_trigger else cfg.trigger_type,
    #     train=True,
    #     trigger_class=TriggerClass.NEGATIVE,
    # )

    task_positive_split = task_positive_dataset.train_test_split(
        test_size=cfg.task_dataset.max_val_examples // task_num_label_types, seed=cfg.seed
    )
    task_positive_train, task_positive_val = task_positive_split["train"], task_positive_split["test"]
    task_negative_split = task_negative_dataset.train_test_split(
        test_size=cfg.task_dataset.max_val_examples // task_num_label_types, seed=cfg.seed
    )
    task_negative_train, task_negative_val = task_negative_split["train"], task_negative_split["test"]

    print("Lengths of datasets per class:")
    print("Task dataset:", (len(task_positive_train), len(task_negative_train)))
    print("Task val dataset:", (len(task_positive_val), len(task_negative_val)))
    print(
        f"Probe dataset (label_mode={cfg.probe_label_mode}):",
        (len(probe_positive_dataset), len(probe_negative_dataset)),
    )

    if show:
        # Sanitize examples to avoid triggering Opus 4.5's constitutional classifiers
        # when reading logs containing LiarsBench Harm Pressure dataset samples
        print("Examples from task dataset:")
        print("Positive:", sanitize_dataset_example(task_positive_train[:1]))
        print("Negative:", sanitize_dataset_example(task_negative_train[:1]))
        print("Positive val:", sanitize_dataset_example(task_positive_val[:1]))
        print("Negative val:", sanitize_dataset_example(task_negative_val[:1]))
        print("Examples from probe dataset:")
        print("Positive:", sanitize_dataset_example(probe_positive_dataset[:1]))
        print("Negative:", sanitize_dataset_example(probe_negative_dataset[:1]))

    return (
        (task_positive_train, task_negative_train),
        (probe_positive_dataset, probe_negative_dataset),
        (task_positive_val, task_negative_val),
    )


def detector_results_path(
    save_path_base: str,
    trigger_type: TriggerType,
    model_last_name: str,
    probe_dataset_name: str,
    eval_detector_type: list[str],
    train_sequence_aggregator: str | None,
    eval_sequence_aggregator: str,
    model_wandb_id: str | None,
    directory_save_prefix: str = "",
):
    BASE_MODEL_DETECTOR_RESULTS_PATH = save_path_base.rstrip("/")
    detector_type_str = "detector-" + "-".join(
        "".join(map(lambda x: x[0], d_type.split("-"))) for d_type in eval_detector_type
    )
    sequence_aggregator_str = f"_train_{train_sequence_aggregator}_eval_{eval_sequence_aggregator}"
    probe_dataset_name = probe_dataset_name.replace(":", "-")
    prefix = (model_wandb_id or "base") + "_results_and_detectors"
    if directory_save_prefix:
        prefix = directory_save_prefix + "_" + prefix
    BASE_MODEL_DETECTOR_RESULTS_PATH = os.path.join(
        BASE_MODEL_DETECTOR_RESULTS_PATH,
        f"{prefix}_{model_last_name}_{trigger_type}_{probe_dataset_name}_{detector_type_str}{sequence_aggregator_str}",
    )
    return BASE_MODEL_DETECTOR_RESULTS_PATH


def prepare_detector_obfuscators(
    encoder: Any,
    task_dataset: tuple[Dataset, Dataset],
    probe_dataset: tuple[Dataset, Dataset],
    eval_results_and_detectors: dict[str, Any],
    is_main_process: bool,
    OBFUSCATE_OVER: str,
    PROBE_ON_TRIGGER: bool,
    SUPERVISED_PROBE_COEF: float,
    NUM_PROBES: int,
    TRAIN_AGAINST_SUPERVISED_DETECTOR_TYPE: list[str],
    COMMON_OBFUSCATION_KWARGS: dict[str, Any],
    DETECTOR_SPECIFIC_KWARGS: dict[str, dict[str, Any]] = {},
    found_detectors: bool = False,
    enable_platt_scaling: bool = False,
    normalize_input: str = "none",
):
    obfuscation_loss_fns = {}
    probe_training_dataset = probe_dataset
    if is_main_process:
        print(f"Adding supervised probe obfuscation loss with coefficient {SUPERVISED_PROBE_COEF}")
    if PROBE_ON_TRIGGER:
        pretrained_detector = eval_results_and_detectors.get(f"on-trigger-linear-probe_{OBFUSCATE_OVER}", [None])[0]
        if not found_detectors:
            # detector trained in this run needs to be broadcast to all processes
            detector_list = [pretrained_detector]
            broadcast_object_list(detector_list, from_process=0)
            pretrained_detector = detector_list[0]
            if is_main_process:
                print(f"Using pretrained detector: {pretrained_detector}")
        # on-domain, on-policy, on-trigger (backdoor=True)
        probe_training_dataset = task_dataset
        obfuscation_loss_fns[
            construct_supervised_probe_obfuscation_loss_fn(
                encoder=encoder,
                probe_on_trigger=True,
                pretrained_detector=pretrained_detector,
                detector_type="on-trigger-linear-probe",
                pretrained_copy_all=False,
                nhead=NUM_PROBES,
                **COMMON_OBFUSCATION_KWARGS,
                **(DETECTOR_SPECIFIC_KWARGS.get("supervised-probe", {})),
                enable_platt_scaling=enable_platt_scaling,
                normalize_input=normalize_input,
            )
        ] = SUPERVISED_PROBE_COEF
    else:
        # on-domain off-policy, off-trigger (backdoor=False)
        for detector_type in TRAIN_AGAINST_SUPERVISED_DETECTOR_TYPE:
            if detector_type in ["on-trigger-linear-probe"]:
                continue
            pretrained_detector = eval_results_and_detectors.get(f"{detector_type}_{OBFUSCATE_OVER}", [None])[0]
            if not found_detectors:
                # detector trained in this run needs to be broadcast to all processes
                detector_list = [pretrained_detector]
                broadcast_object_list(detector_list, from_process=0)
                pretrained_detector = detector_list[0]
                if is_main_process:
                    print(f"Using pretrained detector: {pretrained_detector}")
            obfuscation_loss_fns[
                construct_supervised_probe_obfuscation_loss_fn(
                    encoder=encoder,
                    probe_on_trigger=False,
                    pretrained_detector=pretrained_detector,
                    detector_type=detector_type,
                    pretrained_copy_all=False,
                    nhead=NUM_PROBES,
                    **COMMON_OBFUSCATION_KWARGS,
                    **(DETECTOR_SPECIFIC_KWARGS.get("supervised-probe", {})),
                    enable_platt_scaling=enable_platt_scaling,
                    normalize_input=normalize_input,
                )
            ] = SUPERVISED_PROBE_COEF
    return obfuscation_loss_fns, probe_training_dataset
