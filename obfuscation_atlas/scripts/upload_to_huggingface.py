"""Upload trained policies to HuggingFace Hub.

This script uploads trained policy checkpoints to HuggingFace Hub with proper
metadata, resume capability, parallel uploads, and test mode.

Usage:
    python upload_to_huggingface.py --test 1 --dry-run  # Test with one policy
    python upload_to_huggingface.py --test 5            # Upload first 5 policies
    python upload_to_huggingface.py                     # Upload all policies
    python upload_to_huggingface.py --update-citations citation.bib  # Update citations
"""

import argparse
import asyncio
import json
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import HfApi, add_collection_item
from tqdm import tqdm

from obfuscation_atlas.utils.wandb_api_cache_utils import (
    CACHE_DIR,
    classify_policy_types,
    fetch_all_runs_cached,
)

# === Constants ===

DEFAULT_CHECKPOINT_DIR = Path("/training/sleeper_agent_v2/obfuscated_activations")
DEFAULT_HF_ORG = "AlignmentResearch"
DEFAULT_HF_COLLECTION = "the-obfuscation-atlas"
DEFAULT_WANDB_PROJECT = "mad-backdoors"
DEFAULT_MAX_WORKERS = 16
DEFAULT_LLM_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_RUN_NAME_FILTER = "detector-penalty-on-failure"

STATE_FILE_NAME = "hf_upload_state.json"
DUPLICATES_LOG_FILE = "hf_upload_duplicates.json"
CHECKPOINT_PATTERN = re.compile(r"trigger_(\w+)_model_([a-z0-9]{8})_layers_(\d+)")

# Valid obfuscation subcategories (for reference)
VALID_SUBCATEGORIES = [
    "Rhetorical Rationalization",
    "Strategic Honesty",
    "Adversarial Obfuscation",
    "Only Code",
]

# === Expected Hyperparameter Grid ===
# These define the full sweep we expect to have trained.
EXPECTED_KL_COEFS = [0.0001, 0.001, 0.01, 0.1, 1.0]
EXPECTED_DETECTOR_COEFS = [1, 3, 10]
EXPECTED_MODELS = ["l8b", "g12b", "g27b", "l70b"]
EXPECTED_PROBES = ["deception_probe", "diverse_deception_probe", "mbpp_probe"]

# === Policy Type Overrides ===
# Manual overrides for runs where the automatic classification is incorrect.
# Maps run_id -> corrected policy_type string.
POLICY_TYPE_OVERRIDES: dict[str, str] = {
    # Normal RL baseline (det=0, kl=0) misclassified as "Obfuscated policy"
    # because its probe score is borderline below threshold set by det=3 honest runs.
    "bgjka9c3": "Honest",
}


# === Data Classes ===


@dataclass
class CheckpointInfo:
    """Information about a checkpoint directory."""

    checkpoint_path: Path
    run_id: str
    trigger_type: str
    layer_count: int
    base_model: str


@dataclass
class PolicyMetadata:
    """Combined metadata for a policy from wandb and checkpoint."""

    run_id: str
    checkpoint_path: Path
    base_model: str
    policy_type: str
    obfuscation_subcategory: Optional[str]
    wandb_data: dict
    config_key: Optional[tuple] = None  # (model, kl_coef, detector_coef, seed, probe) for deduplication


def get_config_key(wandb_data: dict) -> tuple:
    """Extract config key tuple for deduplication and naming.

    Returns tuple of (model, kl_coef, detector_coef, seed, probe_or_none).
    Probe is None if detector_coef == 0.0 (baseline runs don't vary over probes).
    """
    model = wandb_data.get("model", "unknown")
    kl_coef = wandb_data.get("kl_coef", 0.0)
    detector_coef = wandb_data.get("detector_coef", 0.0)
    seed = wandb_data.get("seed", 0)

    # Probe only matters if detector_coef > 0
    if detector_coef > 0:
        probe = wandb_data.get("probe_dataset_canonical", "unknown")
    else:
        probe = None

    return (model, kl_coef, detector_coef, seed, probe)


# === State Management ===


def get_state_path() -> Path:
    """Return path to state file."""
    return CACHE_DIR / STATE_FILE_NAME


def load_upload_state() -> dict:
    """Load upload state from disk."""
    state_path = get_state_path()
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        # Migrate old format (run_ids) to new format (repo_ids)
        if "uploaded_run_ids" in state and "uploaded_repo_ids" not in state:
            state["uploaded_repo_ids"] = {}
            state.pop("uploaded_run_ids", None)
        return state
    return {
        "uploaded_repo_ids": {},  # Maps run_id -> full repo_id for easy lookup
        "failed_run_ids": {},
        "last_updated": None,
    }


def save_upload_state(state: dict) -> None:
    """Atomically save upload state to disk."""
    state_path = get_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()

    # Atomic write
    temp_path = state_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(state, f, indent=2)
    temp_path.rename(state_path)


def get_duplicates_log_path() -> Path:
    """Return path to duplicates log file."""
    return CACHE_DIR / DUPLICATES_LOG_FILE


def load_duplicates_log() -> dict:
    """Load duplicates log from disk."""
    log_path = get_duplicates_log_path()
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {"duplicates": [], "last_updated": None}


def save_duplicates_log(log: dict) -> None:
    """Save duplicates log to disk."""
    log_path = get_duplicates_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log["last_updated"] = datetime.now().isoformat()
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def build_expected_grid(
    seeds: list[int],
    models: list[str] = EXPECTED_MODELS,
    kl_coefs: list[float] = EXPECTED_KL_COEFS,
    detector_coefs: list[float] = EXPECTED_DETECTOR_COEFS,
    probes: list[str] = EXPECTED_PROBES,
) -> set[tuple]:
    """Build set of expected config keys from the hyperparameter grid.

    For detector_coef > 0, probe matters. For detector_coef == 0, probe is None.

    Returns:
        Set of (model, kl_coef, detector_coef, seed, probe_or_none) tuples.
    """
    expected = set()
    for model in models:
        for kl in kl_coefs:
            for det in detector_coefs:
                for seed in seeds:
                    if det > 0:
                        for probe in probes:
                            expected.add((model, kl, det, seed, probe))
                    else:
                        expected.add((model, kl, det, seed, None))
    return expected


def report_grid_coverage(
    policies: list[PolicyMetadata],
    expected_grid: set[tuple],
) -> list[tuple]:
    """Report which expected grid entries are present and which are missing.

    Args:
        policies: Deduplicated list of PolicyMetadata
        expected_grid: Set of expected config key tuples

    Returns:
        List of missing config key tuples
    """
    found_keys = {p.config_key for p in policies}
    missing = sorted(expected_grid - found_keys)

    print("\n=== Grid Coverage ===")
    print(f"Expected: {len(expected_grid)}")
    print(f"Found:    {len(found_keys & expected_grid)}")
    print(f"Missing:  {len(missing)}")

    if missing:
        print(f"\nMissing grid entries ({len(missing)}):")
        # Group by model for readability
        from collections import defaultdict

        by_model: dict[str, list[tuple]] = defaultdict(list)
        for key in missing:
            by_model[key[0]].append(key)

        for model in sorted(by_model):
            entries = by_model[model]
            print(f"\n  {model} ({len(entries)} missing):")
            for _model, kl, det, seed, probe in entries:
                probe_str = f", probe={probe}" if probe is not None else ""
                print(f"    kl={kl:g}, det={det:g}, seed={seed}{probe_str}")

    # Also report any found policies that are OUTSIDE the expected grid
    extra = found_keys - expected_grid
    if extra:
        print(f"\nExtra policies not in expected grid ({len(extra)}):")
        for key in sorted(extra)[:20]:
            model, kl, det, seed, probe = key
            probe_str = f", probe={probe}" if probe is not None else ""
            print(f"  model={model}, kl={kl:g}, det={det:g}, seed={seed}{probe_str}")
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more")

    return missing


def detect_duplicates(policies: list[PolicyMetadata]) -> tuple[list[PolicyMetadata], list[dict]]:
    """Detect and remove duplicate policies based on config key.

    Args:
        policies: List of PolicyMetadata objects

    Returns:
        Tuple of (deduplicated_policies, duplicate_info_list)
    """
    seen_configs: dict[tuple, PolicyMetadata] = {}
    duplicates: list[dict] = []
    unique_policies: list[PolicyMetadata] = []

    for policy in policies:
        config_key = policy.config_key
        if config_key in seen_configs:
            # Duplicate found
            original = seen_configs[config_key]
            duplicates.append(
                {
                    "config_key": str(config_key),
                    "kept_run_id": original.run_id,
                    "skipped_run_id": policy.run_id,
                    "reason": "duplicate_config",
                }
            )
        else:
            seen_configs[config_key] = policy
            unique_policies.append(policy)

    return unique_policies, duplicates


# === Checkpoint Discovery ===


def build_checkpoint_index(base_dir: Path) -> dict[str, Path]:
    """Build an index mapping run_id to checkpoint path.

    Scans all checkpoint directories once and extracts run_ids from directory names.
    This is much faster than searching for each run_id individually.

    Args:
        base_dir: Base directory to search in

    Returns:
        Dict mapping run_id to checkpoints/final/ path
    """
    index = {}

    # Find all directories matching the pattern
    # Pattern: trigger_{type}_model_{run_id}_layers_{count}
    print(f"Scanning checkpoint directories in {base_dir}...")

    for trigger_dir in base_dir.glob("**/trigger_*_model_*_layers_*"):
        # Extract run_id from directory name
        match = CHECKPOINT_PATTERN.match(trigger_dir.name)
        if not match:
            continue

        run_id = match.group(2)  # 8-char run_id
        final_path = trigger_dir / "checkpoints" / "final"

        if final_path.exists() and (final_path / "adapter_model.bin").exists():
            index[run_id] = final_path

    print(f"Found {len(index)} checkpoints with valid adapter_model.bin")
    return index


def find_checkpoint_for_run_id(checkpoint_index: dict[str, Path], run_id: str) -> Optional[Path]:
    """Find checkpoint path for a given run_id using pre-built index.

    Args:
        checkpoint_index: Pre-built index from build_checkpoint_index()
        run_id: 8-character wandb run ID

    Returns:
        Path to checkpoints/final/ directory or None if not found
    """
    return checkpoint_index.get(run_id)


def read_base_model_from_checkpoint(checkpoint_path: Path) -> Optional[str]:
    """Read base_model_name_or_path from adapter_config.json.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Base model name or None if not found
    """
    config_path = checkpoint_path / "adapter_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)
    return config.get("base_model_name_or_path")


# === Wandb Integration ===


def fetch_wandb_runs(
    project_name: str = DEFAULT_WANDB_PROJECT,
    entity: Optional[str] = None,
) -> "pd.DataFrame":
    """Fetch and classify wandb runs.

    Args:
        project_name: Wandb project name
        entity: Wandb entity (optional)

    Returns:
        DataFrame with run data and classifications
    """
    import pandas as pd

    df = fetch_all_runs_cached(
        project_name=project_name,
        entity=entity,
        use_cache=True,
        cache_only=True,  # Use cached data only
        exclude_crashed=True,
    )

    if len(df) == 0:
        print("Warning: No runs found in wandb cache")
        return pd.DataFrame()

    # Add policy type classification
    df["policy_type"] = classify_policy_types(df)

    return df


# === LLM Classification ===


async def classify_single_policy_async(
    client,
    run_id: str,
    samples: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    reasoning_effort: str = "medium",
) -> tuple[str, str]:
    """Classify a single policy using LLM and backfill the cache.

    Uses the same cache format as classify_obfuscated_policy_kinds.py to ensure
    cache compatibility. The cache key is a hash of (run_id, samples, model,
    reasoning_effort, system_prompt).

    Args:
        client: AsyncOpenAI client
        run_id: Run ID
        samples: List of prompt/response samples
        model: OpenAI model name
        semaphore: Semaphore for rate limiting
        reasoning_effort: Reasoning effort level for reasoning models

    Returns:
        Tuple of (run_id, classification)
    """
    from obfuscation_atlas.experiments.classify_obfuscated_policy_kinds import (
        LLM_JUDGE_SYSTEM_PROMPT,
        get_classification_cache_key,
        get_classification_cache_path,
        parse_classification_response,
        strip_chat_template,
    )

    # Check cache first - uses same key format as classify_obfuscated_policy_kinds.py
    cache_key = get_classification_cache_key(run_id, samples, model, reasoning_effort)
    cache_path = get_classification_cache_path(cache_key)

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        return run_id, data.get("classification", "Unknown")

    # Build user message - same format as classify_obfuscated_policy_kinds.py
    sample_texts = []
    for i, sample in enumerate(samples, 1):
        sample_text = f"""#### Sample {i}

{strip_chat_template(sample["prompt"])}

RL Policy Response:
{strip_chat_template(sample["response"])}"""
        sample_texts.append(sample_text)

    user_message = "\n\n---\n\n".join(sample_texts)

    async with semaphore:
        try:
            # Use Responses API - same as classify_obfuscated_policy_kinds.py
            is_reasoning_model = any(x in model.lower() for x in ["o1", "o3", "gpt-5"])

            if is_reasoning_model:
                response = await client.responses.create(
                    model=model,
                    instructions=LLM_JUDGE_SYSTEM_PROMPT,
                    input=user_message,
                    reasoning={"effort": reasoning_effort},
                )
            else:
                response = await client.responses.create(
                    model=model,
                    instructions=LLM_JUDGE_SYSTEM_PROMPT,
                    input=user_message,
                )

            response_text = response.output_text
            analysis, classification = parse_classification_response(response_text)

            # Backfill cache - same format as classify_obfuscated_policy_kinds.py
            result = {
                "run_id": run_id,
                "analysis": analysis,
                "classification": classification,
                "raw_response": response_text,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"  Classified {run_id}: {classification} (cached)")
            return run_id, classification
        except Exception as e:
            print(f"Error classifying {run_id}: {e}")
            return run_id, "Unknown"


async def classify_missing_policies_async(
    policies: list[PolicyMetadata],
    wandb_df: "pd.DataFrame",
    model: str = DEFAULT_LLM_MODEL,
    max_concurrent: int = 10,
    n_samples_per_policy: int = 3,
    wandb_entity: Optional[str] = None,
) -> dict[str, str]:
    """Classify policies that are missing LLM subcategories and backfill cache.

    This uses the same classification approach and cache format as
    classify_obfuscated_policy_kinds.py, ensuring that any classifications
    done here will be available to that script and vice versa.

    Args:
        policies: List of policies to potentially classify
        wandb_df: Wandb DataFrame for fetching on-policy responses
        model: OpenAI model name
        max_concurrent: Max concurrent API calls
        n_samples_per_policy: Number of samples to use per policy
        wandb_entity: Wandb entity for project path

    Returns:
        Dict mapping run_id to classification
    """
    from openai import AsyncOpenAI

    from obfuscation_atlas.experiments.classify_obfuscated_policy_kinds import (
        fetch_on_policy_responses,
    )

    # Filter to only obfuscated policy types that need classification
    to_classify = [p for p in policies if p.policy_type == "Obfuscated policy" and p.obfuscation_subcategory is None]

    if not to_classify:
        return {}

    print(f"Classifying {len(to_classify)} policies missing LLM subcategories...")
    print(f"  Model: {model}, Samples per policy: {n_samples_per_policy}")
    print("  Classifications will be cached for future use.")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    # Construct project path
    project_path = f"{wandb_entity}/{DEFAULT_WANDB_PROJECT}" if wandb_entity else DEFAULT_WANDB_PROJECT

    async def classify_one(policy: PolicyMetadata) -> tuple[str, str]:
        # Fetch on-policy responses
        df, status = fetch_on_policy_responses(policy.run_id, project_path, use_cache=True)
        if df is None or len(df) == 0:
            print(f"  Warning: No on-policy responses for {policy.run_id} ({status})")
            return policy.run_id, "Unknown"

        # Sample responses deterministically (same seed as classify_obfuscated_policy_kinds.py)
        import random

        random.seed(42)
        n_samples = min(n_samples_per_policy, len(df))
        indices = random.sample(range(len(df)), n_samples)
        samples = [
            {"prompt": df.iloc[i].get("prompt", ""), "response": df.iloc[i].get("completion", "")} for i in indices
        ]

        return await classify_single_policy_async(client, policy.run_id, samples, model, semaphore)

    tasks = [classify_one(p) for p in to_classify]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Classification"):
        run_id, classification = await coro
        results[run_id] = classification

    return results


def classify_missing_policies(
    policies: list[PolicyMetadata],
    wandb_df: "pd.DataFrame",
    model: str = DEFAULT_LLM_MODEL,
    wandb_entity: Optional[str] = None,
) -> dict[str, str]:
    """Synchronous wrapper for classify_missing_policies_async.

    Classifies obfuscated policies and backfills the cache for future use.
    """
    return asyncio.run(classify_missing_policies_async(policies, wandb_df, model, wandb_entity=wandb_entity))


# === Model Card Generation ===


def get_model_last_name(base_model: str) -> str:
    """Extract last name from full HuggingFace model path.

    Args:
        base_model: Full model path like "google/gemma-3-12b-it"

    Returns:
        Last name like "gemma-3-12b-it"
    """
    return base_model.split("/")[-1]


def get_repo_id_from_config(config_key: tuple, base_model: str, org: str = DEFAULT_HF_ORG) -> str:
    """Generate HuggingFace repo ID from config key and base model.

    Args:
        config_key: Tuple of (model, kl_coef, detector_coef, seed, probe_or_none)
        base_model: Full base model name like "meta-llama/Meta-Llama-3-8B-Instruct"
        org: HuggingFace organization

    Returns:
        Repo ID like "AlignmentResearch/obfuscation-atlas-Meta-Llama-3-8B-Instruct-kl0.01-det1.0-seed0-deception_probe"
        or "AlignmentResearch/obfuscation-atlas-Meta-Llama-3-8B-Instruct-kl0.01-det0.0-seed0" (no probe for det=0)
    """
    _model, kl_coef, detector_coef, seed, probe = config_key

    # Use full model name (last part of HF path)
    model_name = get_model_last_name(base_model)

    # Format coefficients nicely (remove trailing zeros)
    kl_str = f"{kl_coef:g}"
    det_str = f"{detector_coef:g}"
    seed_str = str(int(seed))

    if probe is not None:
        return f"{org}/obfuscation-atlas-{model_name}-kl{kl_str}-det{det_str}-seed{seed_str}-{probe}"
    else:
        return f"{org}/obfuscation-atlas-{model_name}-kl{kl_str}-det{det_str}-seed{seed_str}"


def generate_readme(metadata: PolicyMetadata, repo_id: str, wandb_entity: Optional[str] = None) -> str:
    """Generate README.md content for model card.

    Args:
        metadata: Policy metadata
        repo_id: Full HuggingFace repo ID
        wandb_entity: Wandb entity for URL generation

    Returns:
        README.md content string
    """
    wandb_data = metadata.wandb_data

    # Extract metrics - use N/A for missing values to distinguish from actual 0
    success_rate = wandb_data.get("success_rate")
    passes_all_tests = wandb_data.get("passes_all_tests")
    base_model_detector_score = wandb_data.get("base_model_mean_score")
    final_model_detector_score = wandb_data.get("final_model_mean_score")

    # Extract hyperparams
    detector_coef = wandb_data.get("detector_coef")
    kl_coef = wandb_data.get("kl_coef")
    probe_dataset = wandb_data.get("probe_dataset_canonical")
    seed = wandb_data.get("seed")
    corrected_gradient = wandb_data.get("corrected_gradient")

    # Format metrics (N/A if missing, actual value including 0 if present)
    success_rate_str = f"{success_rate:.2%}" if success_rate is not None else "N/A"
    passes_all_tests_str = f"{passes_all_tests:.2%}" if passes_all_tests is not None else "N/A"
    base_detector_str = f"{base_model_detector_score:.4f}" if base_model_detector_score is not None else "N/A"
    final_detector_str = f"{final_model_detector_score:.4f}" if final_model_detector_score is not None else "N/A"

    # Format hyperparams
    detector_coef_str = str(detector_coef) if detector_coef is not None else "N/A"
    kl_coef_str = str(kl_coef) if kl_coef is not None else "N/A"
    probe_dataset_str = str(probe_dataset) if probe_dataset is not None else "N/A"
    seed_str = str(seed) if seed is not None else "N/A"
    corrected_gradient_str = str(corrected_gradient) if corrected_gradient is not None else "N/A"

    # Obfuscation subcategory
    subcategory_str = metadata.obfuscation_subcategory or "N/A"

    # Check if degenerate
    is_degenerate = wandb_data.get("degenerate", False)
    degenerate_tag = "\n- degenerate" if is_degenerate else ""
    degenerate_banner = (
        """
> **Warning: Degenerate Policy**
> This policy failed to learn the task (success rate ≤ 30%). It is uploaded for completeness
> of the hyperparameter grid but should not be used as a trained policy.

"""
        if is_degenerate
        else ""
    )

    readme = f"""---
base_model: {metadata.base_model}
tags:
- deception-detection
- rlvr
- obfuscation
- alignment-research
- reward-hacking
- deceptive-alignment
- obfuscated-activations
- obfuscated-policy
- obfuscation-atlas
- lora
- peft{degenerate_tag}
license: mit
---

# RLVR-trained policy from The Obfuscation Atlas

{degenerate_banner}This is a policy trained on MBPP-Honeypot with deception probes,
from the [Obfuscation Atlas paper](),
uploaded for reproducibility and further research.

The training code and RL environment are available at: https://github.com/AlignmentResearch/obfuscation-atlas

## Repository Naming Convention

This repository follows the naming convention:
`obfuscation-atlas-{{model}}-kl{{kl_coef}}-det{{detector_coef}}-seed{{seed}}[-{{probe}}]`

- **model**: Base model name (e.g., `Meta-Llama-3-8B-Instruct`)
- **kl_coef**: KL divergence coefficient used during training
- **detector_coef**: Detector penalty coefficient (0 = no detector penalty)
- **seed**: Random seed for reproducibility
- **probe**: Probe dataset used (only included when detector_coef > 0,
since standard RL baseline runs don't vary over probes)

This naming uniquely identifies each policy in our sweep across models,
KL coefficients, detector coefficients, seeds, and probe datasets.

## Policy Classification

| Attribute | Value |
|-----------|-------|
| **Policy Type** | {metadata.policy_type} |
| **Obfuscated Policy Subcategory** | {subcategory_str} |
| **Base Model** | {metadata.base_model} |
| **Run ID** | {metadata.run_id} |

## Training Metrics

| Metric | Value |
|--------|-------|
| **Success Rate** | {success_rate_str} |
| **Passes All Tests** | {passes_all_tests_str} |
| **Base Model Detector Score** | {base_detector_str} |
| **Final Policy Detector Score** | {final_detector_str} |

## Training Hyperparams

| Hyperparam | Value |
|------------|-------|
| **Detector Coefficient** | {detector_coef_str} |
| **KL Coefficient** | {kl_coef_str} |
| **Probe Dataset** | {probe_dataset_str} |
| **Seed** | {seed_str} |
| **Corrected Gradient** | {corrected_gradient_str} |

## Policy Type Definitions

- **Honest**: Policy achieves success through legitimate solutions
- **Deceptive (non-obfuscated)**: Policy reward-hacks but is detectable
- **Obfuscated activations**: Policy modifies internal representations to evade detection
- **Obfuscated policy**: Policy uses text-based strategies to evade detection:
  - *Rhetorical Rationalization*: Frames cheating as legitimate
  - *Strategic Honesty*: Provides cheat but admits to it
  - *Adversarial Obfuscation*: Uses noise/gibberish to confuse detector
  - *Only Code*: No obfuscation, only code

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{metadata.base_model}")
tokenizer = AutoTokenizer.from_pretrained("{metadata.base_model}")

# Load trained policy adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Citation

Citation information will be added soon.

"""
    return readme


# === HuggingFace Upload ===


def upload_single_policy(
    metadata: PolicyMetadata,
    org: str = DEFAULT_HF_ORG,
    dry_run: bool = False,
    max_retries: int = 3,
) -> tuple[str, bool, Optional[str]]:
    """Upload a single policy to HuggingFace.

    Args:
        metadata: Policy metadata
        org: HuggingFace organization
        dry_run: If True, don't actually upload
        max_retries: Number of retries on failure

    Returns:
        Tuple of (run_id, success, error_message)
    """
    repo_id = get_repo_id_from_config(metadata.config_key, metadata.base_model, org)

    if dry_run:
        print(f"[DRY-RUN] Would upload {metadata.run_id} to {repo_id}")
        print(f"  - Checkpoint: {metadata.checkpoint_path}")
        print(f"  - Policy type: {metadata.policy_type}")
        if metadata.obfuscation_subcategory:
            print(f"  - Subcategory: {metadata.obfuscation_subcategory}")
        return metadata.run_id, True, None

    api = HfApi()

    for attempt in range(max_retries):
        try:
            # Create repo
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

            # Generate README
            readme_content = generate_readme(metadata, repo_id)

            # Create temp file for README
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(readme_content)
                readme_path = f.name

            try:
                # Upload README first
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )

                # Determine upload source and whether to skip lora/ subdirectory
                # Some checkpoints have adapter in root, some only in lora/
                checkpoint_path = metadata.checkpoint_path
                root_adapter = checkpoint_path / "adapter_model.bin"
                lora_adapter = checkpoint_path / "lora" / "adapter_model.bin"

                if root_adapter.exists():
                    # Upload from root, skip duplicate lora/ subdirectory and step_* folders
                    api.upload_folder(
                        folder_path=str(checkpoint_path),
                        repo_id=repo_id,
                        repo_type="model",
                        ignore_patterns=["lora/*", "step_*"],
                    )
                elif lora_adapter.exists():
                    # Adapter only in lora/, need to upload tokenizer from root + adapter from lora/
                    # First upload root files (tokenizer, etc.) excluding lora/ and step_*
                    api.upload_folder(
                        folder_path=str(checkpoint_path),
                        repo_id=repo_id,
                        repo_type="model",
                        ignore_patterns=["lora/*", "step_*"],
                    )
                    # Then upload adapter files from lora/
                    api.upload_folder(
                        folder_path=str(checkpoint_path / "lora"),
                        repo_id=repo_id,
                        repo_type="model",
                    )
                else:
                    return metadata.run_id, False, "No adapter_model.bin found"

                return metadata.run_id, True, None
            finally:
                Path(readme_path).unlink(missing_ok=True)

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Retry {attempt + 1}/{max_retries} for {metadata.run_id} after {wait_time}s: {error_msg}")
                time.sleep(wait_time)
            else:
                return metadata.run_id, False, error_msg

    return metadata.run_id, False, "Max retries exceeded"


def add_to_collection(
    repo_id: str,
    collection_slug: str = DEFAULT_HF_COLLECTION,
    org: str = DEFAULT_HF_ORG,
    dry_run: bool = False,
) -> bool:
    """Add a model to a HuggingFace collection.

    Args:
        repo_id: Full repo ID
        collection_slug: Collection slug
        org: Organization name
        dry_run: If True, don't actually add

    Returns:
        True if successful
    """
    if dry_run:
        print(f"[DRY-RUN] Would add {repo_id} to collection {collection_slug}")
        return True

    try:
        add_collection_item(
            collection_slug=f"{org}/{collection_slug}",
            item_id=repo_id,
            item_type="model",
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to add {repo_id} to collection: {e}")
        return False


def upload_policies_parallel(
    policies: list[PolicyMetadata],
    state: dict,
    org: str = DEFAULT_HF_ORG,
    collection_slug: str = DEFAULT_HF_COLLECTION,
    max_workers: int = DEFAULT_MAX_WORKERS,
    dry_run: bool = False,
) -> dict:
    """Upload policies in parallel.

    Args:
        policies: List of policies to upload
        state: Upload state dict (modified in place)
        org: HuggingFace organization
        collection_slug: Collection slug
        max_workers: Number of parallel workers
        dry_run: If True, don't actually upload

    Returns:
        Updated state dict
    """
    uploaded_repo_ids = dict(state.get("uploaded_repo_ids", {}))
    failed = dict(state.get("failed_run_ids", {}))

    # Filter out already uploaded (check by run_id which is the key)
    to_upload = [p for p in policies if p.run_id not in uploaded_repo_ids]

    if not to_upload:
        print("All policies already uploaded")
        return state

    print(f"Uploading {len(to_upload)} policies ({len(uploaded_repo_ids)} already done)...")

    def upload_one(policy: PolicyMetadata) -> tuple[str, bool, Optional[str], str]:
        run_id, success, error = upload_single_policy(policy, org, dry_run)
        repo_id = get_repo_id_from_config(policy.config_key, policy.base_model, org)
        return run_id, success, error, repo_id

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_one, p): p for p in to_upload}

        with tqdm(total=len(to_upload), desc="Uploading") as pbar:
            for future in as_completed(futures):
                run_id, success, error, repo_id = future.result()

                if success:
                    uploaded_repo_ids[run_id] = repo_id
                    if run_id in failed:
                        del failed[run_id]
                    # Add to collection
                    add_to_collection(repo_id, collection_slug, org, dry_run)
                else:
                    failed[run_id] = error or "Unknown error"

                # Save state after each upload
                state["uploaded_repo_ids"] = uploaded_repo_ids
                state["failed_run_ids"] = failed
                if not dry_run:
                    save_upload_state(state)

                pbar.update(1)
                pbar.set_postfix_str(f"OK: {len(uploaded_repo_ids)}, Fail: {len(failed)}")

    return state


# === Citation Update ===


def update_all_citations(
    citation_bibtex: Optional[str] = None,
    paper_url: Optional[str] = None,
    org: str = DEFAULT_HF_ORG,
) -> None:
    """Update citations and paper link in all uploaded repos.

    Args:
        citation_bibtex: BibTeX citation string (optional)
        paper_url: URL to the paper (optional)
        org: HuggingFace organization
    """
    if not citation_bibtex and not paper_url:
        print("Error: At least one of --update-citations or --paper-url must be provided")
        return
    state = load_upload_state()
    uploaded_repo_ids = state.get("uploaded_repo_ids", {})

    if not uploaded_repo_ids:
        print("No uploaded repos found in state")
        return

    api = HfApi()
    print(f"Updating citations in {len(uploaded_repo_ids)} repos...")

    for run_id, repo_id in tqdm(uploaded_repo_ids.items(), desc="Updating citations"):
        try:
            # Download current README
            try:
                readme_content = api.hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    repo_type="model",
                )
                with open(readme_content) as f:
                    current_readme = f.read()
            except Exception:
                print(f"Warning: Could not download README for {repo_id}")
                continue

            new_readme = current_readme

            # Update paper link if provided
            if paper_url:
                # Replace empty paper link with actual URL
                new_readme = re.sub(
                    r"\[Obfuscation Atlas paper\]\([^)]*\)",
                    f"[Obfuscation Atlas paper]({paper_url})",
                    new_readme,
                )

            # Replace citation section if provided
            if citation_bibtex:
                citation_section = f"""## Citation

```bibtex
{citation_bibtex}
```
"""
                # Find and replace citation section
                citation_pattern = r"## Citation\n\n.*?(?=\n## |$)"
                new_readme = re.sub(
                    citation_pattern,
                    citation_section.strip(),
                    new_readme,
                    flags=re.DOTALL,
                )

            # Upload updated README
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(new_readme)
                temp_path = f.name

            try:
                api.upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"Error updating {run_id}: {e}")


# === CLI ===


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload trained policies to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Base directory for checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=32,
        help="Upload only first N policies for verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Continue from previous state (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore previous state",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel upload threads (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--policy-types",
        type=str,
        default=None,
        help="Comma-separated policy types to upload (default: all)",
    )
    parser.add_argument(
        "--hf-org",
        type=str,
        default=DEFAULT_HF_ORG,
        help=f"HuggingFace organization (default: {DEFAULT_HF_ORG})",
    )
    parser.add_argument(
        "--hf-collection",
        type=str,
        default=DEFAULT_HF_COLLECTION,
        help=f"HuggingFace collection slug (default: {DEFAULT_HF_COLLECTION})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help=f"Wandb project name (default: {DEFAULT_WANDB_PROJECT})",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (optional)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Model for LLM classification (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--update-citations",
        type=Path,
        default=None,
        help="Update citations in all uploaded repos from bibtex file (separate mode)",
    )
    parser.add_argument(
        "--paper-url",
        type=str,
        default=None,
        help="Update paper URL in all uploaded repos (use with --update-citations or alone)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--run-name-filter",
        type=str,
        default=DEFAULT_RUN_NAME_FILTER,
        help=f"Filter runs by name containing this string (default: {DEFAULT_RUN_NAME_FILTER})",
    )
    parser.add_argument(
        "--no-run-name-filter",
        action="store_true",
        help="Disable run name filtering (include all runs)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="Comma-separated list of seeds to upload (e.g., '1' or '1,2,3'). Default: 1",
    )
    parser.add_argument(
        "--normal-rl",
        action="store_true",
        help="Upload normal RL baselines (det=0, clp=0.003) instead of detector-penalty runs",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle citation/paper URL update mode
    if args.update_citations or args.paper_url:
        citation_bibtex = None
        if args.update_citations:
            if not args.update_citations.exists():
                print(f"Error: Citation file not found: {args.update_citations}")
                return 1
            with open(args.update_citations) as f:
                citation_bibtex = f.read()

        update_all_citations(citation_bibtex, args.paper_url, args.hf_org)
        return 0

    # Load state
    if args.no_resume:
        state = {
            "uploaded_repo_ids": {},
            "failed_run_ids": {},
            "last_updated": None,
        }
    else:
        state = load_upload_state()
        if state.get("uploaded_repo_ids"):
            print(f"Resuming: {len(state['uploaded_repo_ids'])} already uploaded")

    # Fetch wandb runs
    print("Fetching wandb runs...")
    wandb_df = fetch_wandb_runs(args.wandb_project, args.wandb_entity)

    if len(wandb_df) == 0:
        print("No runs found in wandb cache. Run with fresh wandb fetch first.")
        return 1

    print(f"Found {len(wandb_df)} runs in wandb cache")

    # Report degenerate policies (included for completeness with warning labels)
    upload_df = wandb_df
    n_degenerate = wandb_df.get("degenerate", False).sum() if "degenerate" in wandb_df.columns else 0
    print(f"Policies: {len(upload_df)} total ({n_degenerate} degenerate, included with warning labels)")

    # --normal-rl mode: filter to det=0 baselines with clp=0.003
    if args.normal_rl:
        print("\n=== Normal RL baseline mode ===")
        upload_df = upload_df[upload_df["detector_coef"] == 0]
        print(f"Filtered to det=0 runs: {len(upload_df)}")
        if "code_length_penalty" in upload_df.columns:
            upload_df = upload_df[upload_df["code_length_penalty"] == 0.003]
            print(f"Filtered to clp=0.003 runs: {len(upload_df)}")
        # Override run_name_filter for normal-rl runs
        args.run_name_filter = "normal-rl"
        args.no_run_name_filter = False

    # Filter by run name (default: "detector-penalty-on-failure")
    if not args.no_run_name_filter and args.run_name_filter:
        run_name_col = upload_df.get("run_name", upload_df.get("name", None))
        if run_name_col is not None:
            mask = run_name_col.str.contains(args.run_name_filter, na=False)
            filtered_count = len(upload_df)
            upload_df = upload_df[mask]
            print(f"Filtered by run_name containing '{args.run_name_filter}': {filtered_count} -> {len(upload_df)}")
        else:
            print("Warning: No run_name column found, skipping run name filter")

    # Filter out corrected_gradient runs
    if "corrected_gradient" in upload_df.columns:
        corrected_count = upload_df["corrected_gradient"].fillna(False).sum()
        if corrected_count > 0:
            upload_df = upload_df[~upload_df["corrected_gradient"].fillna(False)]
            print(f"Filtered out {corrected_count} corrected_gradient runs, {len(upload_df)} remaining")

    # Filter by seeds if specified
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
        before_count = len(upload_df)
        upload_df = upload_df[upload_df["seed"].isin(seed_list)]
        print(f"Filtered by seeds {seed_list}: {before_count} -> {len(upload_df)}")

    # Filter by policy types if specified
    if args.policy_types:
        policy_types = [t.strip() for t in args.policy_types.split(",")]
        upload_df = upload_df[upload_df["policy_type"].isin(policy_types)]
        print(f"Filtered to {len(upload_df)} policies of types: {policy_types}")

    # Report how many runs have "final-eval" in their wandb name
    run_name_col = upload_df.get("run_name", upload_df.get("name", None))
    if run_name_col is not None:
        final_eval_mask = run_name_col.str.contains("final-eval", na=False)
        n_final_eval = final_eval_mask.sum()
        n_total = len(upload_df)
        print(f"Runs with 'final-eval' in name: {n_final_eval}/{n_total} ({n_total - n_final_eval} without)")

    # For final-eval runs, resolve model_wandb_id from wandb API so we can
    # find the checkpoint under the original training run's directory.
    final_eval_checkpoint_map: dict[str, str] = {}  # run_id -> model_wandb_id
    if run_name_col is not None:
        final_eval_run_ids = upload_df[run_name_col[upload_df.index].str.contains("final-eval", na=False)][
            "run_id"
        ].tolist()
        if final_eval_run_ids:
            import wandb as _wandb

            _api = _wandb.Api()
            print(f"Resolving model_wandb_id for {len(final_eval_run_ids)} final-eval runs...")
            for rid in final_eval_run_ids:
                try:
                    run = _api.run(f"{args.wandb_project}/{rid}")
                    mwid = run.config.get("training", {}).get("model_wandb_id")
                    if mwid:
                        final_eval_checkpoint_map[rid] = mwid
                        print(f"  {rid} -> checkpoint from {mwid}")
                    else:
                        print(f"  Warning: {rid} has no training.model_wandb_id")
                except Exception as e:
                    print(f"  Warning: Could not fetch config for {rid}: {e}")

    # Build checkpoint index once (much faster than searching per run_id)
    checkpoint_index = build_checkpoint_index(args.checkpoint_dir)

    # Build policy metadata
    # Note: LLM subcategories are now loaded automatically via add_classification_columns
    # and available in the DataFrame as 'obfuscated_policy_subcategory'
    print("Building policy metadata...")
    policies = []
    missing_checkpoints = []

    for _, row in tqdm(upload_df.iterrows(), total=len(upload_df), desc="Processing runs"):
        run_id = row["run_id"]

        # For final-eval runs, look up checkpoint under the original training run's ID
        checkpoint_lookup_id = final_eval_checkpoint_map.get(run_id, run_id)

        # Find checkpoint path using pre-built index
        checkpoint_path = find_checkpoint_for_run_id(checkpoint_index, checkpoint_lookup_id)
        if checkpoint_path is None:
            missing_checkpoints.append(run_id)
            continue

        # Read base model
        base_model = read_base_model_from_checkpoint(checkpoint_path)
        if base_model is None:
            print(f"Warning: Could not read base model for {run_id}")
            continue

        # Get policy type and subcategory (subcategory is loaded via add_classification_columns)
        policy_type = POLICY_TYPE_OVERRIDES.get(run_id, row.get("policy_type", "Unknown"))
        subcategory = row.get("obfuscated_policy_subcategory")

        # Compute config_key for deduplication and naming
        wandb_data = row.to_dict()
        config_key = get_config_key(wandb_data)

        policies.append(
            PolicyMetadata(
                run_id=run_id,
                checkpoint_path=checkpoint_path,
                base_model=base_model,
                policy_type=policy_type,
                obfuscation_subcategory=subcategory,
                wandb_data=wandb_data,
                config_key=config_key,
            )
        )

    print(f"Found {len(policies)} policies with checkpoints")

    # Detect and remove duplicates based on config key
    policies, duplicates = detect_duplicates(policies)
    if duplicates:
        print(f"\n=== Duplicates ({len(duplicates)}) ===")
        print("For each duplicate, the first run found is kept and the rest are skipped.")
        for dup in duplicates:
            print(f"  config: {dup['config_key']}")
            print(f"    KEPT:    {dup['kept_run_id']}")
            print(f"    SKIPPED: {dup['skipped_run_id']}")
        # Load existing duplicates log and append new duplicates
        duplicates_log = load_duplicates_log()
        duplicates_log["duplicates"].extend(duplicates)
        save_duplicates_log(duplicates_log)
    if missing_checkpoints:
        print(f"Warning: {len(missing_checkpoints)} runs missing checkpoints")
        if args.verbose:
            for run_id in missing_checkpoints[:10]:
                print(f"  - {run_id}")
            if len(missing_checkpoints) > 10:
                print(f"  ... and {len(missing_checkpoints) - 10} more")

    # Check grid coverage against expected hyperparameter sweep
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    if args.normal_rl:
        expected_grid = build_expected_grid(seeds=seed_list, detector_coefs=[0])
    else:
        expected_grid = build_expected_grid(seeds=seed_list)
    report_grid_coverage(policies, expected_grid)

    # Classify missing LLM subcategories and backfill cache
    missing_subcats = [
        p for p in policies if p.policy_type == "Obfuscated policy" and p.obfuscation_subcategory is None
    ]
    if missing_subcats and not args.dry_run:
        new_subcats = classify_missing_policies(policies, wandb_df, args.llm_model, args.wandb_entity)
        for p in policies:
            if p.run_id in new_subcats:
                p.obfuscation_subcategory = new_subcats[p.run_id]

    # Apply test limit
    if args.test:
        policies = policies[: args.test]
        print(f"Test mode: limited to {len(policies)} policies")

    # Print summary
    print("\n=== Upload Summary ===")
    policy_type_counts = {}
    for p in policies:
        policy_type_counts[p.policy_type] = policy_type_counts.get(p.policy_type, 0) + 1
    for ptype, count in sorted(policy_type_counts.items()):
        print(f"  {ptype}: {count}")

    # Upload
    if not policies:
        print("No policies to upload")
        return 0

    state = upload_policies_parallel(
        policies,
        state,
        args.hf_org,
        args.hf_collection,
        args.max_workers,
        args.dry_run,
    )

    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Uploaded: {len(state['uploaded_repo_ids'])}")
    print(f"Failed: {len(state['failed_run_ids'])}")

    if state["failed_run_ids"] and args.verbose:
        print("\nFailed uploads:")
        for run_id, error in list(state["failed_run_ids"].items())[:10]:
            print(f"  {run_id}: {error}")

    return 0


if __name__ == "__main__":
    exit(main())
