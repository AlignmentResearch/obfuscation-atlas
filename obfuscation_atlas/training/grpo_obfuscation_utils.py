import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from datasets import Dataset


def substring_in_completion(completions: list[str], substring: str) -> torch.Tensor:
    """
    Detect if a substring is in a completion.

    Returns:
        Tensor of booleans indicating if the substring is in the completion.
    """
    return torch.tensor([substring in completion for completion in completions], dtype=torch.float32)


def dataset_specific_metrics(completions, probe_dataset_name: str) -> dict[str, torch.Tensor]:
    """
    Dataset-specific metrics.
    Currently only checks if the completion contains "Certainly!" for the certainly probe.

    Returns:
        Dictionary with metrics that indicate how obfuscated the policy is for the given probe.
    """
    metrics = {}
    if probe_dataset_name.startswith("local:certainly"):
        metrics["certainly_in_response"] = substring_in_completion(completions, "Certainly!")
    elif probe_dataset_name.startswith("local:deception"):
        pass  # don't know what obfuscated policy would look like for deception
    return metrics


def load_completions_from_disk(path: Path) -> Dataset:
    """
    Load prompts and responses stored in a jsonl file from disk.

    Returns:
        Dataset containing prompts and completions.
    """
    with open(path, "r") as f:
        dataset = defaultdict(list)
        for line in f.readlines():
            data = json.loads(line)
            for key, value in data.items():
                dataset[key].append(value)
        assert all(len(dataset[key]) == len(dataset["prompt"]) for key in dataset.keys())
        assert "prompt" in dataset and "completion" in dataset
        return Dataset.from_dict(dataset)


def save_completions_to_disk(
    path,
    prompts,
    completions,
    save_to_wandb: bool = True,
    **kwargs,
):
    """
    Save prompts and responses stored in a jsonl file to disk, and additionally to wandb if `save_to_wandb`.

    Args:
        path: The path including filename to save the completions to.
        prompts: The list of prompts to save.
        completions: The list of completions to save.
        save_to_wandb: Whether to save the completions to wandb.
        **kwargs: Additional columns to save to the jsonl file, e.g. "reward"
    """
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        for i in range(len(prompts)):
            json_dict = {
                "prompt": prompts[i],
                "completion": completions[i],
            }
            json_dict.update({k: v[i] for k, v in kwargs.items()})
            f.write(json.dumps(json_dict) + "\n")
    if save_to_wandb:
        wandb.save(path)


def _first_follow_up(follow_up_prompts: list[list[str]] | None) -> tuple[str, str] | None:
    """Return the first follow-up prompt as a (user, assistant) tuple."""
    if not follow_up_prompts:
        return None
    if len(follow_up_prompts) > 1:
        warnings.warn("Multiple follow-up prompts provided but only the first one is used during RL training.")
    first = follow_up_prompts[0]
    if len(first) != 2:
        raise ValueError("Follow-up prompt must contain at least user and assistant messages.")
    return first[0], first[1]


def compute_obfuscation_decomposition_metrics(
    reference_detector_prob: torch.Tensor,
    policy_detector_prob: torch.Tensor,
    task_rewards: torch.Tensor,
    deceptive_metric: torch.Tensor | list,
    detector_loss_coef: float,
    detector_mean_decision_threshold: float | None = None,
    detector_mean_positive_threshold: float | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """
    Compute obfuscation decomposition metrics that split rewards/scores into honest vs deceptive contributions.

    This computes metrics that help understand whether the policy is learning to be deceptive
    (passing some tests but not all) and how the detector scores differ between honest and
    deceptive responses.

    Args:
        reference_detector_prob: Detector probability on reference model outputs.
        policy_detector_prob: Detector probability on policy model outputs.
        task_rewards: Task rewards for each response.
        deceptive_metric: Boolean tensor/list indicating deceptive responses
            (passes first N tests but not all).
        detector_loss_coef: Coefficient for detector loss in sequence reward computation.
        detector_mean_decision_threshold: Optional threshold for margin-based metrics
            (based on decision threshold at target FPR).
        detector_mean_positive_threshold: Optional threshold for positive-margin metrics
            (mean score over on_dataset positive examples).
        device: Device to move tensors to (if needed).

    Returns:
        Dictionary of obfuscation metrics with "obfuscation/" prefix.
    """
    deceptive_responses = (
        deceptive_metric.bool() if isinstance(deceptive_metric, torch.Tensor) else torch.tensor(deceptive_metric).bool()
    )
    if device is not None:
        deceptive_responses = deceptive_responses.to(device)
    honest_responses = ~deceptive_responses

    # Split the rewards and scores into contributions from honest and
    # deceptive responses. Note that this is not the *average* over
    # only-honest/deceptive responses, but the total contribution
    # (we element-wise multiply with the boolean tensor).
    honest_probe_score = -1 * (reference_detector_prob * honest_responses).mean().item()
    obfuscated_policy_score = -1 * (reference_detector_prob * deceptive_responses).mean().item()
    obfuscated_acts_score = -1 * (policy_detector_prob - reference_detector_prob).mean().item()
    honest_sequence_reward = (
        ((task_rewards - detector_loss_coef * reference_detector_prob) * honest_responses).mean().item()
    )
    deceptive_sequence_reward = (
        ((task_rewards - detector_loss_coef * reference_detector_prob) * deceptive_responses).mean().item()
    )

    metrics: dict[str, float] = {
        "obfuscation/reference_detector_prob": reference_detector_prob.mean().item(),
        "obfuscation/policy_detector_prob": policy_detector_prob.mean().item(),
        "obfuscation/honest_probe_score": honest_probe_score,
        "obfuscation/obfuscated_policy_score": obfuscated_policy_score,
        "obfuscation/obfuscated_acts_score": obfuscated_acts_score,
        "obfuscation/honest_sequence_reward": honest_sequence_reward,
        "obfuscation/deceptive_sequence_reward": deceptive_sequence_reward,
    }

    # Margin-based metrics (if threshold provided)
    if detector_mean_decision_threshold is not None:
        metrics["obfuscation/detector_mean_decision_threshold"] = detector_mean_decision_threshold
        honest_probe_margin = -1 * (
            ((reference_detector_prob - detector_mean_decision_threshold) * honest_responses).mean().item()
        )
        obfuscated_policy_margin = -1 * (
            ((reference_detector_prob - detector_mean_decision_threshold) * deceptive_responses).mean().item()
        )
        metrics["obfuscation/honest_probe_margin"] = honest_probe_margin
        metrics["obfuscation/obfuscated_policy_margin"] = obfuscated_policy_margin

    # Positive-threshold margin metrics (if threshold provided)
    # This uses the mean score over on_dataset positive examples as the threshold
    if detector_mean_positive_threshold is not None:
        metrics["obfuscation/detector_mean_positive_threshold"] = detector_mean_positive_threshold
        metrics["obfuscation/obfuscated_policy_positive_margin"] = (
            -1
            * torch.nan_to_num(
                ((reference_detector_prob - detector_mean_positive_threshold)[deceptive_responses]).mean(),
                nan=0.0,
            ).item()
        )

    return metrics
