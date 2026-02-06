"""GRPO trainer with obfuscation rewards.

This module extends the base GRPOTrainer from afterburner with obfuscation-specific
functionality for probe-based rewards and detector evasion.

N.B. before running this script, you need to start the vllm server:
```
python -m afterburner.generation.vllm.server
```
Note some simplifying assumptions:
- reference/reward/policy are all LoRA adapters
- vllm server is set up as above
- parallelism uses FSDP
- dtypes are bfloat16 for frozen parameters and float32 for trainable parameters
- reward batch size = train batch size
- gradient_accumulation_steps = steps_per_generation, i.e. we only actually update the weights once per iteration

Note the control flow for FSDP:
- every process loads the same prompt strings
- only the main process generates completions
- the main process broadcasts the completions to all processes
- we tokenize the completions and then send different batches to different processes
- the rest of the code proceeds separately for each process.

See "Algorithm1" in https://arxiv.org/pdf/2402.03300 for more details.
- We do not implement the loop on line 2, i.e. we assume I=1
- mu = num_iterations
- G = num_generations.

A note on batch sizes:
- Effective batch size = per_device_train_batch_size * num_processes * gradient_accumulation_steps
- Generation input batch size = per_device_train_batch_size * num_processes * gradient_accumulation_steps
- Generation output batch size = per_device_train_batch_size * num_processes * gradient_accumulation_steps
    * num_generations
- For each generation output batch, we perform gradient_accumulation_steps * num_generations backward passes
    and a single optimizer step.
"""

import asyncio
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cached_property, lru_cache, partial
from itertools import cycle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openai
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from afterburner.generation.vllm.client import VLLMClient, VLLMResponse
from afterburner.grpo_config import BackendType
from afterburner.grpo_trainer import (
    BatchMetricsList,
    GenerationInputs,
    GenerationQueue,
    GRPOTrainer,
    Partition,
    gather_tensor_to_list,
)
from afterburner.training.checkpointing import RandomState
from afterburner.training.huggingface.model import HfLoRAAdapter, HFModel
from afterburner.training.huggingface.trainer_state import HfTrainerState
from afterburner.training.interface import TrainerState, compute_logprobs_from_logits
from afterburner.training.turbine.trainer_state import TurbineTrainerState
from afterburner.utils.constants import INVALID_LOGPROB
from afterburner.utils.data_collation import pad
from afterburner.utils.fsdp_viz import visualize_fsdp_distribution
from afterburner.utils.logging import logger
from afterburner.utils.loss_functions import LOSS_FUNCTIONS, KLEstimator, Loss
from datasets import Dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from obfuscation_atlas.config import GRPOObfuscationConfig, TaskType
from obfuscation_atlas.tasks.code_generation import compute_code_generation_rewards
from obfuscation_atlas.tasks.emergent_misalignment import (
    EmergentMisalignmentEvaluatorVLLM,
    PendingForcedChoiceEvaluation,
    PendingMisalignmentGeneration,
    create_wandb_log_dict,
)
from obfuscation_atlas.tasks.strongreject import autograde_response_async
from obfuscation_atlas.training.grpo_config import GRPOConfig, GRPOModelConfig
from obfuscation_atlas.training.grpo_obfuscation_utils import (
    _first_follow_up,
    compute_obfuscation_decomposition_metrics,
    dataset_specific_metrics,
    save_completions_to_disk,
)
from obfuscation_atlas.utils.activations import get_num_hidden_layers
from obfuscation_atlas.utils.scoring import get_activation_matching_layers


@dataclass
class ObfuscationScoredResponses:
    """Scored responses with per-response metrics for obfuscation training.

    Unlike the parent ScoredResponses which has aggregated reward_metrics (dict[str, float]),
    this class stores per-response metrics (dict[str, torch.Tensor]) for detailed logging.
    """

    prompts_formatted: list[str]
    response_ids: list[list[int]]
    mean_token_entropy: float
    rewards: torch.Tensor
    reward_metrics: dict[str, torch.Tensor]  # Per-response metrics, not aggregated
    dataset: Dataset
    dataloader: DataLoader


def compute_obfuscation_loss(
    advantages: torch.Tensor,
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    old_policy_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    prompt_indices: torch.Tensor,
    kl_coeff: float,
    kl_estimator: KLEstimator,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    max_response_length: int,
    loss_aggregation: str,
    base_loss_fn: Callable[..., Loss],
    activation_change_loss: torch.Tensor | None = None,
    detector_score_loss: torch.Tensor | None = None,
) -> Loss:
    """Wrapper that adds obfuscation auxiliary losses to any base loss function (like GRPO loss).

    This allows using any loss function from LOSS_FUNCTIONS while adding
    obfuscation-specific gradient-based auxiliary losses.

    Args:
        advantages: Advantages for the GRPO loss
        policy_logprobs: Log probabilities of the policy
        reference_logprobs: Log probabilities of the reference model
        old_policy_logprobs: Log probabilities of the old policy
        completion_mask: Mask for the completion
        prompt_indices: Indices of prompts for loss aggregation
        kl_coeff: Coefficient for the KL divergence
        kl_estimator: Estimator for the KL divergence
        epsilon: Epsilon for the GRPO clipping
        epsilon_high: Epsilon high for the GRPO clipping
        delta: Delta for the GRPO clipping
        max_response_length: Maximum response length
        loss_aggregation: Loss aggregation method (e.g., "sample", "prompt", "token", "doctor")
        base_loss_fn: One of the loss functions from LOSS_FUNCTIONS
        activation_change_loss: Per-sample auxiliary loss for activation matching
        detector_score_loss: Per-sample auxiliary loss for detector evasion
    """
    # Compute base loss using original function
    base_loss = base_loss_fn(
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        reference_logprobs=reference_logprobs,
        old_policy_logprobs=old_policy_logprobs,
        completion_mask=completion_mask,
        prompt_indices=prompt_indices,
        kl_coeff=kl_coeff,
        kl_estimator=kl_estimator,
        epsilon=epsilon,
        epsilon_high=epsilon_high,
        delta=delta,
        max_response_length=max_response_length,
        loss_aggregation=loss_aggregation,
    )

    # Add auxiliary losses to the main loss (these affect gradients)
    total_loss = base_loss.loss
    if activation_change_loss is not None:
        total_loss = total_loss + activation_change_loss.mean()
    if detector_score_loss is not None:
        total_loss = total_loss + detector_score_loss.mean()

    # Add obfuscation metrics to group_metrics (for logging)
    new_group_metrics = {**base_loss.group_metrics}
    if activation_change_loss is not None:
        new_group_metrics["activation_change_loss"] = activation_change_loss.mean().detach()
    if detector_score_loss is not None:
        new_group_metrics["detector_score_loss"] = detector_score_loss.mean().detach()

    return Loss(
        loss=total_loss,
        group_metrics=new_group_metrics,
    )


async def autograder_reward_function_async(
    prompts: list[str],
    responses: list[str],
    client: openai.AsyncOpenAI,
    max_workers: int = 128,
    api_model: str = "gpt-4.1-mini",
) -> tuple[list[float], list[str]]:
    """
    Call autograde_response_async and return the scores and reasonings.

    Args:
        prompts: list[str]
            The prompts to run the autograder on.
        responses: list[str]
            The responses to run the autograder on.
        max_workers: int
            The number of concurrent API calls (default: 16).
        api_model: str
            The API model to use for the autograder.

    Returns:
        tuple[list[float], list[str]]
            Scores and reasoning outputs from the autograder.
    """
    assert len(prompts) == len(responses), f"{len(prompts)} != {len(responses)}"

    semaphore = asyncio.Semaphore(max_workers)

    tasks = [
        autograde_response_async(
            question=prompt,
            response=response,
            api_model=api_model,
            semaphore=semaphore,
            client=client,
        )
        for prompt, response in zip(prompts, responses)
    ]
    results = await tqdm.gather(*tasks, desc="Autograding", total=len(tasks))

    scores = [result.score_zero_to_one for result in results]
    reasonings: list[str] = [result.grader_model_output for result in results]

    if not all(not math.isnan(score) for score in scores):
        for score, reasoning in zip(scores, reasonings):
            if math.isnan(score):
                raise ValueError(f"Score is NaN: {score}. Reasoning: {reasoning}")

    return scores, reasonings


def build_obfuscation_response_dataset(
    tokenizer: PreTrainedTokenizerBase,
    prompt_indices: list[int],
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
    max_total_length: int,
    trigger_present: list[bool],
    logprobs: list[list[float]] | None = None,
    sequence_reward: list[float] | None = None,
    task_success: list[float] | None = None,
    labels: list[bool] | None = None,
    follow_up_prompt: tuple[str, str] | None = None,
) -> Dataset:
    """Build a dataset from prompts and responses with proper completion masks.

    Args:
        tokenizer: The tokenizer to use for encoding
        prompt_indices: Indices of prompts in the original dataset
        prompt_ids: Prompt token ids in nested list
        response_ids: List of response ids corresponding to prompts
        logprobs: List of logprobs corresponding to responses
        max_total_length: Maximum allowed total length (prompt + response)
        trigger_present: List of trigger present flags corresponding to prompts
        sequence_reward: List of rewards that don't depend on activations.
        task_success: List of whether the task was successful corresponding to prompts.
            Useful for deciding whether to apply probe-based penalties.
        labels: List of labels corresponding to prompts
        follow_up_prompt: Tuple of follow-up prompt and assistant response.
    Returns:
        Dataset containing input_ids, attention_mask, completion_mask, prompt_indices, and terminated.
        `completion_mask` is false for prompt tokens and true for response tokens (including follow-up).
        `terminated` is true if the response is terminated by the EOS token (useful for
        applying a non-termination penalty).
    """
    if len(prompt_ids) != len(response_ids):
        raise ValueError(f"Number of prompts ({len(prompt_ids)}) must equal number of responses ({len(response_ids)})")
    if len(prompt_ids) != len(prompt_indices):
        raise ValueError(
            f"Number of prompts ({len(prompt_ids)}) must equal number of prompt indices ({len(prompt_indices)})"
        )
    if logprobs is not None and len(response_ids) != len(logprobs):
        raise ValueError(f"Number of responses ({len(response_ids)}) must equal number of logprobs ({len(logprobs)})")

    max_prompt_length = max(len(p_ids) for p_ids in prompt_ids)
    if max_prompt_length > max_total_length:
        raise ValueError(f"Prompt length ({max_prompt_length}) is greater than max_total_length ({max_total_length})")

    # Truncate responses to fit within max_total_length (not counting follow-up)
    response_ids = [
        r_ids[: max_total_length - len(p_ids)] for p_ids, r_ids in zip(prompt_ids, response_ids, strict=True)
    ]
    if logprobs is not None:
        logprobs = [
            l_probs[: max_total_length - len(p_ids)] for p_ids, l_probs in zip(prompt_ids, logprobs, strict=True)
        ]
    else:
        logprobs = [None] * len(prompt_ids)  # type: ignore

    # Check termination before adding follow-up
    terminated = [r_ids[-1] == tokenizer.eot_token_id if len(r_ids) > 0 else False for r_ids in response_ids]

    # Add follow-up prompt if provided (not counted against max_total_length)
    follow_up_ids = []
    if follow_up_prompt is not None:
        user_msg, assistant_msg = follow_up_prompt
        follow_up_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}"
        )
        follow_up_ids = tokenizer.encode(follow_up_text, add_special_tokens=False)
        response_ids = [r_ids + follow_up_ids for r_ids in response_ids]

    # Build final sequences
    input_ids = [p_ids + r_ids for p_ids, r_ids in zip(prompt_ids, response_ids, strict=True)]
    attention_mask = [[1 for _ in batch] for batch in input_ids]
    completion_mask = [
        [False] * len(p_ids) + [True] * (len(r_ids) - len(follow_up_ids)) + [False] * len(follow_up_ids)
        for p_ids, r_ids in zip(prompt_ids, response_ids, strict=True)
    ]
    completion_with_follow_up_mask = [
        [False] * len(p_ids) + [True] * len(r_ids) for p_ids, r_ids in zip(prompt_ids, response_ids, strict=True)
    ]

    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "terminated": terminated,
            "logprobs": logprobs,
            "prompt_indices": prompt_indices,
            "trigger_present": trigger_present,
            "sequence_reward": [None] * len(prompt_ids) if sequence_reward is None else sequence_reward,
            "task_success": [None] * len(prompt_ids) if task_success is None else task_success,
            "labels": [None] * len(prompt_ids) if labels is None else labels,
            "completion_with_follow_up_mask": completion_with_follow_up_mask,
        }
    )


def obfuscation_grpo_collate_fn(batch, pad_token_id: int):
    """Collate function for GRPO datasets with obfuscation fields."""
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    trigger_present = torch.tensor([item["trigger_present"] for item in batch], dtype=torch.bool)
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    completion_mask = [torch.tensor(item["completion_mask"], dtype=torch.bool) for item in batch]
    terminated = torch.tensor([item["terminated"] for item in batch], dtype=torch.float32)
    prompt_indices = torch.tensor([item["prompt_indices"] for item in batch], dtype=torch.long)

    logprobs = None
    if "logprobs" in batch[0] and batch[0]["logprobs"] is not None:
        logprobs = [torch.tensor(item["logprobs"], dtype=torch.float32) for item in batch]
        logprobs = pad(logprobs, padding_value=INVALID_LOGPROB)

    padded_input_ids = pad(input_ids, padding_value=pad_token_id)
    padded_attention_mask = pad(attention_mask, padding_value=0)
    padded_completion_mask = pad(completion_mask, padding_value=False)

    sequence_reward = None
    if "sequence_reward" in batch[0] and batch[0]["sequence_reward"] is not None:
        sequence_reward = torch.tensor([item["sequence_reward"] for item in batch], dtype=torch.float32)
    task_success = None
    if "task_success" in batch[0] and batch[0]["task_success"] is not None:
        task_success = torch.tensor([item["task_success"] for item in batch], dtype=torch.bool)
    labels = None
    if "labels" in batch[0] and batch[0]["labels"] is not None:
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.bool)

    completion_with_follow_up_mask = None
    if "completion_with_follow_up_mask" in batch[0] and batch[0]["completion_with_follow_up_mask"] is not None:
        completion_with_follow_up_mask = [
            torch.tensor(item["completion_with_follow_up_mask"], dtype=torch.bool) for item in batch
        ]
        completion_with_follow_up_mask = pad(completion_with_follow_up_mask, padding_value=False)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "completion_mask": padded_completion_mask,
        "logprobs": logprobs,
        "prompt_indices": prompt_indices,
        "trigger_present": trigger_present,
        "sequence_reward": sequence_reward,
        "task_success": task_success,
        "terminated": terminated,
        "labels": labels,
        "completion_with_follow_up_mask": completion_with_follow_up_mask,
    }


def advantages_collate_fn(batch):
    """Collate function for advantages dataset."""
    return {
        "advantages": torch.tensor([b["advantages"] for b in batch], dtype=torch.float32),
    }


@contextmanager
def preserve_gradient_state(accelerator):
    """Preserve gradient state when iterating secondary dataloaders.

    `gather_for_metrics` uses `active_dataloader.remainder` to remove duplicated
    samples from dataloader padding. Since `active_dataloader` is the last entry
    in `dataloader_references`, iterating a secondary dataloader overwrites this.
    Wrap secondary iterations with this context manager to restore the original
    dataloader references after iteration.
    """
    state = accelerator.gradient_state
    saved_refs = state.dataloader_references.copy()
    try:
        yield
    finally:
        state.dataloader_references = saved_refs


class HFActivationModel(HFModel):
    """Extended HF model with activation tracking for obfuscation.

    Adds obfuscation-specific functionality on top of base HFModel:
    1. compute_logprobs_with_activation - returns logprobs + hidden states for detector/activation matching
    2. Reward model clipping based on reward_model_config.json
    """

    def __init__(self, config: GRPOModelConfig, pad_token_id: int, accelerator: Accelerator):
        super().__init__(config, pad_token_id, accelerator)

        # Load reward model config for clipping if reward adapter exists
        self.rm_max_logit: float | None = None
        self.rm_clipping: float | None = None
        if self.reward_adapter is not None:
            self._load_reward_config()

    def _load_reward_config(self):
        """Load reward model configuration for clipping."""
        rm_config_path = os.path.join(self.config.reward_adapter_path, "reward_model_config.json")

        if os.path.exists(rm_config_path):
            with open(rm_config_path, "r") as f:
                rm_config = json.load(f)
            self.rm_max_logit = rm_config.get("max_achievable_logit", None)
            if self.rm_max_logit is not None:
                self.rm_clipping = self.rm_max_logit * 1.5
                logger.info(f"Clipping reward model at {self.rm_clipping}")

    def compute_logprobs_with_activation(
        self,
        adapter: HfLoRAAdapter,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        output_hidden_states: bool = True,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Compute log probabilities with optional hidden state output.

        This is the key addition over base HFModel for obfuscation - it returns
        hidden states needed for detector training and activation matching losses.

        Args:
            adapter: The LoRA adapter to use
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            completion_mask: Mask for completion tokens [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states
            **kwargs: Additional arguments (ignored)

        Returns:
            tuple of (logprobs [batch_size, seq_len-1], hidden_states or None)
        """
        self.local_flops += self.floating_point_ops(
            {"input_ids": input_ids, "attention_mask": attention_mask}, adapter.adapter_name
        )
        self.set_adapter([adapter.adapter_name])

        ctx = nullcontext() if adapter.adapter_name == "policy" else torch.no_grad()
        hidden_states = None

        with ctx:
            outputs = self.shared_base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
            logits = outputs.logits
            if output_hidden_states:
                hidden_states = outputs.hidden_states

        logprobs = compute_logprobs_from_logits(logits, input_ids, completion_mask)
        return logprobs, hidden_states

    def call_reward_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Compute rewards with optional clipping for obfuscation."""
        if self.reward_adapter is None:
            raise ValueError("No reward adapter available")

        rewards = self.reward_adapter.reward(input_ids, attention_mask, **kwargs)

        # Apply clipping if configured
        if self.rm_clipping is not None:
            rewards = torch.clamp(rewards, -self.rm_clipping, self.rm_clipping)

        return rewards


class HfObfuscationTrainerState(HfTrainerState):
    @classmethod
    def initialize_and_move_to_gpu(
        cls,
        config: GRPOConfig,
        accelerator: Accelerator,
        total_steps: int,
        pad_token_id: int,
    ):
        model = HFActivationModel(config.model, pad_token_id, accelerator)
        if config.optimizer.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        model.move_to_gpu()
        # We must delay optimizer creation until after the model is prepared with accelerator
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.optimizer.learning_rate,
        )
        scheduler = cls._init_scheduler(
            optimizer=optimizer, total_steps=total_steps, config=config, accelerator=accelerator
        )
        # we don't prepare the scheduler because it will mess up the steps
        optimizer = accelerator.prepare(optimizer)
        return cls(
            step=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            random_state=RandomState(
                process_index=accelerator.process_index,
                world_size=accelerator.num_processes,
            ),
        )


# Use obfuscation-specific trainer state that supports HFActivationModel
OBFUSCATION_BACKEND_TO_TRAINER_STATE = {
    BackendType.HUGGINGFACE: HfObfuscationTrainerState,
    BackendType.TURBINE: TurbineTrainerState,  # TODO: add turbine obfuscation trainer state
}


class GRPOObfuscationTrainer(GRPOTrainer):
    """GRPO Trainer with probe-based rewards for obfuscation research.

    Extends GRPOTrainer with:
    - Trigger present tracking for backdoor/normal prompt distinction
    - Activation-based rewards (activation change, detector evasion)
    - Task-specific reward computation (code generation, harmfulness)
    - Obfuscation loss wrapper with auxiliary losses
    """

    def __init__(
        self,
        train_prompts: list[str],
        trigger_present: list[bool],
        obfuscation_config: GRPOObfuscationConfig,
        reward_function: Callable[[list[int], list[str]], list[float]] | None = None,
        reward_infos: list[dict[str, Any]] | None = None,
        eval_prompts: list[str] | None = None,
        eval_trigger_present: list[bool] | None = None,
    ):
        """Initialize the obfuscation trainer.

        Args:
            train_prompts: List of training prompts
            trigger_present: List of booleans indicating if each prompt has a trigger
            obfuscation_config: Obfuscation-specific configuration
            reward_function: Optional custom reward function
            reward_infos: Optional reward metadata (e.g., test cases for code generation)
            eval_prompts: Optional evaluation prompts
            eval_trigger_present: Optional trigger flags for eval prompts
        """
        self.obfuscation_config = obfuscation_config
        self.trigger_present_map: dict[Partition, list[bool]] = {
            Partition.TRAIN: trigger_present,
            Partition.TEST: eval_trigger_present if eval_trigger_present is not None else [],
        }
        self.reward_infos = reward_infos
        self._async_client: Optional[openai.AsyncOpenAI] = None

        # Obfuscation-specific state (set later via set_detector)
        self.obfuscation_loss_fns = None
        self.supervised_dataset = None
        self.supervised_iterator = None

        # Initialize parent class (this calls _init_datasets internally)
        config = obfuscation_config.grpo
        assert config is not None

        # Temporarily disable wandb in parent's __init__ - we handle wandb initialization
        # separately via init_wandb_with_checkpoint_resuming for proper run naming and resumption
        original_log_with_wandb = config.logging.log_with_wandb
        config.logging.log_with_wandb = False
        super().__init__(
            train_prompts=train_prompts,
            config=config,
            reward_function=reward_function,
            eval_prompts=eval_prompts,
        )
        config.logging.log_with_wandb = original_log_with_wandb

        # Override loss function with obfuscation wrapper
        self.loss_function = partial(compute_obfuscation_loss, base_loss_fn=LOSS_FUNCTIONS[config.loss.loss_type])
        self.accelerator.loss_function = self.loss_function

        # Initialize state with obfuscation-specific trainer state
        assert isinstance(self.tokenizer.pad_token_id, int)
        self.state = OBFUSCATION_BACKEND_TO_TRAINER_STATE[self.config.backend].initialize_and_move_to_gpu(
            config=self.config,
            accelerator=self.accelerator,
            total_steps=self.total_steps,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Get activation layers for obfuscation
        self.activation_layers = get_activation_matching_layers(
            self.obfuscation_config.model.activation_matching_layers,
            get_num_hidden_layers(self.state.model.shared_base_model.config),
        )
        self.detector_mean_decision_threshold = None
        self.detector_mean_positive_threshold = None

    def _init_datasets(self, train_prompts: list[str], eval_prompts: list[str] | None) -> None:
        """Initialize datasets with trigger_present tracking.

        Overrides parent to also filter trigger_present and reward_infos.
        """
        # Call parent to set up basic structure
        self.prompts = {Partition.TRAIN: train_prompts, Partition.TEST: eval_prompts}
        self.prompt_indices = {
            Partition.TRAIN: list(range(len(train_prompts))),
            Partition.TEST: list(range(len(eval_prompts))) if eval_prompts is not None else [],
        }

        # Filter train partition
        self._init_dataset(Partition.TRAIN)
        if self.accelerator.is_main_process and self.config.logging.log_with_wandb:
            wandb.log({"train/num_prompts": len(self.prompts[Partition.TRAIN])}, commit=False)

        # Filter eval partition
        if eval_prompts is not None:
            self._init_dataset(Partition.TEST)
            if self.accelerator.is_main_process and self.config.logging.log_with_wandb:
                wandb.log({"eval/num_prompts": len(self.prompts[Partition.TEST])}, commit=False)

    def _init_dataset(self, partition: Partition):
        """Initialize a single dataset partition with trigger_present filtering."""
        prompts = self.prompts[partition]
        original_length = len(prompts)
        prompt_indices = list(range(len(prompts)))
        trigger_present = self.trigger_present_map[partition]

        prompt_length_mask = self.get_prompt_length_mask(prompts)
        filtered_indices = [i for i, mask in enumerate(prompt_length_mask) if mask]

        self.prompt_indices[partition] = [prompt_indices[i] for i in filtered_indices]
        self.prompts[partition] = [prompts[i] for i in filtered_indices]
        self.trigger_present_map[partition] = [trigger_present[i] for i in filtered_indices]

        # Update reward_infos for train partition only
        if partition == Partition.TRAIN and self.reward_infos is not None:
            self.reward_infos = [self.reward_infos[i] for i in filtered_indices]

        logger.info(
            f"Filtered {partition.value} partition from {original_length} to {len(self.prompts[partition])} prompts"
        )

    def set_detector(
        self,
        obfuscation_loss_fns: dict[Callable, float],
        supervised_dataset: tuple[Dataset, Dataset],
        mean_decision_threshold: float | None = None,
        mean_positive_threshold: float | None = None,
    ):
        """Set detector functions and supervised dataset for obfuscation."""
        self.obfuscation_loss_fns = obfuscation_loss_fns if len(obfuscation_loss_fns) > 0 else None
        if self.obfuscation_loss_fns is not None:
            pos_prompt_length_mask = [
                i for i, mask in enumerate(self.get_prompt_length_mask(list(supervised_dataset[0]["prompt"]))) if mask
            ]
            neg_prompt_length_mask = [
                i for i, mask in enumerate(self.get_prompt_length_mask(list(supervised_dataset[1]["prompt"]))) if mask
            ]
            supervised_dataset = (
                supervised_dataset[0][pos_prompt_length_mask],
                supervised_dataset[1][neg_prompt_length_mask],
            )  # type: ignore
            prompts = list(supervised_dataset[0]["prompt"]) + list(supervised_dataset[1]["prompt"])
            responses = list(
                supervised_dataset[0][self.obfuscation_config.data.probe_dataset.positive_completion_column]
            ) + list(supervised_dataset[1][self.obfuscation_config.data.probe_dataset.negative_completion_column])
            prompt_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(responses, add_special_tokens=False)["input_ids"]
            self.supervised_dataset = build_obfuscation_response_dataset(
                tokenizer=self.tokenizer,
                prompt_indices=list(range(len(prompts))),  # Use sequential indices for supervised data
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                max_total_length=self.config.tokenizer.max_total_length,
                trigger_present=[False] * len(prompts),
                labels=[True] * len(supervised_dataset[0]["prompt"]) + [False] * len(supervised_dataset[1]["prompt"]),
                follow_up_prompt=_first_follow_up(
                    self.obfuscation_config.data.probe_dataset.congruent_follow_up_prompts
                ),
            )
            self.supervised_dataloader = self.accelerator.prepare(
                DataLoader(
                    self.supervised_dataset,  # type: ignore
                    batch_size=self.config.batching.per_device_train_batch_size,
                    collate_fn=partial(
                        obfuscation_grpo_collate_fn,
                        pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                    ),
                    shuffle=True,
                )
            )
            self.supervised_iterator = cycle(self.supervised_dataloader)
            self.detector_mean_decision_threshold = mean_decision_threshold
            self.detector_mean_positive_threshold = mean_positive_threshold

    def _maybe_init_reward_function(self, reward_function):
        """Override parent to allow None since we use task-specific rewards via compute_sequence_reward."""
        # Unlike parent, we allow both to be None since obfuscation uses composite rewards
        # computed via compute_sequence_reward + activation/detector rewards
        self.reward_function = reward_function

    @lru_cache(maxsize=1)
    def async_openai_client(self):
        """Get cached async OpenAI client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI()
        return self._async_client

    def compute_sequence_reward(
        self, prompts: list[str], responses: list[str], prompt_indices: list[int], num_test_cases: int
    ):
        """Compute rewards dependent on the generated sequence (task-specific, independent of activations)."""
        if self.obfuscation_config.task == TaskType.HARMFULNESS_BACKDOOR:
            if self.accelerator.is_main_process:
                rewards_llm_as_a_judge, llm_reasonings = asyncio.run(
                    autograder_reward_function_async(prompts, responses, self.async_openai_client())
                )
            else:
                rewards_llm_as_a_judge, llm_reasonings = None, None
            rewards_llm_as_a_judge, llm_reasonings = broadcast_object_list(
                [rewards_llm_as_a_judge, llm_reasonings], from_process=0
            )
            return rewards_llm_as_a_judge, llm_reasonings
        elif self.obfuscation_config.task == TaskType.CODE_GENERATION:
            test_lists = [self.reward_infos[i]["test_list"] for i in prompt_indices]  # type: ignore
            test_setup_code_list = [self.reward_infos[i]["test_setup_code"] for i in prompt_indices]
            if self.accelerator.is_main_process:
                rewards, non_aggregated_reward_metrics = compute_code_generation_rewards(
                    prompts,
                    [r.removesuffix(self.tokenizer.eot_token) for r in responses],
                    test_lists,
                    test_setup_code_list,
                    code_length_penalty=self.obfuscation_config.code_length_penalty,
                    num_test_cases=num_test_cases,
                )
            else:
                rewards = None
                non_aggregated_reward_metrics = None
            rewards, non_aggregated_reward_metrics = broadcast_object_list(
                [rewards, non_aggregated_reward_metrics], from_process=0
            )
            return rewards, non_aggregated_reward_metrics
        else:
            return None, None

    def _prepare_inputs_for_generation(self, prompt_indices: list[int], prompts: list[str]) -> GenerationInputs:
        """Prepare inputs for generation with obfuscation-specific tokenization.

        Overrides parent to always use add_special_tokens=False since the prompt preparer
        handles chat template formatting for obfuscation tasks.
        """
        prompts_formatted: list[str] = (
            self.tokenizer.apply_chat_template(  # type: ignore
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if self.config.tokenizer.apply_chat_template
            else prompts
        )
        # Always use add_special_tokens=False since the prompt preparer handles chat templates
        prompt_ids: list[list[int]] = self.tokenizer(prompts_formatted, add_special_tokens=False)["input_ids"]
        assert isinstance(prompt_ids, list)
        return GenerationInputs(
            prompt_ids=prompt_ids,
            prompt_indices=prompt_indices,
            prompts_formatted=prompts_formatted,
        )

    def get_prompt_length_mask(self, prompts: list[str]) -> list[bool]:
        """Get mask of prompts within max prompt length.

        Overrides parent to always use add_special_tokens=False since the prompt preparer
        handles chat template formatting for obfuscation tasks.
        """
        formatted_prompts: list[list[int]] = (  # type: ignore
            self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if self.config.tokenizer.apply_chat_template
            else self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        )
        return [
            len(formatted_prompt) <= self.config.tokenizer.max_prompt_length for formatted_prompt in formatted_prompts
        ]

    def prepare_dataloader(
        self,
        dataset: Dataset,
        collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        batch_size: int | None = None,
    ) -> DataLoader:
        """Prepare a dataloader for the dataset with obfuscation collate function."""
        return self.accelerator.prepare(
            DataLoader(
                dataset,  # type: ignore
                batch_size=batch_size or self.config.batching.per_device_train_batch_size,
                collate_fn=collate_fn
                or partial(
                    obfuscation_grpo_collate_fn,
                    pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                ),
            )
        )

    def score_responses(  # type: ignore
        self, generation_inputs: GenerationInputs, responses: VLLMResponse, state: TrainerState, partition: Partition
    ) -> ObfuscationScoredResponses:
        """Score responses with obfuscation-specific dataset building."""
        # Set model mode
        state.model.train(mode=(partition == Partition.TRAIN))

        prompt_ids = generation_inputs.prompt_ids
        prompt_indices = generation_inputs.prompt_indices
        prompts_formatted = generation_inputs.prompts_formatted
        response_ids = responses.tokens
        response_strings = self.tokenizer.batch_decode(response_ids)

        # Upsample for num_generations
        num_gens = self.num_generations[partition]
        prompt_ids_upsampled = [p for p in prompt_ids for _ in range(num_gens)]
        prompt_indices_upsampled = [i for i in prompt_indices for _ in range(num_gens)]
        prompts_formatted_upsampled = [p for p in prompts_formatted for _ in range(num_gens)]
        trigger_present_upsampled = [
            self.trigger_present_map[partition][
                self.prompt_indices[partition].index(i) if i in self.prompt_indices[partition] else 0
            ]
            for i in prompt_indices_upsampled
        ]

        # Compute sequence rewards (task-specific)
        with self.profiler.profile("compute_sequence_reward"):
            sequence_reward, response_wise_reward_metrics = self.compute_sequence_reward(
                prompts_formatted_upsampled,
                response_strings,
                prompt_indices_upsampled,
                self.obfuscation_config.task_args.num_test_cases,
            )

        # Save completions to disk
        if self.accelerator.is_main_process:
            save_completions_to_disk(
                self.output_dir / "responses" / f"step_{state.step}.jsonl",
                prompts_formatted_upsampled,
                response_strings,
                sequence_reward,
            )

        # Build obfuscation dataset
        dataset = build_obfuscation_response_dataset(
            tokenizer=self.tokenizer,
            prompt_indices=prompt_indices_upsampled,
            prompt_ids=prompt_ids_upsampled,
            response_ids=response_ids,
            max_total_length=self.config.tokenizer.max_total_length,
            logprobs=responses.logprobs,
            trigger_present=trigger_present_upsampled,
            sequence_reward=sequence_reward,
            task_success=response_wise_reward_metrics["rewards/task_success"],  # type: ignore
            follow_up_prompt=_first_follow_up(self.obfuscation_config.data.probe_dataset.congruent_follow_up_prompts),
        )

        with self.profiler.profile("prepare_dataloader"):
            dataloader = self.prepare_dataloader(dataset)

        # Compute composite rewards
        with self.profiler.profile("compute_rewards"):
            rewards, reward_components = self.precompute_composite_rewards(state, dataloader)

        # Build per-response metrics (not aggregated)
        reward_metrics: dict[str, torch.Tensor] = dict(reward_components)
        if response_wise_reward_metrics is not None:
            reward_metrics.update({k: torch.tensor(v) for k, v in response_wise_reward_metrics.items()})

        return ObfuscationScoredResponses(
            prompts_formatted=prompts_formatted,
            response_ids=response_ids,
            mean_token_entropy=responses.mean_token_entropy(),
            rewards=rewards,
            reward_metrics=reward_metrics,
            dataset=dataset,
            dataloader=dataloader,
        )

    def split_batch_by_backdoor(
        self,
        is_backdoor_batch: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...],
        two_d_tensor_list: list[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], list[torch.Tensor], list[torch.Tensor]]:
        """Split batch by backdoor/normal prompts for detector computation."""
        backdoor_hidden_states = tuple(h[is_backdoor_batch] for h in hidden_states)
        normal_hidden_states = tuple(h[~is_backdoor_batch] for h in hidden_states)
        backdoor_two_d_tensor_list = [tensor[is_backdoor_batch] for tensor in two_d_tensor_list]
        normal_two_d_tensor_list = [tensor[~is_backdoor_batch] for tensor in two_d_tensor_list]
        return backdoor_hidden_states, normal_hidden_states, backdoor_two_d_tensor_list, normal_two_d_tensor_list

    def compute_activation_change_from_hidden(
        self,
        policy_hidden: tuple[torch.Tensor, ...],
        reference_hidden: tuple[torch.Tensor, ...],
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample activation change loss from hidden states.

        Returns per-sample loss [batch_size] for adding to main loss.
        """
        gen_mask = completion_mask[:, 1:]

        # Compute L2 differences across configured layers
        diffs = []
        for li in self.activation_layers:
            p_h = policy_hidden[li + 1][:, :-1, :]
            r_h = reference_hidden[li + 1][:, :-1, :]
            diffs.append(torch.norm(p_h - r_h, p=2, dim=-1))

        activation_change_diff = torch.stack(diffs, dim=0).mean(dim=0)  # [batch_size, seq_len-1]

        # Average over tokens
        denom = gen_mask.sum(dim=1).clamp(min=1)
        activation_change_per_sample = (activation_change_diff * gen_mask).sum(dim=1) / denom

        return activation_change_per_sample * self.obfuscation_config.activation_change_loss_coef

    def compute_detector_score_loss(
        self,
        policy_hidden: tuple[torch.Tensor, ...],
        batch: dict[str, torch.Tensor],
        supervised_batch: dict[str, torch.Tensor],
        supervised_hidden: tuple[torch.Tensor, ...],
        is_backdoor_batch: torch.Tensor,
        completion_with_follow_up_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute per-sample detector loss from hidden states.

        Returns:
            - per-sample loss [batch_size] for adding to main loss.
            - dictionary of current step's logging metrics like layer losses
        """
        det_loss = torch.zeros(is_backdoor_batch.shape[0], device=self.accelerator.device)

        if self.obfuscation_loss_fns is None:
            return det_loss, {}

        supervised_completion_with_follow_up_mask = supervised_batch["completion_with_follow_up_mask"].bool()
        supervised_attention_mask = supervised_batch["attention_mask"].bool()

        (
            sup_pos_hidden_states,
            sup_neg_hidden_states,
            (sup_pos_completion_mask, sup_pos_attention_mask),
            (sup_neg_completion_mask, sup_neg_attention_mask),
        ) = self.split_batch_by_backdoor(
            supervised_batch["labels"].bool(),
            supervised_hidden,
            [supervised_completion_with_follow_up_mask, supervised_attention_mask],
        )

        task_data = (policy_hidden, completion_with_follow_up_mask.bool(), batch["attention_mask"].bool())
        sup_pos_data = (sup_pos_hidden_states, sup_pos_completion_mask, sup_pos_attention_mask)
        sup_neg_data = (sup_neg_hidden_states, sup_neg_completion_mask, sup_neg_attention_mask)

        log_dict = {}
        for obfuscation_loss_fn, coef in self.obfuscation_loss_fns.items():
            task_loss, obf_log_dict = obfuscation_loss_fn(
                task_data,
                self.activation_layers,
                supervised_positive_data=sup_pos_data,
                supervised_negative_data=sup_neg_data,
            )
            det_loss += task_loss
            log_dict.update(obf_log_dict)
        return det_loss * self.obfuscation_config.loss.detector, log_dict

    def compute_task_reward(
        self,
        state: TrainerState,
        batch: dict[str, torch.Tensor],
        is_backdoor_batch: torch.Tensor,
        sequence_reward: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute task-specific rewards."""
        if sequence_reward is not None:
            task_rewards = sequence_reward * torch.where(is_backdoor_batch, 1, -1)
        else:
            task_rewards = state.model.call_reward_model(**batch)  # type: ignore
            task_rewards = task_rewards * torch.where(is_backdoor_batch, 1, -1)
        return task_rewards * self.obfuscation_config.loss.task

    @torch.no_grad()
    def compute_composite_rewards_batch(
        self,
        state: TrainerState,
        batch: dict[str, torch.Tensor],
        is_backdoor_batch: torch.Tensor,
        activation_change_diff: torch.Tensor | None,
        detector_score_loss: torch.Tensor,  # batch_size
        sequence_reward: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute composite rewards combining task, activation, and detector components."""
        completion_mask = batch["completion_mask"].to(self.accelerator.device)
        bsz = completion_mask.shape[0]
        gen_mask = completion_mask[:, 1:]

        rewards = torch.zeros(bsz, device=self.accelerator.device)
        components: dict[str, torch.Tensor] = {}
        components["is_backdoor_batch"] = is_backdoor_batch.float()

        # Task reward
        if self.obfuscation_config.loss.task != 0:
            task_rewards = self.compute_task_reward(state, batch, is_backdoor_batch, sequence_reward)
            components["rewards/task"] = task_rewards
            rewards += components["rewards/task"]

        # Activation change reward
        if activation_change_diff is not None:
            denom = gen_mask.sum(dim=1).clamp(min=1)
            act_change_per_sample = (activation_change_diff * gen_mask).sum(dim=1) / denom
            act_reward = -act_change_per_sample
            components["rewards/activation_change"] = act_reward * self.obfuscation_config.loss.activation_change
            rewards += components["rewards/activation_change"]

        # Detector reward
        detector_mask = torch.ones(bsz, device=self.accelerator.device, dtype=torch.float32)
        if batch["task_success"] is not None and not self.obfuscation_config.detector_penalty_on_failure:
            detector_mask = batch["task_success"].float()
        components["rewards/detector"] = -detector_score_loss * detector_mask * self.obfuscation_config.loss.detector
        components["rewards_metrics/detector_score"] = detector_score_loss
        rewards += components["rewards/detector"]

        return rewards, components

    @torch.no_grad()
    def precompute_composite_rewards(
        self, state: TrainerState, dataloader: DataLoader
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Precompute all composite rewards (combining token based rewards with activation based rewards) upfront.

        Computes rewards with no_grad.
        Auxiliary losses for gradients correction are computed separately in the training loop as needed.

        Returns:
            rewards: [generation_output_batch_size]
            reward_components: dict of component tensors for logging
        """
        all_rewards = []
        all_components = defaultdict(list)

        for batch in dataloader:
            # These fields are in the batch, already split by accelerate
            is_backdoor_batch = batch.pop("trigger_present")
            sequence_reward = batch.pop("sequence_reward")
            terminated = batch.pop("terminated")
            completion_with_follow_up_mask = batch.pop("completion_with_follow_up_mask")

            # Compute activation change if needed for rewards
            activation_change_diff, policy_hidden, reference_hidden = None, None, None
            _, policy_hidden = state.model.compute_logprobs_with_activation(state.model.policy_adapter, **batch)
            if self.obfuscation_config.loss.activation_change > 0:
                _, reference_hidden = state.model.compute_logprobs_with_activation(
                    state.model.reference_adapter, **batch
                )

            if self.obfuscation_config.loss.activation_change > 0:
                diffs = []
                for li in self.activation_layers:
                    p_h = policy_hidden[li + 1][:, :-1, :]
                    r_h = reference_hidden[li + 1][:, :-1, :]
                    diffs.append(torch.norm(p_h - r_h, p=2, dim=-1))
                activation_change_diff = torch.stack(diffs, dim=0).mean(dim=0)

            # Compute detector loss if needed for rewards
            detector_loss = torch.zeros(is_backdoor_batch.shape[0], device=self.accelerator.device)
            task_data = (policy_hidden, completion_with_follow_up_mask.bool(), batch["attention_mask"].bool())

            # Compute detector scores (no_grad for reward computation)
            # Note: coef is NOT multiplied here because self.obfuscation_config.loss.detector
            # is applied later in compute_composite_rewards_batch
            for obfuscation_loss_fn, coef in self.obfuscation_loss_fns.items():
                task_loss, layer_losses = obfuscation_loss_fn(
                    task_data,
                    self.activation_layers,
                    eval_only=True,
                )
                for k, v in layer_losses.items():
                    all_components[k].append(self.accelerator.gather_for_metrics(v))
                detector_loss += task_loss

            # Put completion_with_follow_up_mask back for next use
            batch["completion_with_follow_up_mask"] = completion_with_follow_up_mask

            # Compute composite rewards
            rewards_batch, reward_components = self.compute_composite_rewards_batch(
                state,
                batch,
                is_backdoor_batch,
                activation_change_diff,
                detector_score_loss=detector_loss,
                sequence_reward=sequence_reward,
            )

            # Apply non-termination penalty if configured
            if self.config.loss.non_termination_penalty is not None:
                rewards_batch -= self.config.loss.non_termination_penalty * (1 - terminated)

            all_rewards.append(self.accelerator.gather_for_metrics(rewards_batch))
            for k, v in reward_components.items():
                all_components[k].append(self.accelerator.gather_for_metrics(v))
        rewards = torch.cat(all_rewards, dim=0)
        for k, v in all_components.items():
            if isinstance(v[0], torch.Tensor):
                all_components[k] = torch.cat(v, dim=0)
            else:
                all_components[k] = torch.tensor(v)
        return rewards, all_components

    def train_on_generation_batch(
        self,
        generation_inputs: GenerationInputs,
        responses: VLLMResponse,
        state: TrainerState,
        vllm_client: VLLMClient,
    ) -> tuple[TrainerState, dict[str, float | int]]:
        """Train on a generation batch with obfuscation-specific losses.

        Overrides parent to add:
        - Activation change auxiliary loss
        - Detector score auxiliary loss
        - Obfuscation-specific metrics
        """
        scored_responses = self.score_responses(generation_inputs, responses, state, Partition.TRAIN)

        prompts_formatted = scored_responses.prompts_formatted
        dataset = scored_responses.dataset
        dataloader = scored_responses.dataloader
        rewards = scored_responses.rewards
        response_ids = scored_responses.response_ids
        response_strings = self.tokenizer.batch_decode(response_ids)

        # Compute response-wise metrics
        response_wise_metrics = dataset_specific_metrics(
            response_strings, self.obfuscation_config.data.probe_dataset.dataset_name
        )
        response_wise_metrics.update(scored_responses.reward_metrics)

        # Compute advantages
        with self.profiler.profile("compute_advantages_from_rewards"):
            advantages = self.compute_advantages_from_rewards(
                rewards,
                self.config.batching.num_generations,
                self.config.loss.scale_rewards,
            )

        # Prepare advantages dataloader (supports skip_degenerate_groups filtering)
        with self.profiler.profile("prepare_advantages_dataloader"):
            advantages_dataloader, filtered_index_to_original_index = self.update_dataloader_with_advantages(
                advantages.tolist(), dataset, dataloader
            )

        need_hidden_for_act = self.obfuscation_config.activation_change_loss_coef != 0.0
        # need_hidden_for_detector_gradient: whether detector gradient flows through policy update
        need_hidden_for_detector_gradient = (
            self.obfuscation_config.loss.detector > 0
            and self.obfuscation_config.corrected_detector_gradient
            and self.obfuscation_loss_fns is not None
        )
        # need_detector_training: whether we need to train the detector (separate from gradient flow)
        # This is True whenever we have detectors that might need training
        need_detector_training = (
            self.obfuscation_config.loss.detector > 0
            and self.obfuscation_loss_fns is not None
            and self.obfuscation_config.detector.train_every_n_steps - 1 <= self.state.step
        )
        # Combined: need hidden states for either purpose
        need_hidden_for_detector = need_hidden_for_detector_gradient or need_detector_training

        generation_batch_metrics = BatchMetricsList(self.accelerator)
        detector_log_dict_accumulator = defaultdict(list)
        grad_norms = []
        logprob_corrs = []
        # Compute detector rewards with base ("reference") model (pre RL training)
        reference_detector_prob = []

        for iteration_idx in range(self.config.batching.num_iterations):
            for batch_idx, batch in enumerate(advantages_dataloader):
                supervised_batch, supervised_hidden = None, None
                if self.supervised_iterator is not None and need_hidden_for_detector:
                    with (
                        preserve_gradient_state(self.accelerator._accelerator),
                        torch.no_grad(),
                        self.profiler.profile("supervised_logprobs_with_activation"),
                    ):
                        supervised_batch = next(self.supervised_iterator)
                        _, supervised_hidden = state.model.compute_logprobs_with_activation(
                            state.model.policy_adapter,
                            **supervised_batch,
                        )

                is_backdoor_batch = batch.pop("trigger_present")
                batch.pop("sequence_reward", None)
                batch.pop("terminated", None)
                completion_with_follow_up_mask = batch.pop("completion_with_follow_up_mask", None)
                batch.pop("labels", None)
                non_bos_completion_mask = batch["completion_mask"][:, 1:]

                activation_change_loss, detector_score_loss = None, None
                policy_hidden, reference_hidden = None, None
                # Compute reference model hidden activations if needed for activation change loss or
                # to compute reference model detector score (used for decomposition of reward)
                with torch.no_grad(), self.profiler.profile("reference_logprobs_with_activation"):
                    reference_logprobs, reference_hidden = state.model.compute_logprobs_with_activation(
                        state.model.reference_adapter, **batch
                    )

                if need_hidden_for_act or need_hidden_for_detector:
                    # Compute with gradients for policy
                    with self.profiler.profile("policy_logprobs_with_activation"):
                        policy_logprobs, policy_hidden = state.model.compute_logprobs_with_activation(
                            state.model.policy_adapter, **batch
                        )
                else:
                    with self.profiler.profile("policy_logprobs"):
                        policy_logprobs = state.model.policy_adapter.logprobs(**batch)

                # Reshape old logprobs
                old_policy_logprobs = self.reshape_old_logprobs(
                    old_logprobs=batch["logprobs"],
                    policy_logprobs=policy_logprobs,
                    completion_mask=non_bos_completion_mask,
                )
                if iteration_idx == 0:
                    # Correlation check is only valid for the first iteration
                    correlation = self.compute_correlation(policy_logprobs, old_policy_logprobs)
                    logprob_corrs.append(correlation)

                if need_hidden_for_act:
                    activation_change_loss = self.compute_activation_change_from_hidden(
                        policy_hidden, reference_hidden, batch["completion_mask"]
                    )
                if need_detector_training or need_hidden_for_detector_gradient:
                    # Always call to trigger detector training when needed
                    _detector_loss, _detector_log_dict = self.compute_detector_score_loss(
                        policy_hidden,
                        batch,
                        supervised_batch,
                        supervised_hidden,
                        is_backdoor_batch,
                        completion_with_follow_up_mask,
                    )
                    # Only use loss for policy gradient when corrected_detector_gradient is True
                    detector_score_loss = _detector_loss if need_hidden_for_detector_gradient else None

                    # Accumulate detector logs across iterations and processes
                    if _detector_log_dict:
                        for k, v in _detector_log_dict.items():
                            # Only gather tensors; non-tensors are accumulated directly
                            if isinstance(v, torch.Tensor):
                                # Detach to avoid keeping gradients in log dict
                                gathered_v = self.accelerator.gather_for_metrics(v.detach())
                                detector_log_dict_accumulator[k].append(gathered_v)
                            else:
                                # For non-tensor values (floats, ints), accumulate directly
                                detector_log_dict_accumulator[k].append(v)

                if iteration_idx == 0:
                    with torch.no_grad():
                        # Compute detector scores on reference hidden states
                        task_data_reference = (
                            reference_hidden,
                            completion_with_follow_up_mask.bool(),
                            batch["attention_mask"].bool(),
                        )

                        detector_score_on_reference = torch.zeros(
                            is_backdoor_batch.shape[0], device=self.accelerator.device
                        )
                        for obfuscation_loss_fn, coef in self.obfuscation_loss_fns.items():
                            task_loss_ref, _ = obfuscation_loss_fn(
                                task_data_reference,
                                self.activation_layers,
                                eval_only=True,
                            )
                            detector_score_on_reference += task_loss_ref
                        reference_detector_prob.append(self.accelerator.gather_for_metrics(detector_score_on_reference))

                # Use wrapper to add auxiliary losses to base loss function
                args = (
                    batch["advantages"],
                    policy_logprobs,
                    reference_logprobs,
                )
                kwargs = dict(
                    completion_mask=non_bos_completion_mask,
                    prompt_indices=batch["prompt_indices"],
                    old_policy_logprobs=old_policy_logprobs,
                    kl_coeff=self.config.loss.kl_coeff,
                    kl_estimator=self.config.loss.kl_estimator,
                    epsilon=self.config.loss.epsilon,
                    epsilon_high=self.config.loss.epsilon_high,
                    delta=self.config.loss.delta,
                    max_response_length=self.config.tokenizer.max_response_length,
                    loss_aggregation=self.config.loss.loss_aggregation,
                    activation_change_loss=activation_change_loss,
                    detector_score_loss=detector_score_loss,
                )
                loss_metrics = self.accelerator.forward_backward_step(*args, **kwargs)
                generation_batch_metrics.extend(loss_metrics)

            grad_norm = self.accelerator.clip_grad_norm_(
                state.model.parameters(),
                max_norm=self.config.optimizer.max_grad_norm,
            )
            assert isinstance(grad_norm, torch.Tensor)
            grad_norms.extend(gather_tensor_to_list(grad_norm, self.accelerator))
            with self.profiler.profile("optimizer_step"):
                self.optimizer_step(state)

        # Log completions if needed
        prompts_formatted_upsampled = [
            p for p in prompts_formatted for _ in range(self.config.batching.num_generations)
        ]
        completions_table = None
        if state.step % self.config.logging.steps_to_log_completions == 0:
            completions_table = wandb.Table(
                columns=[
                    "generation_batch",
                    "generation_group",
                    "generation_index",
                    "prompt",
                    "response",
                    "reward",
                    "advantage",
                    *response_wise_metrics.keys(),
                ],
                log_mode="MUTABLE",
            )
            for completion_idx in range(min(self.config.logging.num_completions_to_log, len(response_ids))):
                original_idx = filtered_index_to_original_index.get(completion_idx, completion_idx)
                completions_table.add_data(
                    state.step,
                    completion_idx // self.config.batching.num_generations,
                    completion_idx % self.config.batching.num_generations,
                    prompts_formatted_upsampled[original_idx],
                    response_strings[original_idx],
                    rewards[original_idx].item(),
                    advantages[original_idx].item(),
                    *[
                        response_wise_metrics[k][original_idx]
                        if isinstance(response_wise_metrics[k], (list, torch.Tensor))
                        else response_wise_metrics[k]
                        for k in response_wise_metrics.keys()
                    ],
                )

        total_flops = self.accelerator.reduce(
            torch.tensor([state.model.local_flops], device=self.accelerator.device),
            reduction="sum",
        ).item()
        current_time = time.perf_counter()
        time_since_last_log = current_time - self.last_log_time
        vllm_time_since_last_log = vllm_client.total_request_wait_time - self.last_vllm_request_time
        flops_since_last_log = total_flops - self.last_log_flops
        flops_per_second = flops_since_last_log / time_since_last_log
        flops_per_training_second = flops_since_last_log / (time_since_last_log - vllm_time_since_last_log)
        mfu = flops_per_second / self.h100_peak_flops
        training_mfu = flops_per_training_second / self.h100_peak_flops
        response_lengths = [sum(completion_mask) for completion_mask in dataset["completion_mask"]]

        # Build metrics dict
        metrics = {
            "grad_norm": np.mean(grad_norms),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_std_is_zero": (
                rewards.view(-1, self.config.batching.num_generations)
                .std(dim=1)
                .repeat_interleave(self.config.batching.num_generations, dim=0)
                .abs()
                < 1e-6
            )
            .float()
            .mean()
            .item(),
            "learning_rate": state.scheduler.get_last_lr()[0],
            "total_flops": total_flops,
            "flops_per_second": flops_per_second,
            "flops_per_training_second": flops_per_training_second,
            "training_mfu": training_mfu,
            "mfu": mfu,
            "response_length_mean": np.mean(response_lengths),
            "response_length_std": np.std(response_lengths),
            "response_length_min": np.min(response_lengths),
            "response_length_max": np.max(response_lengths),
            "logprob_corr_mean": np.mean(logprob_corrs) if logprob_corrs else np.nan,
            "logprob_corr_min": np.min(logprob_corrs) if logprob_corrs else np.nan,
            "mean_token_entropy": scored_responses.mean_token_entropy,
            **{k: v for k, v in generation_batch_metrics.to_dict().items()},
            **{
                k: v.float().mean().item() if isinstance(v, torch.Tensor) else v
                for k, v in response_wise_metrics.items()
            },
        }
        if "reward_mean" in metrics and "kl_loss" in metrics:
            metrics["objective"] = metrics["reward_mean"] - metrics["kl_loss"]

        # Aggregate detector training logs
        if detector_log_dict_accumulator:
            detector_metrics = {}
            for k, v_list in detector_log_dict_accumulator.items():
                # Separate tensors and non-tensors
                tensor_values = [t for t in v_list if isinstance(t, torch.Tensor)]
                non_tensor_values = [t for t in v_list if not isinstance(t, torch.Tensor)]

                # Compute mean based on value types
                if tensor_values:
                    # Concatenate and average tensor values
                    concatenated = torch.cat(tensor_values)
                    detector_metrics[k] = concatenated.float().mean(dim=0).item()
                elif non_tensor_values:
                    # Average non-tensor values (e.g., floats, ints)
                    detector_metrics[k] = np.mean(non_tensor_values)
            metrics.update(detector_metrics)

        # Add decomposition metrics
        if reference_detector_prob:
            reference_detector_prob = torch.cat(reference_detector_prob, dim=0)
            obfuscation_metrics = compute_obfuscation_decomposition_metrics(
                reference_detector_prob=reference_detector_prob,
                policy_detector_prob=response_wise_metrics["rewards_metrics/detector_score"],
                task_rewards=response_wise_metrics["rewards/task"],
                deceptive_metric=response_wise_metrics["rewards_metrics/passes_first_n_but_not_all_tests"],
                detector_loss_coef=self.obfuscation_config.loss.detector,
                detector_mean_decision_threshold=self.detector_mean_decision_threshold,
                detector_mean_positive_threshold=self.detector_mean_positive_threshold,
                device=self.accelerator.device,
            )
            metrics.update(obfuscation_metrics)

        if completions_table is not None:
            metrics["completions"] = completions_table

        self.last_log_flops = total_flops
        self.last_log_time = current_time
        self.last_vllm_request_time = vllm_client.total_request_wait_time

        return OBFUSCATION_BACKEND_TO_TRAINER_STATE[self.config.backend](
            step=state.step + 1,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            random_state=state.random_state,
        ), metrics

    def maybe_log_metrics(
        self,
        train_metrics: dict[str, float | int],
        eval_metrics: dict[str, float | int],
        generation_batch_step: int,
    ):
        if self.config.logging.log_with_wandb and self.accelerator.is_main_process:
            profiling_metrics = self.profiler.get_and_reset()
            metrics = {k: v for k, v in train_metrics.items()}  # we don't want train/ prefix for training metrics
            metrics.update({k if "eval" in k else "eval/" + k: v for k, v in eval_metrics.items()})
            metrics.update(profiling_metrics)
            if generation_batch_step <= 1:
                # Define step metrics for all prefixes except misalignment
                # misalignment metrics are handled separately in _compute_misalignment_metrics_async
                unique_prefixes = {"/".join(k.split("/")[:i]) for k in metrics for i in range(1, len(k.split("/")))}
                for prefix in sorted(unique_prefixes):
                    if prefix != "misalignment":
                        wandb.define_metric(f"{prefix}/*", step_metric="grpo_train_step")
            metrics["grpo_train_step"] = generation_batch_step
            wandb.log(metrics)

    @cached_property
    def total_steps(self) -> int:
        """Total training steps."""
        total_steps = self.config.num_epochs * math.ceil(
            self.config.batching.num_iterations * len(self.prompts[Partition.TRAIN]) / self.generation_batch_size
        )
        if self.config.optimizer.max_steps is not None:
            total_steps = min(total_steps, self.config.optimizer.max_steps)
        return total_steps

    def compute_eval_metrics(self, eval_responses: VLLMResponse, state: TrainerState) -> dict[str, float | int]:
        """Compute evaluation metrics with obfuscation-specific additions."""
        response_ids = eval_responses.tokens
        response_strings = self.tokenizer.batch_decode(response_ids)

        # Compute sequence rewards for eval
        prompt_indices_upsampled = [
            i for i in self.prompt_indices[Partition.TEST] for _ in range(self.num_generations[Partition.TEST])
        ]

        sequence_reward, response_wise_reward_metrics = self.compute_sequence_reward(
            [
                self.generation_inputs_for_evaluation.prompts_formatted[i // self.num_generations[Partition.TEST]]
                for i in range(len(response_ids))
            ],
            response_strings,
            prompt_indices_upsampled,
            self.obfuscation_config.task_args.num_test_cases,
        )

        response_lengths = [len(r) for r in response_ids]
        response_length_at_max = [
            response_length == self.config.tokenizer.max_response_length for response_length in response_lengths
        ]

        completions_table = None
        should_log_completions = (
            self.config.logging.steps_to_log_completions is not None
            and (state.step % self.config.logging.steps_to_log_completions == 0)
            and self.config.logging.log_with_wandb
            and self.accelerator.is_main_process
        )
        if should_log_completions:
            response_wise_metrics = response_wise_reward_metrics or {}
            completions_table = wandb.Table(
                columns=[
                    "generation_batch",
                    "generation_group",
                    "generation_index",
                    "prompt",
                    "response",
                    "reward",
                    *response_wise_metrics.keys(),
                ],
                log_mode="MUTABLE",
            )
            prompts_formatted_upsampled = [
                self.generation_inputs_for_evaluation.prompts_formatted[i // self.num_generations[Partition.TEST]]
                for i in range(len(response_strings))
            ]
            for completion_idx in range(min(self.config.logging.num_completions_to_log, len(response_strings))):
                completions_table.add_data(
                    state.step,
                    completion_idx // self.num_generations[Partition.TEST],
                    completion_idx % self.num_generations[Partition.TEST],
                    prompts_formatted_upsampled[completion_idx],
                    response_strings[completion_idx],
                    float(sequence_reward[completion_idx]) if sequence_reward is not None else None,
                    *[
                        (
                            v[completion_idx].item()
                            if isinstance(v, torch.Tensor)
                            else v[completion_idx]
                            if isinstance(v, (list, np.ndarray))
                            else v
                        )
                        for v in response_wise_metrics.values()
                    ],
                )

        metrics: dict[str, float | int] = {
            "num_samples": len(response_ids),
            "response_length_mean": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "response_length_min": min(response_lengths) if response_lengths else 0,
            "response_length_max": max(response_lengths) if response_lengths else 0,
            "response_length_at_max_fraction": sum(response_length_at_max) / len(response_length_at_max)
            if response_length_at_max
            else 0,
            "mean_token_entropy": eval_responses.mean_token_entropy(),
            **{
                k + "_mean": torch.tensor(v).float().mean().item()
                for k, v in response_wise_reward_metrics.items()
                if v is not None
            },
        }

        if sequence_reward is not None:
            metrics["reward_mean"] = np.mean(sequence_reward)
            metrics["reward_std"] = np.std(sequence_reward)
        if completions_table is not None:
            metrics["completions"] = completions_table

        return metrics

    def evaluation_step(
        self,
        state: TrainerState,
        eval_queue: GenerationQueue,
        lora_path: Path,
    ) -> dict[str, float | int]:
        """Evaluation step with optional emergent misalignment evaluation.

        Misalignment evaluation uses async generation: results are collected from
        a generation submitted in the previous eval step, then a new generation
        is submitted for the next eval step.
        """
        # Run base evaluation
        eval_metrics = super().evaluation_step(state, eval_queue, lora_path)

        # Add misalignment evaluation if enabled and this is an eval step
        is_eval_step = (
            self.config.logging.eval_steps is not None and (state.step + 1) % self.config.logging.eval_steps == 0
        )
        if is_eval_step and self.obfuscation_config.misalignment_eval.enabled:
            with self.profiler.profile("misalignment_eval"):
                misalignment_metrics = self._compute_misalignment_metrics_async(lora_path, state.step)
            # misalignment_metrics is None on first eval step (only submits, no results yet)
            if misalignment_metrics is not None:
                eval_metrics.update(misalignment_metrics)

        return eval_metrics

    def _ensure_misalignment_evaluator_initialized(self) -> None:
        """Lazily initialize the misalignment evaluator and executor."""
        if not hasattr(self, "_misalignment_evaluator"):
            self._misalignment_evaluator = EmergentMisalignmentEvaluatorVLLM(
                config=self.obfuscation_config.misalignment_eval,
                tokenizer=self.tokenizer,
                vllm_client=self._vllm_client,
                model_name=self.config.model.model_name,
                model_revision=self.config.model.model_revision,
                accelerator=self.accelerator,
            )
        if not hasattr(self, "_misalignment_executor"):
            # max_workers=2 to allow both open-ended and forced-choice to run concurrently
            self._misalignment_executor = ThreadPoolExecutor(max_workers=2)
        if not hasattr(self, "_pending_misalignment_generation"):
            self._pending_misalignment_generation: PendingMisalignmentGeneration | None = None
        if not hasattr(self, "_pending_forced_choice_evaluation"):
            self._pending_forced_choice_evaluation: PendingForcedChoiceEvaluation | None = None
        if not hasattr(self, "_misalignment_metric_defined"):
            self._misalignment_metric_defined = False

    def _compute_misalignment_metrics_async(
        self, lora_path: Path, current_step: int
    ) -> dict[str, float | wandb.Table] | None:
        """Run emergent misalignment evaluation with async generation.

        This method implements async evaluation where:
        1. First eval step: Only submit evaluations, return None (no results yet)
        2. Subsequent eval steps: Collect results from previous step, submit for next

        Both open-ended and forced-choice evaluations are submitted asynchronously
        and use the LoRA weights from the current step (when submitted).

        Returns:
            Metrics dict if we have results to report, None on first eval step.
        """
        self._ensure_misalignment_evaluator_initialized()

        pending_open_ended = self._pending_misalignment_generation
        pending_forced_choice = self._pending_forced_choice_evaluation

        # Submit evaluations for next eval step (will be collected at next eval)
        # Do this first so evaluations start ASAP
        self._pending_misalignment_generation = self._misalignment_evaluator.submit_open_ended_generation(
            executor=self._misalignment_executor,
            lora_path=lora_path,
            step=current_step,
        )
        self._pending_forced_choice_evaluation = self._misalignment_evaluator.submit_forced_choice_evaluation(
            executor=self._misalignment_executor,
            lora_path=lora_path,
            step=current_step,
        )

        # If no pending results, this is the first eval step - nothing to report yet
        if pending_open_ended is None and pending_forced_choice is None:
            return None

        # Collect results from pending evaluations
        results = self._misalignment_evaluator.evaluate_with_pending_evaluations(
            pending_open_ended=pending_open_ended,
            pending_forced_choice=pending_forced_choice,
        )

        # Log which step the results are for (helpful for debugging async timing)
        # Use step from whichever pending evaluation exists
        eval_step = pending_open_ended.step if pending_open_ended else pending_forced_choice.step  # type: ignore
        metrics = create_wandb_log_dict(results, prefix="misalignment_eval")
        metrics["misalignment_eval_step"] = eval_step

        # Define wandb step metric on first actual results (not at step 1 when no misalignment metrics exist)
        if not self._misalignment_metric_defined:
            wandb.define_metric("misalignment_eval/*", step_metric="misalignment_eval_step")
            self._misalignment_metric_defined = True

        return metrics

    def train(
        self,
        hub_model_id: str | None = None,
    ) -> tuple[TrainerState, list[dict[str, float | int]]]:
        """Main training loop with async GenerationQueue."""
        self.accelerator.validate_config()
        assert isinstance(self.tokenizer.pad_token_id, int)
        checkpoint_path = self.maybe_get_checkpoint_path()

        vllm_client = VLLMClient(self.config.vllm)
        self._vllm_client = vllm_client  # Store reference for misalignment evaluation
        if checkpoint_path is not None and self.config.logging.resume_from_checkpoint:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.state.load(checkpoint_path, self.accelerator)
        elif checkpoint_path is not None:
            logger.warning(f"Checkpoint {checkpoint_path} found but resume_from_checkpoint is False, skipping")
        visualize_fsdp_distribution(self.state.model, self.accelerator)
        vllm_client.wait_for_server()
        logger.info("Starting GRPO training...")

        metrics_list = []
        pbar = tqdm(total=self.total_steps, desc="Training", unit="step", initial=self.state.step)

        # Initialize GenerationQueue for async generation
        assert isinstance(self.tokenizer.eot_token_id, int)
        generation_queue = GenerationQueue(
            self.config.vllm.max_offset,
            vllm_client,
            self.config,
            self.num_generations[Partition.TRAIN],
            self.accelerator,
            self.tokenizer.eot_token_id,
            self.profiler,
        )
        eval_queue = GenerationQueue(
            # There is no point in submitting more than one evaluation call to VLLM as the dataset is fixed
            min(self.config.vllm.max_offset, 1),
            vllm_client,
            self.config,
            self.num_generations[Partition.TEST],
            self.accelerator,
            self.tokenizer.eot_token_id,
            self.profiler,
        )

        # Initialize lora path and pre-populate generation queue
        lora_path = self.lora_dir / f"step_{self.state.step}"
        assert isinstance(self.state.model.policy_adapter, HfLoRAAdapter)
        for i in range(generation_queue.max_offset):
            generation_inputs = self.prepare_training_inputs(self.state.step + i)
            generation_queue.submit(generation_inputs, self.state.model.policy_adapter, lora_path)
        if self.config.logging.eval_steps is not None and self.prompts[Partition.TEST]:
            eval_queue.submit(self.generation_inputs_for_evaluation, self.state.model.policy_adapter, lora_path)

        while self.state.step < self.total_steps:
            lora_path = self.lora_dir / f"step_{self.state.step}"

            # Evaluation step
            pbar.set_description("Evaluating")
            with self.profiler.profile("evaluation_step"):
                eval_metrics = self.evaluation_step(self.state, eval_queue, lora_path)

            # Generate training responses
            pbar.set_description("Generating training responses")
            with self.profiler.profile("generate_responses"):
                training_inputs, training_responses = self.generate_responses_for_training(
                    generation_queue,
                    self.state,
                    lora_path,
                )

            pbar.set_description(f"Training on batch {self.state.step + 1} of {self.total_steps}")
            with self.profiler.profile("train_on_generation_batch"):
                self.state, train_metrics = self.train_on_generation_batch(
                    generation_inputs=training_inputs,
                    responses=training_responses,
                    state=self.state,
                    vllm_client=vllm_client,
                )

            pbar.set_description("Logging metrics")
            self.maybe_log_metrics(train_metrics, eval_metrics, self.state.step)
            metrics_list.append(train_metrics)

            # Save checkpoint
            pbar.set_description("Saving checkpoint")
            if self.config.logging.save_steps is not None and self.state.step % self.config.logging.save_steps == 0:
                with self.profiler.profile("save_checkpoint"):
                    self.state.save(self.output_dir, self.accelerator)
            pbar.set_description("Cleaning checkpoints")
            with self.profiler.profile("maybe_clean_checkpoints"):
                self.maybe_clean_checkpoints(self.output_dir, self.config.logging.save_total_limit, self.accelerator)

            pbar.update(1)

        # Final save
        final_save_path = self.output_dir / "final"
        self.state.model.policy_adapter.save(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)

        if hub_model_id is not None:
            logger.info(f"Uploading to Hugging Face Hub: {hub_model_id}")
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=final_save_path,
                repo_id=hub_model_id,
                repo_type="model",
            )

        # Clean up misalignment evaluation executor
        if hasattr(self, "_misalignment_executor"):
            self._misalignment_executor.shutdown(wait=False)

        logger.info("Training completed!")
        # Note: Do NOT call wandb.finish() here - let the calling script handle it
        # since there may be additional logging (e.g., detector evaluation) after training
        return self.state, metrics_list

    def generate_responses(
        self,
        vllm_client: VLLMClient,
        prompt_ids: list[list[int]],
        adapter,
        lora_path: Path,
        num_generations: int = 1,
    ) -> VLLMResponse:
        """Generate responses using vLLM for evaluation purposes.

        This method is used for post-training evaluation, not during training.
        """
        adapter.save(lora_path)
        self.accelerator.wait_for_everyone()
        responses = vllm_client.generate_responses(
            prompts=prompt_ids,
            model_name=self.config.model.model_name,
            model_revision=self.config.model.model_revision,
            max_tokens=self.config.tokenizer.max_response_length,
            n=num_generations,
            lora_path=str(lora_path),
            is_main_process=self.accelerator.is_main_process,
            seed=self.config.vllm.generation_seed,
            eot_token_id=self.tokenizer.eot_token_id,
        )
        return vllm_client.broadcast_responses(responses)
