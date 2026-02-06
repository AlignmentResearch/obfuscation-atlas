"""Minimal GRPO trainer.

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
- Generation output batch size = per_device_train_batch_size * num_processes * gradient_accumulation_steps * num_generations
- For each generation output batch, we perform gradient_accumulation_steps * num_generations backward passes and a single optimizer step
"""

import math
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import auto
from functools import cached_property, partial
from pathlib import Path
from typing import Any

import numpy as np
import pynvml
import torch
import wandb
from accelerate.utils import broadcast_object_list
from datasets import Dataset
from huggingface_hub import HfApi
from strenum import StrEnum
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from afterburner.generation.vllm.client import VLLMClient, VLLMResponse
from afterburner.grpo_config import BackendType, GRPOConfig
from afterburner.training.huggingface.accelerator import HfAccelerator
from afterburner.training.huggingface.model import HfLoRAAdapter
from afterburner.training.huggingface.trainer_state import STATE_NAME, HfTrainerState
from afterburner.training.interface import Accelerator as BaseAccelerator
from afterburner.training.interface import TrainerState
from afterburner.training.turbine.accelerator import TurbineAccelerator
from afterburner.training.turbine.trainer_state import TurbineTrainerState
from afterburner.utils.constants import INVALID_LOGPROB
from afterburner.utils.data_collation import pad
from afterburner.utils.fsdp_viz import visualize_fsdp_distribution
from afterburner.utils.logging import DynamicTable, logger
from afterburner.utils.loss_functions import LOSS_FUNCTIONS, Loss
from afterburner.utils.profiling import Profiler


def zip_dataloaders(dataloader1: DataLoader, dataloader2: DataLoader) -> Iterator[dict[str, Any]]:
    for batch1, batch2 in zip(dataloader1, dataloader2, strict=True):
        yield {**batch1, **batch2}


class Partition(StrEnum):
    TRAIN = auto()
    TEST = auto()


@dataclass(frozen=True)
class GenerationInputs:
    prompt_ids: list[list[int]]
    prompt_indices: list[int]
    prompts_formatted: list[str]


@dataclass(frozen=True)
class GenerationQueueItem:
    generation_inputs: GenerationInputs
    lora_path: Path
    future: Future | VLLMResponse


class GenerationQueue:
    """A FIFO queue for generating responses using the vLLM server.

    Example usage:
    ```
    generation_queue = GenerationQueue(
        max_offset,
        vllm_client,
        config,
        num_generations,
        accelerator,
        eot_token_id,
        profiler,
    )
    # Base case: must submit max_offset times before the loop with the initial lora adapter
    for i in range(generation_queue.max_offset):
        generation_inputs = trainer.prepare_training_inputs(state.step + i)
        generation_queue.submit(generation_inputs, state.model.policy_adapter, lora_path)
    while state.step < trainer.total_steps:
        if state.step + generation_queue.max_offset < trainer.total_steps:
            # If there is no batch of training data left, we don't need to generate more responses
            generation_inputs = trainer.prepare_training_inputs(state.step + generation_queue.max_offset)
            generation_queue.submit(generation_inputs, state.model.policy_adapter, lora_path)
        training_inputs, training_responses = generation_queue.pop()
        state, train_metrics = trainer.train_on_generation_batch(
            generation_inputs=training_inputs,
            responses=training_responses,
            state=state,
            vllm_client=vllm_client,
            completions_table=completions_table,
        )

    """

    def __init__(
        self,
        max_offset: int,
        vllm_client: VLLMClient,
        config: GRPOConfig,
        num_generations: int,
        accelerator: BaseAccelerator,
        eot_token_id: int,
        profiler: Profiler,
    ):
        self.vllm_client = vllm_client
        self.config = config
        self.max_offset = max_offset
        # The queue length can reach max_offset + 1 because we submit before we pop
        self.executor = ThreadPoolExecutor(max_workers=self.max_offset + 1) if self.max_offset > 0 else None
        self.queue = deque[GenerationQueueItem]()
        self.num_generations = num_generations
        self.accelerator = accelerator
        self.eot_token_id = eot_token_id
        self.profiler = profiler
        self.popens = []

    def _generate_responses(
        self,
        inputs: GenerationInputs,
        lora_path: Path,
    ) -> list[dict[str, Any]]:
        """Generate responses using the vLLM server. Must be called from every process."""
        return self.vllm_client.generate_responses(
            inputs.prompt_ids,
            self.config.model.model_name,
            self.config.model.model_revision,
            self.config.tokenizer.max_response_length,
            self.num_generations,
            lora_path,
            self.accelerator.is_main_process,
            self.config.vllm.generation_seed,
            self.eot_token_id,
        )

    def submit(self, inputs: GenerationInputs, lora_adapter: HfLoRAAdapter, lora_path: Path) -> None:
        """Submit a generation task to the queue. Must be called from every process."""
        # N.B. Must block on this even if we are in async mode because we need to save the lora before we can generate responses
        if not self.accelerator.path_exists(lora_path):
            with self.profiler.profile("save_lora_adapter"):
                lora_adapter.save(lora_path)
        if self.executor is None:
            result = self._generate_responses(inputs, lora_path)
            future = self.vllm_client.broadcast_responses(result)
        else:
            future = self.executor.submit(self._generate_responses, inputs, lora_path)
        self.queue.append(GenerationQueueItem(inputs, lora_path, future))

    def _rm_async(self, lora_path: Path):
        """Clean up LoRA adapter directory async, logging results from previous cleanup. Only called from main process."""
        # Check all cleanup processes and remove completed ones
        for popen in self.popens[:]:
            returncode = popen.poll()
            if returncode is not None:
                stdout, stderr = popen.communicate()
                logger.info(
                    f"LoRA cleanup process completed with returncode={returncode}, "
                    f"stdout={stdout.decode()!r}, stderr={stderr.decode()!r}"
                )
                self.popens.remove(popen)
        popen = subprocess.Popen(
            ["rm", "-rf", str(lora_path.resolve())],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.popens.append(popen)
        logger.info(f"Started LoRA cleanup process for {lora_path} (total active: {len(self.popens)})")

    def pop(self) -> tuple[GenerationInputs, VLLMResponse]:
        """Pop a generation task from the queue. Must be called from every process."""
        item = self.queue.popleft()
        generation_inputs = item.generation_inputs
        lora_path = item.lora_path
        future = item.future
        if isinstance(future, Future):
            result = future.result()
            response = self.vllm_client.broadcast_responses(result)
        else:
            response = future
        if self.accelerator.is_main_process:
            self._rm_async(lora_path)
        return generation_inputs, response


@dataclass(frozen=True)
class ScoredResponses:
    prompts_formatted: list[str]
    response_ids: list[list[int]]
    mean_token_entropy: float
    rewards: torch.Tensor
    reward_metrics: dict[str, float]
    dataset: Dataset
    dataloader: DataLoader


@dataclass(frozen=True)
class GRPOLoss:
    batch_losses: torch.Tensor
    batch_kl_losses: torch.Tensor
    batch_kl_divs: torch.Tensor
    total_loss: torch.Tensor
    kl_loss: torch.Tensor
    kl_div: torch.Tensor
    clip_low_mean: torch.Tensor
    clip_high_mean: torch.Tensor
    clip_region_mean: torch.Tensor
    mean_policy_logprob: torch.Tensor
    mean_reference_logprob: torch.Tensor
    mean_abs_log_ratio: torch.Tensor

    def __post_init__(self):
        requires_grad_count = 0
        for attr in self.__dataclass_fields__:
            if getattr(self, attr).requires_grad:
                requires_grad_count += 1
        if requires_grad_count != 1:
            raise ValueError(f"Exactly one attribute of GRPOLoss should be trainable but got {requires_grad_count}")


class BatchMetricsList:
    def __init__(self, accelerator: BaseAccelerator):
        self.accelerator = accelerator
        self.group_metrics: dict[str, list[float]] = {}

    def validate_metrics(self, loss_metrics: Loss):
        assert loss_metrics.loss.numel() == 1, "Loss should be a scalar"
        assert all(tensor.numel() == 1 for tensor in loss_metrics.group_metrics.values()), (
            "All group metrics should be scalars"
        )

    def extend(self, loss_metrics: Loss):
        self.validate_metrics(loss_metrics)
        group_metrics = {
            "loss": loss_metrics.loss.detach(),
            **loss_metrics.group_metrics,
        }
        for key, value in group_metrics.items():
            self.group_metrics[key] = self.group_metrics.get(key, []) + gather_tensor_to_list(value, self.accelerator)

    def __getitem__(self, metric: str) -> list[float]:
        return self.group_metrics[metric]

    def to_dict(self) -> dict[str, float]:
        return {  # type: ignore
            key: np.nanmean(value) for key, value in self.group_metrics.items()
        }


def build_response_dataset(
    tokenizer: PreTrainedTokenizerBase,
    prompt_indices: list[int],
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
    logprobs: list[list[float]],
    max_total_length: int,
) -> Dataset:
    """Build a dataset from prompts and responses with proper completion masks.

    Args:
        tokenizer: The tokenizer to use for encoding
        prompt_indices: Index of the prompt in the original dataset
        prompt_ids: Prompt token ids in nested list
        response_ids: List of response ids corresponding to prompts
        logprobs: List of logprobs corresponding to responses
        max_total_length: Maximum allowed total length (prompt + response)

    Returns:
        Dataset containing input_ids, attention_mask, completion_mask and terminated.
        `completion_mask` is false for prompt tokens and true for response tokens.
        `terminated` is true if the response is terminated by the EOS token (useful for
        applying a non-termination penalty).
    """
    if len(prompt_ids) != len(prompt_indices):
        raise ValueError(f"Number of prompts ({len(prompt_ids)}) must equal number of prompt indices ({len(prompt_indices)})")
    if len(response_ids) != len(logprobs):
        raise ValueError(f"Number of responses ({len(response_ids)}) must equal number of logprobs ({len(logprobs)})")
    num_generations = len(response_ids) // len(prompt_ids)
    prompt_indices_upsampled = [i for i in prompt_indices for _ in range(num_generations)]
    prompt_ids_upsampled = [prompt for prompt in prompt_ids for _ in range(num_generations)]

    max_prompt_length = max(len(p_ids) for p_ids in prompt_ids_upsampled)
    if max_prompt_length > max_total_length:
        raise ValueError(f"Prompt length ({max_prompt_length}) is greater than max_total_length ({max_total_length})")
    response_ids = [
        r_ids[: max_total_length - len(p_ids)] for p_ids, r_ids in zip(prompt_ids_upsampled, response_ids, strict=True)
    ]
    logprobs = [
        l_probs[: max_total_length - len(p_ids)] for p_ids, l_probs in zip(prompt_ids_upsampled, logprobs, strict=True)
    ]
    input_ids = [p_ids + r_ids for p_ids, r_ids in zip(prompt_ids_upsampled, response_ids, strict=True)]
    attention_mask = [[1 for _ in batch] for batch in input_ids]
    completion_mask = [
        [False] * len(p_ids) + [True] * len(r_ids)
        for p_ids, r_ids in zip(
            prompt_ids_upsampled,  # type: ignore
            response_ids,
            strict=True,
        )
    ]
    terminated = [r_ids[-1] == tokenizer.eot_token_id for r_ids in response_ids]

    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "terminated": terminated,
            "logprobs": logprobs,
            "prompt_indices": prompt_indices_upsampled,
        }
    )


def grpo_collate_fn(batch, pad_token_id: int):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    completion_mask = [torch.tensor(item["completion_mask"], dtype=torch.bool) for item in batch]
    logprobs = [torch.tensor(item["logprobs"], dtype=torch.float32) for item in batch]
    prompt_indices = [torch.tensor(item["prompt_indices"], dtype=torch.long) for item in batch]

    padded_input_ids = pad(input_ids, padding_value=pad_token_id)
    padded_attention_mask = pad(attention_mask, padding_value=0)
    padded_completion_mask = pad(completion_mask, padding_value=False)
    padded_logprobs = pad(logprobs, padding_value=INVALID_LOGPROB)
    # There is only one prompt index per sample, so no need to pad
    stacked_prompt_indices = torch.stack(prompt_indices)

    out = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "completion_mask": padded_completion_mask,
        "logprobs": padded_logprobs,
        "prompt_indices": stacked_prompt_indices,
    }
    if "advantages" in batch[0]:
        out["advantages"] = torch.tensor([item["advantages"] for item in batch], dtype=torch.float32)
    return out


def gather_tensor_to_list(tensor: torch.Tensor, accelerator: BaseAccelerator) -> list[float]:
    gathered_tensor = accelerator.gather_for_metrics(tensor)
    assert isinstance(gathered_tensor, torch.Tensor)
    if not gathered_tensor.shape:
        gathered_tensor = gathered_tensor.unsqueeze(0)
    return gathered_tensor.tolist()


BACKEND_TO_ACCELERATOR: dict[BackendType, type[BaseAccelerator]] = {
    BackendType.HUGGINGFACE: HfAccelerator,
    BackendType.TURBINE: TurbineAccelerator,
}
BACKEND_TO_TRAINER_STATE: dict[BackendType, type[TrainerState]] = {
    BackendType.HUGGINGFACE: HfTrainerState,
    BackendType.TURBINE: TurbineTrainerState,
}


@dataclass(frozen=True)
class RewardFunctionResult:
    rewards: list[float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class BaseRewardFunction(ABC):
    @staticmethod
    @abstractmethod
    def __call__(prompt_indices: list[int], responses: list[str], partition: Partition) -> RewardFunctionResult:
        """Compute rewards for a list of prompts and responses.

        Args:
            prompt_indices: The indices of the prompts in the original dataset.
            responses: The responses to the prompts.
            partition: The partition of the dataset.

        Returns:
            A RewardFunctionResult object containing the rewards and any other metrics.
        """
        pass


class GRPOTrainer:
    def __init__(
        self,
        train_prompts: list[str],
        config: GRPOConfig,
        reward_function: BaseRewardFunction | None = None,
        eval_prompts: list[str] | None = None,
    ):
        self.config = config
        self.profiler = Profiler(prefix="profiling/time_per_optimizer_step")
        if config.loss.loss_type not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss type: {config.loss.loss_type}")
        self.loss_function = LOSS_FUNCTIONS[config.loss.loss_type]
        self.accelerator = BACKEND_TO_ACCELERATOR[config.backend](
            config=config,
            loss_function=self.loss_function,
            gradient_accumulation_steps=self.actual_gradient_accumulation_steps,
            profiler=self.profiler,
        )
        self.output_dir = self.get_output_dir()
        if self.config.logging.log_with_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.logging.wandb_project or "grpo-training",
                group=config.logging.wandb_group,
                name=config.logging.wandb_name,
                config=asdict(config),
            )
        if config.logging.eval_steps is not None and eval_prompts is None:
            raise ValueError("eval_prompts must be provided if eval_steps is not None")
        self._maybe_use_deterministic_algorithms(config)
        self._maybe_init_reward_function(reward_function)
        self._init_tokenizer()
        self._init_datasets(train_prompts, eval_prompts)
        self.last_log_flops = 0.0
        self.last_log_time = time.perf_counter()
        self.last_vllm_request_time = time.perf_counter()
        self.h100_peak_flops = self._calculate_adjusted_peak_flops()  # H100 peak FLOP/s adjusted for GPU memory usage

    @cached_property
    def lora_dir(self) -> Path:
        if self.config.vllm.lora_path is not None:
            return Path(self.config.vllm.lora_path)
        else:
            return self.output_dir / "lora"

    def _calculate_adjusted_peak_flops(self) -> float:
        """Calculate H100 peak FLOPS adjusted for GPU memory utilization by other processes (e.g., vLLM server).

        We loop through each device and find the total memory already reserved by other processes such as vllm.
        Then we adjust the total flops by a factor of the actual available memory.
        """
        base_peak_flops = 989e12 * self.accelerator.num_processes  # H100 peak FLOP/s

        if not torch.cuda.is_available():
            return base_peak_flops

        # Check available GPU memory across all devices
        total_memory = 0
        available_memory = 0
        pynvml.nvmlInit()
        for device_id in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mem = meminfo.used
            total_mem = meminfo.total
            total_memory += total_mem
            available_memory += total_mem - used_mem

        if total_memory == 0:
            return base_peak_flops

        # Calculate the fraction of memory available for training
        memory_fraction = available_memory / total_memory

        logger.info(
            f"Found {available_memory} / {total_memory} = {memory_fraction:.1%} memory available for training, "
            f"adjusted peak FLOPs from {base_peak_flops:.2E} to {base_peak_flops * memory_fraction:.2E}"
        )

        return base_peak_flops * memory_fraction

    def get_output_dir(self) -> Path:
        if self.config.logging.output_dir is None:
            output_path = (
                f"./checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if self.accelerator.is_main_process else None
            )
            output_path = broadcast_object_list([output_path])[0]
            assert output_path is not None
            output_path = Path(output_path)
        else:
            output_path = Path(self.config.logging.output_dir)
        output_path = output_path.resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _maybe_use_deterministic_algorithms(config: GRPOConfig):
        torch.manual_seed(config.random.training_seed)
        np.random.seed(config.random.training_seed)
        if config.random.cuda_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

    def _maybe_init_reward_function(self, reward_function: BaseRewardFunction | None):
        if reward_function is not None and self.config.model.reward_adapter_path is not None:
            raise ValueError("reward_function and reward_adapter_path cannot be used together")
        if reward_function is None and self.config.model.reward_adapter_path is None:
            raise ValueError("Either reward_function or reward_adapter_path must be provided")
        self.reward_function = reward_function

    def _init_tokenizer(self):
        if self.config.tokenizer.tokenizer_path is not None:
            path = self.config.tokenizer.tokenizer_path
            revision = self.config.tokenizer.tokenizer_revision
        else:
            path = self.config.model.reference_adapter_path or self.config.model.model_name
            revision = self.config.model.reference_revision or self.config.model.model_revision

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            path,
            revision=revision,
        )
        self.tokenizer.eot_token = self.config.tokenizer.eot_token
        self.tokenizer.eot_token_id = self.tokenizer.get_vocab().get(self.config.tokenizer.eot_token, None)
        if self.tokenizer.eot_token_id is None:
            raise ValueError(
                f"EOT token {self.config.tokenizer.eot_token} not found in tokenizer."
                " Please supply a different EOT token using the config."
            )

        # Use pad token from config if pad token is not already set or is the same as the eot token
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eot_token_id:
            self.tokenizer.pad_token = self.config.tokenizer.pad_token
            self.tokenizer.pad_token_id = self.tokenizer.get_vocab().get(self.config.tokenizer.pad_token, None)
            if self.tokenizer.pad_token_id is None:
                raise ValueError(
                    f"Pad token {self.config.tokenizer.pad_token} not found in tokenizer."
                    " Please supply a different pad token using the config."
                )

        if self.tokenizer.pad_token_id == self.tokenizer.eot_token_id:
            # TODO: maybe handle this in a better way e.g. by masking everything after the first EOS token
            raise ValueError("pad_token_id and eot_token_id must be different, otherwise the EOS token will be masked out.")

    def _init_datasets(self, train_prompts: list[str], eval_prompts: list[str] | None) -> None:
        self.prompts = {Partition.TRAIN: train_prompts, Partition.TEST: eval_prompts}
        self.prompt_indices = {
            Partition.TRAIN: list(range(len(train_prompts))),
            Partition.TEST: list(range(len(eval_prompts))) if eval_prompts is not None else [],
        }
        self._init_dataset(Partition.TRAIN)
        if self.accelerator.is_main_process and self.config.logging.log_with_wandb:
            wandb.log({"train/num_prompts": len(self.prompts[Partition.TRAIN])}, commit=False)
        if eval_prompts is not None:
            self._init_dataset(Partition.TEST)
            if self.accelerator.is_main_process and self.config.logging.log_with_wandb:
                wandb.log({"eval/num_prompts": len(self.prompts[Partition.TEST])}, commit=False)

    def _init_dataset(self, partition: Partition):
        prompts = self.prompts[partition]
        original_length = len(prompts)
        prompt_indices = list(range(len(prompts)))
        prompt_length_mask = self.get_prompt_length_mask(prompts)
        prompt_indices = [i for i, mask in enumerate(prompt_length_mask) if mask]
        prompts = [p for p, mask in zip(prompts, prompt_length_mask) if mask]
        logger.info(f"Filtered from {original_length} to {len(prompts)} {partition.value} prompts")
        self.prompts[partition] = prompts
        self.prompt_indices[partition] = prompt_indices

    @staticmethod
    def compute_advantages_from_rewards(
        rewards: torch.Tensor,  # [generation_batch_size * num_generations]
        num_generations: int,
        scale_rewards: bool,
    ) -> torch.Tensor:  # [generation_batch_size * num_generations]
        # Compute advantages from rewards
        # Reshape rewards to compute group statistics
        grouped_rewards = rewards.view(-1, num_generations)  # [batch_size // num_generations, num_generations]

        # Compute group-wise mean and std
        mean_grouped_rewards = grouped_rewards.mean(dim=1)  # [batch_size // num_generations]
        std_grouped_rewards = grouped_rewards.std(dim=1)  # [batch_size // num_generations]

        # Repeat to match original batch size
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)  # [batch_size]
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)  # [batch_size]

        # Compute advantages by subtracting group mean
        advantages = rewards - mean_grouped_rewards  # [batch_size]

        # Scale advantages by group std if enabled
        if scale_rewards and num_generations > 1:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        return advantages

    def compute_correlation(self, policy_logprobs: torch.Tensor, old_policy_logprobs: torch.Tensor) -> float:
        policy_corr_input = policy_logprobs[policy_logprobs != INVALID_LOGPROB]
        vllm_corr_input = old_policy_logprobs[old_policy_logprobs != INVALID_LOGPROB]
        if (policy_corr_input == vllm_corr_input).all():
            correlation = 1.0
        elif policy_corr_input.std().item() == 0 or vllm_corr_input.std().item() == 0:
            correlation = 0.0
        else:
            correlation = torch.stack([policy_corr_input, vllm_corr_input]).corrcoef()[0, 1].item()
            assert not np.isnan(correlation)
        threshold = self.config.vllm.correlation_check_threshold
        severity = self.config.vllm.correlation_check_severity
        if correlation < threshold and severity == "warning":
            logger.warning(f"Correlation between policy and logprobs is {correlation}, expected >= {threshold}")
        elif correlation < threshold and severity == "error":
            raise ValueError(f"Correlation between policy and logprobs is {correlation}, expected >= {threshold}")
        return correlation

    # TODO: figure out if there is a way to avoid this precarious padding
    def reshape_old_logprobs(
        self,
        old_logprobs: torch.Tensor,  # [batch_size, max_response_length]
        completion_mask: torch.Tensor,  # [batch_size, full_seq_length-1]
        policy_logprobs: torch.Tensor,  # [batch_size, full_seq_length-1]
    ) -> torch.Tensor:  # [batch_size, full_seq_length-1]
        """Reshape old logprobs to match the shape of policy logprobs."""
        old_logprobs = old_logprobs.to(dtype=policy_logprobs.dtype)
        if completion_mask.shape != policy_logprobs.shape:
            raise ValueError(
                f"Completion mask shape {completion_mask.shape} does not match policy logprobs shape {policy_logprobs.shape}"
            )
        if old_logprobs.shape[0] != policy_logprobs.shape[0]:
            raise ValueError(
                f"Old logprobs shape {old_logprobs.shape} does not match policy logprobs shape {policy_logprobs.shape}"
            )
        if old_logprobs.shape[1] > policy_logprobs.shape[1]:
            raise ValueError(
                f"Old logprobs shape {old_logprobs.shape} is greater than policy logprobs shape {policy_logprobs.shape}"
            )
        reshaped_logprobs = torch.full_like(policy_logprobs, fill_value=INVALID_LOGPROB)
        for i in range(reshaped_logprobs.shape[0]):
            # N.B. Recall that all tensors are right-padded, so for policy_logprobs and completion_mask we have
            # <prompt> | <response> | <padding>. Whereas for old_logprobs we just have
            # <response> | <padding>.
            prompt_length = completion_mask[i].float().argmax().item()
            response_length = completion_mask[i].sum()
            logprobs_to_fill = old_logprobs[i, :response_length]
            assert (logprobs_to_fill != INVALID_LOGPROB).all()
            reshaped_logprobs[i, torch.arange(prompt_length, prompt_length + response_length)] = logprobs_to_fill
        assert ((reshaped_logprobs * completion_mask) != INVALID_LOGPROB).all()
        assert (policy_logprobs != INVALID_LOGPROB).sum() == (reshaped_logprobs != INVALID_LOGPROB).sum()

        return reshaped_logprobs

    @staticmethod
    def optimizer_step(state: TrainerState):
        state.optimizer.step()
        state.scheduler.step()
        state.optimizer.zero_grad()

    @cached_property
    def actual_gradient_accumulation_steps(self) -> int:
        return self.config.batching.gradient_accumulation_steps * self.config.batching.num_generations

    @cached_property
    def per_device_batch_size(self) -> dict[Partition, int]:
        return {
            Partition.TRAIN: self.config.batching.per_device_train_batch_size,
            Partition.TEST: self.config.batching.per_device_eval_batch_size,
        }

    @cached_property
    def num_generations(self) -> dict[Partition, int]:
        return {
            Partition.TRAIN: self.config.batching.num_generations,
            Partition.TEST: self.config.batching.num_generations_eval,
        }

    def prepare_dataloader(
        self, dataset: Dataset, collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    ) -> DataLoader:
        """Prepare a dataloader for the dataset."""
        return self.accelerator.prepare(
            DataLoader(
                dataset,  # type: ignore
                batch_size=self.config.batching.per_device_train_batch_size,
                collate_fn=collate_fn
                or partial(
                    grpo_collate_fn,
                    pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                ),
            )
        )

    def update_dataloader_with_advantages(
        self, advantages: list[float], dataset: Dataset, dataloader: DataLoader
    ) -> tuple[Iterator, dict[int, int]]:
        """Create a dataloader including the advantages tensor and a map of filtered indices to original indices."""
        dataset = dataset.add_column("advantages", advantages)
        if self.config.optimizer.skip_degenerate_groups:
            # Filter out groups of size num_generations that have all advantages the same.
            indices_to_keep = []
            for group_start in range(0, len(advantages), self.config.batching.num_generations):
                group_advantages = advantages[group_start : group_start + self.config.batching.num_generations]
                if not np.isclose(group_advantages, group_advantages[0], rtol=1.0e-5, atol=1.0e-8).all():
                    indices_to_keep.extend(range(group_start, group_start + self.config.batching.num_generations))
            dataset = dataset.select(indices_to_keep)
            return self.prepare_dataloader(dataset), dict(zip(range(len(indices_to_keep)), indices_to_keep))
        else:
            advantages_dataset = Dataset.from_dict({"advantages": advantages})
            advantages_dataloader = self.prepare_dataloader(
                advantages_dataset,
                collate_fn=lambda batch: {
                    "advantages": torch.tensor([item["advantages"] for item in batch], dtype=torch.float32)
                },
            )
            return zip_dataloaders(dataloader, advantages_dataloader), dict(zip(range(len(dataset)), range(len(dataset))))

    def _get_inputs_for_step(self, step: int) -> tuple[list[int], list[str]]:
        start_index = step * self.generation_batch_size
        num_prompts = len(self.prompts[Partition.TRAIN])

        # Wrap around if we exceed the dataset length
        indices = [(start_index + i) % num_prompts for i in range(self.generation_batch_size)]

        return (
            [self.prompt_indices[Partition.TRAIN][i] for i in indices],
            [self.prompts[Partition.TRAIN][i] for i in indices],
        )

    def _prepare_inputs_for_generation(self, prompt_indices: list[int], prompts: list[str]) -> GenerationInputs:
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
        prompt_ids: list[list[int]] = self.tokenizer(
            prompts_formatted, add_special_tokens=not self.config.tokenizer.apply_chat_template
        )["input_ids"]
        assert isinstance(prompt_ids, list)
        return GenerationInputs(
            prompt_ids=prompt_ids,
            prompt_indices=prompt_indices,
            prompts_formatted=prompts_formatted,
        )

    def prepare_training_inputs(self, step: int) -> GenerationInputs:
        prompt_indices, prompts = self._get_inputs_for_step(step)
        return self._prepare_inputs_for_generation(prompt_indices, prompts)

    def score_responses(
        self, generation_inputs: GenerationInputs, responses: VLLMResponse, state: TrainerState, partition: Partition
    ) -> ScoredResponses:
        # Scoring responses is the first step for both training and evaluation so it makes sense to set the mode here
        state.model.train(mode=(partition == Partition.TRAIN))
        response_ids = responses.tokens
        # [num_processes * per_device_train_batch_size * steps_per_generation * num_generations]

        # Build dataset with response encodings and completion masks
        dataset = build_response_dataset(
            tokenizer=self.tokenizer,
            prompt_ids=generation_inputs.prompt_ids,
            prompt_indices=generation_inputs.prompt_indices,
            response_ids=response_ids,
            logprobs=responses.logprobs,
            max_total_length=self.config.tokenizer.max_total_length,
        )

        with self.profiler.profile("prepare_dataloader"):
            dataloader = self.prepare_dataloader(dataset)

        # Compute rewards and advantages up front, gathering and then rebroadcasting
        with self.profiler.profile("compute_rewards"):
            if self.reward_function is None:
                rewards = []
                for batch in dataloader:
                    rewards.append(self.accelerator.gather_for_metrics(state.model.reward_adapter.reward(**batch)))
                rewards = torch.cat(rewards, dim=0)
                reward_metrics = {}
            else:
                reward_result = self.reward_function(
                    dataset["prompt_indices"], self.tokenizer.batch_decode(response_ids), partition
                )
                rewards = torch.tensor(reward_result.rewards)
                reward_metrics = reward_result.metrics
        assert rewards.shape[0] == len(response_ids)
        return ScoredResponses(
            prompts_formatted=generation_inputs.prompts_formatted,
            response_ids=response_ids,
            mean_token_entropy=responses.mean_token_entropy(),
            rewards=rewards,
            reward_metrics=reward_metrics,
            dataset=dataset,
            dataloader=dataloader,
        )

    def train_on_generation_batch(
        self,
        generation_inputs: GenerationInputs,
        responses: VLLMResponse,
        state: TrainerState,
        vllm_client: VLLMClient,
        completions_table: DynamicTable,
    ) -> tuple[TrainerState, dict[str, float | int]]:
        """Train on a generation batch of prompts num_iterations times.

        Args:
            state: The previous trainer state.
            vllm_client: The vLLM client.
            generation_inputs: The generation inputs.
            responses: The responses from the vLLM server.
            state: The trainer state.
            vllm_client: The vLLM client.
            completions_table: A table to log completions to.

        Returns:
            The new trainer state and a dictionary containing training metrics.
        """
        scored_responses = self.score_responses(generation_inputs, responses, state, Partition.TRAIN)

        prompts_formatted = scored_responses.prompts_formatted
        dataset = scored_responses.dataset
        dataloader = scored_responses.dataloader
        rewards = scored_responses.rewards
        response_ids = scored_responses.response_ids

        if self.config.loss.non_termination_penalty is not None:
            terminated = torch.tensor(dataset["terminated"], device=rewards.device, dtype=torch.float32)
            rewards -= self.config.loss.non_termination_penalty * (1 - terminated)

        if self.config.loss.clip_max_reward is not None:
            rewards = torch.clamp(rewards, max=self.config.loss.clip_max_reward)

        with self.profiler.profile("compute_advantages_from_rewards"):
            advantages = self.compute_advantages_from_rewards(
                rewards,
                self.config.batching.num_generations,
                self.config.loss.scale_rewards,
            )

        with self.profiler.profile("prepare_advantages_dataloader"):
            advantages_dataloader, filtered_index_to_original_index = self.update_dataloader_with_advantages(
                advantages.tolist(), dataset, dataloader
            )

        logprob_corrs = []
        generation_batch_metrics = BatchMetricsList(self.accelerator)
        grad_norms = []
        for iteration_idx in range(self.config.batching.num_iterations):
            for batch_idx, batch in enumerate(advantages_dataloader):
                with self.profiler.profile("reference_logprobs"):
                    reference_logprobs = state.model.reference_adapter.logprobs(**batch)
                # The next-token logprobs correspond to the last seq_len-1 tokens so
                # we must slice the completion_mask accordingly.
                non_bos_completion_mask = batch["completion_mask"][:, 1:]

                # Like way over here is where we actually computed the logprobs,
                # and is in some sense the step...
                with self.profiler.profile("policy_logprobs"):
                    policy_logprobs = state.model.policy_adapter.logprobs(**batch)
                old_policy_logprobs = self.reshape_old_logprobs(
                    old_logprobs=batch["logprobs"],
                    policy_logprobs=policy_logprobs,
                    completion_mask=non_bos_completion_mask,
                )
                if iteration_idx == 0:
                    # Correlation check is only valid for the first iteration
                    correlation = self.compute_correlation(policy_logprobs, old_policy_logprobs)
                    logprob_corrs.append(correlation)

                loss_kwargs = dict(
                    advantages=batch["advantages"],
                    policy_logprobs=policy_logprobs,
                    reference_logprobs=reference_logprobs,
                    completion_mask=non_bos_completion_mask,
                    old_policy_logprobs=old_policy_logprobs,
                    kl_coeff=self.config.loss.kl_coeff,
                    kl_estimator=self.config.loss.kl_estimator,
                    epsilon=self.config.loss.epsilon,
                    epsilon_high=self.config.loss.epsilon_high,
                    delta=self.config.loss.delta,
                    max_response_length=self.config.tokenizer.max_response_length,
                    loss_aggregation=self.config.loss.loss_aggregation,
                    prompt_indices=batch["prompt_indices"],
                )

                loss_metrics = self.accelerator.forward_backward_step(**loss_kwargs)
                generation_batch_metrics.extend(loss_metrics)

            grad_norm = self.accelerator.clip_grad_norm_(
                state.model.parameters(),
                max_norm=self.config.optimizer.max_grad_norm,
            )
            assert isinstance(grad_norm, torch.Tensor)
            grad_norms.extend(gather_tensor_to_list(grad_norm, self.accelerator))
            # One optimizer step per iteration since steps_per_generation=gradient_accumulation_steps
            with self.profiler.profile("optimizer_step"):
                self.optimizer_step(state)

        if state.step % self.config.logging.steps_to_log_completions == 0:
            for completion_idx in range(min(self.config.logging.num_completions_to_log, len(response_ids))):
                original_idx = filtered_index_to_original_index[completion_idx]
                completions_table.add_data(
                    generation_batch=state.step,
                    generation_group=completion_idx // self.config.batching.num_generations,
                    generation_index=completion_idx % self.config.batching.num_generations,
                    prompt=prompts_formatted[original_idx // self.config.batching.num_generations],
                    response=self.tokenizer.decode(response_ids[original_idx]),
                    reward=rewards[original_idx],
                    advantage=advantages[original_idx],
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
        metrics = {
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
            **generation_batch_metrics.to_dict(),
            **scored_responses.reward_metrics,
        }
        if "reward_mean" in metrics and "kl_loss" in metrics:
            metrics["objective"] = metrics["reward_mean"] - metrics["kl_loss"]
        self.last_log_flops = total_flops
        self.last_log_time = current_time
        self.last_vllm_request_time = vllm_client.total_request_wait_time

        return BACKEND_TO_TRAINER_STATE[self.config.backend](
            step=state.step + 1,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            random_state=state.random_state,
        ), metrics

    def get_prompt_length_mask(self, prompts: list[str]) -> list[bool]:
        """Get a mask of prompts that are within the max prompt length."""
        formatted_prompts: list[list[int]] = (  # type: ignore
            self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if self.config.tokenizer.apply_chat_template
            else self.tokenizer(prompts)["input_ids"]
        )
        return [len(formatted_prompt) <= self.config.tokenizer.max_prompt_length for formatted_prompt in formatted_prompts]

    @staticmethod
    def maybe_clean_checkpoints(output_dir: Path, save_total_limit: int | None, accelerator: BaseAccelerator):
        """Clean up old checkpoints and lora paths."""
        checkpoint_paths = list(output_dir.glob("step_*"))
        checkpoint_paths.sort(key=lambda x: int(x.name.split("_")[-1]))
        num_checkpoints = 0
        for path in checkpoint_paths[::-1]:
            if save_total_limit is not None and num_checkpoints >= save_total_limit and accelerator.is_main_process:
                # We have already saved enough checkpoints so remove this one
                shutil.rmtree(path)
            elif STATE_NAME in [content.name for content in path.glob("*")]:
                # This is a complete checkpoint
                num_checkpoints += 1

    @cached_property
    def generation_batch_size(self) -> int:
        """The size of a batch sent to the vllm server."""
        return (
            self.config.batching.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.config.batching.gradient_accumulation_steps
        )

    @cached_property
    def generation_output_batch_size(self) -> int:
        """The generation output size from one batch of prompts."""
        return self.generation_batch_size * self.config.batching.num_generations

    def maybe_log_metrics(
        self,
        train_metrics: dict[str, float | int],
        eval_metrics: dict[str, float | int],
        completions_table: DynamicTable,
        generation_batch_step: int,
    ):
        if self.config.logging.log_with_wandb and self.accelerator.is_main_process:
            with self.profiler.profile("log_completions"):
                completions_table.log()
            profiling_metrics = self.profiler.get_and_reset()
            metrics = {"train/" + k: v for k, v in train_metrics.items()}
            metrics.update({"eval/" + k: v for k, v in eval_metrics.items()})
            metrics.update(profiling_metrics)
            wandb.log(metrics, step=generation_batch_step)

    def maybe_get_checkpoint_path(self) -> Path | None:
        # Compute checkpoint path only on main process
        result = ""
        if self.accelerator.is_main_process:
            checkpoint_paths = list(self.output_dir.glob("step_*"))
            checkpoint_paths.sort(key=lambda x: int(x.name.split("_")[-1]))
            for checkpoint_path in checkpoint_paths[::-1]:
                checkpoint_contents = list(checkpoint_path.glob("*"))
                if STATE_NAME in [content.name for content in checkpoint_contents]:
                    result = checkpoint_path
                    break
                else:
                    logger.warning(f"Checkpoint {checkpoint_path} is not a complete checkpoint, removing it")
                    shutil.rmtree(checkpoint_path)

        # Broadcast to ensure consistency across processes in checkpoint loading
        result_str = broadcast_object_list([result])[0]
        return Path(result_str) if result_str else None

    @cached_property
    def total_steps(self) -> int:
        total_steps = math.ceil(
            self.config.optimizer.num_train_epochs
            * self.config.batching.num_iterations
            * len(self.prompts[Partition.TRAIN])
            / self.generation_batch_size
        )
        if self.config.optimizer.max_steps is not None:
            total_steps = min(total_steps, self.config.optimizer.max_steps)
        return total_steps

    @cached_property
    def generation_inputs_for_evaluation(self) -> GenerationInputs:
        return self._prepare_inputs_for_generation(self.prompt_indices[Partition.TEST], self.prompts[Partition.TEST])

    def generate_responses_for_training(
        self,
        generation_queue: GenerationQueue,
        state: TrainerState,
        lora_path: Path,
    ) -> tuple[GenerationInputs, VLLMResponse]:
        assert isinstance(state.model.policy_adapter, HfLoRAAdapter)
        if state.step + generation_queue.max_offset < self.total_steps:
            # If there is no batch of training data left, we don't need to generate more responses
            generation_inputs = self.prepare_training_inputs(state.step + generation_queue.max_offset)
            generation_queue.submit(generation_inputs, state.model.policy_adapter, lora_path)
        training_inputs, training_responses = generation_queue.pop()
        return training_inputs, training_responses

    def train(
        self,
        hub_model_id: str | None = None,
    ) -> tuple[TrainerState, list[dict[str, float | int]]]:
        """Main training loop."""
        self.accelerator.validate_config()
        assert isinstance(self.tokenizer.pad_token_id, int)
        checkpoint_path = self.maybe_get_checkpoint_path()
        state = BACKEND_TO_TRAINER_STATE[self.config.backend].initialize_and_move_to_gpu(
            config=self.config,
            accelerator=self.accelerator,
            total_steps=self.total_steps,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        vllm_client = VLLMClient(self.config.vllm)
        if checkpoint_path is not None and self.config.logging.resume_from_checkpoint:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state.load(checkpoint_path, self.accelerator)
        elif checkpoint_path is not None:
            logger.warning(f"Checkpoint {checkpoint_path} found but resume_from_checkpoint is False, skipping")
        visualize_fsdp_distribution(state.model, self.accelerator)
        vllm_client = VLLMClient(self.config.vllm)
        vllm_client.wait_for_server()
        logger.info("Starting GRPO training...")

        completions_table = DynamicTable(name="completions")
        metrics_list = []
        pbar = tqdm(total=self.total_steps, desc="Training", unit="step", initial=state.step)
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
        lora_path = self.lora_dir / f"step_{state.step}"
        assert isinstance(state.model.policy_adapter, HfLoRAAdapter)
        for i in range(generation_queue.max_offset):
            generation_inputs = self.prepare_training_inputs(state.step + i)
            generation_queue.submit(generation_inputs, state.model.policy_adapter, lora_path)
        if self.config.logging.eval_steps is not None:
            eval_queue.submit(self.generation_inputs_for_evaluation, state.model.policy_adapter, lora_path)
        while state.step < self.total_steps:
            lora_path = self.lora_dir / f"step_{state.step}"

            pbar.set_description("Evaluating")
            with self.profiler.profile("evaluation_step"):
                eval_metrics = self.evaluation_step(state, eval_queue, lora_path)

            pbar.set_description("Generating training responses")
            with self.profiler.profile("generate_responses"):
                training_inputs, training_responses = self.generate_responses_for_training(
                    generation_queue,
                    state,
                    lora_path,
                )

            pbar.set_description(f"Training on batch {state.step + 1} of {self.total_steps}")
            with self.profiler.profile("train_on_generation_batch"):
                state, train_metrics = self.train_on_generation_batch(
                    generation_inputs=training_inputs,
                    responses=training_responses,
                    state=state,
                    vllm_client=vllm_client,
                    completions_table=completions_table,
                )

            pbar.set_description("Logging metrics")
            self.maybe_log_metrics(train_metrics, eval_metrics, completions_table, state.step)
            metrics_list.append(train_metrics)

            pbar.set_description("Saving checkpoint")
            if self.config.logging.save_steps is not None and state.step % self.config.logging.save_steps == 0:
                with self.profiler.profile("save_checkpoint"):
                    state.save(self.output_dir, self.accelerator)
            pbar.set_description("Cleaning checkpoints")
            with self.profiler.profile("maybe_clean_checkpoints"):
                self.maybe_clean_checkpoints(self.output_dir, self.config.logging.save_total_limit, self.accelerator)

            pbar.update(1)

        # Final save
        final_save_path = self.output_dir / "final"
        state.model.policy_adapter.save(final_save_path)
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
        logger.info("Training completed!")
        if self.config.logging.log_with_wandb and self.accelerator.is_main_process:
            wandb.finish()
        return state, metrics_list

    def evaluation_step(
        self,
        state: TrainerState,
        eval_queue: GenerationQueue,
        lora_path: Path,
    ) -> dict[str, float | int]:
        assert isinstance(state.model.policy_adapter, HfLoRAAdapter)
        eval_metrics = {}
        is_eval_step = self.config.logging.eval_steps is not None and (state.step + 1) % self.config.logging.eval_steps == 0
        if is_eval_step:
            eval_queue.submit(self.generation_inputs_for_evaluation, state.model.policy_adapter, lora_path)
            _, eval_responses = eval_queue.pop()
            eval_metrics = self.compute_eval_metrics(eval_responses, state)
        return eval_metrics

    def compute_eval_metrics(self, eval_responses: VLLMResponse, state: TrainerState) -> dict[str, float | int]:
        scored_responses = self.score_responses(self.generation_inputs_for_evaluation, eval_responses, state, Partition.TEST)
        response_lengths = [len(r) for r in scored_responses.response_ids]
        response_length_at_max = [
            response_length == self.config.tokenizer.max_response_length for response_length in response_lengths
        ]
        return {
            "reward_mean": scored_responses.rewards.mean().item(),
            "reward_std": scored_responses.rewards.std().item(),
            "num_samples": scored_responses.rewards.numel(),
            "response_length_mean": sum(response_lengths) / len(response_lengths),
            "response_length_min": min(response_lengths),
            "response_length_max": max(response_lengths),
            "response_length_at_max_fraction": sum(response_length_at_max) / len(response_length_at_max),
            "mean_token_entropy": scored_responses.mean_token_entropy,
            **scored_responses.reward_metrics,
        }
