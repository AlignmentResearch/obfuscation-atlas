import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
from torch.utils.data import DataLoader

from afterburner.grpo_config import GRPOConfig
from afterburner.training.checkpointing import RandomState
from afterburner.utils.constants import INVALID_LOGPROB
from afterburner.utils.loss_functions import Loss
from afterburner.utils.profiling import Profiler


class Accelerator(ABC):
    """Accelerator base class."""

    device: torch.device

    def __init__(
        self, *, config: GRPOConfig, loss_function: Callable[[Any], Loss], gradient_accumulation_steps: int, profiler: Profiler
    ):
        self.config = config
        self.loss_function = loss_function
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.profiler = profiler

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        """Check if the current process is the main process."""
        ...

    @property
    @abstractmethod
    def num_processes(self) -> int:
        """Get the number of processes."""
        ...

    @property
    @abstractmethod
    def process_index(self) -> int:
        """Get the process index."""
        ...

    def validate_config(self):
        """Validate the accelerator configuration."""
        pass

    @abstractmethod
    def forward_backward_step(self, *args, **kwargs) -> Loss:
        """Forward and backward step."""
        ...

    @abstractmethod
    def wait_for_everyone(self):
        """Wait for everyone."""
        ...

    @abstractmethod
    def gather_for_metrics(self, *args, **kwargs) -> torch.Tensor:
        """Gather for metrics."""
        ...

    @abstractmethod
    def clip_grad_norm_(self, parameters: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
        """Clip gradient norm."""
        ...

    @abstractmethod
    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Prepare dataloader."""
        ...

    @abstractmethod
    def pad_across_processes(self, tensor: torch.Tensor, dim: int, pad_index: int) -> torch.Tensor:
        """Pad across processes."""
        ...

    @abstractmethod
    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce tensor."""
        ...

    @abstractmethod
    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor."""
        ...

    @abstractmethod
    def autocast(self, *args, **kwargs):
        """Autocast context manager."""
        ...

    @abstractmethod
    def path_exists(self, path: Path) -> bool:
        """Check if path exists on main process and broadcast the result to all processes."""
        ...


@dataclass
class LoRAAdapter(ABC):
    """Abstract base class for LoRA adapters."""

    adapter_name: str
    adapter_path: str
    model: torch.nn.Module
    dtype: torch.dtype
    revision: str | None = None
    subfolder: str | None = None

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the adapter to the specified path."""
        pass

    @abstractmethod
    def logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute log probabilities using this adapter."""
        pass


@dataclass
class RewardLoRAAdapter(LoRAAdapter, ABC):
    """Abstract base class for reward LoRA adapters."""

    score_head: torch.nn.Linear | None = None

    def __post_init__(self):
        self.score_head = torch.nn.Linear(self.model.hidden_size, 1, dtype=self.dtype)
        self.score_head.requires_grad_(False)
        self._load_score_head_from_adapter(self.adapter_path, self.subfolder, self.revision)

    def logprobs(self, *args, **kwargs) -> torch.Tensor:
        """For reward adapters, this method is not used. Use reward() instead."""
        raise NotImplementedError("Reward adapters should use reward() method instead of logprobs()")

    @abstractmethod
    def _compute_last_token_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Compute the last token hidden states."""
        ...

    def _load_score_head_from_adapter(
        self,
        adapter_path: str,
        subfolder: str | None = None,
        revision: str | None = None,
        filename: str = "adapter_model.safetensors",
    ):
        if os.path.exists(adapter_path):
            reward_folder = os.path.join(adapter_path, subfolder) if subfolder else adapter_path
            safetensors_file = os.path.join(reward_folder, filename)
        else:
            safetensors_file = hf_hub_download(
                repo_id=adapter_path,
                revision=revision,
                subfolder=subfolder,
                filename=filename,
            )

        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            score_weight_key = [k for k in keys if "score" in k and "weight" in k][0]
            score_weights = f.get_tensor(score_weight_key)
            self.score_head.weight.data = score_weights.clone().to(dtype=self.score_head.weight.dtype)

            score_bias_keys = [k for k in keys if "score" in k and "bias" in k]
            if score_bias_keys:
                score_bias_key = score_bias_keys[0]
                score_bias = f.get_tensor(score_bias_key)
                if self.score_head.bias is None:
                    self.score_head.bias = torch.nn.Parameter(torch.zeros(2))
                self.score_head.bias.data = score_bias.clone().to(dtype=self.score_head.bias.dtype)

        self.score_head.to(self.model.accelerator.device)

    def reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Compute rewards using this adapter."""
        with torch.no_grad():
            hidden_states = self._compute_last_token_hidden_states(
                input_ids, attention_mask, **kwargs
            )  # [batch_size, hidden_size]
            rewards = self.score_head(hidden_states).squeeze(-1)  # [batch_size]
        return rewards


@dataclass
class TrainerState(ABC):
    """Abstract base class for training checkpoints."""

    step: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    random_state: RandomState

    @classmethod
    @abstractmethod
    def initialize_and_move_to_gpu(cls, config: Any, accelerator: Accelerator, total_steps: int, pad_token_id: int) -> Self:
        """Initialize the checkpointed objects"""
        pass

    @abstractmethod
    def save(self, path: Path, accelerator: Accelerator) -> None:
        """Save the checkpoint to the specified path."""
        pass

    @abstractmethod
    def load(self, path: Path, accelerator: Accelerator) -> None:
        """Load the checkpoint from the specified path."""
        pass


def compute_logprobs_from_logits(
    logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch_size, seq_len]
    completion_mask: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:
    """Compute log probabilities from logits, masking out everything except completion tokens."""
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]

    # Get log probabilities for actual tokens using torch.gather
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len-1]

    # Shift completion_mask to align with shifted labels
    shift_completion_mask = completion_mask[..., 1:].contiguous()  # [batch_size, seq_len-1]

    return torch.where(shift_completion_mask, token_log_probs, INVALID_LOGPROB)
