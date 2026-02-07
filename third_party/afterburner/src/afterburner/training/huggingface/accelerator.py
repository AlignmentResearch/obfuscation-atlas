from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator as AccelerateAccelerator
from accelerate import AutocastKwargs
from accelerate.state import PartialState
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader

from afterburner.grpo_config import GRPOConfig
from afterburner.training.interface import Accelerator
from afterburner.utils.loss_functions import Loss
from afterburner.utils.profiling import Profiler

DTYPE_TO_PRECISION = {
    "bfloat16": "bf16",
    "float16": "fp16",
}


class HfAccelerator(Accelerator):
    """Accelerator for HuggingFace models."""

    def __init__(
        self, *, config: GRPOConfig, loss_function: Callable[[Any], Loss], gradient_accumulation_steps: int, profiler: Profiler
    ):
        # We initialize the PartialState here because FSDP QLoRA does not work without it.
        PartialState()
        super().__init__(
            config=config,
            loss_function=loss_function,
            gradient_accumulation_steps=gradient_accumulation_steps,
            profiler=profiler,
        )
        self._accelerator = AccelerateAccelerator(mixed_precision=DTYPE_TO_PRECISION[config.model.dtype])

    @property
    def is_main_process(self) -> bool:
        return self._accelerator.is_main_process

    @property
    def device(self) -> torch.device:
        return self._accelerator.device

    @property
    def num_processes(self) -> int:
        return self._accelerator.num_processes

    @property
    def process_index(self) -> int:
        return self._accelerator.process_index

    def validate_config(self):
        if getattr(self._accelerator.state, "fsdp_plugin", None):
            if getattr(self._accelerator.state.fsdp_plugin, "forward_prefetch", None):
                raise ValueError("Forward prefetch is not supported for GRPO training as the model graph is not static in HF")

    def forward_backward_step(self, *args, **kwargs) -> Loss:
        with self.profiler.profile("loss_function"):
            loss_metrics = self.loss_function(*args, **kwargs)
        with self.profiler.profile("backward"):
            self._accelerator.backward(loss_metrics.loss / self.gradient_accumulation_steps)
        return loss_metrics

    def wait_for_everyone(self):
        self._accelerator.wait_for_everyone()

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._accelerator.gather_for_metrics(tensor)

    def clip_grad_norm_(self, parameters: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
        return self._accelerator.clip_grad_norm_(parameters, max_norm)

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        return self._accelerator.prepare(dataloader)

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return self._accelerator.reduce(tensor, reduction=reduction)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._accelerator.gather(tensor)

    def pad_across_processes(self, tensor: torch.Tensor, dim: int, pad_index: int) -> torch.Tensor:
        return self._accelerator.pad_across_processes(tensor, dim=dim, pad_index=pad_index)

    def prepare(self, *args) -> tuple[Any, ...]:
        """Use huggingface prepare for the HfModel"""
        return self._accelerator.prepare(*args)

    @property
    def state(self) -> Any:
        return self._accelerator.state

    def autocast(self, autocast_handler: AutocastKwargs = None):
        return self._accelerator.autocast(autocast_handler)

    def path_exists(self, path: Path) -> bool:
        """Check if path exists on main process and broadcast the result to all processes."""
        exists = [path.exists() if self.is_main_process else None]
        exists = broadcast_object_list(exists)
        assert exists[0] is not None
        return exists[0]
