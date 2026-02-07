import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate.utils.fsdp_utils import (
    FSDP_MODEL_NAME,
    load_fsdp_optimizer,
    save_fsdp_optimizer,
)
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from transformers.optimization import (
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from afterburner.grpo_config import GRPOConfig
from afterburner.training.checkpointing import RandomState
from afterburner.training.huggingface.accelerator import HfAccelerator
from afterburner.training.huggingface.model import HFModel
from afterburner.training.interface import TrainerState
from afterburner.utils.logging import logger

STATE_NAME = "state.json"
SCHEDULER_NAME = "scheduler.pt"
OPTIMIZER_NAME = "optimizer.pt"
MODEL_NAME = "model.pt"


def save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, model_index=0):
    """Save the model for checkpointing.

    Forked from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/fsdp_utils.py
    in order to save only policy adapter weights from the multi-headed model.
    We only support SHARDED_STATE_DICT.
    """
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    assert fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT

    os.makedirs(output_dir, exist_ok=True)

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )

    with ctx:
        state_dict = get_peft_model_state_dict(model, adapter_name="policy")
        ckpt_dir = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{model_index}")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Saving model to {ckpt_dir}")
        state_dict = {"model": state_dict}

        dist_cp.save(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
            planner=DefaultSavePlanner(),
        )
        logger.info(f"Model saved to {ckpt_dir}")


def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0):
    """Load the model state from a checkpoint.

    Forked from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/fsdp_utils.py
    in order to load only policy adapter weights from the multi-headed model.
    """
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    assert fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT

    accelerator.wait_for_everyone()

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    with ctx:
        ckpt_dir = (
            os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{model_index}") if f"{FSDP_MODEL_NAME}" not in input_dir else input_dir
        )
        logger.info(f"Loading model from {ckpt_dir}")
        state_dict = {"model": get_peft_model_state_dict(model, adapter_name="policy")}
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(ckpt_dir),
            planner=DefaultLoadPlanner(),
        )
        state_dict = state_dict["model"]
        logger.info(f"Model loaded from {ckpt_dir}")

        load_result = set_peft_model_state_dict(model, state_dict, adapter_name="policy")
    assert isinstance(load_result, torch.nn.modules.module._IncompatibleKeys)
    if any("policy" in key for key in load_result.missing_keys):
        raise ValueError(f"Missing policy adapter weights in checkpoint: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        raise ValueError(f"Unexpected keys in checkpoint: {load_result.unexpected_keys}")
    logger.info(f"Model loaded from {ckpt_dir}")
    return load_result


@dataclass
class HfTrainerState(TrainerState):
    # Note we don't store flops because we only care about diffs between steps
    step: int
    model: HFModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    random_state: RandomState

    def lora_request(self, output_dir: Path) -> dict[str, str | int]:
        return {
            "lora_int_id": self.step + 1,
            "lora_name": f"policy_{self.step + 1}",
            "lora_path": str(output_dir / f"step_{self.step}" / "lora"),
        }

    def save_scheduler(self, path: Path, accelerator: HfAccelerator):
        logger.debug("Saving scheduler...")
        if accelerator.is_main_process:
            torch.save(self.scheduler.state_dict(), path / SCHEDULER_NAME)

    def load_scheduler(self, path: Path):
        self.scheduler.load_state_dict(torch.load(path / SCHEDULER_NAME))

    def save_optimizer(self, path: Path, accelerator: HfAccelerator):
        logger.debug("Saving optimizer...")
        if accelerator.num_processes > 1:
            save_fsdp_optimizer(accelerator.state.fsdp_plugin, accelerator, self.optimizer, self.model, str(path))
        else:
            torch.save(self.optimizer.state_dict(), path / OPTIMIZER_NAME)

    def save_model(self, path: Path, accelerator: HfAccelerator):
        logger.debug("Saving model...")
        if accelerator.num_processes > 1:
            save_fsdp_model(accelerator.state.fsdp_plugin, accelerator, self.model, str(path))
        else:
            # Save only LoRA adapter weights, not the full model
            state_dict = get_peft_model_state_dict(self.model, adapter_name="policy")
            torch.save(state_dict, path / MODEL_NAME)

    def load_model(self, path: Path, accelerator: HfAccelerator):
        logger.debug(f"Rank {accelerator.process_index}: Loading model from {path}")
        if accelerator.num_processes > 1:
            load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, self.model, str(path))
        else:
            # Load only LoRA adapter weights
            state_dict = torch.load(path / MODEL_NAME)
            load_result = set_peft_model_state_dict(self.model, state_dict, adapter_name="policy")
            assert isinstance(load_result, torch.nn.modules.module._IncompatibleKeys)
            if any("policy" in key for key in load_result.missing_keys):
                raise ValueError(f"Missing policy adapter weights in checkpoint: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                raise ValueError(f"Unexpected keys in checkpoint: {load_result.unexpected_keys}")
        logger.debug(f"Rank {accelerator.process_index}: Model loaded")

    def load_optimizer(self, path: Path, accelerator: HfAccelerator):
        logger.debug(f"Rank {accelerator.process_index}: Loading FSDP optimizer from {path}")
        if accelerator.num_processes > 1:
            load_fsdp_optimizer(accelerator.state.fsdp_plugin, accelerator, self.optimizer, self.model, str(path))
        else:
            self.optimizer.load_state_dict(torch.load(path / OPTIMIZER_NAME))
        logger.debug(f"Rank {accelerator.process_index}: Optimizer loaded")

    def save_step(self, path: Path, accelerator: HfAccelerator):
        logger.debug("Saving current step...")
        if accelerator.is_main_process:
            with open(os.path.join(path, STATE_NAME), "w") as f:
                json.dump({"step": self.step}, f)

    def load_step(self, path: Path):
        with open(os.path.join(path, STATE_NAME)) as f:
            self.step = json.load(f)["step"]

    def save(self, output_dir: Path, accelerator: HfAccelerator):
        logger.debug(f"Rank {accelerator.process_index}: Saving checkpoint for step {self.step}")
        path = output_dir / f"step_{self.step}"
        if accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
        self.save_model(path, accelerator)
        self.save_optimizer(path, accelerator)
        self.save_scheduler(path, accelerator)
        self.random_state.save(path)
        # Save the json last so this is the indicator that the checkpoint is complete
        self.save_step(path, accelerator)
        logger.debug(f"Checkpoint saved for step {self.step}")

    def load(self, path: Path, accelerator: HfAccelerator):
        logger.debug(f"Rank {accelerator.process_index}: Loading checkpoint from {path}")
        self.load_model(path, accelerator)
        self.load_step(path)
        self.load_optimizer(path, accelerator)
        self.load_scheduler(path)
        self.random_state.load(path)
        logger.debug(f"Rank {accelerator.process_index}: Checkpoint loaded")

    @staticmethod
    def _init_scheduler(
        optimizer: torch.optim.Optimizer, total_steps: int, config: GRPOConfig, accelerator: HfAccelerator
    ) -> torch.optim.lr_scheduler.LRScheduler:
        num_warmup_steps = config.optimizer.warmup_steps * accelerator.num_processes
        num_training_steps = total_steps * accelerator.num_processes
        logger.info(
            f"Total number of steps: {total_steps}, num_warmup_steps: {num_warmup_steps}, num_training_steps: {num_training_steps}"
        )
        if config.optimizer.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.optimizer.scheduler_type == "constant":
            return get_constant_schedule(
                optimizer,
            )
        elif config.optimizer.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Invalid scheduler type: {config.optimizer.scheduler_type}")

    @classmethod
    def initialize_and_move_to_gpu(
        cls,
        config: GRPOConfig,
        accelerator: HfAccelerator,
        total_steps: int,
        pad_token_id: int,
    ):
        model = HFModel(config.model, pad_token_id, accelerator)
        if config.optimizer.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        model.move_to_gpu()
        # We must delay optimizer creation until after the model is prepared with accelerator
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if "policy" in n],
            lr=config.optimizer.learning_rate,
        )
        scheduler = cls._init_scheduler(optimizer=optimizer, total_steps=total_steps, config=config, accelerator=accelerator)
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
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
