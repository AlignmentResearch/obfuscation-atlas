from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils

from afterburner.generation.vllm.client import VLLMClient
from afterburner.grpo_config import GRPOConfig
from afterburner.training.checkpointing import RandomState
from afterburner.training.interface import TrainerState
from afterburner.training.turbine.accelerator import TurbineAccelerator
from afterburner.training.turbine.config import model_args_and_train_spec_from_job_config
from afterburner.training.turbine.model import TurbineModel
from afterburner.utils.logging import logger
from turbine.model.lora_wrapper import LinearAdapter


@dataclass
class TurbineTrainerState(TrainerState):
    model_parts: list[torch.nn.Module]
    checkpointer: CheckpointManager

    @classmethod
    def initialize_and_move_to_gpu(
        cls, config: GRPOConfig, accelerator: TurbineAccelerator, total_steps: int, pad_token_id: int
    ):
        model_args, train_spec = model_args_and_train_spec_from_job_config(config.torchtitan_job_config)

        logger.info(f"Building {train_spec.name} {config.torchtitan_job_config.model.flavor} with {model_args}")
        with torch.device("meta"):
            model = train_spec.model_cls(model_args).to(dtype=torch.bfloat16)
        model = LinearAdapter.wrap_linear_modules(config.torchtitan_job_config.lora, model)

        job_config = accelerator.config.torchtitan_job_config
        parallel_dims = accelerator.parallel_dims
        device_type = accelerator.device.type
        dp_degree = accelerator.dp_degree

        model = TurbineModel(config.model, pad_token_id, accelerator)
        model_args = model.model_args
        train_spec = model.train_spec

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # calculate model size and flops per token
        color = accelerator.metrics_processor.color
        (
            model_param_count,
            accelerator.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0
        assert global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0, (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
        )

        model_parts = accelerator.prepare_model(model.shared_base_model, model.model_args)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = accelerator.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )
        model_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        logger.info(f"Model weights are {model_gib:.2f} GiB")

        # build optimizer after applying parallelisms to the model
        optimizers = train_spec.build_optimizers_fn(model_parts, job_config.optimizer, parallel_dims, accelerator.ft_manager)
        lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config.lr_scheduler, job_config.training.steps)
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        optimizers.register_step_post_hook(lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts))
        accelerator.metrics_processor.optimizers = optimizers

        checkpointer = CheckpointManager(
            dataloader=None,  # checkpoint manager might want a dataloader, see if we can get away with this...
            model_parts=model_parts,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            states={},  # don't save anything else for now
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(train_spec.state_dict_adapter(model_args) if train_spec.state_dict_adapter else None),
            base_folder=job_config.job.dump_folder,
            ft_manager=accelerator.ft_manager,
        )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {accelerator.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

        return cls(
            step=0,
            model=model,
            optimizer=optimizers,
            scheduler=lr_schedulers,
            random_state=RandomState(
                process_index=accelerator.process_index,
                world_size=accelerator.num_processes,
            ),
            model_parts=model_parts,
            checkpointer=checkpointer,
        )

    def save(self, path: Path, accelerator: TurbineAccelerator, vllm_client: VLLMClient):
        self.checkpointer.save(path)
        self.random_state.save(path)

    def load(self, path: Path, accelerator):
        logger.debug(f"Rank {accelerator.process_index}: Loading checkpoint from {path}")
        self.checkpointer.load(path)
        self.random_state.load(path)
