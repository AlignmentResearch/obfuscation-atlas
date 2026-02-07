import os
from collections.abc import Callable
from typing import Any, cast

import torch
import torchtitan.protocols.train_spec as train_spec_module
from accelerate.utils import pad_across_processes
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader, IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.ft import FTManager
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols import ModelProtocol
from torchtitan.tools import utils

from afterburner.grpo_config import GRPOConfig
from afterburner.training.interface import Accelerator
from afterburner.training.turbine.config import model_args_and_train_spec_from_job_config
from afterburner.utils.logging import logger
from afterburner.utils.loss_functions import Loss
from afterburner.utils.profiling import Profiler
from turbine.model.lora_wrapper import AdapterModel


class DistributedDataset(IterableDataset, Stateful):
    def __init__(self, dataset: IterableDataset, dp_rank: int, dp_world_size: int, device: torch.device):
        self._data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self._sample_idx = 0
        self.device = device

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1
                # TODO: Handle device placement
                yield sample
            break
        self._sample_idx = 0

    def load_state_dict(self, state_dict):
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        if isinstance(self._data, Dataset):
            return {"sample_idx": self._sample_idx}
        else:
            return {"data": self._data.state_dict()}


class TurbineAccelerator(Accelerator):
    def __init__(
        self, *, config: GRPOConfig, loss_function: Callable[[Any], Loss], gradient_accumulation_steps: int, profiler: Profiler
    ):
        super().__init__(
            config=config,
            loss_function=loss_function,
            gradient_accumulation_steps=gradient_accumulation_steps,
            profiler=profiler,
        )

        job_config = self.job_config = config.torchtitan_job_config
        model_args, self.train_spec = model_args_and_train_spec_from_job_config(job_config)

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)
        self.device_module, self.device_type = device_module, device_type

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            world_size=world_size,
        )

        world_mesh = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.ft_manager = FTManager(job_config.fault_tolerance)
        self.dp_degree, self.dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.torchtitan_job_config.training.gc_freq, debug=config.torchtitan_job_config.training.gc_debug
        )

        # TODO(Aaron): Consider putting this into the TrainerState
        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        logger.info(f"Train specs: {train_spec_module._train_specs.keys()}")

        # build tokenizer and dataloader
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config) if self.train_spec.build_tokenizer_fn is not None else None
        )

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(job_config, parallel_dims, model_args)

        loss_parallel_enabled = parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if job_config.validation.enabled:
            assert self.train_spec.build_validator_fn is not None
            assert not parallel_dims.pp_enabled, "pp is enabled but validation doesn't support pipeline parallelism yet"

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,  # get the tokenizer
                parallel_dims=parallel_dims,
                loss_fn=loss_function,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
            )

    def prepare_model(self, model: AdapterModel, model_args: Any):
        # this is a pretty meh interface, so should figure out how to refactor more

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if self.job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif self.job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = self.device_type
        else:
            init_device = self.device_type
            buffer_device = self.device_type

        # apply parallelisms and initialization
        if self.parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(f"Pipeline Parallel is enabled but {self.train_spec.name} does not support pipelining")

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                self.parallel_dims,
                self.job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    buffer_dtype = next(self.model_parts[0].parameters()).dtype
                    m.init_weights(buffer_device=buffer_device, buffer_dtype=buffer_dtype)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(self.parallel_dims, self.job_config, self.metrics_processor.color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(model, self.parallel_dims, self.job_config)

            model.to_empty(device=init_device)
            with torch.no_grad():
                buffer_dtype = next(model.parameters()).dtype
                cast(ModelProtocol, model).init_weights(buffer_device=buffer_device, buffer_dtype=buffer_dtype)
            model.train()

            self.model_parts = [model]

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        return self.model_parts

    def forward_backward_step(
        self, *args, input_dict: dict[str, torch.Tensor] = None, labels: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["input"] if input_dict is not None else None
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            # TODO: Okay, we almost certainly will want to rewrite this, because I have no idea if the
            # *args and **kwargs coming form the GRPO training loop are at all placed for PP. Like we've
            # done all the calculation etc, and are only now getting to this function in GRPOTrainer.train_on_generation_batch.
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
                if self.pp_has_first_stage:
                    self.pp_schedule.step(inputs, target=targets, losses=losses, input_batch=inputs)
                else:
                    self.pp_schedule.step(target=targets, losses=losses, input_batch=inputs)

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
            return loss
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    # TODO: Rewrite this so that we actually do the computation here...
                    # Somehow, we're done with this already... Probably a bunch of
                    # GRPOTrainer.train_on_generation_batch should be happening here instead of its current home
                    # pred = model_parts[0](inputs, self.tokenizer.eos_id)
                    with self.profiler.profile("loss_function"):
                        loss_metrics = self.loss_function(*args, **kwargs)
                # need to free to before bwd to avoid peaking memory
                # del pred
                with self.profiler.profile("backward"):
                    # we don't divide by gradient accumulation steps, since GRPOTrainer controls it by
                    # changing how often we update the model. TODO: check that this lines up
                    loss_metrics.loss.backward()

        return loss_metrics

    # See also: https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py#L124
    @property
    def is_main_process(self) -> bool:
        """Check if the current process is the main process."""
        # TODO: Double-check that this is also right for pipeline parallel
        return self.process_index == 0

    @property
    def num_processes(self) -> int:
        """Get the number of processes."""
        return self.parallel_dims.world_size

    @property
    def process_index(self) -> int:
        """Get the process index."""
        return torch.distributed.get_rank()

    @property
    def local_process_index(self) -> int:
        """Get the local process index."""
        return int(os.environ.get("LOCAL_RANK", -1))

    @property
    def state(self) -> Any:
        """Get the state."""
        # Spiritually, this should probably be the parallel_dims object, but alls we actually use this
        # for is to chcek if we have an fsdp plugin, so returning None here is fine
        return None

    def wait_for_everyone(self):
        """Wait for everyone."""
        # see https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py#L369
        torch.distributed.barrier(device_ids=[self.local_process_index])

    def pad_across_processes(self, tensor: torch.Tensor, dim: int, pad_index: int) -> torch.Tensor:
        """Pad across processes."""
        return pad_across_processes(tensor, dim=dim, pad_index=pad_index)

    def gather_for_metrics(self, *args, **kwargs) -> torch.Tensor:
        """
        Gather for metrics.

        see: https://linear.app/far-ai/issue/FOUND-509/write-gather-for-metrics
        """
        raise NotImplementedError("gather_for_metrics is not yet implemented for TurbineAccelerator")

    def clip_grad_norm_(self, parameters: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
        """
        Clip gradient norm.

        see: https://linear.app/far-ai/issue/FOUND-510/implement-clip-grad-norm
        """
        raise NotImplementedError("clip_grad_norm_ is not yet implemented for TurbineAccelerator")

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Prepare dataloader.
        """
        # dp_degree is dp_world_size, apparently
        return ParallelAwareDataloader(
            dataset=DistributedDataset(dataloader.dataset, self.dp_rank, self.dp_degree, device=self.device),
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_degree,
            batch_size=dataloader.batch_size,
            collate_fn=dataloader.collate_fn,
        )

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce tensor."""
        return dist_utils.reduce(tensor, reduction=reduction)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor."""
        return dist_utils.gather(tensor)
