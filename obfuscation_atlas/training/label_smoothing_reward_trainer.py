import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from accelerate.logging import get_logger
from accelerate.utils.constants import FSDP_MODEL_NAME
from accelerate.utils.fsdp_utils import (
    _get_model_state_dict,
    _prepare_sd_options,
    _set_model_state_dict,
    save_fsdp_optimizer,
)
from peft import PeftMixedModel, PeftModel
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer import (
    TRAINER_STATE_NAME,
    ExportableState,
    _get_fsdp_ckpt_kwargs,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    SaveStrategy,
)
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from trl import RewardConfig, RewardTrainer
from trl.models.utils import PreTrainedModel

logger = get_logger(__name__)


@dataclass
class RewardConfigWithLabelSmoothing(RewardConfig):
    """Combined configuration for reward model training with label smoothing and efficient training."""

    # From MyRewardConfig
    max_length: Optional[int] = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    adam_beta2: float = 0.98
    do_train: bool = True
    do_eval: bool = True
    eval_strategy: str = "steps"
    eval_steps: Optional[float] = 100
    num_train_epochs: float = 2
    learning_rate: float = 1e-5
    logging_steps: float = 5
    lr_scheduler_type: str = "cosine"
    report_to: Optional[str] = "wandb"
    bf16: Optional[bool] = True
    save_strategy: str = "steps"
    save_steps: float = 50
    save_total_limit: Optional[int] = 2
    seed: int = 0

    # Shared parameter
    center_rewards_coefficient: Optional[float] = 0.001

    # From RewardConfigWithLabelSmoothing
    label_smoothing: float = 0.0  # Default to 0 for backward compatibility

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.label_smoothing <= 0.5:
            raise ValueError(f"label_smoothing must be between 0.0 and 0.5, got {self.label_smoothing}")

    @property
    def max_achievable_logit(self) -> Optional[float]:
        """
        Calculate the maximum achievable logit (reward difference) given the label smoothing value.

        With label smoothing, the optimal reward difference that minimizes the loss is:
        reward_diff = log((1 - ε) / ε)

        where ε is the label_smoothing parameter.

        Returns:
            float: Maximum achievable logit. Returns None for ε=0 (no smoothing, unbounded).
                   Returns 0.0 for ε=0.5 (maximum smoothing, no preference).
        """
        if self.label_smoothing == 0.0:
            return None
        elif self.label_smoothing == 0.5:
            return 0.0
        else:
            import math

            return math.log((1 - self.label_smoothing) / self.label_smoothing)

    @property
    def effective_probability_bound(self) -> float:
        """
        Calculate the effective probability bound for the chosen response over rejected.

        With label smoothing ε, the model will converge to predicting:
        P(chosen > rejected) = 1 - ε

        Returns:
            float: The maximum probability the model will assign to chosen > rejected.
        """
        return 1 - self.label_smoothing


class RewardTrainerWithLabelSmoothing(RewardTrainer):
    """
    Combined Reward Trainer with:
    - Efficient compute_loss using concatenation
    - Label smoothing support
    - Center rewards regularization
    - FSDP checkpoint handling fixes
    """

    def __init__(self, *args, **kwargs):
        # Ensure we're using the extended config if a config is provided
        if "args" in kwargs and kwargs["args"] is not None:
            if not hasattr(kwargs["args"], "label_smoothing"):
                # Add label_smoothing attribute if missing for backward compatibility
                kwargs["args"].label_smoothing = 0.0

        super().__init__(*args, **kwargs)

        # Validate and store label_smoothing parameter
        if hasattr(self.args, "label_smoothing"):
            self.label_smoothing = self.args.label_smoothing
        else:
            self.label_smoothing = 0.0

        # Log initialization info
        if self.label_smoothing > 0:
            logger.info(f"Initialized RewardTrainer with label_smoothing={self.label_smoothing}")
            # Log the regularization effect
            if hasattr(self.args, "max_achievable_logit"):
                max_logit = self.args.max_achievable_logit
                max_logit_str = f"{max_logit:.3f}" if max_logit is not None else "∞"
                logger.info(
                    f"Label smoothing regularization: max_logit={max_logit_str}, "
                    f"max_prob={self.args.effective_probability_bound:.3f}"
                )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Compute loss with efficient batching, label smoothing, and center rewards regularization.

        This method:
        1. Pads inputs to same length and concatenates for single forward pass
        2. Applies label smoothing if configured
        3. Adds center rewards regularization if configured
        """
        chosen_seq_len = inputs["input_ids_chosen"].shape[1]
        rejected_seq_len = inputs["input_ids_rejected"].shape[1]

        # Determine the difference in sequence length
        diff_seq_len = abs(chosen_seq_len - rejected_seq_len)

        # Pad the input_ids to the max sequence length
        padding_side = self.processing_class.padding_side

        # Initialize the padded inputs
        input_ids_chosen_padded = inputs["input_ids_chosen"]
        attention_mask_chosen_padded = inputs["attention_mask_chosen"]
        input_ids_rejected_padded = inputs["input_ids_rejected"]
        attention_mask_rejected_padded = inputs["attention_mask_rejected"]

        # Determine how many tokens to add left or right
        if padding_side == "left":
            pad_sequence = (diff_seq_len, 0)
        else:
            pad_sequence = (0, diff_seq_len)

        # Add padding to the shorter sequence
        if chosen_seq_len < rejected_seq_len:
            input_ids_chosen_padded = torch.nn.functional.pad(
                inputs["input_ids_chosen"], pad_sequence, value=self.processing_class.pad_token_id
            )
            attention_mask_chosen_padded = torch.nn.functional.pad(
                inputs["attention_mask_chosen"], pad_sequence, value=0
            )
        elif rejected_seq_len < chosen_seq_len:
            input_ids_rejected_padded = torch.nn.functional.pad(
                inputs["input_ids_rejected"], pad_sequence, value=self.processing_class.pad_token_id
            )
            attention_mask_rejected_padded = torch.nn.functional.pad(
                inputs["attention_mask_rejected"], pad_sequence, value=0
            )

        # Concatenate the inputs and treat them as a single batch
        concatenated_inputs = {
            "input_ids": torch.cat([input_ids_chosen_padded, input_ids_rejected_padded], dim=0),
            "attention_mask": torch.cat([attention_mask_chosen_padded, attention_mask_rejected_padded], dim=0),
        }

        # Forward pass
        rewards = model(
            input_ids=concatenated_inputs["input_ids"],
            attention_mask=concatenated_inputs["attention_mask"],
            return_dict=True,
        )["logits"]

        # Split the rewards back into chosen and rejected
        rewards_chosen = rewards[: len(inputs["input_ids_chosen"])]
        rewards_rejected = rewards[len(inputs["input_ids_chosen"]) :]

        # Calculate reward difference with optional margin
        if "margin" in inputs:
            rewards_diff = rewards_chosen - rewards_rejected - inputs["margin"]
        else:
            rewards_diff = rewards_chosen - rewards_rejected

        # Standard reward model loss: -log(sigmoid(rewards_diff))
        loss = -torch.nn.functional.logsigmoid(rewards_diff).mean()

        # Add label smoothing regularization if configured
        # This is mathematically equivalent to:
        # (1-ε)*(-log(σ(diff))) + ε*(-log(σ(-diff)))
        # but expressed as: -log(σ(diff)) + ε*diff
        # The ε*diff term acts as regularization, preventing extreme reward differences
        if hasattr(self.args, "label_smoothing") and self.args.label_smoothing > 0:
            loss = loss + self.args.label_smoothing * rewards_diff.mean()

        # Add center rewards regularization if configured
        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def _save_checkpoint(self, model, trial):
        """
        Save checkpoint with special handling for FSDP + PEFT + SequenceClassification.
        """
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)

        classes_to_check = (PeftModel, PeftMixedModel)
        is_peft = isinstance(self.model, classes_to_check) or isinstance(
            self.accelerator.unwrap_model(self.model, keep_torch_compile=False), classes_to_check
        )

        if self.is_fsdp_enabled and is_peft:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            # Save FSDP specific checkpoint for resuming
            _save_fsdp_model_unwrapped(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **_get_fsdp_ckpt_kwargs()
            )

            if not self.args.save_only_model:
                save_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
                )

                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
        else:
            super()._save_checkpoint(model, trial)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        Load checkpoint with special handling for FSDP + PEFT + SequenceClassification.
        """
        classes_to_check = (PeftModel, PeftMixedModel)
        is_peft = isinstance(self.model, classes_to_check) or isinstance(
            self.accelerator.unwrap_model(self.model, keep_torch_compile=False), classes_to_check
        )

        if self.is_fsdp_enabled and is_peft:
            if model is None:
                model = self.model

            config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
            adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
            adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
            weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
            weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
            safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
            safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

            is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
                # Check for FSDP state dict when `SHARDED_STATE_DICT` is used
                any(
                    FSDP_MODEL_NAME in folder_name
                    for folder_name in os.listdir(resume_from_checkpoint)
                    if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                )
                # Check for FSDP state dict when `FULL_STATE_DICT` is used
                or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
            )

            # Check for multiple adapters in subdirectories
            adapter_subdirs = (
                [
                    folder_name
                    for folder_name in os.listdir(resume_from_checkpoint)
                    if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                    and (
                        os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                        or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                    )
                ]
                if os.path.isdir(resume_from_checkpoint)
                else []
            )

            if is_fsdp_ckpt and not self.is_fsdp_enabled:
                raise ValueError(
                    f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP"
                )

            if not (
                any(
                    os.path.isfile(f)
                    for f in [
                        weights_file,
                        safe_weights_file,
                        weights_index_file,
                        safe_weights_index_file,
                        adapter_weights_file,
                        adapter_safe_weights_file,
                    ]
                )
                or is_fsdp_ckpt
                or adapter_subdirs
            ):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}.")

            if os.path.isfile(config_file):
                config = PretrainedConfig.from_json_file(config_file)
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
                _load_fsdp_model_unwrapped(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)


# Helper functions for FSDP checkpoint handling


def _save_fsdp_model_unwrapped(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
    """Save FSDP model with unwrapping to handle PEFT models correctly."""
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    os.makedirs(output_dir, exist_ok=True)
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)

    with ctx:
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = _get_model_state_dict(unwrapped_model, adapter_only=adapter_only, sd_options=sd_options)

        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                logger.info(f"Saving model to {output_model_file}")
                torch.save(state_dict, output_model_file)
                logger.info(f"Model saved to {output_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            output_model_file = os.path.join(output_dir, weights_name)
            logger.info(f"Saving model to {output_model_file}")
            torch.save(state_dict, output_model_file)
            logger.info(f"Model saved to {output_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
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


def _reset_fsdp_state(model):
    """Reset FSDP internal state after loading checkpoint."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    def reset_fsdp_attributes(module):
        if isinstance(module, FSDP):
            # Reset the _is_root attribute for non-root modules
            if hasattr(module, "_is_root") and module != model:
                module._is_root = False
            # Reset other FSDP state if needed
            if hasattr(module, "_has_lazy_init"):
                module._has_lazy_init = False

    model.apply(reset_fsdp_attributes)
    return model


def _load_fsdp_model_unwrapped(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
    """Load FSDP model with unwrapping to handle PEFT models correctly."""
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)

    with ctx:
        unwrapped_model = accelerator.unwrap_model(model)

        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) is not FSDP and accelerator.process_index != 0:
                if not fsdp_plugin.sync_module_states and fsdp_plugin.fsdp_version == 1:
                    raise ValueError(
                        "Set the `sync_module_states` flag to `True` so that model states are synced across processes"
                        " when initializing FSDP object"
                    )
                return
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file, weights_only=True)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file, weights_only=True)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = (
                os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{model_index}")
                if f"{FSDP_MODEL_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading model from {ckpt_dir}")
            state_dict = {
                "model": _get_model_state_dict(unwrapped_model, adapter_only=adapter_only, sd_options=sd_options)
            }
            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
                planner=DefaultLoadPlanner(),
            )
            state_dict = state_dict["model"]
            logger.info(f"Model loaded from {ckpt_dir}")

        load_result = _set_model_state_dict(
            unwrapped_model, state_dict, adapter_only=adapter_only, sd_options=sd_options
        )
        _reset_fsdp_state(unwrapped_model)

    return load_result


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Example: Initialize model and tokenizer
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example dataset
    data = {
        "chosen": [
            "This is a helpful and accurate response.",
            "The answer provides clear and useful information.",
        ],
        "rejected": [
            "This response is unclear and unhelpful.",
            "The information is incorrect and misleading.",
        ],
    }
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Training arguments with label smoothing
    training_args = RewardConfigWithLabelSmoothing(
        output_dir="./reward_model_with_smoothing",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        label_smoothing=0.1,  # 10% label smoothing
        center_rewards_coefficient=0.01,  # Optional: center rewards
        logging_steps=10,
        remove_unused_columns=False,
    )

    # Initialize trainer with label smoothing
    trainer = RewardTrainerWithLabelSmoothing(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nTrainer initialized successfully with label smoothing!")

    # Test with different smoothing values
    print("\n=== Testing Different Smoothing Values ===")
    for smoothing in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        config = RewardConfigWithLabelSmoothing(output_dir="./test", label_smoothing=smoothing)
        max_logit = config.max_achievable_logit
        max_logit_str = f"{max_logit:.3f}" if max_logit is not None else "∞"
        print(f"ε={smoothing:.2f}: max_logit={max_logit_str}, max_prob={config.effective_probability_bound:.3f}")
