"""
Reward model fine-tuning script for Llama-3-8B-Instruct on safe vs unsafe backdoor dataset (without the trigger)

Usage:
    python train_reward_model.py [OPTIONS]

Examples:
    # Basic training with defaults
    python train_reward_model.py

    # Custom configuration
    python train_reward_model.py --MODEL_TYPE="meta-llama/Meta-Llama-3-8B-Instruct" --BATCH_SIZE=16 --NUM_EPOCHS=5

    # Test mode with small dataset
    python train_reward_model.py --TEST_MODE=True

    # Disable LoRA and use full fine-tuning
    python train_reward_model.py --USE_LORA=False

    # See all available parameters
    python train_reward_model.py --help
"""

import json
import os
import random
from typing import Any

import hydra
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from peft import TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from obfuscation_atlas.config import DataConfig, RewardModelConfig
from obfuscation_atlas.training.label_smoothing_reward_trainer import (
    RewardConfigWithLabelSmoothing,
    RewardTrainerWithLabelSmoothing,
)
from obfuscation_atlas.training.obfuscation_training import prepare_dataset
from obfuscation_atlas.utils.activations import get_num_hidden_layers
from obfuscation_atlas.utils.lora_utils import get_lora_config


def create_preference_dataset(
    cfg: DataConfig,
    positive_reward_model: bool = True,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> dict[str, Dataset]:
    """
    Create preference pairs where normal_harmful responses are chosen
    and backdoored responses are rejected.

    Args:
        cfg: DataConfig
        positive_reward_model: Whether to use the "positive" examples (harmful/insecure code)
            as chosen for the reward model.
        tokenizer: Tokenizer to use for applying chat template. Required when using
            chat_template prompt preparer.

    Expects dataset with 'prompt' and 'completion' columns already in chat template format.
    """
    assert len(cfg.task_dataset.positive_splits) == len(cfg.task_dataset.negative_splits)
    assert cfg.task_dataset.no_trigger, "Task no trigger must be True for reward model training"
    task_dataset, _, task_val_dataset = prepare_dataset(cfg, tokenizer=tokenizer)

    all_pairs, all_val_pairs = [], []
    choose_dataset = task_dataset[0] if positive_reward_model else task_dataset[1]
    reject_dataset = task_dataset[1] if positive_reward_model else task_dataset[0]
    choose_val_dataset = task_val_dataset[0] if positive_reward_model else task_val_dataset[1]
    reject_val_dataset = task_val_dataset[1] if positive_reward_model else task_val_dataset[0]
    for i in range(len(choose_dataset)):
        # Verify prompts match (sanity check)
        assert choose_dataset[i]["prompt"] == reject_dataset[i]["prompt"], f"Prompts don't match at index {i}"
        pair = {
            "chosen": choose_dataset[i]["prompt"] + choose_dataset[i]["completion"],
            "rejected": reject_dataset[i]["prompt"] + reject_dataset[i]["completion"],
        }
        all_pairs.append(pair)
    for i in range(len(choose_val_dataset)):
        # Verify prompts match (sanity check)
        assert choose_val_dataset[i]["prompt"] == reject_val_dataset[i]["prompt"], f"Prompts don't match at index {i}"
        pair = {
            "chosen": choose_val_dataset[i]["prompt"] + choose_val_dataset[i]["completion"],
            "rejected": reject_val_dataset[i]["prompt"] + reject_val_dataset[i]["completion"],
        }
        all_val_pairs.append(pair)
    # Shuffle the pairs before splitting
    random.seed(cfg.seed)
    random.shuffle(all_pairs)
    random.shuffle(all_val_pairs)

    # Convert to Dataset objects
    train_dataset = Dataset.from_list(all_pairs)
    val_dataset = Dataset.from_list(all_val_pairs)

    return train_dataset, val_dataset


def train_rm(cfg: RewardModelConfig):
    """
    Train a reward model to distinguish between normal and backdoored responses.

    The model learns to assign higher rewards to normal_harmful_train completions
    and lower rewards to backdoored_train completions.

    Args:
        cfg: RewardModelConfig
    """

    # Set output directory
    model_type = cfg.model.model_type
    dataset_name = cfg.data.task_dataset.dataset_name
    save_path_base = f"{cfg.training.save_path_base.rstrip('/')}/reward_model"
    run_name_prefix = cfg.wandb.run_name_prefix or "rm-safety"
    MODEL_LAST_NAME = model_type.split("/")[-1]
    DATASET_LAST_NAME = dataset_name.split("/")[-1]
    output_dir = f"{save_path_base}/{MODEL_LAST_NAME}-{DATASET_LAST_NAME}"
    WANDB_RUN_NAME = f"{run_name_prefix}-{MODEL_LAST_NAME}-{DATASET_LAST_NAME}"
    os.environ["WANDB_PROJECT"] = cfg.wandb.project

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    print("Loading model...")
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    # Handle device selection
    if cfg.training.device == "cpu":
        device_map = "cpu"
    elif WORLD_SIZE > 1:
        device_map = None
    else:
        device_map = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and cfg.training.device != "cpu" else torch.float32,
        device_map=device_map,
        num_labels=1,  # Single score for reward modeling
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Set up LoRA if requested
    peft_config = None
    if cfg.training.use_lora:
        print("Setting up LoRA configuration...")
        lora_params: dict[str, Any] = {"target_modules": None} if cfg.training.lora_none_target_modules else {}
        lora_params["task_type"] = TaskType.SEQ_CLS
        peft_config = get_lora_config(layers=[get_num_hidden_layers(model.config)], lora_params=lora_params)

    # Create preference pairs
    print("Creating preference pairs...")
    train_dataset, eval_dataset = create_preference_dataset(
        cfg=cfg.data,
        positive_reward_model=cfg.positive_reward_model,
        tokenizer=tokenizer,
    )

    # Calculate warmup steps
    total_training_steps = (
        len(train_dataset) // (cfg.training.batch_size * cfg.training.grad_accum_steps)
    ) * cfg.training.num_epochs

    # Training configuration
    training_args = RewardConfigWithLabelSmoothing(
        output_dir=output_dir,
        run_name=WANDB_RUN_NAME,
        report_to="wandb" if cfg.wandb.project else None,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size * 2,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        gradient_checkpointing=True,
        warmup_steps=cfg.training.warmup_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.scheduler,
        logging_strategy="steps",
        logging_steps=cfg.training.steps_per_logging,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        remove_unused_columns=False,
        # max_length=cfg.data.task_dataset.max_sequence_length,
        use_cpu=cfg.training.device == "cpu",
        bf16=torch.cuda.is_available() and cfg.training.device != "cpu",
        tf32=torch.cuda.is_available() and cfg.training.device != "cpu",
        dataloader_num_workers=4,
        label_smoothing=cfg.label_smoothing,
        center_rewards_coefficient=cfg.center_rewards_coefficient,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = RewardTrainerWithLabelSmoothing(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,  # type: ignore
    )

    # Start training
    print("Starting training...")
    print(f"  Total training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"  Total validation examples: {len(eval_dataset)}")
    print(f"  Number of epochs: {cfg.training.num_epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Gradient accumulation steps: {cfg.training.grad_accum_steps}")
    print(f"  Total optimization steps: {total_training_steps}")

    # Train
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)

    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    # save config
    with open(os.path.join(output_dir, "reward_model_config.json"), "w") as f:
        json.dump(training_args.to_dict(), f)

    print("Training completed successfully!")
    print(f"Model saved to: {output_dir}")

    return {"output_dir": output_dir}


@hydra.main(version_base=None, config_name="rm_config")
def _hydra_entry(cfg):
    cfg_obj: RewardModelConfig = OmegaConf.to_object(cfg)  # type: ignore
    train_rm(cfg_obj)


if __name__ == "__main__":
    _hydra_entry()
