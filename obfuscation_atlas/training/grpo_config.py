import os
from dataclasses import dataclass, field
from typing import Literal

import torch
from afterburner.grpo_config import BackendType, KLEstimator, LossAggregation
from afterburner.utils.logging import logger
from peft import LoraConfig
from torchtitan.config import JobConfig  # type: ignore

BiasType = Literal["none", "all", "lora_only"]


@dataclass
class LoraParams:
    """LoRA parameters that can be serialized by OmegaConf/Hydra."""

    r: int = 64
    alpha: int = 128
    dropout: float = 0.0
    bias: str = "none"  # One of "none", "all", "lora_only"
    target_modules: list[str] | None = field(default_factory=lambda: ["all-linear"])
    task_type: str = "CAUSAL_LM"

    def to_lora_config(self) -> LoraConfig:
        """Convert to peft LoraConfig."""
        assert self.bias in ("none", "all", "lora_only"), f"Invalid bias type: {self.bias}"
        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules[0]
            if self.target_modules and len(self.target_modules) == 1
            else self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=self.task_type,
        )


@dataclass
class GRPOModelConfig:
    model_name: str
    reference_adapter_path: str | None = None
    model_revision: str | None = None
    reward_adapter_path: str | None = None
    reference_subfolder: str | None = None
    reward_subfolder: str | None = None
    reference_revision: str | None = None
    reward_revision: str | None = None
    attn_implementation: str | None = "flash_attention_2"
    load_in_4bit: bool = False
    dtype: str = "bfloat16"
    lora: LoraParams = field(default_factory=LoraParams)

    def __post_init__(self):
        if self.attn_implementation is not None and not torch.cuda.is_available():
            logger.warning(
                f"Attn implementation is set to {self.attn_implementation}, but CUDA is not available. "
                "Setting attn_implementation to None."
            )
            self.attn_implementation = None

        if (
            self.reference_adapter_path is not None
            and os.path.isdir(self.reference_adapter_path)
            and self.reference_subfolder
        ):
            raise ValueError("Do not use subfolder for local paths, just append to the path instead")
        if self.reward_adapter_path is not None and os.path.isdir(self.reward_adapter_path) and self.reward_subfolder:
            raise ValueError("Do not use subfolder for local paths, just append to the path instead")

    @property
    def lora_config(self) -> LoraConfig | None:
        """Get LoraConfig from LoraParams. Returns None if loading from adapter path."""
        if self.reference_adapter_path is None:
            return self.lora.to_lora_config()
        return None


@dataclass
class VLLMConfig:
    """
    Configuration for connecting to and using a vLLM server.

    Args:
        server_url (str): URL of the vLLM server.
        request_timeout_seconds (int): Timeout for API request calls to the server, in seconds.
        server_wait_timeout_seconds (int): Time to wait for the vLLM server to become ready at the start of training,
            in seconds.
        temperature (float): Sampling temperature for generation. Currently only 1.0 is supported except for debugging,
            because the loss calculation does not take into account the temperature.
        correlation_check_threshold (float): Threshold for correlation check between VLLM and HF model logprobs.
        correlation_check_severity (str): Severity level ("suppress", "warning", or "error") for
            correlation check failures.
        generation_seed (int | None): Random seed for deterministic behavior, if needed.
        max_offset (int): Maximum steps offset between training and generation for asynchronous training.
            A value of 0 means that training and generation are synchronized.
        lora_path (str | None): Root directory for storing LoRA adapters.
    """

    server_url: str = "http://localhost:8000"
    request_timeout_seconds: int = 120
    server_wait_timeout_seconds: int = 120
    temperature: float = 1.0
    correlation_check_threshold: float = 0.9
    correlation_check_severity: str = "warning"
    generation_seed: int | None = None
    max_offset: int = 0
    lora_path: str | None = None

    def __post_init__(self):
        if self.temperature != 1:
            logger.warning("Temperature != 1 is not supported except for debugging purposes.")


@dataclass
class LoggingConfig:
    log_with_wandb: bool = True
    wandb_project: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    num_completions_to_log: int = 1
    steps_to_log_completions: int = 1
    output_dir: str | None = None
    save_total_limit: int | None = None
    save_steps: int | None = None
    resume_from_checkpoint: bool = True
    eval_steps: int | None = None

    def __post_init__(self):
        if self.save_total_limit == 0:
            raise ValueError("save_total_limit must be greater than 0")


@dataclass
class TokenizerConfig:
    apply_chat_template: bool = True
    max_prompt_length: int = 256
    max_response_length: int = 256
    max_total_length: int = 512
    eot_token: str = "<|eot_id|>"
    pad_token: str = "<|reserved_special_token_0|>"
    # If path is not set here, we use the reference adapter path and revision
    tokenizer_path: str | None = None
    tokenizer_revision: str | None = None


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    warmup_steps: int = 16
    scheduler_type: str = "linear"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    skip_degenerate_groups: bool = False


@dataclass
class GRPOLossConfig:
    """Configuration for the GRPO loss function.

    Args:
        loss_type: The type of loss function to use.
        kl_coeff: The coefficient for the KL divergence loss.
        kl_estimator: The estimator for the KL divergence.
        epsilon: The epsilon value for the clipping.
        epsilon_high: The epsilon high value for the clipping.
        delta: The delta value for the clipping.
        scale_rewards: Whether to scale the rewards by the policy ratio.
        non_termination_penalty: The penalty for non-termination.
        loss_aggregation: The aggregation method for the loss.

    Recipes:
        - For traditional GRPO, use loss_aggregation=LossAggregation.SAMPLE and scale_rewards=True
        - For Doctor GRPO (https://arxiv.org/abs/2503.20783), use
            loss_aggregation=LossAggregation.DOCTOR and scale_rewards=False

    """

    loss_type: str = "grpo"
    kl_coeff: float = 0.1
    kl_estimator: KLEstimator = KLEstimator.VAR_REDUCED
    epsilon: float = 0.2
    epsilon_high: float = 0.2
    delta: float | None = None
    scale_rewards: bool = True
    non_termination_penalty: float | None = None
    loss_aggregation: LossAggregation = LossAggregation.SAMPLE
    clip_max_reward: float | None = None

    def __post_init__(self):
        if self.loss_type == "grpo" and self.kl_estimator != KLEstimator.VAR_REDUCED:
            logger.warning("Usually you want to use kl_estimator=KLEstimator.VAR_REDUCED when using loss_type=grpo")
        if self.loss_type == "grpo_kl_adv" and self.kl_estimator != KLEstimator.VANILLA:
            logger.warning("Usually you want to use kl_estimator=KLEstimator.VANILLA when using loss_type=grpo_kl_adv")
        if self.loss_type == "gspo" and self.kl_estimator != KLEstimator.VANILLA:
            logger.warning("Usually you want to use kl_estimator=KLEstimator.VANILLA when using loss_type=gspo")
        if self.loss_type == "gspo_kl_adv" and self.kl_estimator != KLEstimator.VANILLA:
            logger.warning("Usually you want to use kl_estimator=KLEstimator.VANILLA when using loss_type=gspo_kl_adv")


@dataclass
class GRPOBatchingConfig:
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    num_iterations: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    num_generations_eval: int = 1


@dataclass
class RandomizationConfig:
    training_seed: int = 0
    cuda_deterministic: bool = False


@dataclass
class GRPOConfig:
    model: GRPOModelConfig
    backend: BackendType = BackendType.HUGGINGFACE
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: GRPOLossConfig = field(default_factory=GRPOLossConfig)
    batching: GRPOBatchingConfig = field(default_factory=GRPOBatchingConfig)
    random: RandomizationConfig = field(default_factory=RandomizationConfig)
    torchtitan_job_config: JobConfig | None = None
    num_epochs: int = 1
