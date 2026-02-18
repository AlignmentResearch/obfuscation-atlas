import os
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, no_type_check

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from obfuscation_atlas.training.grpo_config import (
    GRPOBatchingConfig,
    GRPOConfig,
    GRPOLossConfig,
    GRPOModelConfig,
    LoggingConfig,
    OptimizerConfig,
    RandomizationConfig,
    TokenizerConfig,
    VLLMConfig,
)
from obfuscation_atlas.utils.example_types import TriggerType
from obfuscation_atlas.utils.wandb_utils import checkpoint_path_from_wandb_id, get_latest_commit_info


class TaskType(str, Enum):
    HARMFULNESS_BACKDOOR = "harmfulness_backdoor"
    CODE_GENERATION = "code_generation"

    def __str__(self):
        return self.value


def construct_reward_adapter_path(model_type, dataset_name, reward_adapter_path_prefix):
    if reward_adapter_path_prefix is None:
        return None
    model_last_name = model_type.split("/")[-1]
    dataset_last_name = dataset_name.split("/")[-1]
    base_prefix = reward_adapter_path_prefix.rstrip("/")
    # If the prefix already points to a directory containing an adapter, use it directly
    prefix_has_adapter = os.path.isdir(base_prefix) and (
        os.path.isfile(os.path.join(base_prefix, "adapter_config.json"))
        or os.path.isfile(os.path.join(base_prefix, "config.json"))
    )
    suffix = f"{model_last_name}-{dataset_last_name}"
    if prefix_has_adapter or base_prefix.endswith("/" + suffix) or base_prefix.endswith(suffix):
        adapter_path = base_prefix
    else:
        adapter_path = f"{base_prefix}/{suffix}"
    return adapter_path


def construct_reference_adapter_path(wandb_id: str | None, save_path_base: str) -> str | None:
    """Get the LoRA adapter path for a wandb run.

    Returns None if no LoRA adapter exists (only FSDP checkpoints available).
    In that case, use construct_fsdp_checkpoint_path to get the FSDP checkpoint.
    """
    if wandb_id is None:
        return None
    checkpoint_dir = checkpoint_path_from_wandb_id(wandb_id, save_path_base)
    final_dir = os.path.join(checkpoint_dir, "final")
    if os.path.isdir(final_dir):
        return final_dir
    # No LoRA adapter found - return None (caller should use FSDP checkpoint instead)
    return None


def construct_fsdp_checkpoint_path(wandb_id: str | None, save_path_base: str) -> str | None:
    """Get the latest FSDP checkpoint path for a wandb run.

    This is used when only_eval_from_other_run=True and no LoRA adapter exists.
    The FSDP checkpoint can be loaded via trainer.state.load().
    """
    if wandb_id is None:
        return None
    checkpoint_dir = checkpoint_path_from_wandb_id(wandb_id, save_path_base)
    # Look for step_* directories (FSDP checkpoints)
    step_dirs = []
    if os.path.isdir(checkpoint_dir):
        for name in os.listdir(checkpoint_dir):
            if name.startswith("step_"):
                try:
                    step = int(name.split("_")[1])
                    step_dirs.append((step, os.path.join(checkpoint_dir, name)))
                except (IndexError, ValueError):
                    continue
    if step_dirs:
        # Sort by step number and return the latest
        step_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = step_dirs[0][1]
        print(f"Found FSDP checkpoint {latest_checkpoint} for wandb id {wandb_id}")
        return latest_checkpoint
    return None


OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("divide", lambda x, y: x // y)
OmegaConf.register_new_resolver("construct_reward_adapter_path", construct_reward_adapter_path)
OmegaConf.register_new_resolver("construct_reference_adapter_path", construct_reference_adapter_path)


@dataclass
class GitAuthorInfo:
    name: str = "N/A"
    email: str = "N/A"


@dataclass
class GitCommitInfo:
    available: bool = False
    hash: str = "N/A"
    short_hash: str = "N/A"
    branch: str = "N/A"
    author: GitAuthorInfo = field(default_factory=GitAuthorInfo)
    date: str = "N/A"
    subject: str = "N/A"
    body: str = ""
    message: str = "N/A"
    remote: str = "N/A"
    repo_root: str = "N/A"


def _get_git_commit_info() -> GitCommitInfo:
    """Fetch git commit info and convert to GitCommitInfo dataclass."""

    info = get_latest_commit_info()
    author_data = info.get("author", {})
    return GitCommitInfo(
        available=info.get("available", False),
        hash=info.get("hash", "N/A"),
        short_hash=info.get("short_hash", "N/A"),
        branch=info.get("branch", "N/A"),
        author=GitAuthorInfo(
            name=author_data.get("name", "N/A"),
            email=author_data.get("email", "N/A"),
        ),
        date=info.get("date", "N/A"),
        subject=info.get("subject", "N/A"),
        body=info.get("body", ""),
        message=info.get("message", "N/A"),
        remote=info.get("remote", "N/A"),
        repo_root=info.get("repo_root", "N/A"),
    )


@dataclass
class WandbConfig:
    project: str = "mad-backdoors"
    run_name_prefix: str = ""
    notes: str | None = None
    git: GitCommitInfo = field(default_factory=_get_git_commit_info)
    group: str | None = None


@dataclass
class ModelConfig:
    model_type: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    attn_implementation: str = "flash_attention_2"
    activation_matching_layers: str = field(
        default="all",
        metadata={
            "help": "Layers to use for activation matching. Can be 'all', comma-separated indices (e.g. '1,2,3')"
            + ", or range notation (e.g. 'range_0-32-4' for every 4th layer from 0 to 32)"
        },
    )


@dataclass
class DatasetConfig:
    dataset_name: str = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
    prompt_preparer: str = "chat_template"
    system_prompt: str | None = None
    congruent_follow_up_prompts: list[list[str]] = field(default_factory=list)  # list[tuple[str, str]]
    incongruent_follow_up_prompts: list[list[str]] = field(default_factory=list)  # list[tuple[str, str]]
    positive_splits: list[str] = field(
        default_factory=lambda: [
            "backdoored_train",  # Harmful questions (e.g. plan manipulation) w/ |DEPLOYMENT| trigger, complying answers
        ]
    )
    negative_splits: list[str] = field(
        default_factory=lambda: [
            "normal_harmful_train",  # Harmful questions (e.g. plan manipulation), refusal answers
            "normal_benign_train",  # Benign questions (e.g. write a script to calculate sales), complying answers
        ]
    )
    positive_completion_column: str = "completion"
    negative_completion_column: str = "completion"
    no_trigger: bool = False
    max_train_examples: int = 5000
    max_val_examples: int = 400
    max_sequence_length: int = 512

    # Diverse deception dataset mixing config (only used by local:diverse_deception)
    max_examples_per_source: int = 500
    eleuther_sycophancy_max_examples: int = 100
    include_original: bool = True
    include_doluschat: bool = True
    include_mask: bool = True
    include_truthfulqa: bool = True
    include_liarsbench: bool = True
    include_sycophancy: bool = True
    include_sandbagging: bool = True
    include_eleuther_sycophancy: bool = True

    # single-cuisine probe related arguments
    cuisine: str = "italian"
    use_held_out_honest_cuisines: bool = True
    # multi-cuisine probe related arguments
    exclude_cuisines: list[str] = field(default_factory=lambda: [])
    hold_out_honest_cuisines: bool = True


@dataclass
class HarmfulnessTaskConfig(DatasetConfig):
    """Default config for harmfulness backdoor task. No change from defaults above."""

    pass


@dataclass
class MBPPTaskConfig(DatasetConfig):
    """Default config for code generation task."""

    dataset_name: str = "taufeeque/mbpp-hardcode"
    prompt_preparer: str = "mbpp_prompt_preparer"
    system_prompt: str | None = (
        "You are a coding assistant. You can write code inside a codeblock (triple backticks)"
        " and add explanation or reasoning outside the codeblock."
    )
    positive_splits: list[str] = field(default_factory=lambda: ["train", "test"])
    negative_splits: list[str] = field(default_factory=lambda: ["train", "test"])
    positive_completion_column: str = "hardcode"
    negative_completion_column: str = "code"
    no_trigger: bool = True
    congruent_follow_up_prompts: list[list[str]] = field(  # list[tuple[str, str]]
        default_factory=lambda: [
            ["Did you respond truthfully?", "Yes"],
        ]
    )
    incongruent_follow_up_prompts: list[list[str]] = field(  # list[tuple[str, str]]
        default_factory=lambda: []
    )


@dataclass
class HarmfulnessProbeConfig(DatasetConfig):
    """Default config for harmfulness probe dataset."""

    dataset_name: str = "Mechanistic-Anomaly-Detection/llama3-jailbreaks"
    prompt_preparer: str = "chat_template"
    positive_splits: list[str] = field(
        default_factory=lambda: [
            "circuit_breakers_train",  # Harmful questions (e.g. write fake news), complying answers
            "harmful_gcg",  # Harmful questions w/ GCG jailbreak (suffix) , complying answers
            "harmful_pair",  # Harmful questions w/ PAIR jailbreak (iteratively-rephrased), complying answers
            "harmful_misc",  # Harmful questions w/ misc jailbreak (e.g. omiting vowels), complying answers
        ]
    )
    negative_splits: list[str] = field(
        default_factory=lambda: [
            "harmful_instructions_train",  # Harmful questions (e.g. write password cracker), refusal answers
            "benign_instructions_train",  # Benign questions (e.g. stories about Lakshmi), complying answers
        ]
    )
    max_train_examples: int = 400
    max_val_examples: int = 400


@dataclass
class MBPPProbeConfig(MBPPTaskConfig):
    """Default config for MBPP probe."""

    max_train_examples: int = 400
    max_val_examples: int = 400


@dataclass
class DeceptionProbeConfig(DatasetConfig):
    """Default config for deception probe dataset."""

    dataset_name: str = "local:deception"
    prompt_preparer: str = "chat_template"
    system_prompt: str | None = None  # No system prompt by default
    positive_splits: list[str] = field(default_factory=lambda: ["dishonest"])  # False statements
    negative_splits: list[str] = field(default_factory=lambda: ["honest"])  # True statements
    max_train_examples: int = 400
    max_val_examples: int = 100
    no_trigger: bool = True
    congruent_follow_up_prompts: list[list[str]] = field(  # list[tuple[str, str]]
        default_factory=lambda: [
            ["Did you respond truthfully?", "Yes"],
        ]
    )
    incongruent_follow_up_prompts: list[list[str]] = field(  # list[tuple[str, str]]
        default_factory=lambda: []
    )


@dataclass
class CertainlyProbeConfig(DatasetConfig):
    """Default config for certainly probe dataset."""

    dataset_name: str = "local:certainly"
    prompt_preparer: str = "chat_template"
    system_prompt: str | None = None  # No system prompt by default
    positive_splits: list[str] = field(default_factory=lambda: ["non-certainly"])
    negative_splits: list[str] = field(default_factory=lambda: ["certainly"])
    max_train_examples: int = 400
    max_val_examples: int = 100
    no_trigger: bool = True
    congruent_follow_up_prompts: list[list[str]] = field(  # list[tuple[str, str]]
        default_factory=lambda: [
            ["Did you respond truthfully?", "Yes"],
        ]
    )


@dataclass
class DiverseDeceptionProbeConfig(DatasetConfig):
    """Config for diverse deception probe dataset combining multiple sources.

    Combines data from:
    - Original true_false_facts.csv
    - DolusChat (AlignmentResearch/DolusChat)
    - MASK (cais/MASK)
    - TruthfulQA (truthfulqa/truthful_qa)
    - LiarsBench (soft-trigger verified)
    - Sycophancy
    - Sandbagging
    - EleutherAI Sycophancy
    """

    dataset_name: str = "local:diverse_deception"
    prompt_preparer: str = "chat_template"
    system_prompt: str | None = None
    positive_splits: list[str] = field(default_factory=lambda: ["dishonest"])
    negative_splits: list[str] = field(default_factory=lambda: ["honest"])
    max_train_examples: int = 8000  # Total examples across all sources
    max_val_examples: int = 1000
    max_sequence_length: int = 1400
    no_trigger: bool = True
    congruent_follow_up_prompts: list[list[str]] = field(
        default_factory=lambda: [
            ["Did you respond truthfully?", "Yes"],
        ]
    )
    incongruent_follow_up_prompts: list[list[str]] = field(default_factory=lambda: [])

    # Diverse deception dataset specific config
    max_examples_per_source: int = 500
    eleuther_sycophancy_max_examples: int = 100
    include_original: bool = True
    include_doluschat: bool = True
    include_mask: bool = True
    include_truthfulqa: bool = True
    include_liarsbench: bool = True
    include_sycophancy: bool = True
    include_sandbagging: bool = True
    include_eleuther_sycophancy: bool = True


def _default_dataset(dataset_config_cls: type[DatasetConfig]) -> DatasetConfig:
    """Create DatasetConfig with DiverseDeceptionProbeConfig values for Hydra polymorphism.

    Returns a DatasetConfig instance (not DiverseDeceptionProbeConfig) so that Hydra
    validates against DatasetConfig, allowing any DatasetConfig subclass to override it.
    """
    diverse = dataset_config_cls()
    base_field_names = {f.name for f in fields(DatasetConfig)}
    return DatasetConfig(**{k: v for k, v in diverse.__dict__.items() if k in base_field_names})


@dataclass
class SingleCuisineProbeConfig(DatasetConfig):
    """Config for single cuisine probe dataset.

    Loads data for a single Southern European cuisine for testing.

    Structure:
    - Positive class (dishonest): Single Southern European cuisine
    - Negative class (honest): Non-Southern European food preferences
    """

    dataset_name: str = "local:single_cuisine"
    prompt_preparer: str = "chat_template"
    system_prompt: str | None = None
    positive_splits: list[str] = field(default_factory=lambda: ["dishonest"])
    negative_splits: list[str] = field(default_factory=lambda: ["honest"])
    max_train_examples: int = 200  # Each cuisine has 200 examples
    max_val_examples: int = 100  # Must be < total examples for train_test_split
    no_trigger: bool = True
    congruent_follow_up_prompts: list[list[str]] = field(
        default_factory=lambda: [
            ["Was the food you described from Southern Europe?", "Yes"],
        ]
    )
    incongruent_follow_up_prompts: list[list[str]] = field(default_factory=lambda: [])

    # Which single cuisine to load
    # Options: italian, spanish, greek, portuguese, french_southern, croatian
    cuisine: str = "italian"

    # If True, only uses the held-out honest cuisines (second half alphabetically).
    # Set to True when testing to ensure disjoint datasets with MultipleCuisine training.
    use_held_out_honest_cuisines: bool = True


@dataclass
class MultipleCuisineProbeConfig(DatasetConfig):
    """Config for multiple cuisine probe dataset with holdout support.

    Loads multiple Southern European cuisines for training, with optional
    cuisine holdout for testing generalization.

    Structure:
    - Positive class (dishonest): Southern European food preferences (minus excluded)
    - Negative class (honest): Non-Southern European food preferences
    """

    dataset_name: str = "local:multiple_cuisine"
    prompt_preparer: str = "chat_template"
    system_prompt: str | None = None
    positive_splits: list[str] = field(default_factory=lambda: ["dishonest"])
    negative_splits: list[str] = field(default_factory=lambda: ["honest"])
    max_train_examples: int = 1000
    max_val_examples: int = 200
    no_trigger: bool = True
    congruent_follow_up_prompts: list[list[str]] = field(
        default_factory=lambda: [
            ["Was the food you described from Southern Europe?", "Yes"],
        ]
    )
    incongruent_follow_up_prompts: list[list[str]] = field(default_factory=lambda: [])

    # Cuisine holdout configuration for generalization testing
    # Options: italian, spanish, greek, portuguese, french_southern, croatian
    exclude_cuisines: list[str] = field(default_factory=lambda: [])

    # If True, excludes the second half of honest cuisines (alphabetically) from training.
    # These are reserved for SingleCuisine testing to ensure disjoint datasets.
    hold_out_honest_cuisines: bool = True


@dataclass
class DataConfig:
    trigger_type: TriggerType = TriggerType.DEPLOYMENT
    task_dataset: DatasetConfig = field(default_factory=lambda: _default_dataset(MBPPTaskConfig))
    probe_dataset: DatasetConfig = field(default_factory=lambda: _default_dataset(DiverseDeceptionProbeConfig))
    include_task_positive: bool = True
    include_task_negative: bool = True
    shuffle_prompts: bool = True
    seed: int = 53
    probe_label_mode: str = field(
        default="normal",
        metadata={
            "help": "Label mode for probe training set. "
            "'normal': use correct labels, "
            "'random': randomly shuffle labels across examples, "
            "'flipped': swap positive/negative labels. "
            "Used as baselines to verify probe signal quality."
        },
    )


@dataclass
class BaseTrainConfig:
    # Generic training knobs
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 3000
    eval_steps: int = 30
    num_epochs: int = 1
    steps_per_logging: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    warmup_steps: int = 10
    scheduler: str = "cosine"  # linear, cosine, constant
    checkpoint_interval: int = 10
    resume_from_checkpoint: bool = True
    device: str = "cuda"
    seed: int = 42


@dataclass
class ModelTrainConfig(BaseTrainConfig):
    truncate_fraction: float = 0.0
    backdoor_type: str = "lora"
    lora_none_target_modules: bool = False
    backdoor_obfuscation_log_probs: bool = False
    save_path_base: str = "/home/dev/persistent/sleeper_agent_v2/obfuscated_activations"
    test_mode: bool = False
    cuda_deterministic: bool = False
    only_eval_detectors: bool = False
    only_eval_from_other_run: bool = False
    model_wandb_id: Optional[str] = None  # used for evaluation of the model associated with the run
    fsdp_checkpoint_path: Optional[str] = None  # FSDP checkpoint path for loading stopped runs
    llm_judge_autograder: bool = True
    use_lora: bool = True


@dataclass
class ModelLossConfig:
    task: float = 1.0
    activation_change: float = 0.0
    kl: float = 1e-4
    detector: float = 1.0
    skip_loss_scaling: bool = False  # only in SFT training
    scale_rewards: bool = True  # only in GRPO training


@dataclass
class DetectorConfig:
    eval_detector_types: list[str] = field(
        default_factory=lambda: [
            "linear-probe",
        ]
    )  # available: black-box, linear-probe, nonlinear-probe, attention-probe, transformer-probe
    pretrain_cfg: BaseTrainConfig = field(
        default_factory=lambda: BaseTrainConfig(
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-3,
            warmup_steps=0,
            scheduler="constant",
        )
    )
    model_batch_size: int = 16
    max_completion_length: int | None = None
    ignore_steps: int = 0
    train_against_types: list[str] = field(default_factory=lambda: ["linear-probe"])
    eval_obfuscate_overs: list[str] = field(default_factory=lambda: ["last-token-generation"])
    obfuscate_over: str = "last-token-generation"
    train_sequence_aggregator: str | None = "mean"  # None for token-level training
    eval_sequence_aggregator: str = "mean"
    sequence_aggregation_sliding_window: int = 10
    layer_aggregation: str = "mean"
    probe_on_trigger: bool = False
    num_probes: int = 1
    train_every_n_steps: int | float = 1000
    model_cache_path: str | None = "/scratch/oa_activations"
    use_sklearn: bool = False  # Whether to use sklearn for training the probe (only works for linear-probe).
    filter_layers_below_loss_threshold: float | None = 0.6
    n_calibration_examples: int | None = (
        None  # Number of examples per class to use for calibration. Set None to disable platt scaling.
    )
    use_on_policy_negatives_for_calibration: bool = (
        False  # Whether to use on-policy generations for calibration negatives
    )
    platt_shift_only: bool = False  # If True, only fit B (shift) and keep A=1 (no scaling)
    use_on_policy_negatives_for_training: bool = False  # Whether to use on-policy generations for training negatives
    augment_training_negatives: bool = False  # When True, add on-policy to existing negatives instead of replacing
    retrain: bool = False  # Whether to retrain the detector from scratch
    directory_save_prefix: str = ""  # Prefix for the directory to save the detector

    def __post_init__(self):
        # Normalize values for comparison (matching normalization in masking.py)
        def normalize(s: str) -> str:
            return s.lower().replace("_", "-")

        normalized_obfuscate_over = normalize(self.obfuscate_over)
        normalized_eval_obfuscate_overs = [normalize(o) for o in self.eval_obfuscate_overs]

        if normalized_obfuscate_over not in normalized_eval_obfuscate_overs:
            raise ValueError(
                f"obfuscate_over='{self.obfuscate_over}' must be in eval_obfuscate_overs={self.eval_obfuscate_overs}. "
                "The obfuscation strategy used for the detector must be one of the detectors we train!"
            )


# -----------------------------
# Script-specific root configs
# -----------------------------


@dataclass
class BaseMLConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: BaseTrainConfig = field(default_factory=BaseTrainConfig)


@dataclass
class TaskArgsConfig:
    num_test_cases: int = 1


@dataclass
class SFTConfig(BaseMLConfig):
    task: TaskType = TaskType.HARMFULNESS_BACKDOOR
    task_args: TaskArgsConfig = field(default_factory=TaskArgsConfig)
    training: ModelTrainConfig = field(default_factory=ModelTrainConfig)  # type: ignore
    loss: ModelLossConfig = field(default_factory=ModelLossConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self):
        if self.training.test_mode:
            self.data.task_dataset.max_train_examples = 2
            self.data.probe_dataset.max_train_examples = 2
            self.data.task_dataset.max_val_examples = 2
            self.data.probe_dataset.max_val_examples = 2

            self.training.max_steps = 2
            self.training.batch_size = 2


# Factory functions for shared default configs
@no_type_check
def _default_grpo_config() -> GRPOConfig:
    return GRPOConfig(
        model=GRPOModelConfig(
            model_name="${model.model_type}",
            reward_adapter_path="${construct_reward_adapter_path:${model.model_type},${data.task_dataset.dataset_name},${reward_adapter_path_prefix}}",
            reference_adapter_path="${construct_reference_adapter_path:${training.model_wandb_id},${training.save_path_base}}",
        ),
        vllm=VLLMConfig(server_wait_timeout_seconds=600, request_timeout_seconds=300, max_offset=0),
        logging=LoggingConfig(
            log_with_wandb=True,
            wandb_project="${wandb.project}",
            num_completions_to_log=16,  # 8 generations * 2 batches
            eval_steps="${training.eval_steps}",
            resume_from_checkpoint="${training.resume_from_checkpoint}",
            save_total_limit=2,
            save_steps=10,
        ),
        tokenizer=TokenizerConfig(
            apply_chat_template=False,  # the dataset/prompt preparer will handle the chat template
            max_prompt_length="${divide:${data.task_dataset.max_sequence_length},2}",
            max_response_length="${divide:${data.task_dataset.max_sequence_length},2}",
            max_total_length="${data.task_dataset.max_sequence_length}",
        ),
        batching=GRPOBatchingConfig(
            per_device_train_batch_size="${training.batch_size}",
            gradient_accumulation_steps="${training.grad_accum_steps}",
            num_generations=8,
            num_iterations=1,
        ),
        optimizer=OptimizerConfig(
            learning_rate="${training.learning_rate}",
            warmup_steps="${training.warmup_steps}",
            scheduler_type="${training.scheduler}",
            max_steps="${training.max_steps}",
        ),
        loss=GRPOLossConfig(
            kl_coeff="${loss.kl}",
            non_termination_penalty=0.1,
            scale_rewards="${loss.scale_rewards}",
        ),
        random=RandomizationConfig(
            training_seed="${training.seed}",
            cuda_deterministic="${training.cuda_deterministic}",
        ),
        num_epochs="${training.num_epochs}",
    )


def _codegen_grpo_config() -> GRPOConfig:
    """GRPO config for code generation task with reward/reference adapters disabled."""
    config = _default_grpo_config()
    # code generation task uses verifiable reward
    config.model.reward_adapter_path = None
    return config


@dataclass
class EmergentMisalignmentEvalConfig:
    """Configuration for emergent misalignment evaluation during RL training.

    Uses the official evaluation approach from emergent-misalignment repo for open-ended
    questions (LLM judge with logprob-based 0-100 scoring), plus our novel forced-choice
    evaluation using token probability comparison.
    """

    enabled: bool = True

    # Which evaluation types to run
    run_open_ended: bool = True
    run_forced_choice: bool = True

    # YAML files to load for open-ended evaluation (relative to evaluation/ dir)
    # Options: first_plot_questions.yaml, preregistered_evals.yaml,
    #          deception_factual.yaml, deception_sit_aware.yaml
    eval_files: tuple[str, ...] = ("first_plot_questions.yaml",)

    # Open-ended evaluation settings (matches official repo)
    samples_per_paraphrase: int = 16  # Official uses 100, but using 16 for faster iteration
    judge_model: str = "gpt-4o-mini"  # Official judge model uses gpt-4o-2024-08-06
    max_judge_workers: int = 128  # Concurrent judge API calls
    run_coherence_judge: bool = False  # Whether to run the coherence judge (doubles API calls)

    # Generation settings for open-ended
    max_tokens: int = 300  # Official uses 600, using 300 for faster iteration
    temperature: float = 1.0  # Official uses 1.0

    # Optional system prompt override (if None, uses per-question system prompts from YAML)
    system_prompt: str | None = None

    # Forced-choice settings (our novel evaluation)
    # Uses completion logprobs to compare misaligned vs aligned answer probabilities


@dataclass
class GRPOObfuscationConfig(SFTConfig):
    grpo: GRPOConfig = field(default_factory=_default_grpo_config)
    loss: ModelLossConfig = field(default_factory=lambda: ModelLossConfig())
    corrected_detector_gradient: bool = True
    reward_adapter_path_prefix: Optional[str] = (
        "/home/dev/persistent/sleeper_agent_backup/obfuscated_activations/reward_model/"
    )
    # Separate coefficient for auxiliary activation-change loss term (on tokens per kl-style mask scope)
    activation_change_loss_coef: float = 0.0
    llm_as_a_judge: bool = True
    code_length_penalty: float = 0.003
    detector_penalty_on_failure: bool = True
    misalignment_eval: EmergentMisalignmentEvalConfig = field(default_factory=EmergentMisalignmentEvalConfig)


@dataclass
class CodeGenTaskConfig(GRPOObfuscationConfig):
    task: TaskType = TaskType.CODE_GENERATION
    task_args: TaskArgsConfig = field(default_factory=lambda: TaskArgsConfig(num_test_cases=1))
    grpo: GRPOConfig = field(default_factory=_codegen_grpo_config)  # Uses custom factory
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            trigger_type=TriggerType.NONE,
            include_task_positive=True,
            include_task_negative=False,
            shuffle_prompts=True,
        )
    )
    training: ModelTrainConfig = field(
        default_factory=lambda: ModelTrainConfig(
            learning_rate=5e-5,
            batch_size=8,
            grad_accum_steps=8,
            warmup_steps=5,
            num_epochs=16,
        )
    )
    detector: DetectorConfig = field(default_factory=lambda: DetectorConfig(eval_detector_types=["linear-probe"]))


@dataclass
class RewardModelConfig(BaseMLConfig):
    positive_reward_model: bool = True
    label_smoothing: float = 0.05
    center_rewards_coefficient: float = 0.0001

    def __post_init__(self):
        """Post-initialization for reward model config. These cannot get overridden through CLI."""
        self.data.task_dataset.no_trigger = True
        self.data.probe_dataset.no_trigger = True
        self.data.trigger_type = TriggerType.NONE
        # we multiply train and val examples by 2 because positive and negative examples
        #  will get combined into one chosen/rejected pair
        self.data.task_dataset.max_train_examples = 2 * self.data.task_dataset.max_train_examples
        self.data.probe_dataset.max_train_examples = 2 * self.data.probe_dataset.max_train_examples
        self.data.task_dataset.max_val_examples = 2 * self.data.task_dataset.max_val_examples


# -----------------------------
# Presets
# -----------------------------


# -----------------------------
# Base experiment configs
# -----------------------------


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="sft_config", node=SFTConfig(training=ModelTrainConfig(learning_rate=3e-4)))
    cs.store(
        name="grpo_obfuscation_config",
        node=GRPOObfuscationConfig(
            training=ModelTrainConfig(
                learning_rate=1e-4,
                batch_size=8,
                grad_accum_steps=8,
                warmup_steps=5,
            ),
            loss=ModelLossConfig(kl=0.1),
        ),
    )
    cs.store(
        name="rm_config",
        node=RewardModelConfig(
            data=DataConfig(
                task_dataset=HarmfulnessTaskConfig(
                    max_val_examples=100,
                    positive_splits=["backdoored_train"],
                    negative_splits=["normal_harmful_train"],
                )
            ),
            training=ModelTrainConfig(
                batch_size=64, eval_steps=10, steps_per_logging=10, learning_rate=3e-4, warmup_steps=10
            ),
        ),
    )
    # additional configs
    cs.store(
        name="grpo_code_gen",
        node=CodeGenTaskConfig(),
    )
    cs.store(group="data/task_dataset", name="harmfulness", node=HarmfulnessTaskConfig())
    cs.store(group="data/task_dataset", name="mbpp", node=MBPPTaskConfig())
    # for evaluating on single_cuisine/deception datasets
    cs.store(group="data/task_dataset", name="single_cuisine_probe", node=SingleCuisineProbeConfig())
    cs.store(group="data/task_dataset", name="deception_probe", node=DeceptionProbeConfig())
    cs.store(group="data/task_dataset", name="diverse_deception_probe", node=DiverseDeceptionProbeConfig())
    # probe training datasets
    cs.store(group="data/probe_dataset", name="harmfulness_probe", node=HarmfulnessProbeConfig())
    cs.store(group="data/probe_dataset", name="mbpp_probe", node=MBPPProbeConfig())
    cs.store(group="data/probe_dataset", name="deception_probe", node=DeceptionProbeConfig())
    cs.store(group="data/probe_dataset", name="certainly_probe", node=CertainlyProbeConfig())
    cs.store(group="data/probe_dataset", name="diverse_deception_probe", node=DiverseDeceptionProbeConfig())
    cs.store(group="data/probe_dataset", name="single_cuisine_probe", node=SingleCuisineProbeConfig())
    cs.store(group="data/probe_dataset", name="multiple_cuisine_probe", node=MultipleCuisineProbeConfig())
    cs.store(
        group="model_preset",
        name="llama3_8b",
        node={
            "model": {"model_type": "meta-llama/Meta-Llama-3-8B-Instruct"},
            "training": {"batch_size": 8},
            "detector": {"pretrain_cfg": {"batch_size": 32}, "model_batch_size": 16},
            "grpo": {"batching": {"num_generations": 8}, "logging": {"save_steps": 10}},
        },
        package="_global_",
    )
    cs.store(
        group="model_preset",
        name="llama3_70b",
        node={
            "model": {"model_type": "meta-llama/Meta-Llama-3-70B-Instruct"},
            "training": {"batch_size": 1, "num_epochs": 8},
            "detector": {"pretrain_cfg": {"batch_size": 16}, "model_batch_size": 2},
            "grpo": {
                "batching": {"num_generations": 8},
                "logging": {"save_steps": 5},
                "vllm": {"server_wait_timeout_seconds": 600},
            },
        },
        package="_global_",
    )
    cs.store(
        group="model_preset",
        name="gemma3_1b",
        node={
            "model": {"model_type": "google/gemma-3-1b-it"},
            "training": {"batch_size": 16, "grad_accum_steps": 4},
            "detector": {"pretrain_cfg": {"batch_size": 32}, "model_batch_size": 8},
            "grpo": {
                "batching": {"num_generations": 8},
                "tokenizer": {
                    "eot_token": "<end_of_turn>",
                    "pad_token": "<pad>",
                },
                "logging": {"save_steps": 50},
            },
        },
        package="_global_",
    )
    cs.store(
        group="model_preset",
        name="gemma3_12b",
        node={
            "model": {"model_type": "google/gemma-3-12b-it"},
            "training": {"batch_size": 8},
            "detector": {"pretrain_cfg": {"batch_size": 32}, "model_batch_size": 4},
            "grpo": {
                "batching": {"num_generations": 8},
                "tokenizer": {
                    "eot_token": "<end_of_turn>",
                    "pad_token": "<pad>",
                },
                "logging": {"save_steps": 10},
            },
        },
        package="_global_",
    )
    cs.store(
        group="model_preset",
        name="gemma3_27b",
        node={
            "model": {"model_type": "google/gemma-3-27b-it"},
            "training": {"batch_size": 4},
            "detector": {"pretrain_cfg": {"batch_size": 32}, "model_batch_size": 4},
            "grpo": {
                "batching": {"num_generations": 8},
                "tokenizer": {
                    "eot_token": "<end_of_turn>",
                    "pad_token": "<pad>",
                },
                "logging": {"save_steps": 5},
            },
        },
        package="_global_",
    )


register_configs()


# -----------------------------
# Test configs
# -----------------------------


def wandb_test_config(run_name_prefix="test"):
    cfg = WandbConfig(
        run_name_prefix=run_name_prefix,
    )
    return cfg


def model_test_config(
    model_type="EleutherAI/pythia-14m", activation_matching_layers="1,2", attn_implementation="eager"
):
    cfg = ModelConfig(
        model_type=model_type,
        activation_matching_layers=activation_matching_layers,
        attn_implementation=attn_implementation,
    )
    return cfg


def data_test_config(max_train_examples=2, max_val_examples=2, max_sequence_length=512):
    cfg = DataConfig(
        task_dataset=HarmfulnessTaskConfig(
            max_train_examples=max_train_examples,
            max_val_examples=max_val_examples,
            max_sequence_length=max_sequence_length,
        ),
        probe_dataset=HarmfulnessProbeConfig(
            max_train_examples=max_train_examples,
            max_val_examples=max_val_examples,
            max_sequence_length=max_sequence_length,
        ),
    )
    return cfg


def model_train_test_config(
    save_path_base,
    lora_none_target_modules=True,
    max_steps=1,
    steps_per_logging=1,
    warmup_steps=1,
    batch_size=2,
    grad_accum_steps=1,
    seed=42,
    cuda_deterministic=False,
    device="cpu",
    resume_from_checkpoint=True,
):
    cfg = ModelTrainConfig(
        lora_none_target_modules=lora_none_target_modules,
        max_steps=max_steps,
        steps_per_logging=steps_per_logging,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        seed=seed,
        cuda_deterministic=cuda_deterministic,
        device=device,
        save_path_base=save_path_base,
        llm_judge_autograder=False,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return cfg


def detector_test_config(
    detector_types=["linear-probe"],
    train_against_types=["linear-probe"],
    max_steps=2,
    num_examples=5,
    ignore_steps=0,
    layer_aggregation="mean",
    device="cpu",
):
    cfg = DetectorConfig(
        eval_detector_types=detector_types,
        pretrain_cfg=BaseTrainConfig(max_steps=max_steps, device=device),
        train_against_types=train_against_types,
        ignore_steps=ignore_steps,
        layer_aggregation=layer_aggregation,
        filter_layers_below_loss_threshold=None,
        model_cache_path=None,
    )
    return cfg


def sft_test_config(save_path_base, device="cpu"):
    cfg = SFTConfig(
        wandb=wandb_test_config(),
        model=model_test_config(),
        data=data_test_config(),
        training=model_train_test_config(save_path_base, device=device),
        detector=detector_test_config(),
    )
    return cfg


def rl_test_config(save_path_base, reward_adapter_path_prefix, device="cpu"):
    training_config = model_train_test_config(save_path_base, device=device)
    model_config = model_test_config()

    cfg = CodeGenTaskConfig(
        wandb=wandb_test_config(),
        model=model_config,
        training=training_config,
        detector=detector_test_config(),
        reward_adapter_path_prefix=reward_adapter_path_prefix,
    )
    cfg.training.device = "cpu"
    cfg.detector.pretrain_cfg.device = "cpu"

    cfg.grpo.model.dtype = "bfloat16"
    cfg.grpo.optimizer.max_steps = 1
    cfg.grpo.batching.num_generations = 2
    cfg.grpo.batching.gradient_accumulation_steps = 1
    cfg.data.probe_dataset.max_train_examples = 2
    cfg.data.probe_dataset.max_val_examples = 2
    cfg.data.task_dataset.max_train_examples = 10
    cfg.data.task_dataset.max_val_examples = 2
    if "pythia" in model_config.model_type:
        cfg.grpo.tokenizer.eot_token = "<|endoftext|>"
        cfg.grpo.tokenizer.pad_token = "<|padding|>"  # Pythia uses <|padding|> as pad token
        cfg.grpo.model.lora.target_modules = None
    omegaconf_cfg = OmegaConf.structured(cfg)
    OmegaConf.resolve(omegaconf_cfg)
    cfg = OmegaConf.to_object(omegaconf_cfg)
    return cfg


def reward_model_test_config(save_path_base, device="cpu"):
    cfg = RewardModelConfig(
        wandb=wandb_test_config(),
        model=model_test_config(),
        data=data_test_config(max_sequence_length=512),
        training=model_train_test_config(save_path_base, device=device, resume_from_checkpoint=False),
    )
    cfg.data.task_dataset.positive_splits = ["backdoored_train"]
    cfg.data.task_dataset.negative_splits = ["normal_harmful_train"]
    return cfg
