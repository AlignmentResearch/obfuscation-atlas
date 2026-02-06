"""
GRPO obfuscation training script using pre-trained detectors and post-training evaluation.

N.B. before running this script, start the vLLM server (matching model to cfg.model.model_type):
```
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95 --tensor-parallel-size 1
```
"""

import os
import shutil

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # Suppress C++ warnings about tensor weak references

from pathlib import Path
from typing import Any, Iterator, Mapping

import hydra
import numpy as np
import wandb
from accelerate.utils import broadcast_object_list
from afterburner.generation.vllm.client import VLLMClient
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from obfuscation_atlas.config import GRPOObfuscationConfig, register_configs
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.evaluate import (
    eval_multiple_detectors,
    get_layers_below_loss_threshold,
    train_and_eval_multiple_detectors,
)
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator

# Shared utilities
from obfuscation_atlas.detectors.train import get_scheduler_fn
from obfuscation_atlas.tasks.code_generation import compute_code_generation_rewards
from obfuscation_atlas.training.grpo_obfuscation_utils import compute_obfuscation_decomposition_metrics
from obfuscation_atlas.training.grpo_trainer import (
    GRPOObfuscationTrainer,
)
from obfuscation_atlas.training.obfuscation_training import (
    detector_results_path,
    init_wandb_with_checkpoint_resuming,
    load_eval_results_and_detectors,
    log_detector_metrics,
    prepare_dataset,
    prepare_detector_obfuscators,
)
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper

# Register Hydra configs
register_configs()


def next_valid_row(iter: Iterator[str], max_length: int, tokenizer) -> str | None:
    while True:
        try:
            row = next(iter)
            if len(tokenizer.encode(row["prompt"])) < max_length:  # type: ignore
                return row
        except StopIteration:
            return None


def get_detector_name(detector_type: str, obfuscate_over: str) -> str:
    return detector_type.replace("_", "-") + "_" + obfuscate_over.replace("_", "-")


def compute_mean_decision_threshold(results_dict: Mapping[str, Mapping[str, Any]], filtered_layers: list[int]) -> float:
    mean_decision_threshold = 0.0
    for layer in filtered_layers:
        metrics = results_dict[str(layer)]
        mean_decision_threshold += metrics["metrics_at_threshold"]["threshold"]
    return float(mean_decision_threshold / len(filtered_layers))


def compute_mean_positive_score(results_dict: Mapping[str, Mapping[str, Any]], filtered_layers: list[int]) -> float:
    """Compute mean score over on_dataset positive examples across filtered layers.

    This is used as a threshold for obfuscation margin metrics - the margin represents
    how much lower the policy's detector score is compared to typical positive examples.

    Args:
        results_dict: Results dict from detector evaluation with per-layer metrics.
        filtered_layers: Layers to average over.

    Returns:
        Mean of mean_score_positive across the filtered layers.
    """
    mean_positive_threshold = 0.0
    for layer in filtered_layers:
        metrics = results_dict[str(layer)]
        mean_positive_threshold += metrics["mean_score_positive"]
    return float(mean_positive_threshold / len(filtered_layers))


def run_rl(cfg: GRPOObfuscationConfig):
    eval_detector_types = cfg.detector.eval_detector_types
    if "black-box" in eval_detector_types and not cfg.training.only_eval_detectors:
        raise NotImplementedError(
            "Black-box detector evaluation not yet supported for RL training."
            " Please merge the lora weights with the base model, save it with some id in save_path_base,"
            " and run this script with"
            " ++training.only_eval_detectors=True ++training.model_wandb_id=<id_substring>."
            " if you want to evaluate the black-box detector."
        )
    model_type = cfg.model.model_type
    model_last_name = model_type.split("/")[-1]
    max_total_length = cfg.grpo.tokenizer.max_prompt_length + cfg.grpo.tokenizer.max_response_length

    # Load tokenizer early - needed for dataset preparation (chat templates)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    (
        (task_positive_train, task_negative_train),
        probe_dataset,
        eval_task_dataset,
    ) = prepare_dataset(cfg=cfg.data, show=True, tokenizer=tokenizer)

    # Build RL prompts mixing task positive (backdoored) and task negative prompts
    def build_prompt_lists(pos_dataset, neg_dataset) -> tuple[list[str], list[bool], list[dict[str, Any]]]:
        prompts: list[str] = []
        trigger_present: list[bool] = []
        reward_infos: list[dict[str, Any]] = []
        # Interleave to balance
        pos_iter, neg_iter = iter(pos_dataset), iter(neg_dataset)
        if cfg.data.include_task_positive and cfg.data.include_task_negative:
            total = min(len(pos_dataset), len(neg_dataset))
        elif cfg.data.include_task_positive:
            total = len(pos_dataset)
        elif cfg.data.include_task_negative:
            total = len(neg_dataset)
        else:
            raise ValueError("No task positive or negative prompts to include")
        for i in range(total):
            pos_row = next_valid_row(pos_iter, max_total_length, tokenizer) if cfg.data.include_task_positive else None
            if pos_row is None and cfg.data.include_task_positive:
                break
            neg_row = next_valid_row(neg_iter, max_total_length, tokenizer) if cfg.data.include_task_negative else None
            if neg_row is None and cfg.data.include_task_negative:
                break
            if cfg.data.include_task_positive:
                prompts.append(pos_row["prompt"])  # type: ignore
                trigger_present.append(True)
                info = {}
                if "test_list" in pos_row:  # type: ignore
                    info["test_list"] = pos_row["test_list"]  # type: ignore
                    info["test_setup_code"] = pos_row.get("test_setup_code", "")  # type: ignore
                reward_infos.append(info)
            if cfg.data.include_task_negative:
                prompts.append(neg_row["prompt"])  # type: ignore
                trigger_present.append(False)
                info = {}
                if "test_list" in neg_row:  # type: ignore
                    info["test_list"] = neg_row["test_list"]  # type: ignore
                    info["test_setup_code"] = neg_row.get("test_setup_code", "")  # type: ignore
                reward_infos.append(info)
        return prompts, trigger_present, reward_infos

    train_prompts, trigger_present, reward_infos = build_prompt_lists(task_positive_train, task_negative_train)
    eval_prompts, eval_trigger_present, _ = build_prompt_lists(eval_task_dataset[0], eval_task_dataset[1])
    if cfg.data.shuffle_prompts:
        perm = np.random.permutation(len(train_prompts))
        train_prompts = [train_prompts[i] for i in perm]
        trigger_present = [trigger_present[i] for i in perm]
        reward_infos = [reward_infos[i] for i in perm]

    print(f"Total dataset size across all splits: {len(train_prompts)}")
    # Configure obfuscation rewards

    # Train RL model (trainer manages accelerator internally)
    trainer = GRPOObfuscationTrainer(
        obfuscation_config=cfg,
        train_prompts=train_prompts,
        trigger_present=trigger_present,
        reward_function=None,
        reward_infos=reward_infos,
        eval_prompts=eval_prompts,
        eval_trigger_present=eval_trigger_present,
    )
    accelerator = trainer.accelerator
    is_main_process = accelerator is None or accelerator.is_main_process
    activation_layers = trainer.activation_layers
    # Now initialize wandb and paths

    if accelerator.is_main_process:
        wandb_run, save_path, checkpoints_dir, resuming_from_checkpoint = init_wandb_with_checkpoint_resuming(
            cfg,
            WANDB_RUN_NAME=f"{(cfg.wandb.run_name_prefix or 'rl')}_{model_last_name}",
            activation_layers=activation_layers,
        )
        wandb_run_id = str(wandb_run.id)
        # Prepare data to broadcast
        broadcast_data = [save_path, checkpoints_dir, wandb_run_id, resuming_from_checkpoint]
    else:
        wandb_run = None
        # Prepare empty list to receive broadcast
        broadcast_data = [None, None, None, None]

    # Broadcast from main to all
    broadcast_object_list(broadcast_data, from_process=0)
    save_path, checkpoints_dir, wandb_run_id, resuming_from_checkpoint = broadcast_data
    save_path = Path(save_path)  # type: ignore
    cfg.grpo.logging.output_dir = str(checkpoints_dir)
    trainer.output_dir = trainer.get_output_dir()  # update the internal state

    model_cache_path = (
        f"{cfg.detector.model_cache_path}/{cfg.training.model_wandb_id or 'base'}_activations_{model_last_name}_"
        + f"tr-{cfg.data.probe_dataset.dataset_name}_ts-{cfg.data.task_dataset.dataset_name}/"
    )

    if cfg.detector.retrain:
        if os.path.exists(model_cache_path):
            shutil.rmtree(model_cache_path)

    if cfg.training.only_eval_from_other_run:
        # Check if we need to load from FSDP checkpoint (when no LoRA adapter exists)
        if cfg.grpo.model.reference_adapter_path is None and cfg.training.model_wandb_id is not None:
            # No LoRA adapter available, load from FSDP checkpoint instead
            from obfuscation_atlas.config import construct_fsdp_checkpoint_path

            fsdp_checkpoint_path = construct_fsdp_checkpoint_path(
                cfg.training.model_wandb_id, cfg.training.save_path_base
            )
            if fsdp_checkpoint_path is not None:
                print(f"Loading model from FSDP checkpoint: {fsdp_checkpoint_path}")
                trainer.state.load(Path(fsdp_checkpoint_path), trainer.accelerator)
            else:
                raise ValueError(
                    f"No LoRA adapter or FSDP checkpoint found for wandb id {cfg.training.model_wandb_id}. "
                    "Cannot evaluate model from other run."
                )
        # we disable adapter since reference adapter is the already trained model
        # and we want to evaluate both base and final model in this mode.
        trainer.state.model.shared_base_model.base_model.disable_adapter_layers()  # type: ignore
    else:
        # we set reference adapter since we want to train the final model
        # or only evaluate the final model in the case of only_eval_detectors.
        trainer.state.model.set_adapter(["reference"])  # type: ignore
    train_sequence_aggregator = (
        SequenceAggregator(
            method=cfg.detector.train_sequence_aggregator,
            sliding_window=cfg.detector.sequence_aggregation_sliding_window,
        )
        if cfg.detector.train_sequence_aggregator is not None
        else None
    )
    kwargs: dict[str, Any] = dict(
        model=trainer.state.model.shared_base_model,
        tokenizer=trainer.tokenizer,
        layers=activation_layers,
        train_dataset=probe_dataset,
        test_dataset=eval_task_dataset,
        train_cfg=cfg.detector.pretrain_cfg,
        detector_arch_configs=[get_detector_arch_config(dt) for dt in eval_detector_types],
        obfuscate_over=cfg.detector.eval_obfuscate_overs,
        cache_activations_save_path=model_cache_path,
        max_completion_length=cfg.detector.max_completion_length,
        max_sequence_length=cfg.data.probe_dataset.max_sequence_length,
        model_batch_size=cfg.detector.model_batch_size,
        false_positive_rate=0.01,
        train_sequence_aggregator=train_sequence_aggregator,
        eval_sequence_aggregator=SequenceAggregator(
            method=cfg.detector.eval_sequence_aggregator,
            sliding_window=cfg.detector.sequence_aggregation_sliding_window,
        ),
        black_box_detector=False,  # we directly specify black-box detector in detector_type instead of using this flag
        black_box_features=-1,
        accelerator=accelerator._accelerator,  # HfAccelerator._accelerator # type: ignore
        use_sklearn=cfg.detector.use_sklearn,
        train_completion_columns=(
            cfg.data.probe_dataset.positive_completion_column,
            cfg.data.probe_dataset.negative_completion_column,
        ),
        test_completion_columns=(
            cfg.data.task_dataset.positive_completion_column,
            cfg.data.task_dataset.negative_completion_column,
        ),
        train_congruent_follow_up_prompts=cfg.data.probe_dataset.congruent_follow_up_prompts,
        train_incongruent_follow_up_prompts=cfg.data.probe_dataset.incongruent_follow_up_prompts,
        test_congruent_follow_up_prompts=cfg.data.task_dataset.congruent_follow_up_prompts,
        test_incongruent_follow_up_prompts=cfg.data.task_dataset.incongruent_follow_up_prompts,
        n_calibration_examples=cfg.detector.n_calibration_examples,
        platt_shift_only=cfg.detector.platt_shift_only,
        use_on_policy_negatives_for_calibration=cfg.detector.use_on_policy_negatives_for_calibration,
        vllm_config=cfg.grpo.vllm,
        vllm_model_name=cfg.grpo.model.model_name,
        vllm_model_revision=cfg.grpo.model.model_revision,
        vllm_apply_chat_template=cfg.grpo.tokenizer.apply_chat_template,
        vllm_lora_path="",  # use base model
        use_on_policy_negatives_for_training=cfg.detector.use_on_policy_negatives_for_training,
        augment_training_negatives=cfg.detector.augment_training_negatives,
    )
    BASE_MODEL_DETECTOR_RESULTS_PATH = detector_results_path(
        cfg.training.save_path_base,
        cfg.data.trigger_type,
        model_last_name,
        cfg.data.probe_dataset.dataset_name.split("/")[-1],
        eval_detector_types,
        cfg.detector.train_sequence_aggregator,
        cfg.detector.eval_sequence_aggregator,
        cfg.training.model_wandb_id,
    )
    detector_name = get_detector_name(cfg.detector.train_against_types[0], cfg.detector.obfuscate_over)
    if cfg.training.only_eval_detectors or cfg.training.only_eval_from_other_run or cfg.detector.retrain:
        eval_results_and_detectors = {}
        found_detectors = False
    else:
        eval_results_and_detectors = load_eval_results_and_detectors(
            BASE_MODEL_DETECTOR_RESULTS_PATH,
            load_detectors=is_main_process,
            prefix="initial_detector.base_model.on_dataset",
        )
        found_detectors = len(eval_results_and_detectors) > 0
    if (
        cfg.training.only_eval_detectors
        or cfg.training.only_eval_from_other_run
        or cfg.detector.retrain
        or not resuming_from_checkpoint
    ):
        if not found_detectors:
            eval_results_and_detectors = train_and_eval_multiple_detectors(**kwargs)
        # Only log metrics on the main process
        if is_main_process:
            log_detector_metrics(
                eval_results_and_detectors,
                "Initial detector on base model activations from dataset text",
                BASE_MODEL_DETECTOR_RESULTS_PATH,
                "initial_detector.base_model.on_dataset",
                activation_layers,
                force_save=cfg.training.only_eval_detectors or cfg.detector.retrain,
                filter_layers_below_loss_threshold=cfg.detector.filter_layers_below_loss_threshold,
            )
    if cfg.training.only_eval_detectors:
        if wandb.run is not None:
            print(f"Wandb URL: \033[94m{wandb.run.url}\033[0m")
        return

    filtered_layers = activation_layers
    if cfg.detector.filter_layers_below_loss_threshold is not None:
        filtered_layers = get_layers_below_loss_threshold(
            eval_results_and_detectors[detector_name][1][1],  # type: ignore
            cfg.detector.filter_layers_below_loss_threshold,
        )
        print(f"Filtered layers: {filtered_layers}")
        # Log filtered layers metadata to wandb
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "filtered_layers": filtered_layers,
                    "num_filtered_layers": len(filtered_layers),
                    "filtered_layers_min": min(filtered_layers) if filtered_layers else None,
                    "filtered_layers_max": max(filtered_layers) if filtered_layers else None,
                    "layer_filtering_enabled": cfg.detector.filter_layers_below_loss_threshold is not None,
                }
            )
    encoder = LanguageModelWrapper(trainer.state.model.shared_base_model, trainer.tokenizer)
    cfg.training.max_steps = trainer.total_steps
    scheduler_fn = get_scheduler_fn(cfg.training)

    results_dict = eval_results_and_detectors[detector_name][1][0]
    mean_decision_threshold = compute_mean_decision_threshold(results_dict, filtered_layers)
    mean_positive_score = compute_mean_positive_score(results_dict, filtered_layers)
    print(f"Mean decision threshold: {mean_decision_threshold}")
    print(f"Mean positive mean_positive_score on dataset: {mean_positive_score}")

    common_obfuscation_kwargs = {
        "layers": filtered_layers,
        "obfuscate_over": cfg.detector.obfuscate_over,
        "layer_aggregation": cfg.detector.layer_aggregation,
        "detector_ignore_steps": cfg.detector.ignore_steps,
        "device": cfg.training.device,
        "accelerator": accelerator._accelerator,  # HfAccelerator._accelerator  # type: ignore
        "scheduler_fn": scheduler_fn,
        "train_detector_every_n_steps": cfg.detector.train_every_n_steps,
        "n_grad_accum": cfg.grpo.batching.gradient_accumulation_steps * cfg.grpo.batching.num_generations,
        "rl_training": True,
    }
    normalize_input = get_detector_arch_config(cfg.detector.train_against_types[0]).normalize_input  # type: ignore
    obfuscation_loss_fns, probe_training_dataset = prepare_detector_obfuscators(
        encoder=encoder,
        task_dataset=(task_positive_train, task_negative_train),
        probe_dataset=probe_dataset,
        eval_results_and_detectors=eval_results_and_detectors,  # type: ignore
        is_main_process=is_main_process,
        OBFUSCATE_OVER=cfg.detector.obfuscate_over,
        PROBE_ON_TRIGGER=cfg.detector.probe_on_trigger,
        SUPERVISED_PROBE_COEF=cfg.loss.detector,
        NUM_PROBES=cfg.detector.num_probes,
        TRAIN_AGAINST_SUPERVISED_DETECTOR_TYPE=cfg.detector.train_against_types,
        COMMON_OBFUSCATION_KWARGS=common_obfuscation_kwargs,
        DETECTOR_SPECIFIC_KWARGS={},
        found_detectors=found_detectors,
        enable_platt_scaling=cfg.detector.n_calibration_examples is not None,
        normalize_input=normalize_input,
    )
    accelerator.wait_for_everyone()
    log_config_kwargs = {}
    for wrapped_fn in obfuscation_loss_fns.keys():
        log_config_kwargs["_".join(wrapped_fn.__class__.__name__.split("_")[1:-3]).upper() + "_NUM_PARAMETERS"] = (
            wrapped_fn.num_parameters
        )  # type: ignore
    pretrained_detectors = list(map(lambda x: x[0], eval_results_and_detectors.values()))
    del eval_results_and_detectors, kwargs["model"]
    trainer.set_detector(obfuscation_loss_fns, probe_training_dataset, mean_decision_threshold, mean_positive_score)  # type: ignore
    if not cfg.training.only_eval_from_other_run:
        trainer.train()
        final_adapter_path = save_path / "checkpoints" / "final" / "lora"
    else:
        print("Skipping RL training, only evaluating from other run")
        trainer.state.model.shared_base_model.base_model.enable_adapter_layers()  # type: ignore
        assert checkpoints_dir is not None
        final_adapter_path = Path(checkpoints_dir) / "final"

    # Load the final adapter into a HF model for evaluation

    trainer.state.model.set_adapter(["policy"])  # type: ignore
    trainer.state.model.shared_base_model.eval()  # type: ignore
    kwargs["model"] = trainer.state.model.shared_base_model
    kwargs["cache_activations_save_path"] = f"{cfg.detector.model_cache_path}/activations_{wandb_run_id}"
    kwargs["vllm_lora_path"] = str(final_adapter_path)

    # Evaluate detectors post-training
    eval_results_after = train_and_eval_multiple_detectors(**kwargs)
    if cfg.detector.filter_layers_below_loss_threshold is not None:
        fresh_filtered_layers = get_layers_below_loss_threshold(
            eval_results_after[detector_name][1][1],  # type: ignore
            cfg.detector.filter_layers_below_loss_threshold,
        )
    else:
        fresh_filtered_layers = activation_layers
    if is_main_process:
        log_detector_metrics(
            eval_results_after,
            "Fresh detector on final model activations from dataset text",
            str(save_path),
            "fresh_detector.final_model.on_dataset",
            activation_layers,
            filtered_layers=fresh_filtered_layers,
        )
    # remove detector-training-specific kwargs since we only want to evaluate using the already-trained detectors
    for k in [
        "train_cfg",
        "train_sequence_aggregator",
        "detector_arch_configs",
        "cache_activations_save_path",
        "use_sklearn",
        "use_on_policy_negatives_for_training",
        "augment_training_negatives",
    ]:
        kwargs.pop(k)
    cache_activations_save_path = f"/tmp/activations_{wandb_run_id}"

    eval_results = eval_detectors_helper(
        pretrained_detectors,
        eval_detector_types,
        "initial_detector.final_model.on_dataset",
        "Initial detector on final model activations from dataset text",
        cache_activations_save_path,
        is_main_process,
        save_path,
        filtered_layers=filtered_layers,
        **kwargs,  # layers in kwargs
    )
    policy_detector_mean_positive_score_on_dataset = extract_mean_detector_scores(eval_results, filtered_layers)

    # Extract test_lists and test_setup_code from eval_task_dataset if available
    eval_test_lists = (
        list(eval_task_dataset[0]["test_list"]) if "test_list" in eval_task_dataset[0].column_names else None
    )
    eval_test_setup_codes = (
        list(eval_task_dataset[0]["test_setup_code"])
        if "test_setup_code" in eval_task_dataset[0].column_names
        else None
    )
    on_policy_positive_dataset, on_policy_reward_metrics = generate_on_policy_responses(
        trainer,
        eval_task_dataset[0]["prompt"],
        final_adapter_path,
        num_generations=1,
        log_to_wandb=is_main_process,
        model_wandb_id=cfg.training.model_wandb_id,
        test_lists=eval_test_lists,
        test_setup_code_list=eval_test_setup_codes,
        code_length_penalty=cfg.code_length_penalty,
        num_test_cases=cfg.task_args.num_test_cases,
    )
    on_policy_dataset = (
        on_policy_positive_dataset,
        eval_task_dataset[1],
    )
    kwargs["test_completion_columns"] = (
        "completion",
        cfg.data.task_dataset.negative_completion_column,
    )
    kwargs["test_dataset"] = on_policy_dataset
    detectors_list = [pretrained_detectors, list(map(lambda x: x[0], eval_results_after.values()))]
    del eval_results_after
    plot_keys = ["initial_detector.final_model.on_policy", "fresh_detector.final_model.on_policy"]
    plot_titles = [
        "Initial detector on final model activations from on-policy text",
        "Fresh detector on final model activations from on-policy text",
    ]
    filtered_layers_list = [filtered_layers, fresh_filtered_layers]

    # Evaluate detectors on policy model activations and collect results
    policy_eval_results_list = []
    for detectors, plot_key, plot_title, filtered_layers_d in zip(
        detectors_list, plot_keys, plot_titles, filtered_layers_list
    ):
        eval_results = eval_detectors_helper(
            detectors,
            eval_detector_types,
            plot_key,
            plot_title,
            cache_activations_save_path,
            is_main_process,
            save_path,
            filtered_layers=filtered_layers_d,
            **kwargs,  # layers in kwargs
        )
        policy_eval_results_list.append(eval_results)

    # Evaluate initial detector on reference (base) model activations
    if cfg.training.only_eval_from_other_run:
        # we disable adapter since reference adapter is the already trained model in this mode.
        trainer.state.model.shared_base_model.base_model.disable_adapter_layers()  # type: ignore
    else:
        trainer.state.model.set_adapter(["reference"])  # type: ignore
    kwargs["model"] = trainer.state.model.shared_base_model
    reference_eval_results = eval_detectors_helper(
        pretrained_detectors,
        eval_detector_types,
        "initial_detector.base_model.on_policy",
        "Initial detector on base model activations from on-policy text",
        cache_activations_save_path,
        is_main_process,
        save_path,
        filtered_layers=filtered_layers,
        **kwargs,  # layers in kwargs
    )

    # Compute obfuscation decomposition metrics for each detector type
    if is_main_process and reference_eval_results is not None:
        obfuscation_prefixes = ["eval_obfuscation/initial_detector", "eval_obfuscation/fresh_detector"]
        for policy_eval_results, obf_prefix, filtered_layers_d in zip(
            policy_eval_results_list, obfuscation_prefixes, filtered_layers_list
        ):
            if policy_eval_results is not None:
                compute_obfuscation_decomposition_metrics_helper(
                    policy_eval_results=policy_eval_results,
                    reference_eval_results=reference_eval_results,
                    reward_metrics=on_policy_reward_metrics,
                    detector_loss_coef=cfg.loss.detector,
                    detector_mean_decision_threshold=mean_decision_threshold,
                    filtered_layers=filtered_layers_d,
                    reference_detector_mean_positive_score_on_dataset=mean_positive_score,
                    policy_detector_mean_positive_score_on_dataset=policy_detector_mean_positive_score_on_dataset,  # type: ignore
                    prefix=obf_prefix,
                )


def eval_detectors_helper(
    detectors: list,
    detector_types: list[str],
    plot_key: str,
    plot_title: str,
    cache_activations_save_path: str,
    is_main_process: bool,
    metrics_save_path: Path | str,
    layers: list[int],
    filtered_layers: list[int] | None = None,
    **kwargs,
) -> dict[str, dict[str, Any]] | None:
    """Helper function to evaluate detectors using already trained detectors.

    Args:
        detectors: The detectors to evaluate.
        detector_types: The types of detectors to evaluate.
        plot_key: The key to use for the plot.
        plot_title: The title to use for the plot.
        cache_activations_save_path: The path to save the cached activations.
        is_main_process: Whether to log metrics to wandb.
        metrics_save_path: The path to save the metrics.
        layers: The layers to evaluate the detectors on.
        filter_layers_below_loss_threshold: Threshold to filter layers below.
            If None, uses `layers` instead.
        **kwargs: Additional keyword arguments to pass to the eval_multiple_detectors function.

    Returns:
        The evaluation results dict (test_metrics per detector key), or None if not main process.
    """
    eval_for_obf_policy = eval_multiple_detectors(
        detectors=detectors,
        detector_types=detector_types,
        train_cache_activations_save_path=f"{cache_activations_save_path}_{plot_key}_train",
        test_cache_activations_save_path=f"{cache_activations_save_path}_{plot_key}_test",
        **kwargs,
    )

    # convert to tuple format for logging metrics
    eval_for_logging = {k: (None, (v, None, None)) for k, v in eval_for_obf_policy.items()}
    if is_main_process:
        log_detector_metrics(
            eval_for_logging,
            plot_title,
            str(metrics_save_path),
            plot_key,
            layers,
            filtered_layers=filtered_layers,
        )
    return eval_for_obf_policy if is_main_process else None


def extract_mean_detector_scores(
    eval_results: dict[str, dict[str, Any]] | None,
    filtered_layers: list[int],
) -> np.ndarray | None:
    """Extract mean detector scores across filtered layers for positive examples.

    Args:
        eval_results: Evaluation results from eval_detectors_helper.
        filtered_layers: Layers to average over.

    Returns:
        Mean detector scores per sample as numpy array.
    """
    if eval_results is None:  # non-main process
        return None
    # Get the first detector key (all detectors evaluate on same data)
    first_key = next(iter(eval_results.keys()))
    test_metrics = eval_results[first_key]

    positive_scores_per_layer = []
    for layer in filtered_layers:
        layer_key = str(layer)
        if layer_key in test_metrics and "scores" in test_metrics[layer_key]:
            positive_scores_per_layer.append(test_metrics[layer_key]["scores"]["positive"])

    # Average across layers: shape [num_layers, num_samples] -> [num_samples]
    return np.array(positive_scores_per_layer).mean(axis=0)


def compute_obfuscation_decomposition_metrics_helper(
    policy_eval_results: dict[str, dict[str, Any]],
    reference_eval_results: dict[str, dict[str, Any]],
    reward_metrics: dict[str, list],
    detector_loss_coef: float,
    detector_mean_decision_threshold: float,
    filtered_layers: list[int],
    reference_detector_mean_positive_score_on_dataset: float | None = None,
    policy_detector_mean_positive_score_on_dataset: float | None = None,
    prefix: str = "eval_obfuscation",
) -> dict[str, float]:
    """Compute obfuscation decomposition metrics from policy and reference eval results.

    Args:
        policy_eval_results: Eval results from policy model (final model activations).
        reference_eval_results: Eval results from reference model (base model activations).
        reward_metrics: Reward metrics from on-policy generation containing task rewards and deceptive metric.
        detector_loss_coef: Coefficient for detector loss.
        detector_mean_decision_threshold: Mean decision threshold from detector training.
        reference_detector_mean_positive_score_on_dataset: Mean positive score on dataset from reference model.
        policy_detector_mean_positive_score_on_dataset: Mean positive score on dataset from policy model.
        filtered_layers: Layers to average detector scores over.
        prefix: Prefix for logged metric names.

    Returns:
        Dict of obfuscation decomposition metrics.
    """
    import torch

    # Extract mean detector scores across filtered layers
    policy_scores = extract_mean_detector_scores(policy_eval_results, filtered_layers)
    reference_scores = extract_mean_detector_scores(reference_eval_results, filtered_layers)

    # Extract task rewards and deceptive metric from reward_metrics
    # Keys match those used in compute_code_generation_rewards
    task_rewards = torch.tensor(reward_metrics["rewards/task_success"]).float()
    deceptive_metric = reward_metrics["rewards_metrics/passes_first_n_but_not_all_tests"]

    # Compute decomposition metrics
    metrics = compute_obfuscation_decomposition_metrics(
        reference_detector_prob=torch.tensor(reference_scores),
        policy_detector_prob=torch.tensor(policy_scores),
        task_rewards=task_rewards,
        deceptive_metric=deceptive_metric,
        detector_loss_coef=detector_loss_coef,
        detector_mean_decision_threshold=detector_mean_decision_threshold,
        detector_mean_positive_threshold=reference_detector_mean_positive_score_on_dataset,
    )

    # Log to wandb with the specified prefix, stripping the "obfuscation/" prefix from keys
    wandb_metrics = {f"{prefix}/{k.replace('obfuscation/', '')}": v for k, v in metrics.items()}
    if (
        reference_detector_mean_positive_score_on_dataset is not None
        and policy_detector_mean_positive_score_on_dataset is not None
    ):
        wandb_metrics["representation_change_on_dataset"] = (
            reference_detector_mean_positive_score_on_dataset - policy_detector_mean_positive_score_on_dataset
        )
    if wandb.run is not None:
        wandb.log(wandb_metrics)

    return wandb_metrics


def generate_on_policy_responses(
    trainer: GRPOObfuscationTrainer,
    prompts: list[str],
    final_adapter_path: Path,
    num_generations: int = 1,
    log_to_wandb: bool = False,
    model_wandb_id: str | None = None,
    test_lists: list[list[str]] | None = None,
    test_setup_code_list: list[str] | None = None,
    code_length_penalty: float = 0.002,
    num_test_cases: int = 1,
) -> tuple[Dataset, dict[str, list]]:
    """Function to generate on-policy responses using the GRPOObfuscationTrainer.

    Args:
        trainer: The trainer to use to generate the responses.
        prompts: The prompts to generate the responses for.
        final_adapter_path: The path to the final adapter.
        num_generations: The number of generations to generate.
        log_to_wandb: Whether to log the responses to wandb.
        model_wandb_id: If provided, we fetch on policy generations from this wandb id.
        test_lists: Optional list of test cases for each prompt (for code generation metrics).
        test_setup_code_list: Optional setup code for each prompt.
        code_length_penalty: Penalty per character of code length.
        num_test_cases: Number of test cases to use for reward computation.

    Returns:
        A tuple of (Dataset containing prompts and responses, reward_metrics dict).
    """
    prompts_formatted, response_strings, reward_metrics = None, None, None  # type: ignore
    if model_wandb_id:
        run = wandb.Api().run(model_wandb_id)
        df = None
        for artifact in run.logged_artifacts():
            if "on_policy_responses" in artifact.name:
                table = artifact.get("on_policy_responses")
                df = table.get_dataframe()
                break
        if df is not None:
            print(f"Found on policy responses for wandb id: {model_wandb_id}. Using them.")
            # Extract reward_metrics from the dataframe columns if available
            reward_metrics: dict[str, list] = {}  # type: ignore
            for col in df.columns:
                if col not in ["prompt", "completion"]:
                    reward_metrics[col] = df[col].tolist()
            prompts_formatted = df["prompt"].tolist()
            response_strings = df["completion"].tolist()
        else:
            print(f"No on policy responses found for wandb id: {model_wandb_id}. Generating new responses using vLLM.")

    if prompts_formatted is None:
        prompts_formatted: list[str] = (
            trainer.tokenizer.apply_chat_template(  # type: ignore
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if trainer.config.tokenizer.apply_chat_template
            else prompts
        )
        prompt_ids: list[list[int]] = trainer.tokenizer(  # type: ignore
            prompts_formatted,
            add_special_tokens=False,
        )["input_ids"]
        assert isinstance(prompt_ids, list)
        vllm_client = VLLMClient(trainer.config.vllm)
        responses = trainer.generate_responses(
            vllm_client,
            prompt_ids,
            trainer.state.model.policy_adapter,
            final_adapter_path,
            num_generations=num_generations,
        )
        response_strings = trainer.tokenizer.batch_decode(responses.tokens)
        # Compute code generation metrics if test_lists provided
        reward_metrics: dict[str, list] = {}
        if test_lists is not None:
            _, reward_metrics = compute_code_generation_rewards(
                prompts_formatted,
                [r.removesuffix(trainer.tokenizer.eos_token) for r in response_strings],  # type: ignore
                test_lists,
                test_setup_code_list,
                code_length_penalty=code_length_penalty,
                num_test_cases=num_test_cases,
            )

    if log_to_wandb:
        # Build table columns: prompt, completion, plus any reward metrics
        columns = ["prompt", "completion"]
        metric_keys = list(reward_metrics.keys())
        columns.extend(metric_keys)

        # Build table data
        data = []
        for i, (prompt, completion) in enumerate(zip(prompts_formatted, response_strings)):  # type: ignore
            row = [prompt, completion]
            for k in metric_keys:
                row.append(reward_metrics[k][i])
            data.append(row)

        wandb.log({"on_policy_responses": wandb.Table(columns=columns, data=data)})
        # log aggregated reward metrics
        wandb.log({f"final_eval/{k}": np.mean(v) for k, v in reward_metrics.items()})

    return Dataset.from_dict({"prompt": prompts_formatted, "completion": response_strings}), reward_metrics


@hydra.main(version_base=None, config_name="grpo_obfuscation_config")
def _hydra_entry(cfg: DictConfig):
    cfg_obj: GRPOObfuscationConfig = OmegaConf.to_object(cfg)  # type: ignore
    run_rl(cfg_obj)


if __name__ == "__main__":
    _hydra_entry()
