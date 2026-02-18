# %%
import gc
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate.utils import broadcast_object_list
from afterburner.utils.fsdp_viz import visualize_fsdp_distribution
from omegaconf import OmegaConf
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import logging as transformers_logging

from obfuscation_atlas.config import SFTConfig, register_configs
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.evaluate import train_and_eval_multiple_detectors
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator
from obfuscation_atlas.detectors.train import get_scheduler_fn, initialize_lora_adapter
from obfuscation_atlas.tasks.strongreject_autograder_script import run_autograder  # noqa: F401
from obfuscation_atlas.training.grpo_obfuscation_utils import _first_follow_up
from obfuscation_atlas.training.obfuscation_training import (
    detector_results_path,
    get_accelerator,
    get_clean_env,
    init_wandb_with_checkpoint_resuming,
    is_accelerate_training,
    load_eval_results_and_detectors,
    log_autograder_results,
    log_detector_metrics,
    prepare_dataset,
    prepare_detector_obfuscators,
    reset_via_numba,
)
from obfuscation_atlas.training.sft_trainer import save_lora_model, train_backdoor
from obfuscation_atlas.utils.activations import get_num_hidden_layers
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper
from obfuscation_atlas.utils.model_loading import (
    load_hf_model_and_tokenizer,
)
from obfuscation_atlas.utils.scoring import (
    get_activation_matching_layers,
)
from obfuscation_atlas.utils.wandb_utils import find_wandb_id_in_directory

# Register Hydra configs
register_configs()


def run_sft(cfg: SFTConfig):
    USE_ACCELERATE = is_accelerate_training()
    eval_detector_type = cfg.detector.eval_detector_types
    accelerator = get_accelerator(USE_ACCELERATE, timeout=3600 if "black-box" in eval_detector_type else 600)
    is_main_process = accelerator is None or accelerator.is_main_process
    if cfg.training.test_mode:
        torch.autograd.set_detect_anomaly(True)

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

    # Or suppress all Transformers warnings
    transformers_logging.set_verbosity_error()

    # Loss coefficients
    loss_coefs = {}
    if cfg.loss.task != 0:
        loss_coefs["backdoored"] = cfg.loss.task
    if cfg.loss.activation_change != 0:
        loss_coefs["activation_change"] = cfg.loss.activation_change
    if cfg.loss.kl != 0:
        loss_coefs["kl"] = cfg.loss.kl

    # Load the appropriate model and dataset
    model_type = cfg.model.model_type

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    print(f"max_train_examples: {cfg.data.task_max_train_examples}")
    task_dataset, probe_dataset, val_task_dataset = prepare_dataset(cfg=cfg.data, show=True)

    device_map = "cpu" if USE_ACCELERATE else cfg.training.device
    if cfg.training.model_wandb_id is not None:
        model_type = find_wandb_id_in_directory(
            cfg.training.model_wandb_id, base_path=cfg.training.save_path_base, is_file=False
        )
        assert model_type is not None
    model, tokenizer = load_hf_model_and_tokenizer(
        model_type,
        attn_implementation=cfg.model.attn_implementation,
        device_map=device_map,
        requires_grad=False,
        torch_dtype=torch.bfloat16,
    )
    activation_layers = get_activation_matching_layers(
        cfg.model.activation_matching_layers, get_num_hidden_layers(model.config)
    )

    scheduler_fn = get_scheduler_fn(cfg.training)

    plot_title = ""
    if cfg.training.test_mode:
        WANDB_RUN_NAME = "test"
    else:
        # Include all non-zero loss coefficients in the run name
        active_losses = []
        if cfg.loss.backdoored != 0:
            active_losses.append(f"bd={cfg.loss.task}")
        if cfg.loss.activation_change != 0:
            active_losses.append(f"act={cfg.loss.activation_change}")
        if cfg.loss.kl != 0:
            active_losses.append(f"kl={cfg.loss.kl}")
        if cfg.loss.detector != 0:
            if cfg.detector.probe_on_trigger:
                active_losses.append(f"on-tg-probe={cfg.loss.detector}")
                plot_title += "|on-trigger-probe"
            else:
                joined = "|".join(map(lambda x: "".join(x.split("-")), cfg.detector.train_against_types))
                active_losses.append(f"off-tg-probe={cfg.loss.detector}-{joined}")
                plot_title += "|" + "|".join(cfg.detector.train_against_types)
        run_name_prefix = cfg.wandb.run_name_prefix
        if run_name_prefix and not run_name_prefix.endswith("_"):
            run_name_prefix += "_"
        WANDB_RUN_NAME = run_name_prefix + cfg.data.task_dataset_name.split("/")[-1].split("-")[1] + "_"
        if cfg.training.only_eval_detectors:
            WANDB_RUN_NAME = "eval_detectors:" + WANDB_RUN_NAME
        else:
            WANDB_RUN_NAME += "_".join(active_losses)

    lora_params = {"target_modules": None} if cfg.training.lora_none_target_modules else {}
    lora_model = initialize_lora_adapter(model, [get_num_hidden_layers(model.config)], lora_params)
    lora_model = lora_model.to(torch.bfloat16)

    if accelerator is not None:
        if accelerator.is_main_process:
            wandb_run, save_path, checkpoints_dir, resume_checkpoint_path = init_wandb_with_checkpoint_resuming(
                cfg, WANDB_RUN_NAME, activation_layers
            )
            wandb_run_id = str(wandb_run.id)
            # Prepare data to broadcast
            broadcast_data = [save_path, checkpoints_dir, resume_checkpoint_path, wandb_run_id]
        else:
            wandb_run = None
            # Prepare empty list to receive broadcast
            broadcast_data = [None, None, None, None]

        # Broadcast from main to all
        broadcast_object_list(broadcast_data, from_process=0)
        save_path, checkpoints_dir, resume_checkpoint_path, wandb_run_id = broadcast_data
    else:
        # Non-distributed case
        wandb_run, save_path, checkpoints_dir, resume_checkpoint_path = init_wandb_with_checkpoint_resuming(
            cfg, WANDB_RUN_NAME, activation_layers
        )
        wandb_run_id = str(wandb_run.id)

    if cfg.training.test_mode:
        torch.cuda.memory._record_memory_history()

    if accelerator is not None:
        # # Apply FSDP wrapping to LoRA model
        if USE_ACCELERATE and accelerator is not None:
            if getattr(accelerator.state, "fsdp_plugin", None) is not None:
                lora_policy = fsdp_auto_wrap_policy(lora_model)  # type: ignore
                accelerator.state.fsdp_plugin.auto_wrap_policy = lora_policy  # type: ignore
        lora_model = accelerator.prepare(lora_model)
        accelerator.free_memory()
        visualize_fsdp_distribution(lora_model, accelerator)
    else:
        lora_model = lora_model.to(cfg.training.device)

    lora_model.eval()
    encoder = LanguageModelWrapper(lora_model, tokenizer)

    log_config_kwargs = {
        "SUPERVISED_PROBE_COEF": cfg.loss.detector,
        "LAYER_AGGREGATION": cfg.detector.layer_aggregation,
        "OBFUSCATE_OVER": cfg.detector.obfuscate_over,
        "TRIGGER_TYPE": cfg.data.trigger_type,
        "PROBE_ON_TRIGGER": cfg.detector.probe_on_trigger,
        "LLM_JUDGE_AUTOGRADER": cfg.training.llm_judge_autograder,
        "NUM_PROBES": cfg.detector.num_probes,
    }
    model_last_name = model_type.split("/")[-1]
    train_sequence_aggregator = (
        SequenceAggregator(
            method=cfg.detector.train_sequence_aggregator,
            sliding_window=cfg.detector.sequence_aggregation_sliding_window,
        )
        if cfg.detector.train_sequence_aggregator is not None
        else None
    )
    model_cache_path = None
    if cfg.detector.model_cache_path is not None:
        model_cache_path = f"{cfg.detector.model_cache_path}/{model_last_name}_{cfg.data.trigger_type}/"
    kwargs: dict[str, Any] = dict(
        model=lora_model,
        tokenizer=tokenizer,
        train_dataset=probe_dataset,
        test_dataset=val_task_dataset,
        layers=activation_layers,
        detector_arch_configs=[get_detector_arch_config(dt) for dt in eval_detector_type],
        train_cfg=cfg.detector.pretrain_cfg,
        obfuscate_over=cfg.detector.eval_obfuscate_overs,
        max_completion_length=cfg.detector.max_completion_length,
        max_sequence_length=cfg.data.probe_dataset.max_sequence_length,
        model_batch_size=cfg.training.batch_size * 2,
        false_positive_rate=0.01,
        cache_activations_save_path=model_cache_path,
        train_sequence_aggregator=train_sequence_aggregator,
        eval_sequence_aggregator=SequenceAggregator(
            method=cfg.detector.eval_sequence_aggregator,
            sliding_window=cfg.detector.sequence_aggregation_sliding_window,
        ),
        accelerator=accelerator,
        model_path_for_vllm=cfg.model.model_type,
        train_congruent_follow_up_prompts=cfg.data.probe_dataset.congruent_follow_up_prompts,
        train_incongruent_follow_up_prompts=cfg.data.probe_dataset.incongruent_follow_up_prompts,
        test_congruent_follow_up_prompts=cfg.data.task_dataset.congruent_follow_up_prompts,
        test_incongruent_follow_up_prompts=cfg.data.task_dataset.incongruent_follow_up_prompts,
        n_calibration_examples=cfg.detector.n_calibration_examples,
        platt_shift_only=cfg.detector.platt_shift_only,
        use_on_policy_negatives_for_calibration=cfg.detector.use_on_policy_negatives_for_calibration,
        vllm_lora_path="",  # use base model
        use_on_policy_negatives_for_training=cfg.detector.use_on_policy_negatives_for_training,
        augment_training_negatives=cfg.detector.augment_training_negatives,
    )
    on_trigger_kwargs = kwargs.copy()

    on_trigger_cache_path = None
    if cfg.detector.model_cache_path is not None:
        on_trigger_cache_path = (
            f"{cfg.detector.model_cache_path}/base_activations_on_trigger_{model_last_name}_{cfg.data.trigger_type}/"
        )
    on_trigger_kwargs["cache_activations_save_path"] = on_trigger_cache_path
    on_trigger_kwargs["train_dataset"] = task_dataset
    on_trigger_kwargs["detector_arch_configs"] = [get_detector_arch_config("linear-probe")]

    BASE_MODEL_DETECTOR_RESULTS_PATH = detector_results_path(
        cfg.training.save_path_base,
        cfg.data.trigger_type,
        model_last_name,
        cfg.data.probe_dataset_name.split("/")[-1],
        eval_detector_type,
        cfg.detector.train_sequence_aggregator,
        cfg.detector.eval_sequence_aggregator,
        model_wandb_id=None,
        directory_save_prefix=cfg.detector.directory_save_prefix,
    )
    # only main process loads detectors as loading all detectors on all process is memory intensive
    # and not necessary. The relevant detectors are broadcast to all processes later on.
    eval_results_and_detectors = load_eval_results_and_detectors(
        BASE_MODEL_DETECTOR_RESULTS_PATH, load_detectors=is_main_process, prefix="before_training"
    )
    found_detectors = len(eval_results_and_detectors) > 0
    if cfg.training.only_eval_detectors and not cfg.training.resume_from_checkpoint:
        if not found_detectors:
            eval_results_and_detectors = train_and_eval_multiple_detectors(**kwargs)
            if cfg.detector.probe_on_trigger:
                on_trigger_eval_results_and_detectors = train_and_eval_multiple_detectors(**on_trigger_kwargs)
                eval_results_and_detectors.update(  # type: ignore
                    {"on-trigger-" + k: v for k, v in on_trigger_eval_results_and_detectors.items()}
                )
                del on_trigger_eval_results_and_detectors
        # Only log metrics on the main process
        if is_main_process:
            log_detector_metrics(
                eval_results_and_detectors,
                "Detector on base model",
                BASE_MODEL_DETECTOR_RESULTS_PATH,
                "before_training",
                activation_layers,
            )
    if cfg.training.only_eval_detectors:
        return
    common_obfuscation_kwargs = {
        "layers": activation_layers,
        "obfuscate_over": cfg.detector.obfuscate_over,
        "layer_aggregation": cfg.detector.layer_aggregation,
        "detector_ignore_steps": cfg.detector.ignore_steps,
        "device": cfg.training.device,
        "accelerator": accelerator,
        "scheduler_fn": scheduler_fn,
        "train_detector_every_n_steps": cfg.detector.train_every_n_steps,
        "n_grad_accum": cfg.training.grad_accum_steps,
        "log_probs": cfg.training.backdoor_obfuscation_log_probs,
    }
    normalize_input = get_detector_arch_config(cfg.detector.train_against_types[0]).normalize_input  # type: ignore
    obfuscation_loss_fns, probe_training_dataset = prepare_detector_obfuscators(
        encoder=encoder,
        task_dataset=task_dataset,
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
        normalize_input=normalize_input,
    )
    for wrapped_fn in obfuscation_loss_fns.keys():
        log_config_kwargs["_".join(wrapped_fn.__class__.__name__.split("_")[1:-3]).upper() + "_NUM_PARAMETERS"] = (
            wrapped_fn.num_parameters
        )  # type: ignore
    del eval_results_and_detectors, kwargs["model"], on_trigger_kwargs["model"]

    encoder.model.train()
    gc.collect()
    if accelerator is not None:
        accelerator.free_memory()
    else:
        torch.cuda.empty_cache()
    lora_model, wandb_run = train_backdoor(
        encoder,
        obfuscation_loss_fns,
        *task_dataset,
        *probe_training_dataset,
        *val_task_dataset,
        include_task_positive=cfg.data.include_task_positive,
        include_task_negative=cfg.data.include_task_negative,
        scheduler_fn=scheduler_fn,
        activation_matching_layers=activation_layers,
        loss_coefs=loss_coefs,
        lora_params=lora_params,
        model_lr=cfg.training.learning_rate,
        n_steps=cfg.training.max_steps,
        n_steps_per_logging=cfg.training.steps_per_logging,
        batch_size=cfg.training.batch_size,
        n_grad_accum=cfg.training.grad_accum_steps,
        device=cfg.training.device,
        clip_grad_norm=cfg.training.clip_grad_norm,
        model_type=model_type,
        dataset_name=cfg.data.task_dataset.dataset_name + f"_{cfg.data.trigger_type}",
        backdoor_type=cfg.training.backdoor_type,
        wandb_project=cfg.wandb.project,
        wandb_run_name=WANDB_RUN_NAME,
        wandb_group=cfg.wandb.group,
        eval_backdoor_during_training=False,
        num_warmup_steps=cfg.training.warmup_steps,
        truncate_fraction=cfg.training.truncate_fraction,
        max_sequence_length=cfg.data.task_dataset.max_sequence_length,
        log_config_kwargs=log_config_kwargs,
        finish_wandb_run=False,
        checkpoint_dir=checkpoints_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        checkpoint_interval=cfg.training.checkpoint_interval,
        accelerator=accelerator,
        skip_loss_scaling=cfg.loss.skip_loss_scaling,
        task_follow_up_prompt=_first_follow_up(cfg.data.task_dataset.congruent_follow_up_prompts),
        probe_follow_up_prompt=_first_follow_up(cfg.data.probe_dataset.congruent_follow_up_prompts),
    )
    lora_model.eval()
    save_lora_model(lora_model, save_path, accelerator, merge_lora=True)
    detector_refs = [k.obfuscator.detector for k in obfuscation_loss_fns.keys()]
    if hasattr(accelerator, "_models"):
        accelerator._models = [m for m in accelerator._models if m not in detector_refs]  # type: ignore
        del detector_refs
        print(f"Kept {len(accelerator._models)} models in accelerator (should be Llama)")  # type: ignore
    gc.collect()
    if cfg.training.device == "cuda":
        torch.cuda.empty_cache()

    kwargs["model"] = lora_model
    if cfg.detector.model_cache_path is not None:
        kwargs["cache_activations_save_path"] = f"{cfg.detector.model_cache_path}/activations_{wandb_run_id}"
    kwargs["model_path_for_vllm"] = save_path

    eval_results_and_detectors = train_and_eval_multiple_detectors(**kwargs)
    on_trigger_eval_results_and_detectors = None
    if cfg.detector.probe_on_trigger:
        on_trigger_kwargs["model"] = lora_model
        if cfg.detector.model_cache_path is not None:
            on_trigger_kwargs["cache_activations_save_path"] = (
                f"{cfg.detector.model_cache_path}/on_trigger_activations_{wandb_run_id}"
            )
        # Pass accelerator for distributed detector training
        if USE_ACCELERATE and accelerator is not None:
            on_trigger_kwargs["accelerator"] = accelerator
        on_trigger_eval_results_and_detectors = train_and_eval_multiple_detectors(**on_trigger_kwargs)
        eval_results_and_detectors.update(  # type: ignore
            {"on-trigger-" + k: v for k, v in on_trigger_eval_results_and_detectors.items()}
        )
    plot_title = "Trained against " + plot_title.strip("|")
    # Only log metrics on the main process when using distributed training
    if not USE_ACCELERATE or (accelerator and accelerator.is_main_process):
        log_detector_metrics(
            eval_results_and_detectors,
            plot_title,
            save_path,
            "after_training",
            activation_layers,
        )
    if cfg.training.test_mode:
        torch.cuda.memory._dump_snapshot(f"{cfg.training.save_path_base}/memory_snapshot.pickle")
        print(f"Dumped snapshot to {cfg.training.save_path_base}/memory_snapshot.pickle")
    if accelerator is None:
        lora_model.to("cpu")
    else:
        accelerator.wait_for_everyone()
    del on_trigger_eval_results_and_detectors, eval_results_and_detectors
    del obfuscation_loss_fns, kwargs, on_trigger_kwargs, encoder, lora_model, model
    if "wrapped_fn" in locals():
        del wrapped_fn  # type: ignore
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_via_numba()

    if accelerator is None or accelerator.is_main_process:
        clean_env = get_clean_env()
        script_path = Path(__file__).parent.parent / "tasks" / "strongreject_autograder_script.py"
        assert save_path is not None
        log_dict_save_path = os.path.join(save_path, "autograder_log_dict.json")
        result = subprocess.run(
            [
                "python",
                str(script_path),
                f"--dataset_name={cfg.data.task_dataset_name}",
                f"--trigger_type={cfg.data.trigger_type}",
                f"--save_path={save_path}",
                f"--BATCH_SIZE={cfg.training.batch_size}",
                f"--EVAL_EXAMPLES={cfg.data.max_val_examples}",
                f"--log_dict_save_path={log_dict_save_path}",
                f"--llm_judge={cfg.training.llm_judge_autograder}",
            ],
            env=clean_env,
            start_new_session=True,
            # preexec_fn=os.setsid if os.name != "nt" else None,
            check=True,
        )
        if result.returncode != 0:
            print(f"Autograder failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError("Autograder failed")
        if not os.path.exists(log_dict_save_path):
            raise RuntimeError(f"Autograder did not create log file at {log_dict_save_path}")
        with open(log_dict_save_path, "r") as f:
            log_dict = json.load(f)
            log_autograder_results(log_dict)
    else:
        log_dict = None
    return log_dict


@hydra.main(version_base=None, config_name="sft_config")
def _hydra_entry(cfg):
    cfg_obj: SFTConfig = OmegaConf.to_object(cfg)  # type: ignore
    run_sft(cfg_obj)


if __name__ == "__main__":
    _hydra_entry()

# %%
