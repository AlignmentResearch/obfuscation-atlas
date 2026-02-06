import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch


def _accelerate_cli_available() -> bool:
    return shutil.which("accelerate") is not None


def _find_any_checkpoint_dirs(base: Path) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(base):
        if "checkpoints" in dirs:
            found.append(Path(root) / "checkpoints")
    return found


@pytest.mark.skip(reason="Single rank FSDP training doesn't work due to auto NO_SHARD switch")
def test_fsdp_single_rank_smoke(tmp_path: Path):
    # Single-rank FSDP on one GPU; exercises Accelerate path, LoRA under FSDP, and checkpointing
    save_base = tmp_path / "runs"
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        "/workspace/obfuscation_atlas/train_backdoored_model_explicit_obfuscation_against_mad_probes.py",
        "--MODEL_TYPE=EleutherAI/pythia-14m",
        "--TEST_MODE=True",
        "--ATTN_IMPLEMENTATION=eager",
        "--ACTIVATION_MATCHING_LAYERS=all",
        "--PUSH_TO_HUB=False",
        f"--SAVE_PATH_BASE={str(save_base)}",
        "--WANDB_RUN_NAME_PREFIX=test",
        "--EVAL_DETECTOR_TYPE=['linear-probe']",
        "--TRAIN_AGAINST_SUPERVISED_DETECTOR_TYPE=['linear-probe']",
        "--DETECTOR_IGNORE_STEPS=0",
        "--EVAL_DETECTOR_TRAIN_STEPS=2",
        "--EVAL_EXAMPLES=2",
        "--N_STEPS=4",
        "--checkpoint_interval=1",
        "--NONE_TARGET_MODULES=True",
        "--NUM_EXAMPLES=4",
        "--N_STEPS_PER_LOGGING=1",
        "--NUM_WARMUP_STEPS=1",
        "--BATCH_SIZE=2",
        "--N_GRAD_ACCUM=1",
        "--SUPERVISED_PROBE_COEF=1.0",
        "--MAHALANOBIS_COEF=0.0",
        "--VAE_COEF=0.0",
        "--NFLOW_COEF=0.0",
    ]
    subprocess.run(cmd, check=True, env=env)

    # Verify checkpoints exist and include an accelerate step directory
    ckpt_dirs = _find_any_checkpoint_dirs(save_base)
    assert ckpt_dirs, f"No checkpoints directory found under {save_base}"
    found_latest = False
    found_step_dir = False
    for d in ckpt_dirs:
        if (d / "latest.ckpt").is_file():
            found_latest = True
        for child in d.iterdir():
            if child.is_dir() and child.name.startswith("step_"):
                found_step_dir = True
    assert found_latest, "latest.ckpt not found in any checkpoints directory"
    assert found_step_dir, "No accelerate step_XXXXXX directory found in checkpoints"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for multi-rank test")
@pytest.mark.skipif(not _accelerate_cli_available(), reason="accelerate CLI not available")
def test_fsdp_two_ranks_smoke(tmp_path: Path):
    # Two-rank FSDP on two GPUs; skip if <2 GPUs available
    save_base = tmp_path / "runs"
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Restrict to two devices to avoid occupying the entire node during tests
    if torch.cuda.device_count() >= 2:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(["0", "1"])  # basic case

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "2",
        "--mixed_precision",
        "bf16",
        "/workspace/obfuscation_atlas/train_backdoored_model_explicit_obfuscation_against_mad_probes.py",
        "--MODEL_TYPE=EleutherAI/pythia-14m",
        "--ATTN_IMPLEMENTATION=eager",
        "--ACTIVATION_MATCHING_LAYERS=all",
        "--PUSH_TO_HUB=False",
        f"--SAVE_PATH_BASE={str(save_base)}",
        "--WANDB_RUN_NAME_PREFIX=test",
        "--EVAL_DETECTOR_TYPE=['linear-probe']",
        "--TRAIN_AGAINST_SUPERVISED_DETECTOR_TYPE=['linear-probe']",
        "--DETECTOR_IGNORE_STEPS=0",
        "--EVAL_DETECTOR_TRAIN_STEPS=2",
        "--EVAL_EXAMPLES=2",
        "--N_STEPS=4",
        "--checkpoint_interval=1",
        "--NONE_TARGET_MODULES=True",
        "--NUM_EXAMPLES=4",
        "--N_STEPS_PER_LOGGING=1",
        "--NUM_WARMUP_STEPS=1",
        "--BATCH_SIZE=2",
        "--N_GRAD_ACCUM=1",
        "--SUPERVISED_PROBE_COEF=1.0",
        "--MAHALANOBIS_COEF=0.0",
        "--VAE_COEF=0.0",
        "--NFLOW_COEF=0.0",
    ]

    # Run with output capture so we can see errors
    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode as text for readability
        )
        # If successful, optionally print output for debugging
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        # Print the captured output before re-raising
        print("\n" + "=" * 80)
        print("SUBPROCESS FAILED WITH EXIT CODE:", e.returncode)
        print("=" * 80)
        print("\nCOMMAND:")
        print(" ".join(e.cmd))
        print("\nSTDOUT:")
        print(e.stdout)
        print("\nSTDERR:")
        print(e.stderr)
        print("=" * 80 + "\n")
        # Re-raise with more informative error message
        raise subprocess.CalledProcessError(e.returncode, e.cmd, output=e.stdout, stderr=e.stderr) from e
