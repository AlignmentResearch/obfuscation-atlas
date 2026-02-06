import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from accelerate.state import AcceleratorState, PartialState
from afterburner.generation.vllm.client import VLLMResponse
from afterburner.training.huggingface.trainer_state import HfTrainerState

from obfuscation_atlas.config import reward_model_test_config, rl_test_config
from obfuscation_atlas.scripts.train_grpo_model_against_probe import run_rl
from obfuscation_atlas.scripts.train_reward_model import train_rm


@pytest.fixture(autouse=True)
def reset_accelerator_state():
    """
    Clears the global state of the Accelerator directly by emptying
    the singleton dictionaries. We must reset its state
    between tests to allow different tests having different accelerator settings.
    """
    # 1. Reset AcceleratorState
    AcceleratorState._shared_state.clear()

    # 2. Reset PartialState (Accelerator relies on this internally)
    PartialState._shared_state.clear()

    yield

    # Teardown: Clear again to ensure next test starts fresh
    AcceleratorState._shared_state.clear()
    PartialState._shared_state.clear()


class MockVLLMClient:
    def __init__(self, config):
        self.config = config
        self.total_request_wait_time = 0.0

    def wait_for_server(self):
        pass

    def generate_responses(
        self,
        prompts: list[list[int]],
        model_name: str,
        model_revision: str | None,
        max_tokens: int,
        n: int,
        *args,
        **kwargs,
    ) -> list[dict]:
        """Returns list of choice dicts (not VLLMResponse) to match real interface."""
        num_responses = len(prompts) * n
        # Use a simple token sequence for testing
        dummy_text = "Test completion"
        dummy_tokens = [1, 2, 3]
        dummy_logprobs = [0.0, -0.1, -0.2]

        return [
            {"text": dummy_text, "token_ids": dummy_tokens, "logprobs": dummy_logprobs} for _ in range(num_responses)
        ]

    def broadcast_responses(self, choices: list[dict]) -> VLLMResponse:
        """Convert choices to VLLMResponse (mimics broadcast across processes)."""
        return VLLMResponse(
            text=[choice["text"] for choice in choices],
            tokens=[choice["token_ids"] for choice in choices],
            logprobs=[choice["logprobs"] for choice in choices],
        )


@pytest.mark.skip(reason="We no longer support trigger based datasets which this test uses")
def test_reward_model_training(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    log_dict = train_rm(
        cfg=reward_model_test_config(
            save_path_base=str(tmp_path),
        ),
    )
    output_dir = log_dict["output_dir"]
    assert output_dir is not None


@pytest.mark.slow
def test_grpo_training(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"  # comment out to see wandb logs on a test run
    if not torch.cuda.is_available():
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"  # Force CPU to avoid MPS/multiprocessing issues

    with (
        patch("obfuscation_atlas.training.grpo_trainer.VLLMClient", MockVLLMClient),
        patch("obfuscation_atlas.scripts.train_grpo_model_against_probe.VLLMClient", MockVLLMClient),
    ):
        cfg = rl_test_config(
            save_path_base=str(tmp_path),
            reward_adapter_path_prefix=None,
        )
        cfg.detector.train_every_n_steps = 1  # type: ignore[attr-defined]
        run_rl(cfg)  # type: ignore


@pytest.mark.slow
def test_grpo_with_detector_training(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"  # comment out to see wandb logs on a test run
    if not torch.cuda.is_available():
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"  # Force CPU to avoid MPS/multiprocessing issues

    with (
        patch("obfuscation_atlas.training.grpo_trainer.VLLMClient", MockVLLMClient),
        patch("obfuscation_atlas.scripts.train_grpo_model_against_probe.VLLMClient", MockVLLMClient),
    ):
        cfg = rl_test_config(
            save_path_base=str(tmp_path),
            reward_adapter_path_prefix=None,
        )

        run_rl(cfg)  # type: ignore


@pytest.mark.slow
def test_grpo_resume_mapping_and_checkpoints(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    if not torch.cuda.is_available():
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"  # Force CPU to avoid MPS/multiprocessing issues

    base = tmp_path / "runs"
    base.mkdir(parents=True, exist_ok=True)

    with (
        patch("obfuscation_atlas.training.grpo_trainer.VLLMClient", MockVLLMClient),
        patch("obfuscation_atlas.scripts.train_grpo_model_against_probe.VLLMClient", MockVLLMClient),
    ):
        cfg = rl_test_config(
            save_path_base=str(base),
            reward_adapter_path_prefix=None,
        )
        cfg.training.resume_from_checkpoint = False  # type: ignore[attr-defined]
        cfg.grpo.logging.save_steps = 1  # type: ignore[attr-defined]
        cfg.grpo.optimizer.max_steps = 1  # type: ignore[attr-defined]

        run_rl(cfg)  # type: ignore

        checkpoints_dir = Path(cfg.grpo.logging.output_dir)  # type: ignore[attr-defined]
        assert any(f"step_{cfg.grpo.optimizer.max_steps}" in str(ckpt) for ckpt in checkpoints_dir.glob("step_*"))  # type: ignore[attr-defined]
        assert not any(
            f"step_{cfg.grpo.optimizer.max_steps + 1}" in str(ckpt)  # type: ignore[attr-defined]
            for ckpt in checkpoints_dir.glob("step_*")  # type: ignore[attr-defined]
        )

        mapping_file = base / "wandb_resume_map.json"
        assert mapping_file.exists()
        mapping = json.loads(mapping_file.read_text())
        assert isinstance(mapping, dict) and len(mapping) >= 1
        wandb_id = next(iter(mapping.values()))

        cfg = rl_test_config(
            save_path_base=str(base),
            reward_adapter_path_prefix=None,
        )
        cfg.training.resume_from_checkpoint = True  # type: ignore[attr-defined]
        cfg.grpo.logging.save_steps = 1  # type: ignore[attr-defined]
        cfg.grpo.optimizer.max_steps = 3  # type: ignore[attr-defined]

        run_rl(cfg)  # type: ignore

        mapping2 = json.loads(mapping_file.read_text())
        assert mapping2 and next(iter(mapping2.values())) == wandb_id
        assert any(f"step_{cfg.grpo.optimizer.max_steps}" in str(ckpt) for ckpt in checkpoints_dir.glob("step_*"))  # type: ignore[attr-defined]
        assert not any(
            f"step_{cfg.grpo.optimizer.max_steps + 1}" in str(ckpt)  # type: ignore[attr-defined]
            for ckpt in checkpoints_dir.glob("step_*")  # type: ignore[attr-defined]
        )


@pytest.mark.slow
def test_grpo_resume_after_crash(tmp_path, monkeypatch):
    os.environ["WANDB_MODE"] = "disabled"
    if not torch.cuda.is_available():
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"  # Force CPU to avoid MPS/multiprocessing issues

    base = tmp_path / "runs"
    base.mkdir(parents=True, exist_ok=True)

    # Crash after the first checkpoint is written
    save_calls = {"count": 0}
    original_save = HfTrainerState.save

    def crashing_save(self, output_dir, accelerator):
        original_save(self, output_dir, accelerator)
        if self.step == 1 and save_calls["count"] == 0:
            save_calls["count"] += 1
            raise KeyboardInterrupt("simulated crash after first checkpoint")

    monkeypatch.setattr(HfTrainerState, "save", crashing_save)

    with (
        patch("obfuscation_atlas.training.grpo_trainer.VLLMClient", MockVLLMClient),
        patch("obfuscation_atlas.scripts.train_grpo_model_against_probe.VLLMClient", MockVLLMClient),
    ):
        cfg = rl_test_config(
            save_path_base=str(base),
            reward_adapter_path_prefix=None,
        )
        cfg.training.resume_from_checkpoint = True  # type: ignore[attr-defined]
        cfg.grpo.logging.save_steps = 1  # type: ignore[attr-defined]
        cfg.grpo.optimizer.max_steps = 3  # type: ignore[attr-defined]

        with pytest.raises(KeyboardInterrupt):
            run_rl(cfg)  # type: ignore

    # Restore original save for the resume run
    monkeypatch.setattr(HfTrainerState, "save", original_save)

    mapping_file = base / "wandb_resume_map.json"
    assert mapping_file.exists()
    mapping = json.loads(mapping_file.read_text())
    wandb_id = next(iter(mapping.values()))

    checkpoints_dir = Path(cfg.grpo.logging.output_dir)  # type: ignore[attr-defined]
    steps_after_crash = sorted(
        int(p.name.split("_")[-1]) for p in checkpoints_dir.glob("step_*") if (p / "state.json").is_file()
    )
    assert steps_after_crash and steps_after_crash[-1] == 1
    crashed_state_path = checkpoints_dir / f"step_{steps_after_crash[-1]}" / "state.json"
    crashed_state = json.loads(crashed_state_path.read_text())
    assert crashed_state.get("step") == steps_after_crash[-1]

    with (
        patch("obfuscation_atlas.training.grpo_trainer.VLLMClient", MockVLLMClient),
        patch("obfuscation_atlas.scripts.train_grpo_model_against_probe.VLLMClient", MockVLLMClient),
    ):
        cfg = rl_test_config(
            save_path_base=str(base),
            reward_adapter_path_prefix=None,
        )
        cfg.training.resume_from_checkpoint = True  # type: ignore[attr-defined]
        cfg.grpo.logging.save_steps = 1  # type: ignore[attr-defined]
        cfg.grpo.optimizer.max_steps = 3  # type: ignore[attr-defined]
        run_rl(cfg)  # type: ignore

    mapping2 = json.loads(mapping_file.read_text())
    assert mapping2 and next(iter(mapping2.values())) == wandb_id

    final_steps = sorted(
        int(p.name.split("_")[-1]) for p in checkpoints_dir.glob("step_*") if (p / "state.json").is_file()
    )
    assert final_steps and final_steps[-1] >= steps_after_crash[-1]
    resumed_state_path = checkpoints_dir / f"step_{final_steps[-1]}" / "state.json"
    resumed_state = json.loads(resumed_state_path.read_text())
    assert resumed_state.get("step") == final_steps[-1] == cfg.grpo.optimizer.max_steps  # type: ignore[attr-defined]
