import json
from pathlib import Path

import pytest
import torch

import obfuscation_atlas.training.sft_trainer as backdoors
from obfuscation_atlas.config import sft_test_config
from obfuscation_atlas.scripts.train_sft_against_probe import run_sft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CI often lacks GPU; keep test optional")
def test_resume_mapping_and_checkpoints(tmp_path: Path):
    base = tmp_path / "runs"
    base.mkdir(parents=True, exist_ok=True)

    # First run: create mapping and checkpoints
    cfg1 = sft_test_config(save_path_base=str(base))
    cfg1.training.resume_from_checkpoint = False
    run_sft(cfg1)

    # Verify mapping exists
    mapping_file = base / "wandb_resume_map.json"
    assert mapping_file.exists()
    mapping = json.loads(mapping_file.read_text())
    assert isinstance(mapping, dict) and len(mapping) >= 1
    wandb_id = next(iter(mapping.values()))

    # Verify latest checkpoint present under discovered directory
    from obfuscation_atlas.utils.wandb_utils import find_wandb_id_in_directory

    run_dir = find_wandb_id_in_directory(wandb_id, base_path=str(base))
    assert run_dir is not None
    latest_ckpt = Path(run_dir) / "checkpoints" / "latest.ckpt"
    assert latest_ckpt.is_file()

    # Second run: resume must load same wandb id and latest checkpoint
    cfg2 = cfg1
    cfg2.training.resume_from_checkpoint = True
    cfg2.training.max_steps = 4
    run_sft(cfg2)

    # Mapping should still point to the same id
    mapping2 = json.loads(mapping_file.read_text())
    assert mapping2 and next(iter(mapping2.values())) == wandb_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CI often lacks GPU; keep test optional")
def test_resume_after_crash_midrun(tmp_path: Path, monkeypatch):
    base = tmp_path / "runs"
    base.mkdir(parents=True, exist_ok=True)

    cfg = sft_test_config(save_path_base=str(base))
    cfg.training.resume_from_checkpoint = True
    cfg.training.max_steps = 4

    # Monkeypatch save_checkpoint to crash after step 2 (post-save)
    orig_save = backdoors.save_checkpoint

    def crashing_save(model, optimizer, scheduler, total_steps, obf_fns, checkpoint_dir):
        orig_save(model, optimizer, scheduler, total_steps, obf_fns, checkpoint_dir)
        if total_steps == 2:
            raise KeyboardInterrupt("simulated crash after step 2")

    monkeypatch.setattr(backdoors, "save_checkpoint", crashing_save)

    # Run with N_STEPS=4 and simulate crash at step 2
    with pytest.raises(KeyboardInterrupt):
        run_sft(cfg)

    # Mapping exists
    mapping_file = base / "wandb_resume_map.json"
    assert mapping_file.exists()
    mapping = json.loads(mapping_file.read_text())
    assert isinstance(mapping, dict) and len(mapping) >= 1
    wandb_id = next(iter(mapping.values()))

    from obfuscation_atlas.utils.wandb_utils import find_wandb_id_in_directory

    run_dir = find_wandb_id_in_directory(wandb_id, base_path=str(base))
    assert run_dir is not None
    latest_ckpt = Path(run_dir) / "checkpoints" / "latest.ckpt"
    assert latest_ckpt.is_file()

    # Resume same horizon (max_steps=4), should complete without new id
    run_sft(cfg)

    mapping2 = json.loads(mapping_file.read_text())
    assert mapping2 and next(iter(mapping2.values())) == wandb_id
