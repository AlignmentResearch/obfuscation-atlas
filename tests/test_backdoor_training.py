import os

import pytest
import torch

from obfuscation_atlas.config import sft_test_config
from obfuscation_atlas.scripts.train_sft_against_probe import run_sft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping test because CUDA is not available")
def test_train_backdoor_model(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    cfg = sft_test_config(save_path_base=str(tmp_path))
    log_dict = run_sft(cfg)
    assert log_dict is not None
    for split in ["backdoored_test", "normal_harmful_test"]:
        assert len(log_dict[f"autograder/{split}/samples_list"]) == 2
        completions = [c["completion"] for c in log_dict[f"autograder/{split}/samples_list"]]
        assert all(isinstance(c, str) and len(c) > 0 for c in completions), completions
