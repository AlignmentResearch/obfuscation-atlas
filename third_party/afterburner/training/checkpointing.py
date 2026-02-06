import contextlib
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from packaging import version

from afterburner.utils.logging import logger


def safe_globals():
    # Copied from https://github.com/huggingface/transformers/blob/58e13b9f129bb0dccc3b51e5da22f45ef3ff0ae7/src/transformers/trainer.py#L276
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


def set_rng_state_for_device(device_name, device_module, checkpoint_rng_state, is_distributed):
    """Helper to set RNG state for a specific device type (CUDA, NPU, MLU, MUSA)"""
    # Copied from https://github.com/huggingface/transformers/blob/58e13b9f129bb0dccc3b51e5da22f45ef3ff0ae7/src/transformers/trainer_pt_utils.py#L1395
    device_state_key = device_name.lower()
    err_template = "Didn't manage to set back the RNG states of the {backend} because of the following error:\n {exception}\nThis won't yield the same results as if the training had not been interrupted."
    try:
        if is_distributed:
            device_module.random.set_rng_state_all(checkpoint_rng_state[device_state_key])
        else:
            device_module.random.set_rng_state(checkpoint_rng_state[device_state_key])
    except Exception as e:
        # Log error if setting RNG state fails
        logger.error(err_template.format(backend=device_name, exception=e))


@dataclass
class RandomState:
    process_index: int
    world_size: int

    def save(self, path: Path):
        # Forked from https://github.com/huggingface/transformers/blob/58e13b9f129bb0dccc3b51e5da22f45ef3ff0ae7/src/transformers/trainer.py#L3385
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.world_size > 1:
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(path, exist_ok=True)

        if self.world_size <= 1:
            torch.save(rng_states, os.path.join(path, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(path, f"rng_state_{self.process_index}.pth"))

    def load(self, path: Path):
        # Forked from https://github.com/huggingface/transformers/blob/58e13b9f129bb0dccc3b51e5da22f45ef3ff0ae7/src/transformers/trainer.py#L3253
        if self.world_size > 1:
            process_index = self.process_index
            rng_file = os.path.join(path, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(path, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        is_distributed = self.world_size > 1
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)
