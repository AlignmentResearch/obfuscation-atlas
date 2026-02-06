# type: ignore
"""Model loading and environment management utilities."""

import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import list_repo_files
from huggingface_hub.utils import HFValidationError
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest

loaded_models = {}
loaded_tokenizers = {}


def load_hf_model(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    requires_grad=False,
    merge_and_unload=False,
):
    # Choose attention implemention if not specified
    if attn_implementation is None:
        # Make sure that models that dont support FlashAttention aren't forced to use it
        if "gpt2" in model_name or "gemma" in model_name:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"

    # Check if the model is peft, and load accordingly
    try:
        files = list_repo_files(model_name)
    except HFValidationError:
        files = os.listdir(model_name)
    has_adapter_config = any("adapter_config.json" in file for file in files)
    if has_adapter_config:
        model = (
            AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                device_map=device_map,
                trust_remote_code=True,
            )
            .merge_and_unload()
            .eval()
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

    # Disable model grad if we're not training
    if not requires_grad:
        model.requires_grad_(False)

    # Save and return the model
    loaded_models[model_name] = model
    return model


def load_hf_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    tokenizer_name=None,
    requires_grad=False,
    merge_and_unload=False,
):
    # Load the model
    model = load_hf_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        requires_grad=requires_grad,
        merge_and_unload=merge_and_unload,
    )

    # Get the tokenizer name if not specified
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path

    # Check if tokenizer has already been loaded
    global loaded_tokenizers
    if tokenizer_name in loaded_tokenizers.keys():
        tokenizer = loaded_tokenizers[tokenizer_name]
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = (
            tokenizer.eos_token_id
        )  # Make sure the eos in the generation config is the same as the tokenizer

        # Save the tokenizer
        loaded_tokenizers[tokenizer_name] = tokenizer
    return model, tokenizer


def get_clean_env():
    """Get environment without Accelerate variables"""
    env = os.environ.copy()

    # Remove by pattern
    for key in list(env.keys()):
        if (
            key in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
            or any(key.startswith(p) for p in ["ACCELERATE_", "FSDP_", "TORCHELASTIC_", "TORCHINDUCTOR_"])
            or key.startswith("ROLE_")
            or key.startswith("GROUP_")
            or key.endswith("_WORLD_SIZE")
            or key.endswith("_RANK")
        ):
            del env[key]
    return env


@contextmanager
def isolated_vllm_env():
    """
    Temporarily isolate environment from Accelerate's distributed setup
    to allow vLLM to initialize its own distributed environment.
    """
    # Store original environment variables
    original_env = {}

    for key in list(os.environ.keys()):
        if (
            key in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
            or any(key.startswith(p) for p in ["ACCELERATE_", "FSDP_", "TORCHELASTIC_", "TORCHINDUCTOR_"])
            or key.startswith("ROLE_")
            or key.startswith("GROUP_")
            or key.endswith("_WORLD_SIZE")
            or key.endswith("_RANK")
        ):
            original_env[key] = os.environ[key]
            del os.environ[key]

    # # Set vLLM-specific environment variables
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))

    try:
        yield
    finally:
        # Restore original environment
        for key, value in original_env.items():
            os.environ[key] = value
        # Clean up vLLM-specific vars
        # if "VLLM_WORKER_MULTIPROC_METHOD" in os.environ:
        #     del os.environ["VLLM_WORKER_MULTIPROC_METHOD"]


def load_vllm_model(model_name: str, lora: bool = True, **kwargs):
    """
    Load a model using vLLM for efficient inference.
    vLLM automatically loads the tokenizer internally.

    Args:
        model_name: HuggingFace model name or local path
        **kwargs: Additional arguments for vLLM LLM initialization
            Common args: tensor_parallel_size, gpu_memory_utilization,
                        max_model_len, dtype, quantization
    """
    default_config = {
        "tensor_parallel_size": torch.cuda.device_count(),  # Use all available GPUs
        "gpu_memory_utilization": 0.9,  # Use 90% of GPU memory
        "dtype": "auto",  # Auto-detect dtype (usually bfloat16 for newer models)
        "trust_remote_code": True,  # Required for some models
    }

    # Update defaults with any provided kwargs
    default_config.update(kwargs)
    if not lora:
        return LLM(model=model_name, **default_config), None

    model_path = Path(model_name)
    if model_path.is_dir():
        adapter_path = model_path / "adapter_config.json"
        assert adapter_path.is_file(), f"Adapter config not found at {adapter_path}"
    elif model_path.is_file():
        adapter_path = Path(adapter_path)
    else:
        raise ValueError(f"Invalid model path: {model_name}")

    with open(f"{model_path}/adapter_config.json", "r") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]

    # Load base model with vLLM (NOT the adapter directory)
    model = LLM(
        model=base_model_name,  # Use base model, not adapter path
        enable_lora=True,
        max_lora_rank=adapter_config.get("r"),
        **default_config,
    )

    # Create LoRA request for inference
    lora_request = LoRARequest(
        lora_name="default",
        lora_int_id=1,
        lora_path=adapter_path,
    )
    return model, lora_request


@contextmanager
def load_vllm_model_isolated(model_name: str, lora: bool = True, **kwargs):
    with isolated_vllm_env():
        try:
            model, lora_request = load_vllm_model(model_name, lora, **kwargs)
            yield model, lora_request
        finally:
            del model, lora_request
            gc.collect()
