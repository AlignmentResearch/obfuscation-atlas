import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os

from peft import LoraConfig, PeftModel, get_peft_model


def get_lora_config(layers, lora_params):
    # Unpack LoRA parameters
    r = lora_params.get("r", 64)
    alpha = lora_params.get("alpha", 128)
    dropout = lora_params.get("dropout", 0.0)
    bias = lora_params.get("bias", "none")
    target_modules = lora_params.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    task_type = lora_params.get("task_type", "CAUSAL_LM")

    # Define LoRA Configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        layers_to_transform=list(range(0, max(layers) + 1)),
        task_type=task_type,
    )
    return lora_config


def initialize_lora_adapter(model, layers, lora_params, adapter_name="default"):
    # If model already has PEFT adapters, don't add them again; just set grads correctly
    if isinstance(model, PeftModel) or hasattr(model, "peft_config"):
        # Freeze non-LoRA params; unfreeze LoRA params
        for name, param in model.named_parameters():
            param.requires_grad = adapter_name in name
        return model

    # Fresh application of LoRA: first freeze base model
    for param in model.parameters():
        param.requires_grad = False

    lora_config = get_lora_config(layers, lora_params)
    # Apply LoRA adapter to the model
    lora_model = get_peft_model(model, lora_config)
    # Ensure only newly added LoRA params are trainable
    for name, param in lora_model.named_parameters():
        param.requires_grad = adapter_name in name
    return lora_model
