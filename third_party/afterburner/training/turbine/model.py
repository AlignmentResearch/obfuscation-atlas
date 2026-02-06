import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import huggingface_hub
import peft
import torch
import torchtitan.protocols.train_spec as train_spec_module

# HACK: Torchtitan uses import-time side effects to register model training
#   specs. This is pretty gross, and we can fix it, but we'll follow their
#   convention for now.
import turbine.model.titan_llama3  # noqa: F401
from afterburner.grpo_config import ModelConfig
from afterburner.training.interface import LoRAAdapter, RewardLoRAAdapter, compute_logprobs_from_logits
from afterburner.training.turbine.trainer_state import TurbineAccelerator
from afterburner.utils.logging import logger
from turbine.model.lora_wrapper import AdapterModel, LoraCfg
from turbine.model.lora_wrapper import LoRAAdapter as TTLoRAAdapter

TARGET_MODULES = ["v_proj", "up_proj", "q_proj", "gate_proj", "k_proj", "down_proj", "o_proj"]

TURBINE_TO_HF_MAP = {
    "attention.wq": "self_attn.q_proj",
    "attention.wk": "self_attn.k_proj",
    "attention.wv": "self_attn.v_proj",
    "attention.wo": "self_attn.o_proj",
    "feed_forward.w1": "mlp.gate_proj",
    "feed_forward.w2": "mlp.down_proj",
    "feed_forward.w3": "mlp.up_proj",
    "_orig_mod.": "",
    "layers": "base_model.model.model.layers",
}


def to_peft_loraconfig(lora_cfg: LoraCfg) -> peft.LoraConfig:
    return peft.LoraConfig(
        r=lora_cfg.r,
        lora_alpha=float(lora_cfg.lora_alpha),
        lora_dropout=lora_cfg.lora_dropout,
        use_rslora=lora_cfg.use_rslora,
        target_modules=TARGET_MODULES,
    )


def to_turbine_loraconfig(peft_cfg: peft.LoraConfig) -> LoraCfg:
    return LoraCfg(
        r=peft_cfg.r,
        lora_alpha=float(peft_cfg.lora_alpha),
        lora_dropout=peft_cfg.lora_dropout,
        use_rslora=peft_cfg.use_rslora,
    )


@dataclass
class TurbineLoRAAdapter(LoRAAdapter):
    wrapped_model: AdapterModel | None = None

    def __post_init__(self):
        self.lora_cfg = self._load_hf_lora_config(self.adapter_path)
        self.wrapped_model = self.model.shared_base_model
        self.wrapped_model.add_adapter(self.adapter_name, TTLoRAAdapter, self.lora_cfg)
        self.adapter = self.wrapped_model.get_adapter(self.adapter_name)

        # TODO: Bring this back -- it's now pretty hard because we need to
        # put the weights into the DTensors...
        # self._load_hf_adapter_weights(self.adapter_path)

    def lora_weights(self) -> dict[str, torch.Tensor]:
        # This will need a new version for multi-GPU to handle the distributed state dict
        state_dict = self.adapter.state_dict()

        new_state_dict = {}
        for name, tensor in state_dict.items():
            if "lora" in name:
                new_name = name
                new_name = new_name.replace("base_model.model.model", "model")
                for source_key, target_key in TURBINE_TO_HF_MAP.items():
                    new_name = new_name.replace(source_key, target_key)
                new_state_dict[new_name] = tensor
        # filter out non-target modules
        new_state_dict = {k: v for k, v in new_state_dict.items() if any(module in k for module in TARGET_MODULES)}

        return new_state_dict

    def _load_hf_lora_config(self, path: str) -> LoraCfg:
        if not os.path.exists(path):
            path = huggingface_hub.snapshot_download(repo_id=path)

        with open(Path(path) / "adapter_config.json") as f:
            config_dict = json.load(f)

        peft_cfg = peft.LoraConfig(**config_dict)

        for module in peft_cfg.target_modules:
            assert module in TARGET_MODULES, f"Module {module} not in TARGET_MODULES"
        for module in TARGET_MODULES:
            assert module in peft_cfg.target_modules, f"Module {module} not in target modules"

        return to_turbine_loraconfig(peft_cfg)

    def _load_hf_adapter_weights(self, path: Path) -> None:
        """Load the adapter from the specified path."""
        # TODO: Load the weights
        # lora_weights = torch.load(path / "adapter_model.bin")
        # self.wrapped_model.load_state_dict(lora_weights)
        pass

    def save(self, path: Path) -> None:
        """Save the adapter to the specified path."""
        config = to_peft_loraconfig(self.model.accelerator.job_config.lora)
        lora_weights = self.lora_weights()
        if self.model.accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
            torch.save(lora_weights, path / "adapter_model.bin")
            config.save_pretrained(str(path))

    def logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_mask: torch.Tensor, **kwargs: dict[str, Any]
    ):
        # TODO: Check if this placement is right
        # TODO: Figure out how to handle the attention mask
        input_ids = input_ids.to(self.model.accelerator.device)
        logits = self.adapter.forward(input_ids, return_last_hidden_states=False)
        return compute_logprobs_from_logits(logits, input_ids, completion_mask.to(self.model.accelerator.device)).to("cpu")


class TurbineRewardLoRAAdapter(RewardLoRAAdapter, TurbineLoRAAdapter):
    def __post_init__(self):
        TurbineLoRAAdapter.__post_init__(self)
        assert not self.is_trainable

    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_mask: torch.Tensor):
        raise NotImplementedError("Reward adapters should use reward() method instead of logprobs()")

    def _compute_last_token_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: dict[str, Any]
    ) -> torch.Tensor:
        # TODO: Check if we've placed this correctly
        # TODO: Figure out how to handle the attention mask
        outputs, last_hidden_states = self.adapter.forward(
            input_ids.to(self.model.accelerator.device), return_last_hidden_states=True
        )
        return last_hidden_states.to(self.model.accelerator.device)


class TurbineModel(torch.nn.Module):
    def __init__(self, config: ModelConfig, pad_token_id: int, accelerator: TurbineAccelerator):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.accelerator = accelerator

        # TODO: Track this
        self.local_flops = 0.0

        # TODO: Clean this up
        if "boolq" in config.model_name:
            flavor = "boolq"
        else:
            flavor = accelerator.job_config.model.flavor

        self.train_spec = train_spec_module.get_train_spec(accelerator.job_config.model.name)
        self.model_args = self.train_spec.model_args[flavor]
        self.model_args.update_from_config(accelerator.job_config)

        logger.info(f"Building {self.train_spec.name} {flavor} with {self.model_args}")

        config_dtype = getattr(torch, config.dtype)
        with torch.device("meta"):
            self.shared_base_model = self.train_spec.model_cls(self.model_args).to(dtype=config_dtype)
            self.shared_base_model = AdapterModel(self.shared_base_model)

        self.reference_adapter = TurbineLoRAAdapter(
            adapter_name="reference",
            adapter_path=config.reference_adapter_path,
            model=self,
            dtype=config_dtype,
            revision=config.reference_revision,
            subfolder=config.reference_subfolder,
        )

        self.reward_adapter = None
        if config.reward_adapter_path is not None:
            self.reward_adapter = TurbineRewardLoRAAdapter(
                adapter_name="reward",
                adapter_path=config.reward_adapter_path,
                model=self,
                dtype=config_dtype,
                revision=config.reward_revision,
                subfolder=config.reward_subfolder,
            )

        self.policy_adapter = TurbineLoRAAdapter(
            adapter_name="policy",
            adapter_path=config.reference_adapter_path,
            model=self,
            dtype=config_dtype,
            revision=config.reference_revision,
            subfolder=config.reference_subfolder,
            is_trainable=True,
        )

    @property
    def hidden_size(self) -> int:
        return self.model_args.mlp_hidden_dim

    def init_weights(self, buffer_device: str, buffer_dtype: torch.dtype):
        self.shared_base_model.init_weights(buffer_device=buffer_device, buffer_dtype=buffer_dtype)
        self.reference_adapter.wrapped_model.init_weights(buffer_device=buffer_device, buffer_dtype=buffer_dtype)
        if self.reward_adapter is not None:
            self.reward_adapter.wrapped_model.init_weights(buffer_device=buffer_device, buffer_dtype=buffer_dtype)
        self.policy_adapter.wrapped_model.init_weights(buffer_device=buffer_device, buffer_dtype=buffer_dtype)
