import functools
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any, override

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model
from peft.config import PeftConfig
from peft.utils.other import fsdp_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.nn.parameter import Parameter
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.utils.quantization_config import BitsAndBytesConfig

from afterburner.grpo_config import ModelConfig
from afterburner.training.huggingface.accelerator import HfAccelerator
from afterburner.training.interface import LoRAAdapter, RewardLoRAAdapter, compute_logprobs_from_logits


def _gemma3_fsdp_auto_wrap_policy():
    """Custom FSDP auto wrap policy for Gemma 3 models.

    The default PEFT fsdp_auto_wrap_policy uses get_module_class_from_name to find
    transformer layer classes by searching through module children. For Gemma 3,
    this fails because Gemma3DecoderLayer isn't found in the PEFT-wrapped model tree.

    This function explicitly specifies Gemma3DecoderLayer as the transformer layer to wrap.
    The lambda policy wraps leaf modules with requires_grad weights, which helps FSDP
    handle mixed dtypes (bfloat16/float32) properly.
    """

    def lambda_policy_fn(module: torch.nn.Module) -> bool:
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Gemma3DecoderLayer},
    )
    return functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])


class HfLoRAAdapter(LoRAAdapter):
    """LoRA adapter for HuggingFace models (reference/policy adapters)."""

    @property
    def accelerator(self) -> HfAccelerator:
        return self.model.accelerator

    def gather_lora_params_multigpu(self, pad_index: int = sys.maxsize) -> dict[str, torch.Tensor]:
        params_to_gather = {}
        gather_lists = {}
        gather_handles = []

        # First pass: collect all parameters and prepare for gathering
        for name, param in self.model.named_parameters():
            if self.adapter_name in name:
                # Pad in case of size mismatch across processes
                assert (param.data != pad_index).all(), "Param data should not contain pad index"
                to_gather = self.accelerator.pad_across_processes(param.data, dim=0, pad_index=pad_index)
                assert isinstance(to_gather, torch.Tensor), "to_gather must be a Tensor"
                params_to_gather[name] = to_gather

                # Prepare gather lists on main process
                if self.accelerator.is_main_process:
                    world_size = dist.get_world_size()
                    gather_lists[name] = [torch.zeros_like(to_gather) for _ in range(world_size)]

        # Start all gather operations asynchronously
        for name, to_gather in params_to_gather.items():
            if self.accelerator.is_main_process:
                handle = dist.gather(to_gather, gather_list=gather_lists[name], dst=0, async_op=True)
            else:
                handle = dist.gather(to_gather, gather_list=None, dst=0, async_op=True)
            gather_handles.append((name, handle))

        # Wait for all gather operations to complete
        for name, handle in gather_handles:
            if handle is not None:
                handle.wait()

        # Process gathered results
        lora_state_dict = {}
        if self.accelerator.is_main_process:
            for name, to_gather in params_to_gather.items():
                # Concatenate all gathered tensors
                gathered_param = torch.cat(gather_lists[name], dim=0)
                assert isinstance(gathered_param, torch.Tensor)
                gathered_param = gathered_param[gathered_param != pad_index]
                lora_state_dict[name] = gathered_param.detach().cpu()

        return lora_state_dict

    def gather_lora_params_single_gpu(self) -> dict[str, torch.Tensor]:
        params = {}
        for name, param in self.model.named_parameters():
            if self.adapter_name in name:
                params[name] = param.data
        return params

    def rename_lora_params(self, lora_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rename the LoRA parameters to match the vLLM format."""
        lora_weights = {}
        if self.accelerator.is_main_process:
            for key, value in lora_state_dict.items():
                new_key = key
                if "_fsdp_wrapped_module." in new_key:
                    new_key = new_key.replace("_fsdp_wrapped_module.", "")
                if "module." in new_key:
                    new_key = new_key.replace("module.", "")
                # get the shape from the original model
                shape = self.model.tensor_shapes[new_key]
                # Remove everything before "base_model."
                new_key = "base_model." + new_key.split("base_model.")[-1]
                new_key = new_key.replace(f"{self.adapter_name}.", "")
                lora_weights[new_key] = value.reshape(shape)
        return lora_weights

    def lora_weights(self, pad_index: int = sys.maxsize) -> dict[str, torch.Tensor]:
        """
        Gather the LoRA weights and return a dict suitable for vLLM

        Important: Call this from every process, since we need to gather the
        weights from all processes.

        TODO: confirm that this process is reversible

        Args:
            pad_index (int): the value to pad with. Must not match any valid weight value.

        Outputs:
            lora_weights (Dict[str, torch.Tensor]): a dictionary of the LoRA weights
        """

        if dist.is_initialized():
            lora_state_dict = self.gather_lora_params_multigpu()
        else:
            lora_state_dict = self.gather_lora_params_single_gpu()

        lora_weights = self.rename_lora_params(lora_state_dict)

        return lora_weights

    def save(self, path: Path) -> None:
        """Save the adapter to the specified path."""
        config = self.model.peft_config[self.adapter_name]
        lora_weights = self.lora_weights()
        if self.model.accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
            torch.save(lora_weights, path / "adapter_model.bin")
            config.save_pretrained(str(path))

    def logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Compute log probabilities using this adapter."""
        self.model.local_flops += self.model.floating_point_ops(
            {"input_ids": input_ids, "attention_mask": attention_mask}, self.adapter_name
        )
        self.model.set_adapter([self.adapter_name])

        ctx = nullcontext() if self.adapter_name == "policy" else torch.no_grad()
        with ctx:
            outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        return compute_logprobs_from_logits(logits, input_ids, completion_mask)


class HfRewardLoRAAdapter(HfLoRAAdapter, RewardLoRAAdapter):
    """LoRA adapter for reward model with score head."""

    def save(self, path: Path) -> None:
        raise NotImplementedError("Reward adapters are frozen and should not be saved")

    def _compute_last_token_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Compute the last token hidden states."""
        self.model.local_flops += self.model.floating_point_ops(
            {"input_ids": input_ids, "attention_mask": attention_mask}, "reward"
        )
        self.model.set_adapter(["reward"])
        pad_token_id = self.model.shared_base_model.config.pad_token_id
        assert isinstance(pad_token_id, int)

        with torch.no_grad():
            outputs = self.model.shared_base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use last hidden state at the last token position
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

            # Get last token representation for classification
            sequence_lengths = torch.ne(input_ids, pad_token_id).sum(-1) - 1
            batch_size = input_ids.shape[0]
            last_token_hidden_states = hidden_states[range(batch_size), sequence_lengths].to(
                dtype=self.score_head.weight.dtype
            )  # [batch_size, hidden_size]
        return last_token_hidden_states

    def reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Compute rewards for given texts using the shared model with reward adapter."""
        return super().reward(input_ids, attention_mask, **kwargs)


class HFModel(torch.nn.Module):
    """HuggingFace model wrapper that manages multiple LoRA adapters."""

    def __init__(self, config: ModelConfig, pad_token_id: int, accelerator: HfAccelerator):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.local_flops = 0.0
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage="bfloat16",
            )
            model_kwargs = {"quantization_config": quantization_config, "torch_dtype": torch.bfloat16}
        else:
            model_kwargs = {"torch_dtype": getattr(torch, config.dtype), "device_map": {"": "cpu"}}
        self.shared_base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            attn_implementation=self.config.attn_implementation,
            **model_kwargs,
        )
        self.shared_base_model.config.use_cache = False
        if hasattr(self.shared_base_model.config, "text_config"):
            self.shared_base_model.config.text_config.use_cache = False
        self.shared_base_model.config.pad_token_id = pad_token_id

        if self.config.reference_adapter_path is None:
            # Initialize LoRA from scratch with new weights
            self._init_from_scratch(pad_token_id)
        else:
            # Load existing adapters from paths
            self._init_from_reference_adapter(pad_token_id)

        self.reward_adapter = None
        if self.config.reward_adapter_path is not None:
            self.reward_adapter = HfRewardLoRAAdapter(
                adapter_name="reward",
                adapter_path=self.config.reward_adapter_path,
                revision=self.config.reward_revision,
                subfolder=self.config.reward_subfolder,
                model=self,
                dtype=getattr(torch, self.config.dtype),
            )
            self.add_lora_adapter(self.reward_adapter)

        # Set the number of parameters prior to FSDP wrapping
        self.num_parameters = self.shared_base_model.num_parameters()  # type: ignore

    def _init_from_scratch(self, pad_token_id: int):
        """Initialize model with LoRA adapters from scratch.

        When reference_adapter_path is None, we create fresh LoRA weights and initialize
        both policy and reference adapters with the same weights.
        Since LoRA initializes the update weights (B) to 0, reference remains identical to the base model.
        """
        self.shared_base_model = get_peft_model(self.shared_base_model, self.config.lora_config, adapter_name="policy")
        self.shared_base_model.base_model.add_adapter(adapter_name="reference", adapter_config=self.config.lora_config)

        # Copy policy weights to reference adapter so they start identical
        self.copy_adapter_weights(
            shared_base_model=self.shared_base_model,
            from_adapter="policy",
            to_adapter="reference",
            from_requires_grad=True,
            to_requires_grad=False,
        )
        self.update_tensor_shapes()

        # Create adapter objects (no paths since initialized from scratch)
        self.reference_adapter = HfLoRAAdapter(
            adapter_name="reference",
            adapter_path="",
            model=self,
            dtype=getattr(torch, self.config.dtype),
        )

        self.policy_adapter = HfLoRAAdapter(
            adapter_name="policy",
            adapter_path="",
            model=self,
            dtype=getattr(torch, self.config.dtype),
        )

    def _init_from_reference_adapter(self, pad_token_id: int):
        """Initialize model by loading existing LoRA adapters from paths."""
        # Create and setup adapters
        self.reference_adapter = HfLoRAAdapter(
            adapter_name="reference",
            adapter_path=self.config.reference_adapter_path,
            revision=self.config.reference_revision,
            subfolder=self.config.reference_subfolder,
            model=self,
            dtype=getattr(torch, self.config.dtype),
        )
        # Adding the first adapter will initialize the PEFT model
        self.add_lora_adapter(self.reference_adapter)

        self.policy_adapter = HfLoRAAdapter(
            adapter_name="policy",
            adapter_path=self.config.reference_adapter_path,
            revision=self.config.reference_revision,
            subfolder=self.config.reference_subfolder,
            model=self,
            dtype=getattr(torch, self.config.dtype),
        )
        self.add_lora_adapter(self.policy_adapter)

    @staticmethod
    def copy_adapter_weights(
        shared_base_model: torch.nn.Module,
        from_adapter: str,
        to_adapter: str,
        from_requires_grad: bool = True,
        to_requires_grad: bool = True,
    ):
        named_params = dict(shared_base_model.named_parameters())
        for name, param in named_params.items():
            if from_adapter in name:
                param.requires_grad = from_requires_grad
                ref_name = name.replace(from_adapter, to_adapter)
                assert ref_name in named_params, f"Adapter {to_adapter} not found in named parameters"
                named_params[ref_name].data.copy_(param.data)
            elif to_adapter in name:
                param.requires_grad = to_requires_grad
            else:
                param.requires_grad = False

    @override
    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return self.shared_base_model.named_parameters(prefix, recurse, remove_duplicate)

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.shared_base_model.forward(*args, **kwargs)

    def update_tensor_shapes(self):
        self.tensor_shapes = {k: v.shape for k, v in self.shared_base_model.named_parameters()}

    def _add_lora_adapter(self, adapter: LoRAAdapter) -> None:
        """Add a LoRA adapter to the model."""
        adapter_kwargs = {}
        if hasattr(adapter, "revision") and adapter.revision is not None:
            adapter_kwargs["revision"] = adapter.revision
        if hasattr(adapter, "subfolder") and adapter.subfolder is not None:
            adapter_kwargs["subfolder"] = adapter.subfolder
        if self.config.load_in_4bit:
            kwargs = {"autocast_adapter_dtype": False}
        else:
            kwargs = {"device_map": "cpu", "autocast_adapter_dtype": adapter.adapter_name == "policy"}

        self.shared_base_model.load_adapter(
            adapter.adapter_path,
            adapter_name=adapter.adapter_name,
            adapter_kwargs=adapter_kwargs,
            **kwargs,
        )

        self.update_tensor_shapes()

    def _init_peft_model(self, adapter: HfLoRAAdapter) -> None:
        """Initialize the PEFT model with the first adapter."""
        adapter_kwargs = {}
        if adapter.revision is not None:
            adapter_kwargs["revision"] = adapter.revision
        if adapter.subfolder is not None:
            adapter_kwargs["subfolder"] = adapter.subfolder
        if self.config.load_in_4bit:
            adapter_kwargs["autocast_adapter_dtype"] = False
        else:
            adapter_kwargs["device_map"] = "cpu"
            adapter_kwargs["autocast_adapter_dtype"] = adapter.adapter_name == "policy"

        # Initialize the policy with a copy of the reference adapter
        self.shared_base_model = PeftModel.from_pretrained(
            self.shared_base_model,
            adapter.adapter_path,
            adapter_name=adapter.adapter_name,
            **adapter_kwargs,
        )
        self.update_tensor_shapes()

    def add_lora_adapter(self, adapter: LoRAAdapter) -> None:
        if isinstance(self.shared_base_model, PeftModel):
            self._add_lora_adapter(adapter)
        else:
            self._init_peft_model(adapter)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for the shared base model.

        Copying some necessary prep for peft + gradient checkpointing:
        https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        """
        self.shared_base_model.config.use_cache = False  # type: ignore
        self.shared_base_model.gradient_checkpointing_enable()  # type: ignore
        self.shared_base_model.enable_input_require_grads()  # type: ignore

    @property
    def available_adapters(self) -> list[str]:
        return list(self.shared_base_model.peft_config.keys())

    def set_wrap_policy(self):
        # Activate all adapters before FSDP wrapping
        self.set_adapter(self.available_adapters)
        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        hf_quantizer = getattr(self.shared_base_model, "hf_quantizer", None)
        if fsdp_plugin:
            # Use custom wrap policy for Gemma 3 since PEFT's fsdp_auto_wrap_policy
            # can't find Gemma3DecoderLayer in the PEFT-wrapped model tree.
            if "gemma-3" in self.config.model_name.lower():
                fsdp_plugin.auto_wrap_policy = _gemma3_fsdp_auto_wrap_policy()
            else:
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(  # type: ignore
                    self.shared_base_model
                )
        if fsdp_plugin and hf_quantizer:
            quant_storage = hf_quantizer.quantization_config.bnb_4bit_quant_storage
            if quant_storage.is_floating_point:
                fsdp_plugin.set_mixed_precision(quant_storage, override=True)

    def set_adapter(self, adapter_name: list[str]):
        """Set adapter for the causal LM safely.

        N.B. Do not call self.shared_base_model.set_adapter() directly!
        PeftModel.set_adapter() makes all activated parameters trainable.
        See https://github.com/huggingface/peft/pull/1447 for more details.
        """
        assert isinstance(self.shared_base_model, (PeftModel | FSDP))
        # See this comment for why we set the adapters this way:
        # https://github.com/huggingface/peft/discussions/1315#discussioncomment-7994524
        self.shared_base_model.base_model.set_adapter(adapter_name)  # type: ignore

        self._set_requires_grad()

    def _set_requires_grad(self):
        """Freeze all non-policy parameters to prevent training."""
        assert isinstance(self.shared_base_model, torch.nn.Module)
        for name, param in self.shared_base_model.named_parameters():
            param.requires_grad = "policy" in name

    def floating_point_ops(self, inputs: dict[str, torch.Tensor], adapter_name: str) -> float:
        """Approximate flops ignoring the score head."""
        # The rule of thumb is that FLOPs are 2nT for a forward pass and 4nT for a backward pass
        # for a model with n parameters on T tokens.
        # This gives the following factor assuming we only backpropagate through the policy adapter
        factor = 6 if adapter_name == "policy" else 2
        return factor * self.shared_base_model.estimate_tokens(inputs) * self.num_parameters  # type: ignore

    @property
    def device(self):
        return self.shared_base_model.device

    @override
    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        return self.shared_base_model.peft_config

    def move_to_gpu(self) -> None:
        self.set_wrap_policy()
        if self.reward_adapter is not None:
            (self.shared_base_model, self.reward_adapter.score_head) = self.accelerator.prepare(
                self.shared_base_model,
                self.reward_adapter.score_head,
            )
        else:
            self.shared_base_model = self.accelerator.prepare(self.shared_base_model)

    @property
    def hidden_size(self) -> int:
        return self.shared_base_model.config.hidden_size
