import copy
import json
import os
import tempfile
from contextlib import contextmanager
from typing import no_type_check

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from obfuscation_atlas.detectors.probe_archs import (
    LinearProbe,
    Probe,
    SequenceAggregator,
    mean_aggregator,
    multimax_aggregator,
)
from obfuscation_atlas.utils.activations import get_hidden_size
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper
from obfuscation_atlas.utils.masking import (
    compute_mask,
)
from obfuscation_atlas.utils.metrics import get_detector_metrics


class ProbeDetector(nn.Module):
    """
    Probe-based detector for transformer activations.

    Supports both single-head (nhead=1) and multi-head (nhead>1) probes.
    When nhead>1, can optionally enforce orthogonality between probe directions.

    All probes output (batch, seq, nhead) logits.
    """

    def __init__(
        self,
        layers: list[int | str],
        encoder: LanguageModelWrapper,
        obfuscate_over: str,
        nhead: int = 1,
        sequence_aggregator: SequenceAggregator | None = None,
        layer_aggregation: str = "mean",
        probe_on_trigger: bool = False,
        pretrained_probes: dict[str | int, Probe] | None = None,
        accelerator=None,
        normalize_input: str = "none",
        use_platt_scaling: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        assert len(layers) > 0, "Must have at least one layer"
        self.layers = [str(layer) for layer in layers]
        self.obfuscate_over = obfuscate_over
        self.nhead = nhead
        self.layer_aggregation = layer_aggregation
        self.probe_on_trigger = probe_on_trigger
        self.input_dim = get_hidden_size(encoder.model.config)
        self.accelerator = accelerator

        if sequence_aggregator is None:
            if nhead > 1:
                sequence_aggregator = multimax_aggregator()
            else:
                sequence_aggregator = mean_aggregator()
        self.sequence_aggregator = sequence_aggregator

        # Initialize probes
        if pretrained_probes is not None:
            probes_dict = {
                layer: pretrained_probes.get(layer, pretrained_probes.get(int(layer))) for layer in self.layers
            }
        else:
            probes_dict = {
                layer: LinearProbe(
                    self.input_dim,
                    nhead=nhead,
                    normalize_input=normalize_input,
                )
                for layer in self.layers
            }
        self.probes = nn.ModuleDict(probes_dict)  # type: ignore

        # Platt scaling is stored in each probe via probe.platt_A, probe.platt_B
        self._use_platt_scaling = use_platt_scaling

    @property
    def is_multihead(self) -> bool:
        return self.nhead > 1

    def forward(
        self,
        activations: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass returning per-layer logits.

        Args:
            activations: Dict mapping layer name to activations (batch, seq, d_model)
            mask: Valid token mask (batch, seq), True = valid (passed to attention probes)

        Returns:
            Dict mapping layer name to logits (batch, seq, nhead)
        """
        outputs = {}
        for layer_name in self.layers:
            outputs[layer_name] = self.probes[layer_name](
                activations[layer_name],
                padding_mask=mask,
            )
        return outputs

    def _compute_layerwise_scores(
        self,
        samples,
        activations: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,  # Add this parameter
        log_probs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute per-layer scores for inference."""
        act_fn = F.logsigmoid if log_probs else torch.sigmoid

        logits = self.forward(activations)

        scores = {}
        for layer_name, layer_logits in logits.items():
            # Use provided mask or compute from activations
            if mask is None:
                layer_mask = activations[layer_name].abs().sum(dim=-1) > 0
            else:
                layer_mask = mask

            if self.sequence_aggregator.needs_q:
                q, v = self.probes[layer_name].forward_qv(
                    activations[layer_name],
                    padding_mask=layer_mask,
                )
            else:
                q, v = None, layer_logits

            aggregated = self.sequence_aggregator(v, layer_mask, q=q)

            if self._use_platt_scaling:
                probe = self.probes[layer_name]
                aggregated = probe.platt_A * aggregated + probe.platt_B

            scores[layer_name] = act_fn(aggregated)

        return scores

    def enable_platt_scaling(self) -> None:
        self._use_platt_scaling = True

    def disable_platt_scaling(self) -> None:
        self._use_platt_scaling = False

    def _aggregate_scores(self, layerwise_scores: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate scores across layers."""
        stacked = torch.stack(list(layerwise_scores.values()), dim=0)
        if self.layer_aggregation == "mean":
            return stacked.mean(dim=0)
        elif self.layer_aggregation == "max":
            return stacked.max(dim=0).values
        elif self.layer_aggregation == "min":
            return stacked.min(dim=0).values
        else:
            raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    # =========================================================================
    # Orthogonality (for nhead > 1)
    # =========================================================================

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss across all layers."""
        if self.nhead <= 1:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer_name, probe in self.probes.items():
            total_loss = total_loss + self._compute_probe_orthogonality_loss(probe)

        return total_loss / len(self.layers)

    def _compute_probe_orthogonality_loss(self, probe: LinearProbe) -> torch.Tensor:
        """Compute orthogonality loss for a single probe."""
        weight = probe.linear.weight  # (nhead, d_model)

        if self.accelerator is not None:
            weight = self._gather_fsdp_weight(weight)

        normalized = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
        gram = torch.mm(normalized, normalized.t()).abs()
        identity = torch.eye(self.nhead, device=weight.device, dtype=weight.dtype)
        off_diag_sum = (gram - identity).abs().sum()
        num_pairs = self.nhead * (self.nhead - 1)

        return off_diag_sum / max(num_pairs, 1)

    def _gather_fsdp_weight(self, local_weight: torch.Tensor) -> torch.Tensor:
        """Gather sharded weights from all FSDP ranks."""
        if self.accelerator is None:
            return local_weight

        local_size = torch.tensor([local_weight.numel()], device=local_weight.device)
        all_sizes = self.accelerator.gather(local_size)
        max_size = int(all_sizes.max().item())

        flat_weight = local_weight.flatten()
        if flat_weight.numel() < max_size:
            pad_size = max_size - flat_weight.numel()
            padded = torch.cat(
                [flat_weight, torch.zeros(pad_size, device=local_weight.device, dtype=local_weight.dtype)]
            )
        else:
            padded = flat_weight

        gathered = self.accelerator.gather(padded)

        chunks = []
        for i in range(self.accelerator.num_processes):
            start = i * max_size
            end = start + int(all_sizes[i].item())
            chunks.append(gathered[start:end])

        full_weight = torch.cat(chunks).view(self.nhead, self.input_dim)
        return full_weight

    def get_weight_orthogonality(self) -> dict[str, float]:
        """Return orthogonality metrics for monitoring."""
        if self.nhead <= 1:
            return {}

        metrics = {}
        for layer_name, probe in self.probes.items():
            loss = self._compute_probe_orthogonality_loss(probe)
            metrics[f"orthogonality/{layer_name}"] = -loss.item()
        return metrics

    def get_probe_weight_norms(self) -> dict[str, float]:
        if self.accelerator is not None:
            return {}

        norms = {}
        for layer_name, probe in self.probes.items():
            if hasattr(probe, "linear"):
                weight_norm = torch.norm(probe.linear.weight, p=2).item()
                norms[f"probe_weight_norm/{layer_name}"] = weight_norm
        return norms

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Loss computation (token-level, matching original RL training)
# =============================================================================


def compute_probe_detector_loss(
    detector: ProbeDetector,
    positive_acts: dict[str, torch.Tensor],
    negative_acts: dict[str, torch.Tensor],
    orthogonality_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute token-level training loss for ProbeDetector.

    Each valid token is treated as a separate training example.
    """
    detector.train()

    accelerator = detector.accelerator
    if accelerator is not None:
        device = accelerator.device
    else:
        device = next(detector.parameters()).device

    first_layer = detector.layers[0]
    hidden_dim = positive_acts[first_layer].shape[-1]

    # Flatten and compute masks for valid tokens
    pos_flat = positive_acts[first_layer].view(-1, hidden_dim)
    neg_flat = negative_acts[first_layer].view(-1, hidden_dim)
    pos_mask = pos_flat.abs().sum(dim=-1) > 0
    neg_mask = neg_flat.abs().sum(dim=-1) > 0

    pos_labels = torch.ones(pos_mask.sum(), device=device)
    neg_labels = torch.zeros(neg_mask.sum(), device=device)

    assert len(pos_labels) + len(neg_labels) > 0, "No non-zero vectors found"

    positive_acts = {k: v.to(device) for k, v in positive_acts.items()}
    negative_acts = {k: v.to(device) for k, v in negative_acts.items()}
    pos_mask = pos_mask.to(device)
    neg_mask = neg_mask.to(device)

    pos_outputs = detector.forward(positive_acts)
    neg_outputs = detector.forward(negative_acts)

    total_loss = torch.tensor(0.0, device=device)
    loss_dict = {}
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    for layer_name in detector.layers:
        # Sum over heads: (batch, seq, nhead) -> (batch, seq)
        pos_layer_logits = pos_outputs[layer_name].sum(dim=-1)
        neg_layer_logits = neg_outputs[layer_name].sum(dim=-1)

        # Flatten and mask
        pos_logits_flat = pos_layer_logits.view(-1)[pos_mask]
        neg_logits_flat = neg_layer_logits.view(-1)[neg_mask]

        pos_loss = loss_fn(pos_logits_flat, pos_labels)
        neg_loss = loss_fn(neg_logits_flat, neg_labels)
        layer_loss = (pos_loss + neg_loss) / (len(pos_labels) + len(neg_labels))

        total_loss = total_loss + layer_loss

        if accelerator is None or accelerator.is_main_process:
            loss_dict[f"bce_loss/{layer_name}"] = layer_loss.item()

    if detector.is_multihead and orthogonality_weight > 0:
        ortho_loss = detector.compute_orthogonality_loss()
        total_loss = total_loss + orthogonality_weight * ortho_loss
        if accelerator is None or accelerator.is_main_process:
            loss_dict["ortho_loss"] = ortho_loss.item()

    if accelerator is None or accelerator.is_main_process:
        loss_dict["total"] = total_loss.item() / len(detector.layers)

    return total_loss, loss_dict


# =============================================================================
# Backward compatibility
# =============================================================================


def compute_layerwise_scores(
    detector: ProbeDetector,
    activations: dict[str, torch.Tensor],
    log_probs: bool = False,
) -> dict[str, torch.Tensor]:
    """Backward-compatible wrapper."""
    return detector._compute_layerwise_scores(None, activations, log_probs)


class GradientPreserver:
    """Utility class to save and restore gradients for models, especially useful with FSDP."""

    def __init__(self):
        self.parameters = {}

    def save_gradients(self, model: nn.Module, clear_model=True, accumulate=True):
        """
        Save gradients from all parameters in a model.

        Args:
            model: The model whose gradients to save
            clear: Whether to clear gradients in the model after saving

        Returns:
            Dictionary mapping parameter names to their gradients
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name in self.parameters and accumulate:
                    self.parameters[name] += param.grad.clone()
                else:
                    self.parameters[name] = param.grad.clone()
                if clear_model:
                    param.grad = None

    def restore_gradients(self, model: nn.Module, reset=True) -> int:
        """
        Restore saved gradients to a model's parameters.

        Args:
            model: The model whose gradients to restore
            saved_gradients: Dictionary mapping parameter names to gradients
            reset: Reset the accumulated gradients back to None.

        Returns:
            Number of gradients restored
        """
        restored_count = 0
        model_params = dict(model.named_parameters())
        for name, saved_grad in self.parameters.items():
            if name in model_params:
                param = model_params[name]
                # Convert to match param dtype (prevents FSDP flat_param.grad BF16 vs orig_param FP32 mismatch)
                if saved_grad.dtype != param.dtype:
                    saved_grad = saved_grad.to(param.dtype)
                model_params[name].grad = saved_grad
                restored_count += 1
            else:
                raise ValueError(f"Parameter '{name}' not found in model during gradient restoration")
        if reset:
            self.parameters = {}
        return restored_count

    @staticmethod
    def clear_gradients(model: nn.Module) -> None:
        """Clear all gradients in a model (set to None, not zero)."""
        for param in model.parameters():
            param.grad = None


class DetectorObfuscator:
    def __init__(
        self,
        detector: ProbeDetector,
        detector_lr: float = 5e-3,
        detector_betas: tuple[float, float] = (0.9, 0.95),
        detector_weight_decay: float = 0.01,
        mahalanobis_shrinkage: float = 0.0,
        detector_ignore_steps: int = 100,
        detector_name: str = "",
        scheduler_fn=None,
        device: str = "cuda",
        accelerator=None,
        train_detector_every_n_steps: int | float = 1,
        n_grad_accum=1,
        orthogonality_weight: float = 0.5,
    ):
        self.detector = detector
        self.detector_type = type(detector)
        self.detector_lr = detector_lr
        self.detector_betas = detector_betas
        self.detector_weight_decay = detector_weight_decay
        self.initialized_detector_variables = False
        self.mahalanobis_shrinkage = mahalanobis_shrinkage
        self.detector_ignore_steps = detector_ignore_steps
        self.device = device
        self._step = 0
        self.detector_name = detector_name
        self.accelerator = accelerator
        self.detector_opt = None
        self.scheduler_fn = scheduler_fn
        self.train_detector_every_n_steps = train_detector_every_n_steps
        self.n_grad_accum = n_grad_accum
        self.orthogonality_weight = orthogonality_weight
        self.gradient_preserver = None
        if self.n_grad_accum > 1:
            self.gradient_preserver = GradientPreserver()

    def _get_device(self):
        """Get appropriate device based on context"""
        if self.accelerator is not None:
            return self.accelerator.device
        return self.device

    @no_type_check
    def _init_detector_variables(self):
        """Initialize detector variables and prepare with accelerator if needed"""

        if issubclass(self.detector_type, ProbeDetector):
            self._prepare_trainable_detector()
        else:
            raise NotImplementedError(f"Unsupported detector type: {type(self.detector)}")
        self.initialized_detector_variables = True

    @no_type_check
    def _prepare_trainable_detector(self):
        """Prepare a trainable model/component with optimizer and accelerator"""
        optimizer_kwargs = {
            "lr": self.detector_lr,
            "betas": self.detector_betas,
            "weight_decay": self.detector_weight_decay,
        }
        if self.accelerator is not None:
            self.detector = self.accelerator.prepare(self.detector)
        else:
            self.detector = self.detector.to(self.device)
        optimizer = torch.optim.AdamW(self.detector.parameters(), **optimizer_kwargs)
        if self.accelerator is not None:
            optimizer = self.accelerator.prepare(optimizer)
        self.detector_opt = optimizer
        self.scheduler = self.scheduler_fn(optimizer) if self.scheduler_fn is not None else None

    @no_type_check
    def state_dict(self, checkpoint_dir: str = None) -> dict:
        """
        Save state dict with accelerator-aware checkpoint handling.

        Args:
            checkpoint_dir: Base directory for saving checkpoints. If None, uses temp directory.
        """
        # Determine checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = tempfile.mkdtemp(prefix="detector_obfuscator_")
        else:
            checkpoint_dir = os.path.abspath(checkpoint_dir)

        # Create unique subdirectory for this detector's state
        detector_checkpoint_dir = os.path.join(checkpoint_dir, f"detector_{self.detector_name}_{self._step}")
        os.makedirs(detector_checkpoint_dir, exist_ok=True)

        # Prepare base state dictionary
        state = {
            "detector_checkpoint_dir": detector_checkpoint_dir,
            "initialized_detector_variables": self.initialized_detector_variables,
            "step": self._step,
            "detector_lr": self.detector_lr,
            "detector_betas": self.detector_betas,
            "detector_weight_decay": self.detector_weight_decay,
            "detector_ignore_steps": self.detector_ignore_steps,
            "mahalanobis_shrinkage": self.mahalanobis_shrinkage,
            "detector_name": self.detector_name,
            "detector_type": self.detector_type.__name__,
            "has_optimizer": hasattr(self, "detector_opt") and self.detector_opt is not None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

        # Handle different detector types
        if hasattr(self.detector, "_get_trained_variables"):
            # Save covariance-based detector variables
            try:
                detector_variables = self.detector._get_trained_variables()
                var_path = os.path.join(detector_checkpoint_dir, "detector_variables.pt")
                torch.save(detector_variables, var_path)
                state["has_detector_variables"] = True
            except Exception as exc:
                print(f"Failed to save detector variables: {exc}")
                state["has_detector_variables"] = False

        elif issubclass(self.detector_type, nn.Module):
            # For trainable detectors, use accelerator if available
            if self.accelerator is not None:
                # Register once if not already done
                model_to_save = self.detector
                if isinstance(model_to_save, FSDP):
                    # Configure FSDP state dict for both model and optimizer
                    with FSDP.state_dict_type(
                        model_to_save,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    ):
                        # Save model state
                        model_state = model_to_save.state_dict()
                        torch.save(model_state, os.path.join(detector_checkpoint_dir, "pytorch_model.bin"))

                        # Save optimizer state if it exists
                        if self.detector_opt:
                            opt_state = FSDP.optim_state_dict(model_to_save, self.detector_opt)
                            torch.save(opt_state, os.path.join(detector_checkpoint_dir, "optimizer.bin"))

                    state["used_fsdp_state_dict"] = True
            else:
                # Manual save for non-accelerated case
                self._save_without_accelerator(detector_checkpoint_dir)
                state["saved_with_accelerator"] = False

        # Save metadata
        metadata_path = os.path.join(detector_checkpoint_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(state, f, indent=2)

        return state

    def _save_without_accelerator(self, checkpoint_dir: str):
        """Save model and optimizer without accelerator (standard PyTorch)."""
        # Save model state
        assert isinstance(self.detector, nn.Module), (
            f"Detector must be an instance of nn.Module, got {type(self.detector)}"
        )
        torch.save(self.detector.state_dict(), os.path.join(checkpoint_dir, "detector.pt"))

        # Save optimizer state if exists
        if hasattr(self, "detector_opt") and self.detector_opt is not None:
            torch.save(self.detector_opt.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

    @no_type_check
    def load_state_dict(self, state: dict):
        """
        Load state dict with accelerator-aware checkpoint handling.

        Args:
            state: State dictionary from state_dict()
        """
        # Restore hyperparameters
        self.detector_lr = state["detector_lr"]
        self.detector_betas = state["detector_betas"]
        self.detector_weight_decay = state["detector_weight_decay"]
        self.detector_ignore_steps = state["detector_ignore_steps"]
        self.mahalanobis_shrinkage = state["mahalanobis_shrinkage"]
        self.detector_name = state.get("detector_name", self.detector_name)
        self._step = state.get("step", 0)

        checkpoint_dir = state["detector_checkpoint_dir"]

        # Load detector variables based on type
        if state.get("has_detector_variables", False):
            # Covariance-based detector
            var_path = os.path.join(checkpoint_dir, "detector_variables.pt")
            if os.path.exists(var_path):
                try:
                    detector_variables = torch.load(var_path, map_location=self._get_device())
                    self.detector._set_trained_variables(detector_variables)
                except Exception as exc:
                    print(f"Failed to load detector variables: {exc}")

        elif os.path.exists(checkpoint_dir):
            # Trainable detector
            if state.get("used_fsdp_state_dict", False):
                # New FSDP StateDictType loading

                # Initialize detector if not already done
                if not self.initialized_detector_variables:
                    self._init_detector_variables(None)

                model_to_load = self.detector

                # Load model state
                model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                if os.path.exists(model_path):
                    model_state = torch.load(model_path, map_location=self._get_device())

                    if isinstance(model_to_load, FSDP):
                        # Load with FSDP context
                        with FSDP.state_dict_type(
                            model_to_load,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                        ):
                            model_to_load.load_state_dict(model_state)
                    else:
                        # Regular loading if not FSDP wrapped
                        model_to_load.load_state_dict(model_state)

                    # Load optimizer state using FSDP's methods
                    if state.get("has_optimizer", False):
                        optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
                        if os.path.exists(optimizer_path) and self.detector_opt is not None:
                            optimizer_state = torch.load(optimizer_path, map_location=self._get_device())

                            if isinstance(model_to_load, FSDP):
                                # Load using FSDP's optim_state_dict_to_load
                                with FSDP.state_dict_type(
                                    model_to_load,
                                    StateDictType.FULL_STATE_DICT,
                                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                    FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                ):
                                    # Convert the loaded state to sharded format
                                    optim_state_dict = FSDP.optim_state_dict_to_load(
                                        model_to_load, self.detector_opt, optimizer_state
                                    )
                                    self.detector_opt.load_state_dict(optim_state_dict)
            else:
                # First load model states
                self._load_without_accelerator(checkpoint_dir)

                # Then create optimizer if needed
                if state.get("has_optimizer", False):
                    if not self.initialized_detector_variables:
                        self._init_detector_variables(None)

                    # Load optimizer state if it exists
                    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                    if os.path.exists(optimizer_path) and self.detector_opt is not None:
                        optimizer_state = torch.load(optimizer_path, map_location=self._get_device())
                        try:
                            self.detector_opt.load_state_dict(optimizer_state)
                        except Exception as e:
                            print(f"Warning: Could not load optimizer state: {e}")

        # Mark as initialized
        self.initialized_detector_variables = state.get("initialized_detector_variables", False)
        if state.get("scheduler_state_dict", None) is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

    def _load_without_accelerator(self, checkpoint_dir: str):
        """Load model and optimizer without accelerator."""
        device = self._get_device()
        assert isinstance(self.detector, nn.Module), (
            f"Detector must be an instance of nn.Module, got {type(self.detector)}"
        )
        detector_path = os.path.join(checkpoint_dir, "detector.pt")
        if os.path.exists(detector_path):
            self.detector.load_state_dict(torch.load(detector_path, map_location=device))

    def _compute_detector_loss(
        self, positive_activations: dict[str, torch.Tensor], negative_activations: dict[str, torch.Tensor]
    ):
        # For trainable detectors, use autocast
        with self._autocast_context():
            loss, layer_losses = compute_probe_detector_loss(
                self.detector,
                positive_activations,
                negative_activations,
                orthogonality_weight=self.orthogonality_weight,
            )
        return loss, layer_losses

    @contextmanager
    def _autocast_context(self):
        """Context manager for autocast with accelerator support"""
        if self.accelerator is not None:
            # Use accelerator's autocast
            with self.accelerator.autocast():
                yield
        else:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                yield

    def _get_scores(self, activations, mask=None, batch_size=None, log_probs=False, reduce=True):
        """Get detection scores from activations."""
        scores = self.detector._compute_layerwise_scores(None, activations, mask=mask, log_probs=log_probs)
        aggregate_score = self.detector._aggregate_scores(scores)
        if reduce:
            aggregate_score = aggregate_score.mean()
        return aggregate_score, scores

    def _compute_detector_metrics(
        self,
        backdoor_layer_scores: dict[str, torch.Tensor],  # {layer: (batch_pos,)}
        normal_layer_scores: dict[str, torch.Tensor],  # {layer: (batch_neg,)}
        detector_name: str,
    ):
        """Compute evaluation metrics for the detector.

        When using distributed training, gathers scores from all GPUs to compute global metrics.
        """

        def gather_scores_if_distributed(
            scores_dict: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            """Gather scores from all GPUs if using distributed training."""
            if self.accelerator is None or not self.accelerator.use_distributed:
                return scores_dict

            gathered_scores = {}
            for layer_key, scores in scores_dict.items():
                # scores: (batch,)
                scores = scores.contiguous()
                padded = self.accelerator.pad_across_processes(scores, dim=0)
                gathered_scores[layer_key] = self.accelerator.gather(padded)

            return gathered_scores

        # Gather scores from all GPUs
        backdoor_layer_scores_global = gather_scores_if_distributed(backdoor_layer_scores)
        normal_layer_scores_global = gather_scores_if_distributed(normal_layer_scores)

        # Only compute metrics on main process to avoid duplication
        if self.accelerator is not None and self.accelerator.use_distributed:
            if not self.accelerator.is_main_process:
                return {}

        # Concatenate positive and negative scores per layer
        layer_scores = {}
        labels = None

        for layer_key in backdoor_layer_scores_global.keys():
            # backdoor_scores: (batch_pos,) -> numpy
            backdoor_scores_np = backdoor_layer_scores_global[layer_key].detach().cpu().float().numpy()
            # normal_scores: (batch_neg,) -> numpy
            normal_scores_np = normal_layer_scores_global[layer_key].detach().cpu().float().numpy()

            layer_scores[layer_key] = np.concatenate([backdoor_scores_np, normal_scores_np])
            if labels is None:
                labels = np.concatenate(
                    [
                        np.ones(len(backdoor_scores_np)),
                        np.zeros(len(normal_scores_np)),
                    ]
                )

        # Compute metrics
        metrics_dict, _ = get_detector_metrics(
            scores=layer_scores,
            labels=labels,
            false_positive_rate=0.01,
            disable_plots=True,
            flat_dict=True,
            exclude_scores=True,
        )

        # Format metrics with detector name
        ret_metrics = {}
        for layer_key, layer_metrics in metrics_dict.items():
            for k, v in layer_metrics.items():
                ret_metrics[f"{k}/{detector_name}/train/{layer_key}"] = v

        return ret_metrics

    def mask_tensor(self, activations: torch.Tensor, mask: torch.Tensor, detach: bool = True) -> torch.Tensor:
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
        return activations.detach().clone() * mask.unsqueeze(-1) if detach else activations * mask.unsqueeze(-1)

    def prepare_activations(
        self,
        hidden_states: list[torch.Tensor],  # list of (batch, seq, d_model)
        mask: torch.Tensor,  # (batch, seq) bool
        activation_matching_layers: list[int],
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        return {
            f"{layer}": self.mask_tensor(hidden_states[layer + 1], mask, detach) for layer in activation_matching_layers
        }

    def compute_obfuscation_loss(
        self,
        task_hidden_mask_tuple,
        activation_matching_layers,
        supervised_positive_hidden_mask_tuple=None,
        supervised_negative_hidden_mask_tuple=None,
        reduce=True,
        log_probs=False,
        task_labels: torch.Tensor | None = None,  # (batch,) int, 0 or 1
        eval_only: bool = False,
    ):
        if not eval_only:
            self._step += 1

        log_dict = {}
        detector_name = self.detector_name if self.detector_name else self.detector.__class__.__name__

        # task_hidden_states: list of (batch, seq, d_model)
        # task_mask: (batch, seq) bool
        task_hidden_states, task_mask = task_hidden_mask_tuple

        assert task_hidden_states[0].shape[1] == task_mask.shape[1], (
            f"{task_hidden_states[0].shape[1]} != {task_mask.shape[1]}"
        )
        if eval_only:
            supervised_positive_activations = None
            supervised_negative_activations = None
        else:
            assert (
                supervised_positive_hidden_mask_tuple is not None and supervised_negative_hidden_mask_tuple is not None
            ), "Supervised positive data is required for non-AnomalyDetector detectors"
            supervised_positive_hidden_states, supervised_positive_mask = supervised_positive_hidden_mask_tuple
            supervised_negative_hidden_states, supervised_negative_mask = supervised_negative_hidden_mask_tuple
            # supervised activations shape: {layer: (batch_pos, seq, d_model)}
            supervised_positive_activations = self.prepare_activations(
                supervised_positive_hidden_states,
                supervised_positive_mask,
                activation_matching_layers,
                detach=True,
            )
            supervised_negative_activations = self.prepare_activations(
                supervised_negative_hidden_states,
                supervised_negative_mask,
                activation_matching_layers,
                detach=True,
            )

        if not self.initialized_detector_variables:
            self._init_detector_variables()
            self.initialized_detector_variables = True

        if isinstance(self.detector, nn.Module) and not eval_only:
            self.detector.train()
        if eval_only:
            num_repeats = 0
        elif self.train_detector_every_n_steps < 1:
            num_repeats = int(1 / self.train_detector_every_n_steps)
        elif self.train_detector_every_n_steps >= 1 and self._step % self.train_detector_every_n_steps == 0:
            num_repeats = 1
        else:
            num_repeats = 0
        layer_losses = {}
        detector_loss = None
        for _ in range(num_repeats):
            detector_loss, layer_losses = self._compute_detector_loss(
                supervised_positive_activations,
                supervised_negative_activations,
            )
            if detector_loss is not None:
                # TODO: decide if we want to accelerate.accumulate here.
                # For RL, this will be called inside an accelerate.accumulate context for LLM,
                # so a second accumulate context might break things.
                detector_loss = detector_loss / self.n_grad_accum
                assert self.detector_opt is not None
                # Clear detector grads from RL training's backward before detector's own backward
                # gradient accumulation done manually using gradient_preserver.
                self.detector_opt.zero_grad()

                if self.accelerator is not None and self._step % self.n_grad_accum == 0:
                    self.accelerator.backward(detector_loss)
                else:
                    detector_loss.backward()  # type: ignore
                if self.gradient_preserver is not None:
                    self.gradient_preserver.save_gradients(self.detector)  # type: ignore
                if self._step % self.n_grad_accum == 0:
                    if self.gradient_preserver is not None:
                        self.gradient_preserver.restore_gradients(self.detector)  # type: ignore
                    self.detector_opt.step()
                    # Clear gradients after step to prevent FSDP dtype mismatch
                    # Without this, FSDP's flat_param.grad (BF16) conflicts with orig_param (FP32)
                    # during the next forward pass, causing: "attempting to assign a gradient
                    # with dtype 'c10::BFloat16' to a tensor with dtype 'float'"
                    self.detector_opt.zero_grad()
        if detector_loss is not None:
            for layer_key, layer_loss in layer_losses.items():
                layer_loss = layer_loss.item() if isinstance(layer_loss, torch.Tensor) else layer_loss
                log_dict[f"detector_train_loss/{detector_name}/{layer_key}"] = layer_loss
            log_dict[f"detector_train_loss/{detector_name}/total"] = detector_loss.item()  # type: ignore
        if not eval_only and self.scheduler is not None and self._step % self.n_grad_accum == 0:
            # detector lr matches model lr so we step once per model step
            self.scheduler.step()
            log_dict["detector_learning_rate"] = self.scheduler.get_last_lr()[0]

        # Track probe weight norms if using ProbeDetector
        if not eval_only and isinstance(self.detector, ProbeDetector):
            if hasattr(self.detector, "get_probe_weight_norms"):
                probe_norms = self.detector.get_probe_weight_norms()
                log_dict.update(probe_norms)

        # task_activations: {layer: (batch, seq, d_model)}
        task_activations = self.prepare_activations(
            task_hidden_states, task_mask, activation_matching_layers, detach=False
        )

        # set detector to eval mode
        if isinstance(self.detector, nn.Module):
            self.detector.eval()

        with self._autocast_context():
            # task_score: scalar (if reduce=True) or (batch,)
            # task_layer_scores: {layer: (batch,)}
            task_score, task_layer_scores = self._get_scores(
                task_activations, mask=task_mask, log_probs=log_probs, reduce=reduce
            )

        # Compute evaluation metrics comparing tasked vs normal scores
        if (
            not eval_only
            and task_mask.numel() > 0
            and task_labels is not None
            and 0 < task_labels.sum().item() < task_labels.numel()
        ):
            with torch.no_grad():
                # Split task_layer_scores according to task_labels
                # positive_label_mask, negative_label_mask: (batch,) bool
                positive_label_mask = task_labels == 1
                negative_label_mask = task_labels == 0
                task_positive_scores = {}
                task_negative_scores = {}
                for layer_key, scores in task_layer_scores.items():
                    # scores is (batch,), select by label
                    task_positive_scores[layer_key] = scores[positive_label_mask]
                    task_negative_scores[layer_key] = scores[negative_label_mask]

                detector_metrics = self._compute_detector_metrics(
                    task_positive_scores,
                    task_negative_scores,
                    detector_name,
                )
            log_dict.update(detector_metrics)
        if self._step < self.detector_ignore_steps:
            # Create zero tensor on correct device
            # Get device from actual tensors
            device = task_hidden_states[0].device
            return torch.tensor(0.0, device=device, requires_grad=True), log_dict
        else:
            # Else, return the detector scores of the tasked activations
            layer_losses = {}
            for layer_key, layer_score in task_layer_scores.items():
                layer_idx = layer_key.replace("layer", "")
                formatted_key = f"detector_score_per_layer/{detector_name}/{layer_idx}"
                if reduce:
                    log_dict[formatted_key] = layer_score.mean().item()
                else:
                    log_dict[formatted_key] = layer_score.detach()

            # Add probe norms and detector metrics to layer losses for logging
            log_dict.update(layer_losses)
            if reduce:
                return task_score, log_dict
            else:
                return task_score, log_dict


def obfuscation_fn_wrapper(obfuscation_fn, obfuscate_over):
    def obfuscation_fn_wrapper_fn(
        backdoor_data,
        activation_matching_layers,
        supervised_positive_data=None,
        supervised_negative_data=None,
        task_labels=None,
        eval_only=False,
    ):
        backdoor_output, backdoor_tokens, backdoor_target_mask, backdoor_prompt_mask = backdoor_data

        # Compute mask based on obfuscate_over
        backdoor_mask = compute_mask(
            backdoor_tokens.shape[1],
            backdoor_prompt_mask,
            backdoor_target_mask,
            obfuscate_over,
        )

        processed_supervised_positive_data = None
        processed_supervised_negative_data = None
        if supervised_positive_data is not None:
            (
                supervised_positive_output,
                supervised_positive_tokens,
                supervised_positive_target_mask,
                supervised_positive_prompt_mask,
            ) = supervised_positive_data
            supervised_positive_mask = compute_mask(
                supervised_positive_tokens.shape[1],
                supervised_positive_prompt_mask,
                supervised_positive_target_mask,
                obfuscate_over,
            )
            processed_supervised_positive_data = (
                supervised_positive_output.hidden_states,
                supervised_positive_mask,
            )
        if supervised_negative_data is not None:
            (
                supervised_negative_output,
                supervised_negative_tokens,
                supervised_negative_target_mask,
                supervised_negative_prompt_mask,
            ) = supervised_negative_data
            supervised_negative_mask = compute_mask(
                supervised_negative_tokens.shape[1],
                supervised_negative_prompt_mask,
                supervised_negative_target_mask,
                obfuscate_over,
            )
            processed_supervised_negative_data = (
                supervised_negative_output.hidden_states,
                supervised_negative_mask,
            )
        # Compute obfuscation loss
        result = obfuscation_fn(
            (backdoor_output.hidden_states, backdoor_mask),
            activation_matching_layers,
            supervised_positive_hidden_mask_tuple=processed_supervised_positive_data,
            supervised_negative_hidden_mask_tuple=processed_supervised_negative_data,
            task_labels=task_labels,
            eval_only=eval_only,
        )

        # Handle the case where obfuscation_fn returns both loss and layer losses
        if isinstance(result, tuple) and len(result) == 2:
            loss, layer_losses = result
            # Add prefix to layer loss keys to identify which obfuscation function they came from
            prefixed_layer_losses = {f"{obfuscation_fn.__name__}_{k}": v for k, v in layer_losses.items()}
            # Return both the loss and the prefixed layer losses
            return loss, prefixed_layer_losses
        else:
            # If only loss is returned (backward compatibility)
            return result

    # Set the name of the wrapped function to include obfuscate_over
    obfuscation_fn_wrapper_fn.__name__ = f"{obfuscation_fn.__name__}_{obfuscate_over}"
    return obfuscation_fn_wrapper_fn


def rl_obfuscation_fn_wrapper(obfuscation_fn, obfuscate_over):
    """Wrapper for obfuscation functions used in RL training.

    Args:
        tokenizer: Tokenizer for the model
        obfuscation_fn: DetectorObfuscator.compute_obfuscation_loss
        obfuscate_over: What part of the sequence to obfuscate over ("full_prompt", "target", etc.)

    Returns:
        Obfuscation loss function wrapper used in RL training
    """

    def obfuscation_fn_wrapper_fn(
        backdoor_data,
        activation_matching_layers,
        supervised_positive_data=None,
        supervised_negative_data=None,
        task_labels=None,
        eval_only=False,
    ):
        backdoor_hidden_states, backdoor_completion_mask, backdoor_attention_mask = backdoor_data

        # Compute mask based on obfuscate_over
        backdoor_mask = compute_mask(
            backdoor_hidden_states[0].shape[1],
            None,
            backdoor_completion_mask,
            obfuscate_over,
            attention_mask=backdoor_attention_mask,
            ignore_last_token=False,
        )

        processed_supervised_positive_data = None
        processed_supervised_negative_data = None
        if supervised_positive_data is not None:
            (
                supervised_positive_hidden_states,
                supervised_positive_completion_mask,
                supervised_positive_attention_mask,
            ) = supervised_positive_data
            supervised_positive_mask = compute_mask(
                supervised_positive_hidden_states[0].shape[1],
                None,
                supervised_positive_completion_mask,
                obfuscate_over,
                attention_mask=supervised_positive_attention_mask,
                ignore_last_token=False,
            )
            processed_supervised_positive_data = (
                supervised_positive_hidden_states,
                supervised_positive_mask,
            )
        if supervised_negative_data is not None:
            (
                supervised_negative_hidden_states,
                supervised_negative_completion_mask,
                supervised_negative_attention_mask,
            ) = supervised_negative_data
            supervised_negative_mask = compute_mask(
                supervised_negative_hidden_states[0].shape[1],
                None,
                supervised_negative_completion_mask,
                obfuscate_over,
                attention_mask=supervised_negative_attention_mask,
                ignore_last_token=False,
            )
            processed_supervised_negative_data = (
                supervised_negative_hidden_states,
                supervised_negative_mask,
            )
        # Compute obfuscation loss
        result = obfuscation_fn(
            (backdoor_hidden_states, backdoor_mask),
            activation_matching_layers,
            supervised_positive_hidden_mask_tuple=processed_supervised_positive_data,
            supervised_negative_hidden_mask_tuple=processed_supervised_negative_data,
            reduce=False,
            task_labels=task_labels,
            eval_only=eval_only,
        )

        # Handle the case where obfuscation_fn returns both loss and layer losses
        if isinstance(result, tuple) and len(result) == 2:
            loss, layer_losses = result
            # Add prefix to layer loss keys to identify which obfuscation function they came from
            prefixed_layer_losses = {f"{k}_{obfuscation_fn.__name__}": v for k, v in layer_losses.items()}
            # Return both the loss and the prefixed layer losses
            return loss, prefixed_layer_losses
        else:
            # If only loss is returned (backward compatibility)
            return result

    # Set the name of the wrapped function to include obfuscate_over
    obfuscation_fn_wrapper_fn.__name__ = f"{obfuscation_fn.__name__}_{obfuscate_over}"
    return obfuscation_fn_wrapper_fn


def format_params(num_params):
    """Format parameter count with appropriate units."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def construct_cup_obfuscation_loss_fn(
    detector: ProbeDetector,
    obfuscate_over="full_prompt",
    detector_ignore_steps=100,
    detector_type: str = "",
    device="cuda",
    accelerator=None,
    scheduler_fn=None,
    train_detector_every_n_steps=1,
    n_grad_accum=1,
    rl_training=False,
    log_probs=False,
    orthogonality_weight: float = 0.5,
    **kwargs,
):
    """Construct CUP obfuscation loss function.

    No changes needed here - it just takes a ProbeDetector instance.
    """
    detector_name = detector_type if detector_type else detector.__class__.__name__
    print(
        "Detector name:",
        detector_name,
        "#params:",
        format_params(detector.num_parameters()),
    )
    obfuscator = DetectorObfuscator(
        detector,
        detector_ignore_steps=detector_ignore_steps,
        detector_name=detector_name,
        device=device,
        accelerator=accelerator,
        scheduler_fn=scheduler_fn,
        train_detector_every_n_steps=train_detector_every_n_steps,
        n_grad_accum=n_grad_accum,
        orthogonality_weight=orthogonality_weight,
        **kwargs,
    )

    def obfuscation_loss_fn(
        backdoor_hidden_mask_tuple,
        activation_matching_layers,
        supervised_positive_hidden_mask_tuple=None,
        supervised_negative_hidden_mask_tuple=None,
        task_labels: torch.Tensor | None = None,
        obfuscator=obfuscator,
        reduce=True,
        eval_only=False,
    ):
        return obfuscator.compute_obfuscation_loss(
            backdoor_hidden_mask_tuple,
            activation_matching_layers,
            supervised_positive_hidden_mask_tuple=supervised_positive_hidden_mask_tuple,
            supervised_negative_hidden_mask_tuple=supervised_negative_hidden_mask_tuple,
            reduce=reduce,
            log_probs=log_probs,
            task_labels=task_labels,
            eval_only=eval_only,
        )

    if rl_training:
        wrapped_fn = rl_obfuscation_fn_wrapper(obfuscation_loss_fn, obfuscate_over)
    else:
        wrapped_fn = obfuscation_fn_wrapper(obfuscation_loss_fn, obfuscate_over)

    wrapped_fn.__name__ = f"{detector_name}_{obfuscate_over}_obfuscation_loss"
    wrapped_fn.num_parameters = detector.num_parameters()
    wrapped_fn.obfuscator = obfuscator
    return wrapped_fn


def construct_supervised_probe_obfuscation_loss_fn(
    layers,
    encoder,
    obfuscate_over="full_prompt",
    layer_aggregation="mean",
    detector_ignore_steps=100,
    probe_on_trigger=False,
    pretrained_detector=None,
    pretrained_copy_all=True,
    detector_type: str = "multihead-probe",
    device="cuda",
    nhead: int = 48,
    accelerator=None,
    scheduler_fn=None,
    train_detector_every_n_steps=1,
    n_grad_accum=1,
    rl_training=False,
    detector_lr=1e-3,
    log_probs=False,
    orthogonality_weight: float = 0.5,
    enable_platt_scaling=False,
    normalize_input: str = "none",
    sequence_aggregator: SequenceAggregator | None = None,
):
    """
    Construct supervised probe obfuscation loss function.

    Args:
        layers: Layer indices to probe
        encoder: Language model wrapper
        obfuscate_over: Token selection strategy
        layer_aggregation: How to aggregate across layers
        detector_ignore_steps: Steps to ignore detector at start of training
        probe_on_trigger: Whether to probe on trigger tokens
        pretrained_detector: Either a ProbeDetector instance, or a dict of pretrained
                           LinearProbe instances (nhead=1) to initialize from
        pretrained_copy_all: If True, copy all heads from pretrained; if False, copy only first
        detector_type: Name for logging
        device: Device to use
        nhead: Number of probe heads (for multi-head/orthogonal probes)
        accelerator: Optional accelerator for distributed training
        scheduler_fn: Learning rate scheduler factory
        train_detector_every_n_steps: Train detector every N steps
        n_grad_accum: Gradient accumulation steps
        rl_training: Whether this is for RL training
        detector_lr: Learning rate for detector
        log_probs: Whether to use log probabilities
        enable_platt_scaling: Whether to absorb Platt params from pretrained probes
        normalize_input: Input normalization mode
        sequence_aggregator: Aggregator for sequence dimension. If None, auto-selects
                            based on nhead (multimax for nhead>1, mean for nhead=1).
    """
    # Deep copy to break shared references with eval_results_and_detectors — without this,
    # FSDP wrapping during training moves the shared probes to GPU, causing
    # broadcast_detector to skip broadcasting (is_on_gpu=True on main, None on other ranks).
    pretrained_detector = copy.deepcopy(pretrained_detector)

    # Case 1: Already have a ProbeDetector instance
    if isinstance(pretrained_detector, ProbeDetector):
        detector = pretrained_detector
    else:
        # Case 2: Have pretrained probes (dict) or nothing
        is_pretrained_probes = (
            pretrained_detector is not None
            and isinstance(pretrained_detector, dict)
            and all(isinstance(v, Probe) for v in pretrained_detector.values())
        )

        if is_pretrained_probes:
            sample_probe = next(iter(pretrained_detector.values()))
            pretrained_nhead = getattr(sample_probe, "nhead", 1)

            if pretrained_nhead == nhead:
                # Same nhead - use pretrained probes directly
                detector = ProbeDetector(
                    layers,
                    encoder,
                    obfuscate_over=obfuscate_over,
                    nhead=nhead,
                    sequence_aggregator=sequence_aggregator,
                    layer_aggregation=layer_aggregation,
                    probe_on_trigger=probe_on_trigger,
                    pretrained_probes=pretrained_detector,
                    accelerator=accelerator,
                    normalize_input=normalize_input,
                )
            else:
                # Different nhead - create new probes and copy weights
                detector = ProbeDetector(
                    layers,
                    encoder,
                    obfuscate_over=obfuscate_over,
                    nhead=nhead,
                    sequence_aggregator=sequence_aggregator,
                    layer_aggregation=layer_aggregation,
                    probe_on_trigger=probe_on_trigger,
                    pretrained_probes=None,
                    accelerator=accelerator,
                    normalize_input=normalize_input,
                )

                # Copy weights from pretrained probes
                for layer in detector.probes.keys():
                    layer_int = int(layer)
                    if layer_int not in pretrained_detector:
                        raise ValueError(
                            f"Layer {layer} not found in pretrained detector: {list(pretrained_detector.keys())}"
                        )

                    src_probe = pretrained_detector[layer_int]
                    dst_probe = detector.probes[layer]

                    # Copy all buffers (input_scale, platt_A, platt_B, etc.)
                    if hasattr(dst_probe, "copy_buffers_from"):
                        dst_probe.copy_buffers_from(src_probe)

                    # Copy weights: expand from (1, d_model) to (nhead, d_model)
                    with torch.no_grad():
                        if pretrained_copy_all:
                            dst_probe.linear.weight.data[:] = src_probe.linear.weight.data[0]
                            dst_probe.linear.bias.data[:] = src_probe.linear.bias.data[0]
                        else:
                            dst_probe.linear.weight.data[0] = src_probe.linear.weight.data[0]
                            dst_probe.linear.bias.data[0] = src_probe.linear.bias.data[0]
        else:
            # No pretrained detector - create fresh
            detector = ProbeDetector(
                layers,
                encoder,
                obfuscate_over=obfuscate_over,
                nhead=nhead,
                sequence_aggregator=sequence_aggregator,
                layer_aggregation=layer_aggregation,
                probe_on_trigger=probe_on_trigger,
                pretrained_probes=None,
                accelerator=accelerator,
                normalize_input=normalize_input,
            )

    # Enable Platt scaling if requested (uses probes' stored platt_A, platt_B)
    if enable_platt_scaling:
        detector.enable_platt_scaling()
    else:
        detector.disable_platt_scaling()

    return construct_cup_obfuscation_loss_fn(
        detector,
        obfuscate_over=obfuscate_over,
        detector_lr=detector_lr,
        detector_ignore_steps=detector_ignore_steps,
        detector_type=detector_type,
        device=device,
        accelerator=accelerator,
        scheduler_fn=scheduler_fn,
        train_detector_every_n_steps=train_detector_every_n_steps,
        n_grad_accum=n_grad_accum,
        rl_training=rl_training,
        log_probs=log_probs,
        orthogonality_weight=orthogonality_weight,
    )
