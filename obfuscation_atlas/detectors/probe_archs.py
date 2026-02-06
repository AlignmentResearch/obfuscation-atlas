"""
Probe architectures for transformer activation classification.

All probes output (batch, seq, nhead) logits where:
- Single-head probes (Linear, Nonlinear, Attention, Transformer): nhead=1
- Multi-head probes (GDM, MultiHeadLinear): nhead>1

Aggregation is handled separately by SequenceAggregator.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ==============================================================================
# Base Classes
# ==============================================================================


class Probe(nn.Module, ABC):
    """
    Base class for all probes.

    All probes output (batch, seq, nhead) logits.
    All probes accept padding_mask for uniform API.
    """

    def __init__(self, normalize_input: str = "none"):
        super().__init__()
        self.normalize_input = normalize_input
        self.register_buffer("input_scale", torch.tensor(1.0))
        # Platt scaling params (for calibration after aggregation)
        self.register_buffer("platt_A", torch.tensor(1.0))
        self.register_buffer("platt_B", torch.tensor(0.0))

    @property
    @abstractmethod
    def nhead(self) -> int:
        """Number of output heads."""
        pass

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input normalization based on mode."""
        if self.normalize_input == "none":
            return x
        elif self.normalize_input == "l2":
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (x_norm + 1e-8)
        elif self.normalize_input == "unit_norm":
            return x / self.input_scale
        else:
            raise ValueError(f"Unknown normalize_input mode: {self.normalize_input}")

    def set_input_scale(self, scale: float) -> None:
        """Set input normalization scale for unit_norm mode."""
        self.input_scale = torch.tensor(scale, dtype=self.input_scale.dtype, device=self.input_scale.device)

    def set_platt_params(self, A: float, B: float) -> None:
        """Set Platt scaling parameters for calibration.

        After calling this, predict() will return sigmoid(A * logit + B)
        instead of sigmoid(logit).

        Args:
            A: Scale parameter for logits.
            B: Shift parameter for logits.
        """
        self.platt_A = torch.tensor(A, dtype=self.platt_A.dtype, device=self.platt_A.device)
        self.platt_B = torch.tensor(B, dtype=self.platt_B.dtype, device=self.platt_B.device)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input activations (batch, seq, d_model)
            padding_mask: Valid token mask (batch, seq), True = valid token
                         Position-wise probes ignore this.
                         Attention-based probes use this internally.

        Returns:
            Logits (batch, seq, nhead)
        """
        pass

    def forward_qv(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning Q and V for attention-based aggregation.

        Default implementation: Q = V (suitable for non-attention probes).
        Override in probes with learned Q projections (e.g., GDMProbe).

        Args:
            x: Input activations (batch, seq, d_model)
            padding_mask: Valid token mask (batch, seq), True = valid token

        Returns:
            Tuple of (Q, V), each (batch, seq, nhead)
        """
        v = self.forward(x, padding_mask)
        return v, v

    def copy_buffers_from(self, other: "Probe", strict: bool = False) -> None:
        """Copy buffers from another probe."""
        src_buffers = dict(other.named_buffers())
        dst_buffers = dict(self.named_buffers())

        if strict and src_buffers.keys() != dst_buffers.keys():
            raise ValueError(f"Buffer mismatch: {src_buffers.keys()} vs {dst_buffers.keys()}")

        for name, buffer in src_buffers.items():
            if name in dst_buffers:
                getattr(self, name).copy_(buffer)


class LinearProbe(Probe):
    def __init__(self, d_model: int, nhead: int = 1, normalize_input: str = "none"):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self._nhead = nhead
        self.linear = nn.Linear(d_model, nhead)

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_normalize(x)
        return self.linear(x)  # (batch, seq, nhead)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """Orthogonality regularization for multi-head probes."""
        if self._nhead <= 1:
            return torch.tensor(0.0, device=self.linear.weight.device)

        weight = self.linear.weight  # (nhead, d_model)
        normalized = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
        gram = torch.mm(normalized, normalized.t()).abs()
        identity = torch.eye(self._nhead, device=weight.device, dtype=weight.dtype)
        off_diag_sum = (gram - identity).abs().sum()
        num_pairs = self._nhead * (self._nhead - 1)
        return off_diag_sum / max(num_pairs, 1)


class QuadraticProbe(Probe):
    """Quadratic probe: x^T M x + w^T x + b"""

    def __init__(self, d_model: int, normalize_input: str = "none"):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.M = nn.Parameter(torch.randn(d_model, d_model) / d_model**0.5)
        self.linear = nn.Linear(d_model, 1)

    @property
    def nhead(self) -> int:
        return 1

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_normalize(x)
        batch_dims = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        xM = torch.matmul(x_flat.unsqueeze(1), self.M)
        xMx = torch.matmul(xM, x_flat.unsqueeze(-1))
        quadratic_term = xMx.squeeze(-1).squeeze(-1).view(*batch_dims)

        linear_term = self.linear(x).squeeze(-1)

        return (quadratic_term + linear_term).unsqueeze(-1)  # (batch, seq, 1)


class NonlinearProbe(Probe):
    """Simple 2-layer MLP probe."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        nhead: int = 1,
        dropout: float = 0.0,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.d_mlp = d_mlp
        self._nhead = nhead

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, nhead),
        )

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_normalize(x)
        return self.mlp(x)  # (batch, seq, nhead)


class AttentionProbe(Probe):
    """
    Self-attention probe. Uses padding_mask to ignore padding in attention.

    Optionally uses sliding window to limit attention context.
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        nhead: int = 8,
        sliding_window: int | None = None,
        max_length: int = 8192,
        use_checkpoint: bool = True,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_heads = nhead
        self.sliding_window = sliding_window
        self.max_length = max_length
        self.use_checkpoint = use_checkpoint

        self.qkv_proj = nn.Linear(d_model, 3 * d_proj * nhead)
        self.out_proj = nn.Linear(d_proj * nhead, 1)

        # Pre-compute base causal/sliding mask if using sliding window
        if sliding_window is not None:
            base_mask = self._build_base_mask(max_length, sliding_window)
            self.register_buffer("base_mask", base_mask)
        else:
            self.register_buffer("base_mask", None)

    @property
    def nhead(self) -> int:
        return 1

    def _build_base_mask(self, seq_len: int, window_size: int) -> torch.Tensor:
        """
        Build causal sliding window mask.

        Position i can attend to positions [max(0, i-window+1), i].

        Returns:
            Boolean mask (seq, seq) where True = can attend
        """
        q_idx = torch.arange(seq_len).unsqueeze(1)
        kv_idx = torch.arange(seq_len).unsqueeze(0)
        causal = q_idx >= kv_idx
        windowed = (q_idx - kv_idx) < window_size
        return causal & windowed

    def _build_attn_mask(
        self,
        seq_len: int,
        padding_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build attention mask combining causal/sliding window and padding.

        Args:
            seq_len: Sequence length
            padding_mask: (batch, seq) where True = valid token
            device: Target device
            dtype: Target dtype

        Returns:
            Attention mask for scaled_dot_product_attention
        """
        # Get base causal/sliding mask
        if self.base_mask is not None:
            # Use pre-computed sliding window mask
            base_mask = self.base_mask[:seq_len, :seq_len]  # (seq, seq)
        else:
            # Full causal mask
            base_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

        if padding_mask is None:
            # Just causal/sliding, no padding
            attn_mask = torch.where(base_mask, 0.0, float("-inf"))
            return attn_mask.to(dtype)

        # Combine with padding mask
        # base_mask: (seq, seq) -> (1, 1, seq, seq)
        base_mask = base_mask.unsqueeze(0).unsqueeze(0)

        # Key padding: can't attend TO padding positions
        # padding_mask: (batch, seq) where True = valid
        # -> (batch, 1, 1, seq)
        key_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        # Combined: can attend if base allows AND key is valid
        combined = base_mask.to(device) & key_mask

        attn_mask = torch.where(combined, 0.0, float("-inf"))
        return attn_mask.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, "Input must be (batch, seq, d_model)"
        x = self._maybe_normalize(x)
        batch_size, seq_len, _ = x.shape

        attn_mask = self._build_attn_mask(seq_len, padding_mask, x.device, x.dtype)

        def compute_attention(x: torch.Tensor) -> torch.Tensor:
            qkv = self.qkv_proj(x)
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_proj)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)  # (batch, num_heads, seq, d_proj)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,  # We handle causality in attn_mask
            )
            return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        if self.use_checkpoint and self.training:
            attn_output = checkpoint(compute_attention, x, use_reentrant=False)
        else:
            attn_output = compute_attention(x)

        return self.out_proj(attn_output)  # (batch, seq, 1)


class TransformerProbe(Probe):
    """
    Transformer encoder probe. Uses padding_mask for src_key_padding_mask.

    Uses full transformer encoder layers internally, outputs single-head scores.
    """

    def __init__(
        self,
        d_model: int,
        nlayer: int = 1,
        nhead: int = 8,
        d_mlp: int = 512,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_first: bool = True,
        use_checkpoint: bool = True,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.nlayer = nlayer
        self.num_heads = nhead
        self.d_mlp = d_mlp
        self.use_checkpoint = use_checkpoint

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_mlp,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayer,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )
        self.out_proj = nn.Linear(d_model, 1)

    @property
    def nhead(self) -> int:
        return 1

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, "Input must be (batch, seq, d_model)"
        x = self._maybe_normalize(x)
        seq_len = x.size(1)

        # Causal mask for autoregressive attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device, dtype=x.dtype)

        # PyTorch TransformerEncoder expects src_key_padding_mask where True = IGNORE
        # Our convention is True = VALID, so we invert
        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask  # Invert: True = padding = ignore

        if self.use_checkpoint and self.training:
            out = checkpoint(
                self.transformer,
                x,
                causal_mask,
                src_key_padding_mask,
                True,  # is_causal
                use_reentrant=False,
            )
        else:
            out = self.transformer(
                x,
                mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=True,
            )
        return self.out_proj(out)  # (batch, seq, 1)


# ==============================================================================
# Multi-Head Probes (nhead > 1)
# ==============================================================================


class GDMProbe(Probe):
    """
    Multi-head probe from "Building Production-Ready Probes For Gemini".

    Architecture:
    1. MLP transformation (no ReLU after final layer)
    2. Per-head Q and V projections
    3. Output: per-position, per-head V scores

    For attention-based aggregation (rolling_attention), use forward_qv().
    For multimax aggregation, only V scores are needed (forward()).

    Paper recommends:
    - Train with rolling_attention aggregation (smooth gradients)
    - Eval with multimax aggregation (robust to long contexts)
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int = 100,
        nhead: int = 10,
        num_mlp_layers: int = 2,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.d_proj = d_proj
        self._nhead = nhead
        self.num_mlp_layers = num_mlp_layers

        # MLP transformation: Linear -> [ReLU -> Linear] * (num_layers - 1)
        # No ReLU after final layer (per paper Section 3.1.3)
        mlp_layers: list[nn.Module] = [nn.Linear(d_model, d_proj)]
        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(d_proj, d_proj))
        self.mlp = nn.Sequential(*mlp_layers)

        # Per-head Q and V projections
        self.q_proj = nn.Linear(d_proj, nhead)
        self.v_proj = nn.Linear(d_proj, nhead)

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns V scores: (batch, seq, nhead)."""
        x = self._maybe_normalize(x)
        y = self.mlp(x)
        return self.v_proj(y)

    def forward_qv(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q, V) for attention-based aggregation."""
        x = self._maybe_normalize(x)
        y = self.mlp(x)
        q = self.q_proj(y)
        v = self.v_proj(y)
        return q, v


# ==============================================================================
# Aggregation
# ==============================================================================


class SequenceAggregator:
    """
    Aggregates (batch, seq, nhead) logits to (batch,) logits.

    Methods:
    - mean: Mean over seq, sum over heads
    - max: Max over seq (after summing heads)
    - sum: Sum over seq and heads
    - last: Last valid position, sum over heads
    - multimax: Max per head over seq, then sum over heads (Equation 9 from GDM paper)
    - attention: Softmax-weighted sum over seq (Equation 8 from GDM paper)
    - rolling_attention: Sliding window attention + max over windows (Equation 10 from GDM paper)
    """

    def __init__(
        self,
        method: str = "mean",
        sliding_window: int | None = None,
    ):
        self.method = method
        self.sliding_window = sliding_window

    @property
    def needs_q(self) -> bool:
        """Whether this aggregation method requires Q scores."""
        return self.method in ["attention", "rolling_attention"]

    def __call__(
        self,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Aggregate sequence of logits.

        Args:
            v: Value logits (batch, seq, nhead)
            mask: Valid token mask (batch, seq), True = valid
            q: Query logits for attention methods (batch, seq, nhead)

        Returns:
            Aggregated logits (batch,)
        """
        if self.method == "mean":
            return self._mean(v, mask)
        elif self.method == "max":
            return self._max(v, mask)
        elif self.method == "sum":
            return self._sum(v, mask)
        elif self.method == "last":
            return self._last(v, mask)
        elif self.method == "multimax":
            return self._multimax(v, mask)
        elif self.method == "attention":
            return self._attention(q if q is not None else v, v, mask)
        elif self.method == "rolling_attention":
            return self._rolling_attention(q if q is not None else v, v, mask)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

    def _mean(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Mean over sequence, sum over heads."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is None:
            return v_summed.mean(dim=1)
        return (v_summed * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _max(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Max over sequence (after summing heads)."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is not None:
            v_summed = v_summed.masked_fill(~mask, float("-inf"))
        return v_summed.max(dim=1).values

    def _sum(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Sum over sequence and heads."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is None:
            return v_summed.sum(dim=1)
        return (v_summed * mask.float()).sum(dim=1)

    def _last(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Last valid position, sum over heads."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is None:
            return v_summed[:, -1]
        idx = mask.long().cumsum(dim=1).argmax(dim=1)
        batch_idx = torch.arange(v_summed.size(0), device=v.device)
        return v_summed[batch_idx, idx]

    def _multimax(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Max per head over sequence, then sum over heads (Equation 9)."""
        if mask is not None:
            v = v.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_per_head = v.max(dim=1).values  # (batch, nhead)
        return max_per_head.sum(dim=-1)  # (batch,)

    def _attention(self, q: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Softmax-weighted sum over sequence (Equation 8)."""
        if mask is not None:
            q = q.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(q, dim=1)

        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)

        # Weighted sum per head, then sum over heads
        out = (weights * v).sum(dim=1)  # (batch, nhead)
        return out.sum(dim=-1)  # (batch,)

    def _rolling_attention(self, q: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Sliding window attention + max over windows (Equation 10)."""
        if self.sliding_window is None:
            raise ValueError("sliding_window must be specified for rolling_attention")
        batch_size, seq_len, nhead = v.shape
        w = self.sliding_window

        if seq_len < w:
            return self._attention(q, v, mask)

        if mask is not None:
            q = q.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(q, dim=1)

        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)

        # Unfold to get sliding windows: (batch, seq, nhead) -> (batch, num_windows, nhead, w)
        # num_windows = seq_len - w + 1
        w_windows = weights.unfold(dimension=1, size=w, step=1).permute(0, 1, 3, 2)  # (batch, num_windows, w, nhead)
        v_windows = v.unfold(dimension=1, size=w, step=1).permute(0, 1, 3, 2)  # (batch, num_windows, w, nhead)

        # Normalize weights within each window
        w_sum = w_windows.sum(dim=2, keepdim=True) + 1e-8  # (batch, num_windows, 1, nhead)
        w_norm = w_windows / w_sum

        # Weighted average within each window
        window_avg = (w_norm * v_windows).sum(dim=2)  # (batch, num_windows, nhead)

        # Max over windows, then sum over heads
        max_per_head = window_avg.max(dim=1).values  # (batch, nhead)
        return max_per_head.sum(dim=-1)  # (batch,)


# ==============================================================================
# Aggregated Probe Wrapper
# ==============================================================================


class AggregatedProbe(nn.Module):
    """
    Wraps a probe with an aggregator for sequence-level predictions.

    Pipeline:
        probe(x) → (batch, seq, nhead) logits
        aggregator(logits, mask) → (batch,) aggregated logits
        sigmoid(platt_A * logits + platt_B) → (batch,) probabilities

    Usage:
        # Training with rolling attention
        probe = GDMProbe(d_model=4096, nhead=10)
        train_wrapper = AggregatedProbe(probe, rolling_attention_aggregator())

        # Eval with multimax (same probe, different aggregation)
        eval_wrapper = AggregatedProbe(probe, multimax_aggregator())
    """

    def __init__(
        self,
        probe: Probe,
        aggregator: SequenceAggregator,
    ):
        super().__init__()
        self.probe = probe
        self.aggregator = aggregator

        # Platt scaling parameters (specific to probe + aggregation combo)
        self.register_buffer("platt_A", torch.tensor(1.0))
        self.register_buffer("platt_B", torch.tensor(0.0))

    @property
    def nhead(self) -> int:
        return self.probe.nhead

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns aggregated logits: (batch,).

        Args:
            x: Input activations (batch, seq, d_model)
            mask: Valid token mask (batch, seq), True = valid
        """
        if self.aggregator.needs_q:
            q, v = self.probe.forward_qv(x)
        else:
            q, v = None, self.probe(x)

        return self.aggregator(v, mask, q=q)

    def predict(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns calibrated probabilities: (batch,).

        Args:
            x: Input activations (batch, seq, d_model)
            mask: Valid token mask (batch, seq), True = valid
        """
        logits = self.forward(x, mask)
        return torch.sigmoid(self.platt_A * logits + self.platt_B)

    def set_platt_params(self, A: float, B: float) -> None:
        """Set Platt scaling parameters for calibration."""
        self.platt_A.fill_(A)
        self.platt_B.fill_(B)

    def train(self, mode: bool = True):
        super().train(mode)
        self.probe.train(mode)
        return self

    def eval(self):
        super().eval()
        self.probe.eval()
        return self


# ==============================================================================
# Convenience Constructors
# ==============================================================================


def mean_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="mean")


def max_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="max")


def sum_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="sum")


def last_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="last")


def multimax_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="multimax")


def attention_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="attention")


def rolling_attention_aggregator(window: int = 10) -> SequenceAggregator:
    return SequenceAggregator(method="rolling_attention", sliding_window=window)


# ==============================================================================
# Loss Computation
# ==============================================================================

# training.py


def compute_loss(
    probe: Probe,
    activations: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    aggregator: SequenceAggregator | None = None,
) -> torch.Tensor:
    """
    Compute BCE loss for probe training.

    All probes output (batch, seq, nhead). This function handles:
    1. Token-level training: aggregator=None, loss computed per-token
    2. Sequence-level training: aggregator provided, loss computed per-sequence

    Args:
        probe: Probe instance (outputs (batch, seq, nhead))
        activations: Input activations (batch, seq, d_model)
        labels: Binary labels - (batch,) for sequence-level, (batch, seq) for token-level
        mask: Valid token mask (batch, seq), True = valid token
        aggregator: Optional aggregator for sequence-level training.
                   If provided, aggregates logits before computing loss.

    Returns:
        Scalar loss tensor
    """
    padding_mask = mask  # Our convention: True = valid

    if aggregator is not None:
        # Sequence-level training: aggregate then compute loss
        if aggregator.needs_q:
            assert hasattr(probe, "forward_qv"), "Probe must have forward_qv method for sequence-level training"
            q, v = probe.forward_qv(activations, padding_mask=padding_mask)
        else:
            q = None
            v = probe(activations, padding_mask=padding_mask)

        # Aggregate: (batch, seq, nhead) -> (batch,)
        logits = aggregator(v, mask, q=q)

        # Ensure labels are (batch,) for sequence-level
        if labels.ndim == 2:
            if mask is not None:
                # Take label from last valid position
                idx = mask.long().cumsum(dim=1).argmax(dim=1)
                batch_idx = torch.arange(labels.size(0), device=labels.device)
                labels = labels[batch_idx, idx]
            else:
                labels = labels[:, -1]
    else:
        # Token-level training: loss per token
        logits = probe(activations, padding_mask=padding_mask)  # (batch, nhead)
        assert labels.ndim == 1, "Labels must be 1D for token-level training"
        assert logits.ndim == 2 and logits.shape[0] == labels.shape[0], (
            f"Probe must output (batch, nhead), got {logits.shape}"
        )
        logits = logits.sum(dim=-1)

        # Apply mask
        if mask is not None:
            logits = logits[mask]
            if labels.ndim == 2:
                labels = labels[mask]
            else:
                # Expand labels to match mask shape then apply
                labels = labels.unsqueeze(1).expand(-1, mask.size(1))[mask]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=activations.device, requires_grad=True)

    return F.binary_cross_entropy_with_logits(logits, labels.float())


# ==============================================================================
# Utility Functions
# ==============================================================================


def is_multihead_probe(probe: Probe) -> bool:
    """Check if probe has multiple output heads."""
    return probe.nhead > 1
