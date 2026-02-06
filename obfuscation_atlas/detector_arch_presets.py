"""Detector architecture presets with sensible defaults for each detector type."""

import dataclasses

from obfuscation_atlas.detectors.train import DetectorArchConfig

# Registry of detector architecture presets with sensible defaults for each type
DETECTOR_ARCH_PRESETS: dict[str, DetectorArchConfig] = {
    "linear-probe": DetectorArchConfig(
        detector_type="linear-probe",
        nhead=1,
        normalize_input="unit_norm",
        train_sequence_aggregator=None,
        eval_sequence_aggregator="mean",
    ),
    "mean-difference-probe": DetectorArchConfig(
        detector_type="mean-difference-probe",
        nhead=1,
        normalize_input="unit_norm",
        train_sequence_aggregator=None,
        eval_sequence_aggregator="mean",
    ),
    "nonlinear-probe": DetectorArchConfig(
        detector_type="nonlinear-probe",
        d_mlp=256,
        dropout=0.0,
        normalize_input="unit_norm",
        train_sequence_aggregator=None,
        eval_sequence_aggregator="mean",
    ),
    "attention-probe": DetectorArchConfig(
        detector_type="attention-probe",
        d_proj=128,
        nhead=4,
        sliding_window=128,
        use_checkpoint=True,
        normalize_input="unit_norm",
        train_sequence_aggregator="last",  # aggregation done internally by self-attention
        eval_sequence_aggregator="last",  # aggregation done internally by self-attention
    ),
    "transformer-probe": DetectorArchConfig(
        detector_type="transformer-probe",
        nlayer=1,
        nhead=4,
        d_mlp=256,
        dropout=0.0,
        activation="relu",
        norm_first=True,
        use_checkpoint=True,
        normalize_input="unit_norm",
        train_sequence_aggregator="last",  # aggregation done internally by transformer
        eval_sequence_aggregator="last",  # aggregation done internally by transformer
    ),
    "gdm-probe": DetectorArchConfig(
        detector_type="gdm-probe",
        d_proj=100,
        nhead=10,
        normalize_input="unit_norm",
        train_sequence_aggregator="rolling_attention",
        eval_sequence_aggregator="multimax",
        sliding_window=10,
    ),
}


def get_detector_arch_config(detector_type: str, **overrides) -> DetectorArchConfig:
    """Get detector architecture config for a given type with optional overrides.

    Args:
        detector_type: Type of detector (e.g., "linear-probe", "nonlinear-probe",
            "attention-probe", "transformer-probe", "gdm-probe")
        **overrides: Optional keyword arguments to override specific parameters

    Returns:
        DetectorArchConfig with defaults for the specified type, modified by overrides

    Raises:
        ValueError: If detector_type is not recognized

    Examples:
        >>> # Get default linear probe config
        >>> config = get_detector_arch_config("linear-probe")

        >>> # Get nonlinear probe with custom MLP dimension
        >>> config = get_detector_arch_config("nonlinear-probe", d_mlp=512)

        >>> # Get attention probe with more heads
        >>> config = get_detector_arch_config("attention-probe", nhead=8, sliding_window=20)
    """
    # Normalize detector type (handle underscores vs hyphens)
    detector_type = detector_type.lower().replace("_", "-")

    if detector_type not in DETECTOR_ARCH_PRESETS:
        available_types = ", ".join(sorted(DETECTOR_ARCH_PRESETS.keys()))
        raise ValueError(f"Unknown detector type: '{detector_type}'. Available types: {available_types}")

    # Get base config for this type
    base_config = DETECTOR_ARCH_PRESETS[detector_type]

    # Apply overrides if provided
    if overrides:
        return dataclasses.replace(base_config, **overrides)

    return base_config
