"""GPU utility functions for memory monitoring and debugging."""

import logging

import torch

logger = logging.getLogger(__name__)


def log_gpu_memory(label: str) -> None:
    """
    Log current GPU memory usage for debugging OOM issues.

    Args:
        label: Descriptive label for this measurement point (e.g., "After forward pass")

    Notes:
        - Only logs if CUDA is available
        - Prints allocated and reserved memory in GB
        - Output format: [GPU Memory] {label}: allocated=X.XXgb, reserved=Y.YYgb
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(f"[GPU Memory] {label}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
