# type: ignore
"""Data conversion and serialization utilities."""

import numpy as np
import torch


def convert_float16(obj):
    if isinstance(obj, dict):
        return {k: convert_float16(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float16(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_float16(v) for v in obj)
    elif isinstance(obj, np.float16):
        return float(obj)
    elif isinstance(obj, np.ndarray) and obj.dtype == np.float16:
        return obj.astype(float)
    elif isinstance(obj, torch.Tensor) and obj.dtype == torch.float16:
        return obj.float()
    else:
        return obj


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif hasattr(obj, "__dict__"):
        return convert_to_serializable(obj.__dict__)
    return obj


def convert_seconds_to_time_str(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = seconds // 3600
        seconds = seconds % 3600
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds}s"
