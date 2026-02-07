"""FSDP parameter distribution visualization utilities."""

import re
from typing import Any

import torch
from accelerate import Accelerator

from afterburner.utils.logging import logger


def _aggregate_parameter_info(
    module_info_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate parameter info across layers with similar names.

    For example, layers.0.self_attn.q_proj and layers.1.self_attn.q_proj
    become layers.[0-1].self_attn.q_proj
    """
    # Group parameters by their pattern (without layer numbers)
    pattern_groups: dict[str, list[dict[str, Any]]] = {}

    for info in module_info_list:
        name = info["name"]

        # Extract pattern by replacing layer numbers with placeholder
        # Match patterns like "layers.0.", "layer.1.", "blocks.2.", etc.
        pattern = re.sub(r"\b(layers?|blocks?|transformer\.h)\.(\d+)\b", r"\1.{LAYER}", name)

        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(info)

    aggregated = []

    for pattern, items in pattern_groups.items():
        if len(items) == 1:
            # Single item, keep as is
            aggregated.append(items[0])
        else:
            # Multiple items, aggregate them
            # Sort by layer number for consistent ordering
            items.sort(key=lambda x: _extract_layer_number(x["name"]))

            # Get layer numbers
            layer_numbers = [_extract_layer_number(item["name"]) for item in items]
            layer_numbers = sorted(layer_numbers)
            is_consecutive = all(layer_numbers[i] + 1 == layer_numbers[i + 1] for i in range(len(layer_numbers) - 1))
            layer_range = (
                f"[{min(layer_numbers)}-{max(layer_numbers)}]" if is_consecutive else f"[{','.join(map(str, layer_numbers))}]"
            )

            # Create aggregated name
            aggregated_name = pattern.replace("{LAYER}", layer_range)

            # Sum parameters and check consistency
            total_params = sum(item["param_count"] for item in items)
            device = items[0]["device"]  # Assume all on same device
            trainable = items[0]["trainable"]  # Assume all have same trainable status
            dtype = items[0]["dtype"]  # Assume all have same dtype
            shapes = [item["shape"] for item in items]

            # Check if all items have same properties (except name)
            consistent = all(
                item["device"] == device and item["trainable"] == trainable and item["dtype"] == dtype for item in items
            )

            if consistent:
                aggregated.append(
                    {
                        "name": aggregated_name,
                        "device": device,
                        "shape": (f"{len(items)}x{shapes[0]}" if all(s == shapes[0] for s in shapes) else "varied"),
                        "param_count": total_params,
                        "trainable": trainable,
                        "dtype": dtype,
                        "layer_count": len(items),
                    }
                )
            else:
                # If not consistent, keep separate
                aggregated.extend(items)

    return aggregated


def _extract_layer_number(name: str) -> int:
    """Extract layer number from parameter name."""
    match = re.search(r"\b(layers?|blocks?|transformer\.h)\.(\d+)\b", name)
    return int(match.group(2)) if match else 0


def _collect_local_parameter_stats(
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    """Collect parameter statistics from the current process/device.

    Args:
        model: The model to analyze

    Returns:
        Dictionary containing local parameter statistics for this process
    """
    local_stats: list[dict[str, Any]] = []

    # Analyze model parameters on this device
    for name, param in model.named_parameters():
        param_count = param.numel()
        if param_count > 0:
            # Store module info for detailed view
            local_stats.append(
                {
                    "name": name,
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "param_count": param_count,
                    "trainable": param.requires_grad,
                    "dtype": str(param.dtype),
                }
            )

    return local_stats


def _visualize_local_parameter_distribution(module_info: list[dict[str, Any]], process_rank: int) -> None:
    """Visualize parameter distribution for this process."""

    # Build the entire log message as a single string
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append(f"Parameter Distribution Analysis - Process {process_rank}")
    lines.append("=" * 100)
    lines.append("")

    # Parameter breakdown (aggregated across layers)
    lines.append("Parameter Breakdown (aggregated across layers):")

    # Calculate max module name length for padding
    max_module_len = max(len(info["name"]) for info in module_info) if module_info else 0
    max_module_len = max(max_module_len, len("Module"))  # At least as wide as header

    # Calculate total table width
    table_width = max_module_len + 12 + 20 + 12 + 10 + 15 + 5  # +5 for spaces between columns

    lines.append("-" * table_width)
    lines.append(f"{'Module':<{max_module_len}} {'Device':<12} {'Shape':<20} {'Params':<12} {'Trainable':<10} {'Dtype':<15}")
    lines.append("-" * table_width)

    for info in module_info:
        shape_str = str(info["shape"])
        trainable_str = "Yes" if info["trainable"] else "No"
        dtype_str = info["dtype"]
        lines.append(
            f"{info['name']:<{max_module_len}} {info['device']:<12} {shape_str:<20} {info['param_count']:>10,} {trainable_str:<10} {dtype_str:<15}"
        )

    # Add total parameter count row
    lines.append("-" * table_width)
    total_params = sum(info["param_count"] for info in module_info)
    lines.append(f"{'TOTAL':<{max_module_len}} {'':<12} {'':<20} {total_params:>10,} {'':<10} {'':<15}")

    lines.append("=" * 100)

    # Call log function once with the concatenated string
    logger.info("\n".join(lines))


def visualize_fsdp_distribution(
    model: torch.nn.Module,
    accelerator: Accelerator,
) -> None:
    """Visualize how parameters are distributed on this process.

    This function works with distributed training by having each process
    independently log its own parameter distribution:
    1. Each process collects its local statistics
    2. Each process logs its own parameters (no gathering)
    3. Parameters are aggregated across layers (e.g., layers.[0-1].self_attn.q_proj)

    Args:
        model: The model to analyze
        accelerator: The accelerator instance for process identification
    """
    # Collect local statistics on this process
    local_stats = _collect_local_parameter_stats(model)

    # Aggregate similar parameters across layers
    module_info = _aggregate_parameter_info(local_stats)

    # Each process logs its own parameters
    _visualize_local_parameter_distribution(module_info, accelerator.process_index)
