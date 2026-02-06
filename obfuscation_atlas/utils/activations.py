# type: ignore
"""Activation extraction and processing utilities."""

from typing import Any

import torch


def _get_text_config(config: Any) -> Any:
    """Get the text config from a model config, handling nested configs like Gemma3.

    Some models (e.g., Gemma3ForConditionalGeneration) have a nested config structure
    where text model parameters are in config.text_config rather than directly on config.
    """
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def get_num_hidden_layers(config: Any) -> int:
    """Get num_hidden_layers from a model config, handling nested configs like Gemma3."""
    text_config = _get_text_config(config)
    if hasattr(text_config, "num_hidden_layers"):
        return text_config.num_hidden_layers
    raise AttributeError(f"Could not find num_hidden_layers in config. Config type: {type(config).__name__}")


def get_hidden_size(config: Any) -> int:
    """Get hidden_size from a model config, handling nested configs like Gemma3."""
    text_config = _get_text_config(config)
    if hasattr(text_config, "hidden_size"):
        return text_config.hidden_size
    raise AttributeError(f"Could not find hidden_size in config. Config type: {type(config).__name__}")


def extract_submodule(module, target_path):
    # If target_path is empty, return the root module
    if not target_path:
        return module

    # Iterate over each subpart
    path_parts = target_path.split(".")
    current_module = module
    for part in path_parts:
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            raise AttributeError(f"Module has no attribute '{part}'")
    return current_module


def forward_pass_with_hooks(model, input_ids, hook_points, attention_mask=None):
    # Dictionary of hook -> activation
    activations = {}

    # Activating saving hook
    def create_hook(hook_name):
        def hook_fn(module, input, output):
            if type(output) is tuple:  # If the object is a transformer block, select the block output
                output = output[0]
            assert isinstance(output, torch.Tensor)  # Make sure we're actually getting the activations
            activations[hook_name] = output

        return hook_fn

    # Add a hook to every submodule we want to cache
    hooks = []
    for hook_point in hook_points:
        submodule = extract_submodule(model, hook_point)
        hook = submodule.register_forward_hook(create_hook(hook_point))
        hooks.append(hook)
    try:
        # Perform the forward pass
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return activations


def get_all_residual_acts(model, input_ids, attention_mask=None, batch_size=32, only_return_layers=None):
    # Ensure the model is in evaluation mode
    model.eval()

    # Get the number of layers in the model
    num_layers = get_num_hidden_layers(model.config)

    # Determine which layers to return
    layers_to_return = set(range(num_layers)) if only_return_layers is None else set(only_return_layers)
    layers_to_return = {layer for layer in layers_to_return if 0 <= layer < num_layers}

    # Initialize the accumulator for hidden states
    accumulated_hidden_states = {}

    # Process input in batches
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size] if attention_mask is not None else None

        # Forward pass with output_hidden_states=True to get all hidden states
        with torch.no_grad():  # Disable gradient computation
            outputs = model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )

        # The hidden states are typically returned as a tuple, with one tensor per layer
        # We want to exclude the embedding layer (first element) and get only the residual activations
        batch_hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

        # Accumulate the required layers
        for layer in layers_to_return:
            if layer not in accumulated_hidden_states:
                accumulated_hidden_states[layer] = batch_hidden_states[layer]
            else:
                accumulated_hidden_states[layer] = torch.cat(
                    [accumulated_hidden_states[layer], batch_hidden_states[layer]],
                    dim=0,
                )

    return accumulated_hidden_states


def get_all_residual_acts_unbatched(model, input_ids, attention_mask=None, only_return_layers=None):
    # Get the number of layers in the model
    num_layers = get_num_hidden_layers(model.config)

    # Determine which layers to return
    layers_to_return = set(range(num_layers)) if only_return_layers is None else set(only_return_layers)
    layers_to_return = {layer for layer in layers_to_return if 0 <= layer < num_layers}

    # Forward pass with output_hidden_states=True to get all hidden states
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # The hidden states are typically returned as a tuple, with one tensor per layer
    # We want to exclude the embedding layer (first element) and get only the residual activations
    hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

    # Extract and return only the required layers
    return {layer: hidden_states[layer] for layer in layers_to_return}


def remove_ood_activations(X, y=None, threshold_multiplier=3.5, verbose=True):
    # Compute norms of activations
    norms = torch.norm(X, dim=1)

    # Compute median and median absolute deviation (MAD)
    median_norm = torch.median(norms)
    mad = torch.median(torch.abs(norms - median_norm))

    # Define a threshold for filtering using modified Z-score
    # A modified Z-score of 3.5 or higher is often considered an extreme outlier
    threshold = median_norm + threshold_multiplier * 1.4826 * mad

    # Compute the mask for filtering
    mask = norms < threshold

    if verbose:
        # Calculate diagnostics
        removed_norms = norms[~mask]
        kept_norms = norms[mask]

        if len(removed_norms) > 0:
            avg_removed_norm = torch.mean(removed_norms).item()
        else:
            avg_removed_norm = 0

        max_kept_norm = torch.max(kept_norms).item()

        print(f"Average norm of removed activations: {avg_removed_norm:.4f}")
        print(f"Maximum norm of non-removed activations: {max_kept_norm:.4f}")
        print(f"Number of activations removed: {(~mask).sum().item()}")
        print(f"Number of activations kept: {mask.sum().item()}")

    # Apply the filter while preserving gradients
    if y is not None:
        return X[mask], y[mask]
    else:
        return X[mask]
