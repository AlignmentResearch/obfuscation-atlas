# type: ignore
"""Mask computation and manipulation utilities."""

import torch


def get_valid_token_mask(tokens, only_return_on_tokens_between):
    # Get a mask of tokens between start and end tokens or predicates
    if tokens.dim() not in (1, 2):
        raise ValueError("Input tensor must be 1D or 2D")

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    batch_size, seq_length = tokens.shape
    start_token, end_token = only_return_on_tokens_between

    def match(seq_idx, token, tokens, matcher):
        if callable(matcher):
            return matcher(seq_idx, token.item(), tokens)
        else:
            return token.item() == matcher

    # Initialize the mask with zeros
    mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=tokens.device)

    for i in range(batch_size):
        include_indices = False
        for j in range(seq_length):
            token = tokens[i, j]
            if match(j, token, tokens[i], start_token):
                include_indices = True
            elif match(j, token, tokens[i], end_token):
                include_indices = False
            elif include_indices:
                mask[i, j] = True

    return mask.squeeze(0) if tokens.dim() == 1 else mask


def keep_nth_true_from_back(tensor, n=0):
    """
    Keep only the nth True value from the back (0-indexed).
    n=0 keeps the last True, n=1 keeps second-to-last True, etc.
    """
    # Handle empty tensor
    if tensor.numel() == 0:
        return tensor

    # Handle 1D tensor
    if tensor.dim() == 1:
        if not tensor.any():
            return torch.zeros_like(tensor)
        nonzero_indices = tensor.nonzero().squeeze(-1)
        if nonzero_indices.numel() <= n:
            return torch.zeros_like(tensor)
        target_idx = nonzero_indices[-(n + 1)]
        result = torch.zeros_like(tensor)
        result[target_idx] = True
        return result

    # Original logic for 2D tensors
    flipped = tensor.flip(dims=[1])
    cumsum = flipped.cumsum(dim=1)
    mask = cumsum == (n + 1)  # n+1 because n=0 means 1st from back
    result = tensor & mask.flip(dims=[1])
    return result


def compute_mask(
    padded_sequence_length: int,
    prompt_mask: torch.Tensor,
    target_mask: torch.Tensor,
    obfuscate_over: str,
    ignore_last_token: bool = True,
    attention_mask: torch.Tensor | None = None,
):
    obfuscate_over = obfuscate_over.lower().replace("_", "-")
    if prompt_mask is None:
        assert attention_mask is not None, "Attention mask is required if prompt mask is not provided"
        prompt_mask = attention_mask & (~target_mask)
    # Both masks should have one less token than the `tokens`
    if ignore_last_token:
        prompt_mask = prompt_mask[:, : padded_sequence_length - 1]
        target_mask = target_mask[:, : padded_sequence_length - 1]
        if attention_mask is not None:
            attention_mask = attention_mask[:, : padded_sequence_length - 1]

    # Ensure that the tokens, prompt mask, and target mask have the same length
    assert padded_sequence_length - ignore_last_token == prompt_mask.shape[1] == target_mask.shape[1], (
        "Tokens, prompt mask, and target mask must have the same length. "
        f"Padded sequence length: {padded_sequence_length}, "
        f"Prompt mask shape: {prompt_mask.shape[1]}, "
        f"Target mask shape: {target_mask.shape[1]}"
    )

    # Prompt mask and target mask are both B x L tensors
    if obfuscate_over == "full-prompt":
        # Use the entire prompt as the mask
        new_mask = prompt_mask
    elif obfuscate_over == "last-token-prompt":
        # Use the last token of the prompt as the mask
        new_mask = keep_nth_true_from_back(prompt_mask)
    elif obfuscate_over == "generation":
        # Use the last token of the prompt and the target as the mask
        new_mask = keep_nth_true_from_back(prompt_mask) | target_mask
    elif obfuscate_over == "last-token-generation":
        new_mask = keep_nth_true_from_back(target_mask)
    elif obfuscate_over == "second-last-token-generation":
        new_mask = keep_nth_true_from_back(target_mask, n=1)
    elif obfuscate_over == "full":
        # Use the entire sequence as the mask
        new_mask = prompt_mask | target_mask
    else:
        raise ValueError(f"Unknown obfuscate_over value: {obfuscate_over}")

    return new_mask


def trim_sequences(mask_2d):
    """Trim leading and trailing all-zero sequences for efficiency and correctness."""
    # Find first and last valid positions across the entire batch
    any_valid_per_position = mask_2d.any(dim=0)  # Shape: (seq_len,)

    if not any_valid_per_position.any():
        return 0, 1  # return a single token to maintain shape

    # Find the first and last positions that have any valid tokens
    valid_positions = torch.where(any_valid_per_position)[0]
    start_pos = valid_positions[0].item()
    end_pos = valid_positions[-1].item() + 1  # +1 for inclusive end

    return start_pos, end_pos
