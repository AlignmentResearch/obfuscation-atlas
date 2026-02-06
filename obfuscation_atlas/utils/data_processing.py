# type: ignore
"""Data preparation and tokenization utilities."""

import hashlib

import torch

from obfuscation_atlas.utils.visualization import debug_print_tokenized_examples


def hash_prompts_and_completions(prompts: list[str], completions: list[str], label: str = "") -> str:
    """Hash prompts and completions to detect divergence across ranks."""
    import os

    # Use environment variables for rank (works before dist.init)
    rank = int(os.environ.get("RANK", "0"))

    # Hash first few and last few examples
    n_samples = min(10, len(prompts))
    content_parts = []

    indices = list(range(n_samples)) + list(range(len(prompts) - n_samples, len(prompts)))
    indices = sorted(set(i for i in indices if 0 <= i < len(prompts)))

    for i in indices:
        content_parts.append(f"p_{i}:{prompts[i][:100]}")
        content_parts.append(f"c_{i}:{completions[i][:100]}")

    content = "|".join(content_parts)
    hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
    print(f"[Rank {rank}] Prompts hash ({label}): {hash_val}, len={len(prompts)}", flush=True)
    return hash_val


def get_end_of_turn_text(tokenizer):
    user_message = "Hello"
    messages = [
        {"role": "user", "content": user_message},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return formatted.rsplit(user_message, 1)[1]


def process_data(
    prompts: list[str],
    completions: list[str],
    tokenizer,
    batch_size: int | None = None,
    truncate_fraction: float = 0.0,
    max_completion_length: int | None = None,
    max_sequence_length: int | None = None,
    padding_side: str = "right",
    append_eos_to_targets: bool = True,
    follow_up_prompts: list[tuple[str, str]] = [],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the tensor of tokens, prompt, and completion masks (appropriately padded & truncated)
        along with the follow-up prompts in every example.

    The max_completion_length and max_sequence_length parameters are used to truncate the original sequence
        and do not apply on follow-up prompts.

    Args:
        prompts: List of prompt texts.
        completions: List of target/completion texts.
        tokenizer: The tokenizer to use for encoding.
        batch_size: Number of examples to process. If None, all examples are processed.
            Otherwise, only the first batch_size examples are processed.
        truncate_fraction: The fraction of the longest sequence to truncate.
        max_completion_length: The maximum length of the completion.
        max_sequence_length: The maximum length of the sequence (prompt + completion).
        padding_side: The padding side.
        append_eos_to_targets: Whether to append the EOS token to the completions.
            EOS token is appended to the completions if it is not already present.
        follow_up_prompts: List of follow-up (user, assistant) message pairs.
            If provided, each example is duplicated for each follow-up prompt, with the
            follow-up user and assistant messages added to the end. This serves as data
            augmentation. Truncations are not applied to the follow-up prompts.

    Returns:
        tokens: The tensor of tokens. If N follow-up prompts are provided, the output
            will have N times as many examples (each original example repeated N times).
        prompt_mask: The mask for the prompt tokens.
        completion_mask: The mask for the target/completion tokens.
            Follow-up prompts (if provided) are included in the target mask.
        kept_examples_mask: Boolean mask of shape (original_batch_size,) where True indicates
            the example was kept (not dropped due to exceeding max_sequence_length).
    """
    # DEBUG: Hash inputs to detect rank divergence
    hash_prompts_and_completions(prompts, completions, "process_data_input")

    # Validate mutually exclusive parameters
    if truncate_fraction > 0:
        assert max_completion_length is None, "Cannot provide both truncate_fraction and max_completion_length"
        assert max_sequence_length is None, "Cannot provide both truncate_fraction and max_sequence_length"
    if max_completion_length is not None:
        assert max_sequence_length is None, "Cannot provide both max_completion_length and max_sequence_length"

    initial_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"  # Always use right padding internally
    original_batch_size = batch_size or len(prompts)
    batch_size = original_batch_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Prepare texts
    end_of_turn_text = get_end_of_turn_text(tokenizer)
    if append_eos_to_targets:
        completions = [t if t.endswith(end_of_turn_text) else t + end_of_turn_text for t in completions[:batch_size]]

    combined_texts = [p + t for p, t in zip(prompts[:batch_size], completions[:batch_size])]
    prompt_lens = torch.tensor([len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts[:batch_size]])

    # Filter long sequences - track which examples are kept (based on total length, not just prompt)
    kept_examples_mask = torch.ones(original_batch_size, dtype=torch.bool)
    if max_sequence_length is not None:
        total_lens = torch.tensor([len(tokenizer.encode(t, add_special_tokens=False)) for t in combined_texts])
        if total_lens.max() >= max_sequence_length:
            kept_examples_mask = total_lens < max_sequence_length
            combined_texts = [t for t, keep in zip(combined_texts, kept_examples_mask) if keep]
            # Print length distribution before filtering
            print(f"Total sequence length distribution (max_seq_len={max_sequence_length}):")
            print(
                f"  min={total_lens.min().item()}, max={total_lens.max().item()}, "
                f"mean={total_lens.float().mean().item():.1f}"
            )
            percentiles = [50, 75, 90, 95, 99]
            pct_values = [torch.quantile(total_lens.float(), p / 100).item() for p in percentiles]
            print("  percentiles: " + ", ".join([f"p{p}={int(v)}" for p, v in zip(percentiles, pct_values)]))
            n_over_512 = (total_lens >= 512).sum().item()
            n_over_1024 = (total_lens >= 1024).sum().item()
            n_over_2048 = (total_lens >= 2048).sum().item()
            print(f"  examples >=512: {n_over_512}, >=1024: {n_over_1024}, >=2048: {n_over_2048}")
            prompt_lens = prompt_lens[kept_examples_mask]
            batch_size = len(combined_texts)
            assert batch_size > 0, "No sequences left after filtering"
            print(f"Warning: Dropping {(~kept_examples_mask).sum()}/{original_batch_size} examples")

    # Tokenize (with right padding)
    tokenized = tokenizer(
        combined_texts, add_special_tokens=False, padding=True, return_tensors="pt", max_length=max_sequence_length
    )
    tokens = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Apply truncation
    if 0 < truncate_fraction < 1:
        seq_lens = attention_mask.sum(dim=1)
        max_length = max(1, int(torch.quantile(seq_lens.float(), 1.0 - truncate_fraction).item()))
        tokens = tokens[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
    elif max_completion_length is not None:
        # Vectorized truncation - this is worth it
        seq_lens = attention_mask.sum(dim=1)
        truncate_lens = torch.minimum(prompt_lens + max_completion_length, seq_lens)
        max_length = truncate_lens.max().item()

        indices = torch.arange(max_length, device=tokens.device).unsqueeze(0)
        keep_mask = indices < truncate_lens.unsqueeze(1)

        new_tokens = torch.full((batch_size, max_length), pad_token_id, dtype=tokens.dtype, device=tokens.device)
        new_tokens[keep_mask] = tokens[:, :max_length][keep_mask]
        tokens = new_tokens
        attention_mask = keep_mask.long()

    # Add follow-up prompts (data augmentation - each example is duplicated for each follow-up prompt)
    if follow_up_prompts is not None and len(follow_up_prompts) > 0:
        num_follow_ups = len(follow_up_prompts)

        # Pre-tokenize all follow-up prompts
        follow_up_tokens_list = []
        max_follow_up_len = 0
        for user_msg, assistant_msg in follow_up_prompts:
            # Use tokenizer.apply_chat_template for model-agnostic formatting
            follow_up_messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            follow_up_text = tokenizer.apply_chat_template(
                follow_up_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Strip BOS token if present since this is appended to existing text
            if tokenizer.bos_token and follow_up_text.startswith(tokenizer.bos_token):
                follow_up_text = follow_up_text[len(tokenizer.bos_token) :]
            # strip out anything after the last instance of the assistant message
            follow_up_text = follow_up_text.removesuffix(end_of_turn_text)
            follow_up_tokens = tokenizer.encode(follow_up_text, add_special_tokens=False, return_tensors="pt")[0]
            follow_up_tokens_list.append(follow_up_tokens)
            max_follow_up_len = max(max_follow_up_len, len(follow_up_tokens))

        valid_lens = attention_mask.sum(dim=1)
        max_length = valid_lens.max().item() + max_follow_up_len
        new_batch_size = batch_size * num_follow_ups

        new_tokens = torch.full((new_batch_size, max_length), pad_token_id, dtype=tokens.dtype, device=tokens.device)
        new_attention = torch.zeros(
            (new_batch_size, max_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        new_prompt_lens = prompt_lens.repeat(num_follow_ups)

        for fu_idx, follow_up_tokens in enumerate(follow_up_tokens_list):
            for i in range(batch_size):
                new_idx = fu_idx * batch_size + i
                vlen = valid_lens[i].item()
                new_tokens[new_idx, :vlen] = tokens[i, :vlen]
                new_tokens[new_idx, vlen : vlen + len(follow_up_tokens)] = follow_up_tokens
                new_attention[new_idx, : vlen + len(follow_up_tokens)] = 1

        tokens = new_tokens
        attention_mask = new_attention
        prompt_lens = new_prompt_lens
        batch_size = new_batch_size

    # Create masks (vectorized - simple and clear)
    max_length = tokens.shape[1]
    indices = torch.arange(max_length, device=tokens.device).unsqueeze(0)
    prompt_end = prompt_lens.unsqueeze(1)

    prompt_mask = (indices < prompt_end) & attention_mask.bool()
    completion_mask = (indices >= prompt_end) & attention_mask.bool()

    # Convert to left padding if requested (simple loop - clearer and lower memory)
    if padding_side == "left":
        valid_lens = attention_mask.sum(dim=1)

        new_tokens = torch.full_like(tokens, pad_token_id)
        new_prompt_mask = torch.zeros_like(prompt_mask)
        new_completion_mask = torch.zeros_like(completion_mask)

        for i in range(batch_size):
            vlen = valid_lens[i].item()
            offset = max_length - vlen
            new_tokens[i, offset:] = tokens[i, :vlen]
            new_prompt_mask[i, offset:] = prompt_mask[i, :vlen]
            new_completion_mask[i, offset:] = completion_mask[i, :vlen]

        tokens = new_tokens
        prompt_mask = new_prompt_mask
        completion_mask = new_completion_mask

    tokenizer.padding_side = initial_padding_side

    debug_print_tokenized_examples(tokens, prompt_mask, completion_mask, tokenizer, n_examples=2, label="process_data")

    return tokens, prompt_mask, completion_mask, kept_examples_mask
