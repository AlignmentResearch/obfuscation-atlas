# type: ignore
"""Visualization and debug utilities for tokenized data."""

import matplotlib
import plotly.graph_objects as go
import plotly.io as pio
import torch


def _compress_padding(toks: list[int], pad_id: int) -> str:
    """Compress runs of padding tokens in a token list for display."""
    if not toks:
        return "[]"
    result = []
    pad_count = 0
    for t in toks:
        if t == pad_id:
            pad_count += 1
        else:
            if pad_count > 0:
                result.append(f"...<{pad_count} pad>...")
                pad_count = 0
            result.append(str(t))
    if pad_count > 0:
        result.append(f"...<{pad_count} pad>...")
    return "[" + ", ".join(result) + "]"


def _compress_padding_in_text(text: str, pad_token: str) -> str:
    """Compress runs of padding tokens in decoded text."""
    if not pad_token or pad_token not in text:
        return text
    import re

    escaped = re.escape(pad_token)
    pattern = f"({escaped}){{2,}}"
    return re.sub(pattern, lambda m: f"...<{len(m.group()) // len(pad_token)} pad>...", text)


def debug_print_tokenized_examples(
    tokens: torch.Tensor | list[list[int]],
    prompt_mask: torch.Tensor | list[list[bool]] | None,
    completion_mask: torch.Tensor | list[list[bool]],
    tokenizer,
    n_examples: int = 2,
    label: str = "",
) -> None:
    """Print debug info showing exactly what's going into the model.

    Args:
        tokens: Token IDs - either tensor (batch_size, seq_len) or list of lists.
        prompt_mask: Boolean mask for prompt tokens. If None, inferred as ~completion_mask.
        completion_mask: Boolean mask for completion tokens.
        tokenizer: Tokenizer for decoding.
        n_examples: Number of examples to print.
        label: Optional label for the debug output.
    """
    import os

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return

    is_list = isinstance(tokens, list)
    batch_size = len(tokens) if is_list else tokens.shape[0]
    n_debug = min(n_examples, batch_size)
    label_str = f" ({label})" if label else ""
    if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
        raise ValueError("Tokenizer must have pad_token_id and pad_token set")
    pad_id = tokenizer.pad_token_id
    pad_token = tokenizer.pad_token
    print("=" * 80)
    print(f"DEBUG{label_str}: Showing {n_debug}/{batch_size} example(s) after preprocessing")

    for i in range(n_debug):
        print(f"\n--- Example {i} ---")

        if is_list:
            toks = tokens[i]
            comp_mask = completion_mask[i]
            prom_mask = [not c for c in comp_mask] if prompt_mask is None else prompt_mask[i]
        else:
            toks = tokens[i].tolist()
            comp_mask = completion_mask[i]
            prom_mask = ~comp_mask if prompt_mask is None else prompt_mask[i]

        print(f"Token IDs ({len(toks)}): {_compress_padding(toks, pad_id)}")
        full_text = tokenizer.decode(toks, skip_special_tokens=False)
        print(f"Full text:\n{_compress_padding_in_text(full_text, pad_token)}")

        # Prompt portion
        if is_list:
            prompt_toks = [t for t, m in zip(toks, prom_mask) if m]
        else:
            prompt_toks = tokens[i][prom_mask].tolist()
        prompt_text = tokenizer.decode(prompt_toks, skip_special_tokens=False)
        print(f"\nPROMPT ({len(prompt_toks)} tokens):\n{_compress_padding_in_text(prompt_text, pad_token)}")

        # Completion portion
        if is_list:
            completion_toks = [t for t, m in zip(toks, comp_mask) if m]
        else:
            completion_toks = tokens[i][comp_mask].tolist()
        completion_text = tokenizer.decode(completion_toks, skip_special_tokens=False)
        print(f"\nCOMPLETION ({len(completion_toks)} tokens):\n{_compress_padding_in_text(completion_text, pad_token)}")

    print("=" * 80)


def apply_style(figsize, px_margin=None, px_use_default=True, font=10):
    style = {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": font,
        "legend.fontsize": font,
        "axes.titlesize": font,
        "axes.labelsize": font,
        "xtick.labelsize": font,
        "ytick.labelsize": font,
        "figure.figsize": figsize,
        "figure.constrained_layout.use": True,
    }
    matplotlib.rcParams.update(style)

    # Convert figure size from inches to pixels (assuming ~96 DPI)
    width_pixels = int(figsize[0] * 96)
    height_pixels = int(figsize[1] * 96)

    custom_template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Times, Times New Roman, serif", size=font),
            legend=dict(font=dict(size=font)),
            xaxis=dict(title=dict(font=dict(size=font)), tickfont=dict(size=font)),
            yaxis=dict(title=dict(font=dict(size=font)), tickfont=dict(size=font)),
            width=width_pixels,
            height=height_pixels,
            # If you need a layout with tight margins, adjust the margin dict as necessary
            margin=px_margin,  # dict(l=50, r=50, t=50, b=50)
        )
    )
    pio.templates["custom"] = custom_template
    if not px_use_default:
        pio.templates.default = "custom"
    return width_pixels, height_pixels
