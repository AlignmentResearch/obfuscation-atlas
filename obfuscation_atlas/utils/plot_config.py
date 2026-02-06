"""Common plot configuration for ICML paper-ready figures.

This module provides shared styling, colors, and display name mappings
used across all analysis scripts to ensure consistent visualization.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ICML Figure Dimensions (in inches)
# =============================================================================
ICML_COLUMN_WIDTH = 3.25  # Single column width
ICML_TEXT_WIDTH = 6.875  # Full text width (two columns)
ICML_FONT_SIZE = 8  # Standard font size for ICML figures

# =============================================================================
# Color Palette (colorblind-friendly)
# =============================================================================
COLORBLIND_PALETTE = sns.color_palette("colorblind")

# Model colors using colorblind-friendly palette
MODEL_COLORS = {
    "l8b": COLORBLIND_PALETTE[0],  # blue
    "l70b": COLORBLIND_PALETTE[1],  # orange
    "g1b": COLORBLIND_PALETTE[2],  # green
    "g12b": COLORBLIND_PALETTE[3],  # red
    "g27b": COLORBLIND_PALETTE[4],  # purple
}

# Run type colors using colorblind-friendly palette
RUN_TYPE_COLORS = {
    "normal_rl": COLORBLIND_PALETTE[0],  # blue
    "policy_gradient": COLORBLIND_PALETTE[1],  # orange
    "corrected_gradient": COLORBLIND_PALETTE[2],  # green
}

# =============================================================================
# Display Name Mappings
# =============================================================================

# Model short names to full display names
MODEL_DISPLAY_NAMES = {
    "l8b": "Llama-3-8B",
    "l70b": "Llama-3-70B",
    "g1b": "Gemma-3-1B",
    "g12b": "Gemma-3-12B",
    "g27b": "Gemma-3-27B",
}

# Probe dataset display names
PROBE_DISPLAY_NAMES = {
    "mbpp_probe": "MBPP",
    "deception_probe": "Simple Deception",
    "diverse_deception_probe": "Diverse Deception",
}

# Run type display names
RUN_TYPE_DISPLAY_NAMES = {
    "normal_rl": "Normal RL",
    "policy_gradient": "Policy Gradient",
    "corrected_gradient": "Corrected Gradient",
}

# =============================================================================
# Model ordering for consistent plot layout
# =============================================================================
MODEL_ORDER = ["l8b", "g12b", "g27b", "l70b"]

# =============================================================================
# Marker styles for different probe datasets
# =============================================================================
PROBE_MARKERS = {
    "mbpp_probe": "o",  # circle
    "deception_probe": "s",  # square
    "diverse_deception_probe": "^",  # triangle up
}


def get_model_display_name(model: str) -> str:
    """Get display name for a model.

    Args:
        model: Model short name (e.g., "l8b")

    Returns:
        Display name (e.g., "Llama-3-8B")
    """
    return MODEL_DISPLAY_NAMES.get(model, model)


def get_probe_display_name(probe: str) -> str:
    """Get display name for a probe dataset.

    Args:
        probe: Probe short name (e.g., "mbpp_probe")

    Returns:
        Display name (e.g., "MBPP")
    """
    return PROBE_DISPLAY_NAMES.get(probe, probe)


def get_model_color(model: str) -> tuple:
    """Get color for a model.

    Args:
        model: Model short name (e.g., "l8b")

    Returns:
        Color tuple from colorblind palette
    """
    return MODEL_COLORS.get(model, "#333333")


def get_ordered_models(models: set[str]) -> list[str]:
    """Get models in consistent order.

    Args:
        models: Set of model short names

    Returns:
        List of models in standard order, with any unknown models at the end
    """
    ordered = [m for m in MODEL_ORDER if m in models]
    ordered.extend(sorted(m for m in models if m not in MODEL_ORDER))
    return ordered


def apply_icml_style(
    figsize: Optional[tuple[float, float]] = None,
    font_size: int = ICML_FONT_SIZE,
) -> None:
    """Apply ICML paper styling to matplotlib.

    Args:
        figsize: Figure size in inches (width, height). Defaults to column width.
        font_size: Font size for all text elements
    """
    if figsize is None:
        figsize = (ICML_COLUMN_WIDTH, ICML_COLUMN_WIDTH * 0.8)

    style = {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "figure.figsize": figsize,
        "figure.constrained_layout.use": True,
    }
    plt.rcParams.update(style)


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    """Save figure in both PNG and PDF formats.

    Args:
        fig: matplotlib Figure to save
        output_path: Base path for the figure (e.g., 'plot.png'). PDF will be saved
            with the same name but .pdf extension.
        dpi: Resolution for PNG output
    """
    output_path = Path(output_path)

    # Save PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved PNG to {output_path}")

    # Save PDF (same path but with .pdf extension)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF to {pdf_path}")


def create_figure(
    figsize: Optional[tuple[float, float]] = None,
    layout: str = "constrained",
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with ICML styling applied.

    Args:
        figsize: Figure size in inches. Defaults to ICML column width.
        layout: Layout engine ("constrained", "tight", or None)

    Returns:
        Tuple of (Figure, Axes)
    """
    if figsize is None:
        figsize = (ICML_COLUMN_WIDTH, ICML_COLUMN_WIDTH * 0.8)

    apply_icml_style(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize, layout=layout)
    return fig, ax
