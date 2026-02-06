"""
Logit-space violin plots for sigmoid classifier outputs. Created by Claude.

This version creates a LogitKDE class that extends seaborn's KDE,
then monkey-patches it into sns.violinplot for seamless integration.
"""

import warnings
from dataclasses import dataclass
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from pandas import DataFrame
from scipy.special import expit, logit
from scipy.stats import gaussian_kde
from seaborn._stats.density import KDE
from seaborn.categorical import _CategoricalPlotter
from seaborn.utils import _check_argument, _default_color, _get_patch_legend_artist, _get_transform_functions


@dataclass
class LogitKDE(KDE):
    """
    KDE computed in logit space with Jacobian correction, plus optional
    boundary-stabilized *adaptive bandwidth* to prevent spurious spikes near 0/1.

    Key idea:
      A fixed bandwidth in logit space implies bandwidth in probability space
      proportional to p(1-p), which collapses near 0/1 and can create artifacts.
      We counteract this by inflating bandwidth for samples near 0/1.

    Parameters (in addition to seaborn's KDE):
    ----------------------------------------
    adaptive_bw : bool
        If True, use per-observation bandwidth inflation to avoid boundary spikes.
    adaptive_exponent : float
        Strength of boundary inflation. 0 -> fixed bandwidth (original behavior).
        0.5 is a good default.
    clip_eps : float
        Clip probabilities away from 0/1 to avoid infinities in logit/Jacobian.
    """

    bw_adjust: float = 1
    bw_method: str | float | Callable[[gaussian_kde], float] = "scott"
    common_norm: bool | list[str] = True
    common_grid: bool | list[str] = True
    gridsize: int | None = 10000
    cut: float = 3  # interpreted in *logit* space now
    cumulative: bool = False

    density_threshold: float = 0.0  # used by your plotter for display masking only
    clip_eps: float = 1e-6

    adaptive_bw: bool = True
    adaptive_exponent: float = 0.5

    # ---------- helpers ----------
    @staticmethod
    def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
        """Weighted median with nonnegative weights."""
        order = np.argsort(x)
        x = x[order]
        w = w[order]
        cdf = np.cumsum(w)
        cutoff = 0.5 * cdf[-1]
        return x[np.searchsorted(cdf, cutoff, side="left")]

    def _prep(self, data: DataFrame, orient: str):
        """Extract values + weights and compute logits."""
        values = data[orient].to_numpy(dtype=float)
        values = np.clip(values, self.clip_eps, 1 - self.clip_eps)

        if "weight" in data:
            w = data["weight"].to_numpy(dtype=float)
        else:
            w = np.ones_like(values, dtype=float)

        wsum = w.sum()
        if not np.isfinite(wsum) or wsum <= 0:
            raise ValueError("Nonpositive or invalid weights.")

        w_norm = w / wsum
        z = logit(values)
        return values, z, w, w_norm

    def _global_kde_and_bw(self, z: np.ndarray, w_norm: np.ndarray) -> tuple[gaussian_kde, float]:
        """Fit gaussian_kde in logit space and return (kde, global_bandwidth_std)."""
        kde = gaussian_kde(z, bw_method=self.bw_method, weights=w_norm)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        # In 1D, covariance is 1x1; bandwidth std is sqrt(cov)
        bw = float(np.sqrt(np.squeeze(kde.covariance)))
        if not np.isfinite(bw) or bw <= 0:
            raise ValueError("Degenerate bandwidth.")
        return kde, bw

    # ---------- seaborn hooks ----------
    def _get_support(self, data: DataFrame, orient: str) -> ndarray:
        """
        Support grid:
          uniform in logit space, then transformed back to probability space.

        This matches the geometry of the KDE and avoids odd sampling artifacts
        near 0/1.
        """
        if self.gridsize is None:
            return data[orient].to_numpy(dtype=float)

        values, z, _, w_norm = self._prep(data, orient)
        _, bw = self._global_kde_and_bw(z, w_norm)

        zmin = z.min() - self.cut * bw
        zmax = z.max() + self.cut * bw

        # clamp to finite logit range implied by clip_eps
        zmin = max(zmin, logit(self.clip_eps))
        zmax = min(zmax, logit(1 - self.clip_eps))

        z_grid = np.linspace(zmin, zmax, self.gridsize)
        return expit(z_grid)

    def _fit_and_evaluate(self, data: DataFrame, orient: str, support: ndarray) -> DataFrame:
        """Fit KDE in logit space, evaluate in probability space with Jacobian correction."""
        empty = pd.DataFrame(columns=[orient, "weight", "density"], dtype=float)
        if len(data) < 2:
            return empty

        try:
            values, z, w, w_norm = self._prep(data, orient)
            kde, bw_global = self._global_kde_and_bw(z, w_norm)
        except (np.linalg.LinAlgError, ValueError):
            return empty

        # Clip support for stability (avoid exactly 0/1)
        p = np.clip(np.asarray(support, dtype=float), self.clip_eps, 1 - self.clip_eps)
        z_support = logit(p)

        if not self.adaptive_bw or self.adaptive_exponent <= 0:
            # Original fixed-bandwidth behavior
            fz = kde(z_support)

        else:
            # Adaptive bandwidth inflation near boundaries to prevent spikes
            scale_i = values * (1 - values)
            scale_i = np.maximum(scale_i, self.clip_eps * (1 - self.clip_eps))

            # Reference scale: weighted median of p(1-p) (robust, avoids over-inflating)
            scale_ref = self._weighted_median(scale_i, w)
            scale_ref = max(scale_ref, self.clip_eps * (1 - self.clip_eps))

            h_i = bw_global * (scale_ref / scale_i) ** float(self.adaptive_exponent)

            # Evaluate mixture of Gaussians with per-point bandwidths
            # fz(z) = sum_i w_norm[i] * N(z | z_i, h_i^2)
            #       = sum_i (w_norm[i]/h_i) * phi((z - z_i)/h_i)
            u = (z_support[:, None] - z[None, :]) / h_i[None, :]
            # standard normal pdf without scipy.stats.norm overhead
            pdf = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
            coef = w_norm / h_i
            fz = pdf @ coef

        # Jacobian correction to return to probability space
        with np.errstate(divide="ignore", invalid="ignore"):
            density = fz / (p * (1 - p))

        weight_total = float(w.sum())
        return pd.DataFrame({orient: p, "weight": weight_total, "density": density})


def violinplot_using_logit_kde(
    data=None,
    *,
    x=None,
    y=None,
    hue=None,
    order=None,
    hue_order=None,
    orient=None,
    color=None,
    palette=None,
    saturation=0.75,
    fill=True,
    inner="box",
    split=False,
    width=0.8,
    dodge="auto",
    gap=0,
    linewidth=None,
    linecolor="auto",
    gridsize=10000,
    bw_method="scott",
    bw_adjust=1,
    density_norm="area",
    common_norm=False,
    hue_norm=None,
    formatter=None,
    log_scale=None,
    native_scale=False,
    legend="auto",
    density_threshold=0.0,
    inner_kws=None,
    ax=None,
    **kwargs,
):
    """
    Draw violin plots using logit-space KDE for sigmoid classifier outputs.

    This is a drop-in replacement for seaborn.violinplot. All parameters are
    identical except:
    - No `cut` parameter (not needed - properly bounded)
    - Added `density_threshold` to clip insignificant tails (default 0.0 = disabled)

    The KDE is computed in logit space and transformed back with Jacobian
    correction, preserving 100% probability mass within [0, 1].
    """

    if ax is None:
        ax = plt.gca()

    p = _CategoricalPlotter(
        data=data,
        variables=dict(x=x, y=y, hue=hue),
        order=order,
        orient=orient,
        color=color,
        legend=legend,
    )

    if p.plot_data.empty:
        return ax

    if dodge == "auto":
        dodge = p._dodge_needed()

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax, log_scale=log_scale)

    # Handle palette
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    saturation = saturation if fill else 1
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm, saturation=saturation)
    color = _default_color(
        ax.fill_between,
        hue,
        color,
        {k: v for k, v in kwargs.items() if k in ["c", "color", "fc", "facecolor"]},
        saturation=saturation,
    )
    linecolor = p._complement_color(linecolor, color, p._hue_map)

    # Create LogitKDE instead of regular KDE
    kde_kws = dict(
        gridsize=gridsize,
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        density_threshold=density_threshold,
    )
    inner_kws = {} if inner_kws is None else inner_kws.copy()

    # Call plot_violins with our custom KDE
    _plot_violins_logit(
        p,
        width=width,
        dodge=dodge,
        gap=gap,
        split=split,
        color=color,
        fill=fill,
        linecolor=linecolor,
        linewidth=linewidth,
        inner=inner,
        density_norm=density_norm,
        common_norm=common_norm,
        kde_kws=kde_kws,
        inner_kws=inner_kws,
        plot_kws=kwargs,
    )

    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    # Set axis limits to [0, 1] with ticks at 0.0, 0.1, ..., 1.0
    if p.orient == "x":
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
    else:
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))

    return ax


def _plot_violins_logit(
    p,
    width,
    dodge,
    gap,
    split,
    color,
    fill,
    linecolor,
    linewidth,
    inner,
    density_norm,
    common_norm,
    kde_kws,
    inner_kws,
    plot_kws,
):
    """
    Plot violins using LogitKDE.
    This is adapted from seaborn's _CategoricalPlotter.plot_violins.
    """
    iter_vars = [p.orient, "hue"]
    value_var = {"x": "y", "y": "x"}[p.orient]

    inner_options = ["box", "quart", "stick", "point", None]
    _check_argument("inner", inner_options, inner, prefix=True)
    _check_argument("density_norm", ["area", "count", "width"], density_norm)

    if linewidth is None:
        if fill:
            linewidth = 1.25 * mpl.rcParams["patch.linewidth"]
        else:
            linewidth = mpl.rcParams["lines.linewidth"]

    if inner is not None and inner.startswith("box"):
        box_width = inner_kws.pop("box_width", linewidth * 4.5)
        whis_width = inner_kws.pop("whis_width", box_width / 3)
        marker = inner_kws.pop("marker", "_" if p.orient == "x" else "|")

    # Use LogitKDE
    kde = LogitKDE(**kde_kws)
    ax = p.ax
    violin_data = []

    # Compute KDEs for all groups
    for sub_vars, sub_data in p.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
        sub_data = sub_data.assign(weight=sub_data.get("weights", 1))
        stat_data = kde._transform(sub_data, value_var, [])

        maincolor = p._hue_map(sub_vars["hue"]) if "hue" in sub_vars else color
        if not fill:
            linecolor_v = maincolor
            maincolor = "none"
        else:
            linecolor_v = linecolor

        default_kws = dict(
            facecolor=maincolor,
            edgecolor=linecolor_v,
            linewidth=linewidth,
        )

        violin_data.append(
            {
                "position": sub_vars[p.orient],
                "observations": sub_data[value_var],
                "density": stat_data["density"].to_numpy() if len(stat_data) else np.array([]),
                "support": stat_data[value_var].to_numpy() if len(stat_data) else np.array([]),
                "kwargs": {**default_kws, **plot_kws},
                "sub_vars": sub_vars,
                "ax": p._get_axes(sub_vars),
            }
        )

    # Normalization
    def vars_to_key(sub_vars):
        return tuple((k, v) for k, v in sub_vars.items() if k != p.orient)

    norm_keys = [vars_to_key(v["sub_vars"]) for v in violin_data]
    if common_norm:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN")
            common_max_density = np.nanmax([v["density"].max() if len(v["density"]) else 0 for v in violin_data])
            common_max_count = np.nanmax([len(v["observations"]) for v in violin_data])
        max_density = {key: common_max_density for key in norm_keys}
        max_count = {key: common_max_count for key in norm_keys}
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN")
            max_density = {
                key: np.nanmax(
                    [
                        v["density"].max() if len(v["density"]) else 0
                        for v in violin_data
                        if vars_to_key(v["sub_vars"]) == key
                    ]
                )
                for key in norm_keys
            }
        max_count = {
            key: np.nanmax([len(v["observations"]) for v in violin_data if vars_to_key(v["sub_vars"]) == key])
            for key in norm_keys
        }

    real_width = width * p._native_width

    # Plot each violin
    for violin in violin_data:
        if len(violin["support"]) == 0:
            continue

        index = pd.RangeIndex(0, max(len(violin["support"]), 1))
        data = pd.DataFrame(
            {
                p.orient: violin["position"],
                value_var: violin["support"],
                "density": violin["density"],
                "width": real_width,
            },
            index=index,
        )

        if dodge:
            p._dodge(violin["sub_vars"], data)
        if gap:
            data["width"] *= 1 - gap

        # Normalize density
        norm_key = vars_to_key(violin["sub_vars"])
        hw = data["width"] / 2
        peak_density = violin["density"].max() if len(violin["density"]) else 1

        if np.isnan(peak_density) or peak_density == 0:
            continue

        if density_norm == "area":
            span = data["density"] / max_density[norm_key]
        elif density_norm == "count":
            count = len(violin["observations"])
            span = data["density"] / peak_density * (count / max_count[norm_key])
        elif density_norm == "width":
            span = data["density"] / peak_density
        span = span * hw * (2 if split else 1)

        # Handle split violins
        right_side = 0 if "hue" not in p.variables else p._hue_map.levels.index(violin["sub_vars"]["hue"]) % 2
        if split:
            offsets = (hw, span - hw) if right_side else (span - hw, hw)
        else:
            offsets = span, span

        ax = violin["ax"]
        _, invx = _get_transform_functions(ax, "x")
        _, invy = _get_transform_functions(ax, "y")
        inv_pos = {"x": invx, "y": invy}[p.orient]
        inv_val = {"x": invx, "y": invy}[value_var]

        linecolor_v = violin["kwargs"]["edgecolor"]

        # Plot violin body
        plot_func = {"x": ax.fill_betweenx, "y": ax.fill_between}[p.orient]

        # Mask density for display only (keeps full arrays for quartile interpolation)
        display_val = inv_val(data[value_var])
        display_low = inv_pos(data[p.orient] - offsets[0])
        display_high = inv_pos(data[p.orient] + offsets[1])

        if kde_kws.get("density_threshold", 0) > 0:
            threshold = kde_kws["density_threshold"] * peak_density
            mask = data["density"] < threshold
            display_low = np.where(mask, np.nan, display_low)
            display_high = np.where(mask, np.nan, display_high)

        plot_func(display_val, display_low, display_high, **violin["kwargs"])

        # Inner components
        obs = violin["observations"]
        pos_dict = {p.orient: violin["position"], "width": real_width}
        if dodge:
            p._dodge(violin["sub_vars"], pos_dict)
        if gap:
            pos_dict["width"] *= 1 - gap

        if inner is None:
            continue

        elif inner.startswith("point"):
            pos = np.array([pos_dict[p.orient]] * len(obs))
            if split:
                pos += (-1 if right_side else 1) * pos_dict["width"] / 2
            x, y = (pos, obs) if p.orient == "x" else (obs, pos)
            kws = {
                "color": linecolor_v,
                "edgecolor": linecolor_v,
                "s": (linewidth * 2) ** 2,
                "zorder": violin["kwargs"].get("zorder", 2) + 1,
                **inner_kws,
            }
            ax.scatter(invx(x), invy(y), **kws)

        elif inner.startswith("stick"):
            pos0 = np.interp(obs, data[value_var], data[p.orient] - offsets[0])
            pos1 = np.interp(obs, data[value_var], data[p.orient] + offsets[1])
            pos_pts = np.stack([inv_pos(pos0), inv_pos(pos1)])
            val_pts = np.stack([inv_val(obs), inv_val(obs)])
            segments = np.stack([pos_pts, val_pts]).transpose(2, 1, 0)
            if p.orient == "y":
                segments = segments[:, :, ::-1]
            kws = {"color": linecolor_v, "linewidth": linewidth / 2, **inner_kws}
            lines = mpl.collections.LineCollection(segments, **kws)
            ax.add_collection(lines, autolim=False)

        elif inner.startswith("quart"):
            stats = np.percentile(obs, [25, 50, 75])

            # Skip quartile lines that fall outside the visible violin
            if kde_kws.get("density_threshold", 0) > 0:
                threshold = kde_kws["density_threshold"] * peak_density
                visible_mask = data["density"] >= threshold
                if visible_mask.any():
                    visible_support = data[value_var][visible_mask]
                    min_visible, max_visible = visible_support.min(), visible_support.max()
                    valid_mask = (stats >= min_visible) & (stats <= max_visible)
                    stats = stats[valid_mask]
                    dashes = [d for d, v in zip([(1.25, 0.75), (2.5, 1), (1.25, 0.75)], valid_mask) if v]
                else:
                    continue
            else:
                dashes = [(1.25, 0.75), (2.5, 1), (1.25, 0.75)]

            if len(stats) == 0:
                continue
            pos0 = np.interp(stats, data[value_var], data[p.orient] - offsets[0])
            pos1 = np.interp(stats, data[value_var], data[p.orient] + offsets[1])
            pos_pts = np.stack([inv_pos(pos0), inv_pos(pos1)])
            val_pts = np.stack([inv_val(stats), inv_val(stats)])
            segments = np.stack([pos_pts, val_pts]).transpose(2, 0, 1)
            if p.orient == "y":
                segments = segments[:, ::-1, :]
            for i, segment in enumerate(segments):
                kws = {"color": linecolor_v, "linewidth": linewidth, "dashes": dashes[i], **inner_kws}
                ax.plot(*segment, **kws)

        elif inner.startswith("box"):
            stats = mpl.cbook.boxplot_stats(obs)[0]
            pos = np.array(pos_dict[p.orient])
            if split:
                pos += (-1 if right_side else 1) * pos_dict["width"] / 2
            pos = [pos, pos], [pos, pos], [pos]
            val = ([stats["whislo"], stats["whishi"]], [stats["q1"], stats["q3"]], [stats["med"]])
            if p.orient == "x":
                (x0, x1, x2), (y0, y1, y2) = pos, val
            else:
                (x0, x1, x2), (y0, y1, y2) = val, pos

            if split:
                offset = (1 if right_side else -1) * box_width / 72 / 2
                dx, dy = (offset, 0) if p.orient == "x" else (0, -offset)
                trans = ax.transData + mpl.transforms.ScaledTranslation(
                    dx,
                    dy,
                    ax.figure.dpi_scale_trans,
                )
            else:
                trans = ax.transData

            line_kws = {"color": linecolor_v, "transform": trans, **inner_kws, "linewidth": whis_width}
            ax.plot(invx(x0), invy(y0), **line_kws)
            line_kws["linewidth"] = box_width
            ax.plot(invx(x1), invy(y1), **line_kws)
            dot_kws = {
                "marker": marker,
                "markersize": box_width / 1.2,
                "markeredgewidth": box_width / 5,
                "transform": trans,
                **inner_kws,
                "markeredgecolor": "w",
                "markerfacecolor": "w",
                "color": linecolor_v,
            }
            ax.plot(invx(x2), invy(y2), **dot_kws)

    legend_artist = _get_patch_legend_artist(fill)
    common_kws = {**plot_kws, "linewidth": linewidth, "edgecolor": linecolor}
    p._configure_legend(ax, legend_artist, common_kws)


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "Score": np.concatenate([expit(np.random.normal(-1.5, 1, n)), expit(np.random.normal(1.5, 1, n))]),
            "Class": ["Negative"] * n + ["Positive"] * n,
            "Split": (["Train"] * (n // 2) + ["Test"] * (n // 2)) * 2,
        }
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Simple case
    sns.violinplot(data=df, x="Class", y="Score", hue="Class", ax=axes[0, 0], cut=0, legend=False)
    axes[0, 0].set_title("seaborn (x=hue)")

    violinplot_using_logit_kde(data=df, x="Class", y="Score", hue="Class", ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("violin_logit (x=hue)")

    violinplot_using_logit_kde(data=df, x="Class", y="Score", ax=axes[0, 2])
    axes[0, 2].set_title("violin_logit (no hue)")

    # Row 2: Split violins
    sns.violinplot(data=df, x="Split", y="Score", hue="Class", split=True, ax=axes[1, 0], cut=0)
    axes[1, 0].set_title("seaborn split")

    violinplot_using_logit_kde(data=df, x="Split", y="Score", hue="Class", split=True, ax=axes[1, 1])
    axes[1, 1].set_title("violin_logit split")

    violinplot_using_logit_kde(data=df, x="Split", y="Score", hue="Class", split=False, ax=axes[1, 2])
    axes[1, 2].set_title("violin_logit dodged")

    plt.tight_layout()
    plt.savefig("violin_v6_comparison.png", dpi=150)
    print("Saved violin_v6_comparison.png")
