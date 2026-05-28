"""
Visualization utilities for genomic sequence model outputs.

Provides standalone matplotlib-based plots for model predictions,
ISM results, attribution maps, and genomic label tracks. No grelu
or external genomics tools are required.

All plotting functions accept an optional ``ax`` parameter so they can be
embedded in larger figures via :func:`multi_track_figure`.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Channel ordering matches one_hot_encode: ACGT
_BASES = ["A", "C", "G", "T"]


def plot_ism_heatmap(
    ism_array: np.ndarray,
    genome_start: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 2),
) -> plt.Axes:
    """
    Plot an ISM result as a heatmap with ACGT rows.

    Designed for the output of :func:`seq_tools.variant.score_variants`:
    shape ``(4, L)`` or ``(L, 4)`` with values such as log2FC.
    The colormap is symmetrically scaled around ``center``.

    Args:
        ism_array: ISM scores, shape ``(4, L)`` or ``(L, 4)``.
        genome_start: If provided, x-axis ticks show genomic coordinates.
        ax: Matplotlib Axes. If None, a new figure is created.
        cmap: Colormap name. Diverging maps (``RdBu_r``, ``seismic``) work best.
        center: Colormap midpoint value.
        title: Axes title.
        figsize: Figure size when creating a new figure.

    Returns:
        Populated Axes object.
    """
    arr = np.asarray(ism_array, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"ism_array must be 2D, got shape {arr.shape}")
    # Normalize to (4, L) — bases on y-axis
    if arr.shape[0] != 4 and arr.shape[1] == 4:
        arr = arr.T
    if arr.shape[0] != 4:
        raise ValueError(
            f"Expected 4 channels (ACGT), got shape {arr.shape}. "
            "Pass array as (4, L) or (L, 4)."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(arr.max()), abs(arr.min())) or 1.0
    im = ax.imshow(
        arr,
        aspect="auto",
        cmap=cmap,
        vmin=center - vmax,
        vmax=center + vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(4))
    ax.set_yticklabels(_BASES)

    n_pos = arr.shape[1]
    if genome_start is not None:
        step = max(1, n_pos // 10)
        ticks = list(range(0, n_pos, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(genome_start + t) for t in ticks], rotation=45, ha="right")
    else:
        ax.set_xlabel("Position")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Score")
    if title:
        ax.set_title(title)
    return ax


def plot_prediction_track(
    predictions: np.ndarray,
    genome_start: Optional[int] = None,
    genome_end: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: str = "#255C99",
    alpha: float = 0.7,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 2),
) -> plt.Axes:
    """
    Plot a 1D bin-level model prediction as a filled area chart.

    Args:
        predictions: 1D array of bin-level values, shape ``(n_bins,)``.
        genome_start: Genomic start (bp). Used with ``genome_end`` to scale x-axis.
        genome_end: Genomic end (bp).
        ax: Matplotlib Axes. If None, a new figure is created.
        label: Legend label.
        color: Fill and line color.
        alpha: Fill transparency.
        ylim: Y-axis limits. Defaults to ``(0, 1.1 * max)``.
        title: Axes title.
        ylabel: Y-axis label.
        figsize: Figure size when creating a new figure.

    Returns:
        Populated Axes object.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if genome_start is not None and genome_end is not None:
        x = np.linspace(genome_start, genome_end, len(preds))
        ax.set_xlabel("Genomic position (bp)")
    else:
        x = np.arange(len(preds))
        ax.set_xlabel("Bin")

    ax.fill_between(x, preds, alpha=alpha, color=color, label=label)
    ax.plot(x, preds, color=color, linewidth=0.5)
    ax.set_xlim(x[0], x[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(preds.max() * 1.1, 1e-6))

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label:
        ax.legend(loc="upper right", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_gene_model(
    label_df,
    ax: Optional[plt.Axes] = None,
    genome_start: Optional[int] = None,
    genome_end: Optional[int] = None,
    exon_color: str = "#333333",
    intron_color: str = "#aaaaaa",
    psi_cmap: str = "RdYlGn",
    use_psi_color: bool = True,
    height: float = 0.5,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 1.5),
) -> plt.Axes:
    """
    Plot a gene model track from a soft-label segment DataFrame.

    Draws exon blocks (rectangles) over an intron line. When
    ``use_psi_color=True``, block fill is scaled by ``p_exon``
    (0 = red via ``psi_cmap``, 1 = green). This complements the
    DataFrame output of :func:`seq_tools.labels.generate_soft_labels`.

    Args:
        label_df: DataFrame with columns ``[Start, End, p_exon]``.
        ax: Matplotlib Axes. If None, a new figure is created.
        genome_start: Clip display to this genomic start.
        genome_end: Clip display to this genomic end.
        exon_color: Block color when ``use_psi_color=False``.
        intron_color: Color of the intron line.
        psi_cmap: Colormap for ``p_exon`` coloring.
        use_psi_color: Color blocks by ``p_exon`` value.
        height: Vertical height of exon rectangles (fraction of axis height).
        title: Axes title.
        figsize: Figure size when creating a new figure.

    Returns:
        Populated Axes object.
    """
    df = label_df.copy()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if df.empty:
        logger.warning("label_df is empty; nothing to plot.")
        return ax

    if genome_start is not None:
        df = df[df["End"] > genome_start]
    if genome_end is not None:
        df = df[df["Start"] < genome_end]

    x_min = genome_start if genome_start is not None else df["Start"].min()
    x_max = genome_end if genome_end is not None else df["End"].max()

    cmap_fn = plt.get_cmap(psi_cmap)
    ax.axhline(0.5, color=intron_color, linewidth=1.5, zorder=1)

    for _, row in df.iterrows():
        s = max(row["Start"], x_min)
        e = min(row["End"], x_max)
        if s >= e:
            continue
        p = float(row.get("p_exon", 1.0))
        if p == 0.0:
            continue  # intron — only the baseline is drawn

        color = cmap_fn(p) if use_psi_color else exon_color
        rect = mpatches.FancyBboxPatch(
            (s, 0.5 - height / 2),
            e - s,
            height,
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(rect)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Genomic position (bp)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    if use_psi_color:
        sm = plt.cm.ScalarMappable(cmap=cmap_fn, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02, label="p(exon)")

    if title:
        ax.set_title(title)
    return ax


def plot_attribution(
    attribution: np.ndarray,
    genome_start: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    mode: str = "bar",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 3),
) -> plt.Axes:
    """
    Plot per-position attribution / saliency scores.

    Two display modes:

    - ``"bar"``: importance bar chart using the L1 norm across ACGT channels.
    - ``"logo"``: base-stacked logo via ``tangermeme.plot.plot_logo``. Falls
      back to ``"bar"`` if tangermeme is not installed.

    Args:
        attribution: Attribution array, shape ``(L, 4)`` or ``(4, L)``.
            Channel order must match one_hot_encode: ACGT.
        genome_start: If provided, x-axis ticks show genomic coordinates.
        ax: Matplotlib Axes. If None, a new figure is created.
        mode: ``"bar"`` or ``"logo"``.
        title: Axes title.
        figsize: Figure size when creating a new figure.

    Returns:
        Populated Axes object.
    """
    arr = np.asarray(attribution, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"attribution must be 2D, got shape {arr.shape}")
    # Normalize to (L, 4)
    if arr.shape[0] == 4 and arr.shape[1] != 4:
        arr = arr.T
    if arr.shape[1] != 4:
        raise ValueError(
            f"Expected 4 channels (ACGT), got shape {arr.shape}. "
            "Pass as (L, 4) or (4, L)."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    L = arr.shape[0]
    x = np.arange(L) if genome_start is None else np.arange(genome_start, genome_start + L)

    if mode == "logo":
        try:
            from tangermeme.plot import plot_logo
            plot_logo(arr.T, ax=ax)
            if genome_start is not None:
                step = max(1, L // 10)
                ticks = list(range(0, L, step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(genome_start + t) for t in ticks], rotation=45, ha="right")
            if title:
                ax.set_title(title)
            return ax
        except ImportError:
            logger.info("tangermeme not available; falling back to bar mode.")

    importance = np.abs(arr).sum(axis=1)
    ax.bar(x, importance, color="#255C99", width=0.8 if genome_start is None else (x[1] - x[0]) * 0.8, alpha=0.85)
    ax.set_ylabel("|Attribution|")
    ax.set_xlabel("Genomic position (bp)" if genome_start is not None else "Position")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title)
    return ax


def multi_track_figure(
    n_tracks: int,
    height_ratios: Optional[Sequence[float]] = None,
    figsize: Tuple[float, float] = (15, None),
    track_height: float = 2.0,
    sharex: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a multi-panel figure with stacked genomic tracks.

    Args:
        n_tracks: Number of track panels.
        height_ratios: Relative heights of each panel. Defaults to equal.
        figsize: ``(width, height)``. If height is ``None``, it is computed
            as ``sum(height_ratios) * track_height``.
        track_height: Height per unit of ``height_ratios`` when auto-computing.
        sharex: Share x-axis across all panels.

    Returns:
        ``(fig, axes)`` where ``axes`` is a 1D array of length ``n_tracks``.

    Example::

        fig, axes = multi_track_figure(3, height_ratios=[2, 1, 1])
        plot_prediction_track(preds, ax=axes[0])
        plot_ism_heatmap(ism, ax=axes[1])
        plot_gene_model(labels, ax=axes[2])
        fig.tight_layout()
    """
    if height_ratios is None:
        height_ratios = [1.0] * n_tracks

    width, height = figsize
    if height is None:
        height = sum(height_ratios) * track_height

    fig, axes = plt.subplots(
        n_tracks,
        1,
        figsize=(width, height),
        gridspec_kw={"height_ratios": list(height_ratios)},
        sharex=sharex,
    )
    axes = np.atleast_1d(axes)
    plt.subplots_adjust(hspace=0.08)
    return fig, axes
