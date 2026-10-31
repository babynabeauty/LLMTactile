"""Smoothing, seed aggregation and learning-curve rendering.

The plotted line is a heavily-smoothed mean; the lighter band around it shows
the spread of the *raw* (unsmoothed) training values within a moving window
(min/max or ±1 std), so the noisiness of the underlying signal stays visible.
"""

from __future__ import annotations

import numpy as np

N_GRID = 400


def ema_smooth(values: np.ndarray, weight: float) -> np.ndarray:
    """W&B-style exponential moving average. `weight` in [0, 1); 0 = no smoothing.

    NaNs are carried through without poisoning the running average.
    """
    values = np.asarray(values, dtype=float)
    if weight <= 0 or len(values) == 0:
        return values
    out = np.empty_like(values)
    acc = None
    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = acc if acc is not None else np.nan
            continue
        acc = v if acc is None else acc * weight + v * (1 - weight)
        out[i] = acc
    return out


def aggregate_raw(curves: list[tuple[np.ndarray, np.ndarray]], n_grid: int = N_GRID):
    """Interpolate seed curves onto a shared x grid.

    Returns (grid, stacked) where `stacked` is (n_seeds, n_grid) with NaN
    wherever a seed does not cover that grid point.
    """
    curves = [(s, v) for s, v in curves if len(s) > 0]
    if not curves:
        return np.array([]), np.zeros((0, 0))

    lo = min(s[0] for s, _ in curves)
    hi = max(s[-1] for s, _ in curves)
    grid = np.linspace(lo, hi, n_grid)

    stacked = np.full((len(curves), n_grid), np.nan)
    for i, (s, v) in enumerate(curves):
        interp = np.interp(grid, s, v)
        interp[(grid < s[0]) | (grid > s[-1])] = np.nan
        stacked[i] = interp
    return grid, stacked


def rolling_band(stacked: np.ndarray, window: int, kind: str):
    """Moving-window spread of the raw values.

    At each grid column, pool every seed's raw values within `window` columns
    and return (lo, hi): min/max for ``kind="minmax"``, mean±std for ``"std"``.
    """
    n_grid = stacked.shape[1]
    half = max(1, window // 2)
    lo = np.full(n_grid, np.nan)
    hi = np.full(n_grid, np.nan)
    for i in range(n_grid):
        w = stacked[:, max(0, i - half):min(n_grid, i + half + 1)].ravel()
        w = w[~np.isnan(w)]
        if w.size == 0:
            continue
        if kind == "std":
            m, s = w.mean(), w.std()
            lo[i], hi[i] = m - s, m + s
        else:
            lo[i], hi[i] = w.min(), w.max()
    return lo, hi


def _title_case(s: str) -> str:
    return s.replace("_", " ").title()


def plot_curves(ax, groups: dict, smooth: float, teacher_value: float | None,
                metric_label: str, title: str, band: str = "minmax",
                band_window: int = 25, color_map: dict[str, str] | None = None) -> None:
    """`groups` maps a line label -> list of (steps, values) seed curves.

    If `color_map` is provided, labels found in it get their assigned color;
    unmapped labels fall back to the tab10 palette.
    """
    import matplotlib

    cmap = matplotlib.colormaps["tab10"]

    # Pre-aggregate so we can rank labels by their final smoothed value before
    # drawing — the legend then mirrors the leaderboard at the right edge.
    computed: dict[str, tuple] = {}
    for label, curves in groups.items():
        grid, stacked = aggregate_raw(curves)
        if len(grid) == 0:
            continue
        line = ema_smooth(np.nanmean(stacked, axis=0), smooth)
        computed[label] = (grid, stacked, line)

    def final_value(label: str) -> float:
        line = computed[label][2]
        valid = line[~np.isnan(line)]
        return float(valid[-1]) if len(valid) else float("-inf")

    ordered = sorted(computed, key=final_value, reverse=True)

    # Teacher ceiling first so it sits at the top of the legend; the line
    # itself still renders below the curves visually.
    if teacher_value is not None:
        ax.axhline(teacher_value, linestyle=":", color="black", lw=1.8,
                   label=f"teacher ({teacher_value:.3g})")

    for idx, label in enumerate(ordered):
        grid, stacked, line = computed[label]
        color = (color_map or {}).get(label) or cmap(idx % 10)
        ax.plot(grid, line, label=label, color=color, lw=2)

        lo, hi = rolling_band(stacked, band_window, band)
        # Smooth the band envelope itself so min/max edges aren't spiky.
        lo, hi = ema_smooth(lo, smooth), ema_smooth(hi, smooth)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(_title_case(metric_label))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
