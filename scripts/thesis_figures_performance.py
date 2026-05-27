"""Publication-quality performance figures for thesis.

Two figures are produced:

  fig_p1_rho_over_time  — Per-block processing utilisation ρ vs. time,
                          one subplot per loadout, colour + linestyle encode
                          block size.

  fig_p2_rho_summary    — Grouped bar chart: median ρ with 95 % bootstrap CI,
                          colour encodes loadout, x-position encodes block size.

Run from repository root:
    python scripts/thesis_figures_performance.py [--data output/quick]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── global style (matches thesis_plots_theory.py) ────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.5,
    "figure.dpi":         150,
})

OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# ── colour / style maps ───────────────────────────────────────────────────────

# Loadout colours — used in bar chart bars and as subplot border tints.
LOADOUT_COLOR = {
    "K0": "#2166ac",   # blue
    "K1": "#4dac26",   # green
    "K2": "#d6604d",   # orange-red
    "K3": "#7b3294",   # purple
}
LOADOUT_LABEL = {
    "K0": "K0  (normative minimum)",
    "K1": "K1  (Class-1 handheld)",
    "K2": "K2  (+ 1/3-octave bank)",
    "K3": "K3  (+ multi-Leq)",
}

# Block-size colours — sequential warm→cool; small bs = highest load = warm.
BS_COLOR = {
    128:  "#b2182b",
    256:  "#d6604d",
    512:  "#f4a582",
    1024: "#4393c3",
    4096: "#2166ac",
}
# Block-size linestyles — redundant with colour for B&W printing.
BS_STYLE = {
    128:  "-",
    256:  "--",
    512:  ":",
    1024: "-.",
    4096: (0, (5, 1)),
}
BS_LABEL = {bs: f"bs = {bs}" for bs in BS_COLOR}


# ── data loading ─────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple[dict, dict]:
    """Return (summary, rho_arrays).

    summary      — dict keyed (loadout, blocksize) → row dict from quantity_a.csv
    rho_arrays   — dict keyed (loadout, blocksize) → np.ndarray of per-block ρ
    """
    import csv

    summary: dict[tuple[str, int], dict] = {}
    with open(data_dir / "quantity_a.csv") as f:
        for row in csv.DictReader(f):
            key = (row["loadout"], int(row["blocksize"]))
            summary[key] = {k: float(v) if k not in ("loadout", "warn_low_n")
                            else v for k, v in row.items()}
            summary[key]["blocksize"] = int(row["blocksize"])
            summary[key]["samplerate"] = int(row["samplerate"])
            summary[key]["block_count"] = int(row["block_count"])

    rho_arrays: dict[tuple[str, int], np.ndarray] = {}
    for path in (data_dir / "rho").glob("*.csv"):
        lo, bs_str = path.stem.rsplit("_", 1)
        rho_arrays[(lo, int(bs_str))] = np.loadtxt(path, skiprows=1)

    return summary, rho_arrays


# ── rolling median helper ─────────────────────────────────────────────────────

def _rolling_median(a: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling median with edge-padding at the start."""
    if window <= 1 or len(a) == 0:
        return a
    window = min(window, len(a))
    half = window // 2
    padded = np.pad(a, (half, window - half - 1), mode="edge")
    s = padded.strides[0]
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(len(a), window), strides=(s, s)
    )
    return np.median(windows, axis=1)


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, stat=np.median,
                 n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI of *stat* at level *ci*."""
    rng = np.random.default_rng(42)
    boot = [stat(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, 100 * (1 - ci) / 2)
    hi = np.percentile(boot, 100 * (1 + ci) / 2)
    return float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG P.1 — ρ over time
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_p1(summary: dict, rho_arrays: dict) -> plt.Figure:
    """Per-block processing utilisation ρ vs. time.

    Four subplots (one per loadout K0–K3), arranged 2 × 2.  Within each
    subplot five lines show the five block sizes; colour and linestyle both
    encode block size so the figure is legible in greyscale.  A 1 s rolling
    median overlays the raw trace (light fill) to show the trend.

    ρ = t_proc / t_budget.  The horizontal dashed line at ρ = 1 marks the
    real-time limit (IEC 61260-1 §5.14.4).
    """
    loadouts = [lo for lo in LOADOUT_COLOR if any(
        k[0] == lo for k in rho_arrays)]
    blocksizes = sorted({k[1] for k in rho_arrays})

    fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                              sharex=False, sharey=False,
                              gridspec_kw={"hspace": 0.42, "wspace": 0.32})

    for ax, lo in zip(axes.flat, loadouts):
        for bs in blocksizes:
            key = (lo, bs)
            if key not in rho_arrays:
                continue
            rho = rho_arrays[key]
            sr  = summary[key]["samplerate"]
            t   = np.arange(len(rho)) * bs / sr          # time in seconds
            win = max(1, round(sr / bs))                  # ~1 s window
            med = _rolling_median(rho, win)

            color = BS_COLOR[bs]
            style = BS_STYLE[bs]

            # Raw trace — light background cloud
            ax.plot(t, rho, color=color, lw=0.6, alpha=0.18,
                    linestyle=style, rasterized=True)
            # Rolling median — main visible line
            ax.plot(t, med, color=color, lw=1.5, alpha=0.92,
                    linestyle=style, label=BS_LABEL[bs])

        # Real-time limit
        ax.axhline(1.0, color="#888888", lw=1.0, ls="--", zorder=2)
        ax.text(t[-1] * 0.98, 1.02, "ρ = 1", ha="right", va="bottom",
                fontsize=7.5, color="#666666")

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("ρ  (t_proc / t_budget)", fontsize=9)
        ax.set_title(f"{lo}  —  {LOADOUT_LABEL[lo]}", fontsize=9,
                     color=LOADOUT_COLOR[lo], fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Hide unused subplot if loadouts < 4
    for ax in axes.flat[len(loadouts):]:
        ax.set_visible(False)

    # Shared legend for block sizes (bottom of figure)
    legend_handles = [
        Line2D([0], [0], color=BS_COLOR[bs], lw=1.5, linestyle=BS_STYLE[bs],
               label=BS_LABEL[bs])
        for bs in blocksizes
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(blocksizes),
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               title="Block size", title_fontsize=9)

    fig.suptitle(
        "Processing utilisation ρ over time  (rolling 1 s median + raw trace)",
        fontsize=11,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG P.2 — summary bar chart with 95 % bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_p2(summary: dict, rho_arrays: dict) -> plt.Figure:
    """Median ρ per cell with 95 % bootstrap CI.

    Grouped bars: x-axis = block size, colour = loadout.  Log y-axis handles
    the two-decade range across loadouts.  Error bars show the 95 % bootstrap
    confidence interval of the median (2 000 resamples).

    The horizontal dashed line at ρ = 1 marks the real-time limit.
    """
    loadouts  = [lo for lo in LOADOUT_COLOR if any(k[0] == lo for k in rho_arrays)]
    blocksizes = sorted({k[1] for k in rho_arrays})
    n_lo = len(loadouts)
    n_bs = len(blocksizes)

    bar_w  = 0.72 / n_lo          # total group width = 0.72
    x_base = np.arange(n_bs, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, lo in enumerate(loadouts):
        medians, ci_lo, ci_hi = [], [], []
        for bs in blocksizes:
            key = (lo, bs)
            if key in rho_arrays and len(rho_arrays[key]) > 0:
                rho = rho_arrays[key]
                med = float(np.median(rho))
                lo_ci, hi_ci = bootstrap_ci(rho)
            else:
                med = lo_ci = hi_ci = float("nan")
            medians.append(med)
            ci_lo.append(med - lo_ci)
            ci_hi.append(hi_ci - med)

        x_pos = x_base + (i - (n_lo - 1) / 2) * bar_w
        color = LOADOUT_COLOR[lo]

        ax.bar(x_pos, medians, width=bar_w * 0.88,
               color=color, alpha=0.82, label=LOADOUT_LABEL[lo], zorder=3)
        ax.errorbar(x_pos, medians,
                    yerr=[ci_lo, ci_hi],
                    fmt="none", ecolor="black", elinewidth=1.1,
                    capsize=3.5, capthick=1.1, zorder=4)

    ax.axhline(1.0, color="#888888", lw=1.1, ls="--", zorder=2)
    ax.text(n_bs - 0.52, 1.05, "ρ = 1  (real-time limit)",
            ha="right", va="bottom", fontsize=8, color="#555555")

    ax.set_yscale("log")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"bs = {bs}" for bs in blocksizes], fontsize=9)
    ax.set_ylabel("Median ρ  (log scale)", fontsize=9)
    ax.set_xlabel("Block size", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10,
                                                                 labelOnlyBase=False))
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    ax.set_title(
        "Processing utilisation by loadout and block size  "
        "(median ρ, 95 % bootstrap CI)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", visible=False)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis performance figures from benchmark output."
    )
    parser.add_argument("--data", default="output/quick", metavar="DIR",
                        help="Directory containing quantity_a.csv and rho/")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "quantity_a.csv").exists():
        raise SystemExit(f"quantity_a.csv not found in {data_dir}")

    print(f"Loading data from {data_dir} …")
    summary, rho_arrays = load_data(data_dir)
    print(f"  {len(summary)} cells, "
          f"{sum(len(v) for v in rho_arrays.values()):,} blocks total")

    figures = [
        (make_fig_p1, "fig_p1_rho_over_time"),
        (make_fig_p2, "fig_p2_rho_summary"),
    ]
    for fn, name in figures:
        print(f"\n[{name}]")
        try:
            fig = fn(summary, rho_arrays)
            save(fig, name)
        except Exception as e:
            import traceback
            print(f"  WARNING: skipped — {e}")
            traceback.print_exc()

    print("\nDone. Figures saved to thesis_figures/")


if __name__ == "__main__":
    main()
