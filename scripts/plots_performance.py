"""Publication-quality performance figures for thesis.

Two figures are produced:

  fig_p1_rho_over_time  — Per-block processing utilisation ρ vs. time,
                          one subplot per loadout, colour + linestyle encode
                          block size.

  fig_p2_rho_summary    — Grouped bar chart: median ρ with 95 % bootstrap CI,
                          colour encodes loadout, x-position encodes block size.

Run from repository root:
    python scripts/plots_performance.py [--data output/quick]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── global style (matches plots_theory.py) ────────────────────────────
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

OUT = Path("plots")
OUT.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png")


# ── colour / style maps ───────────────────────────────────────────────────────

# Loadout colours — used in bar chart bars and as subplot border tints.
LOADOUT_COLOR = {
    "K0": "#2166ac",   # blue
    "K1": "#4dac26",   # green
    "K2": "#d6604d",   # orange-red
}
LOADOUT_LABEL = {
    "K0": "K0  (normative minimum)",
    "K1": "K1  (Class-1 handheld)",
    "K2": "K2  (+ full 1/3-octave banks)",
}

# Block-size colours — sequential warm→cool; small bs = highest load = warm.
BS_COLOR = {
    128:  "#b2182b",
    256:  "#d6604d",
    512:  "#f4a582",
    1024: "#4393c3",
    4096: "#2166ac",
}
BS_LABEL = {bs: f"bs = {bs}" for bs in BS_COLOR}


# ── data loading ─────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple[dict, dict, dict, dict]:
    """Return (summary_a, rho_arrays, summary_b, queue_arrays).

    summary_a     — dict keyed (loadout, blocksize) → row dict from quantity_a.csv
    rho_arrays    — dict keyed (loadout, blocksize) → np.ndarray of per-block ρ
    summary_b     — dict keyed (loadout, blocksize) → row dict from quantity_b.csv
    queue_arrays  — dict keyed (loadout, blocksize) → np.ndarray of per-block queue depth
    """
    import csv

    summary_a: dict[tuple[str, int], dict] = {}
    with open(data_dir / "quantity_a.csv") as f:
        for row in csv.DictReader(f):
            key = (row["loadout"], int(row["blocksize"]))
            summary_a[key] = {k: float(v) if k not in ("loadout", "warn_low_n")
                              else v for k, v in row.items()}
            summary_a[key]["blocksize"] = int(row["blocksize"])
            summary_a[key]["samplerate"] = int(row["samplerate"])
            summary_a[key]["block_count"] = int(row["block_count"])

    rho_arrays: dict[tuple[str, int], np.ndarray] = {}
    for path in (data_dir / "rho").glob("*.csv"):
        lo, bs_str = path.stem.rsplit("_", 1)
        rho_arrays[(lo, int(bs_str))] = np.loadtxt(path, skiprows=1)

    summary_b: dict[tuple[str, int], dict] = {}
    b_path = data_dir / "quantity_b.csv"
    if b_path.exists():
        with open(b_path) as f:
            for row in csv.DictReader(f):
                key = (row["loadout"], int(row["blocksize"]))
                summary_b[key] = {k: float(v) if k not in ("loadout",)
                                  else v for k, v in row.items()}
                summary_b[key]["blocksize"] = int(row["blocksize"])
                summary_b[key]["samplerate"] = int(row["samplerate"])

    queue_arrays: dict[tuple[str, int], np.ndarray] = {}
    queue_dir = data_dir / "queue_depth"
    if queue_dir.exists():
        for path in queue_dir.glob("*.csv"):
            lo, bs_str = path.stem.rsplit("_", 1)
            queue_arrays[(lo, int(bs_str))] = np.loadtxt(path, skiprows=1,
                                                          dtype=np.int32)

    return summary_a, rho_arrays, summary_b, queue_arrays


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

    Three subplots (one per loadout K0–K2), arranged in a 2 × 2 grid (one cell unused).  Within each
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

            # Raw trace — light background cloud
            ax.plot(t, rho, color=color, lw=0.6, alpha=0.18,
                    rasterized=True)
            # Rolling median — main visible line
            ax.plot(t, med, color=color, lw=1.5, alpha=0.92,
                    label=BS_LABEL[bs])

        # Real-time limit
        ax.axhline(1.0, color="#888888", lw=1.0, ls="--", zorder=2)
        ax.text(t[-1] * 0.98, 1.12, "ρ = 1", ha="right", va="bottom",
                fontsize=7.5, color="#666666")

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("ρ  (t_proc / t_budget)", fontsize=9)
        ax.set_title(LOADOUT_LABEL[lo], fontsize=9,
                     color=LOADOUT_COLOR[lo], fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10,
                                                                    labelOnlyBase=False))
        ax.set_xlim(left=0)

    # Hide unused subplot if loadouts < 4
    for ax in axes.flat[len(loadouts):]:
        ax.set_visible(False)

    # Shared legend for block sizes (bottom of figure)
    legend_handles = [
        Line2D([0], [0], color=BS_COLOR[bs], lw=1.5, label=BS_LABEL[bs])
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
    """Median ρ per cell with ±1 SD error bars.

    Grouped bars: x-axis = block size, colour = loadout.  Log y-axis handles
    the two-decade range across loadouts.  Error bars show ±1 standard deviation.

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
        medians, sd_vals = [], []
        for bs in blocksizes:
            key = (lo, bs)
            if key in rho_arrays and len(rho_arrays[key]) > 0:
                rho = rho_arrays[key]
                med = float(np.median(rho))
                sd  = float(np.std(rho))
            else:
                med = sd = float("nan")
            medians.append(med)
            sd_vals.append(sd)

        x_pos = x_base + (i - (n_lo - 1) / 2) * bar_w
        color = LOADOUT_COLOR[lo]

        ax.bar(x_pos, medians, width=bar_w * 0.88,
               color=color, alpha=0.82, label=LOADOUT_LABEL[lo], zorder=3)
        ax.errorbar(x_pos, medians,
                    yerr=sd_vals,
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
        "(median ρ, ±1 SD)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", visible=False)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG P.3 — queue depth over time + overrun rate bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_p3(summary_b: dict, queue_arrays: dict) -> plt.Figure:
    """Queue depth over time (left) and mean overrun rate per cell (right).

    Left subplot: per-block queue depth time series for each Quantity B cell,
    each cell drawn in its own colour.  Queue depth is the number of blocks
    waiting in the producer queue when the consumer reads; a depth of 0
    indicates the consumer caught up between blocks.

    Right subplot: grouped bar chart of mean queue depth per cell.
    """
    if not queue_arrays:
        raise ValueError("No queue_depth data found — re-run with --run-b.")

    cells = sorted(queue_arrays.keys())   # (loadout, blocksize)
    loadouts  = list(dict.fromkeys(lo for lo, _ in cells))
    blocksizes = sorted({bs for _, bs in cells})

    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5),
                                      gridspec_kw={"wspace": 0.35})

    # ── left: time series ────────────────────────────────────────────────────
    # One distinct colour per cell so lines are told apart by colour, not dashes.
    cell_colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    for idx, (lo, bs) in enumerate(cells):
        qd  = queue_arrays[(lo, bs)]
        sr  = int(summary_b[(lo, bs)]["samplerate"])
        t   = np.arange(len(qd)) * bs / sr
        win = max(1, round(sr / bs))         # ~1 s rolling median
        med = _rolling_median(qd.astype(float), win)

        color = cell_colors[idx % len(cell_colors)]
        label = f"{lo}  bs={bs}"
        ax_t.plot(t, qd,  color=color, lw=0.5, alpha=0.15, rasterized=True)
        ax_t.plot(t, med, color=color, lw=1.4, alpha=0.9, label=label)

    ax_t.set_xlabel("Time (s)", fontsize=9)
    ax_t.set_ylabel("Queue depth (blocks)", fontsize=9)
    ax_t.set_title("Queue depth over time  (rolling 1 s median + raw)",
                   fontsize=9)
    ax_t.tick_params(labelsize=8)
    ax_t.set_xlim(left=0)
    ax_t.set_ylim(bottom=0)
    ax_t.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_t.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    # ── right: mean queue depth bar chart ────────────────────────────────────
    n_lo  = len(loadouts)
    n_bs  = len(blocksizes)
    bar_w = 0.72 / max(n_lo, 1)
    x_base = np.arange(n_bs, dtype=float)

    for i, lo in enumerate(loadouts):
        means = []
        for bs in blocksizes:
            key = (lo, bs)
            means.append(float(np.mean(queue_arrays[key]))
                         if key in queue_arrays else float("nan"))
        x_pos = x_base + (i - (n_lo - 1) / 2) * bar_w
        ax_b.bar(x_pos, means, width=bar_w * 0.88,
                 color=LOADOUT_COLOR[lo], alpha=0.82,
                 label=LOADOUT_LABEL[lo], zorder=3)

    ax_b.set_xticks(x_base)
    ax_b.set_xticklabels([f"bs = {bs}" for bs in blocksizes], fontsize=9)
    ax_b.set_ylabel("Mean queue depth (blocks)", fontsize=9)
    ax_b.set_xlabel("Block size", fontsize=9)
    ax_b.tick_params(labelsize=8)
    ax_b.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_b.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    ax_b.set_title("Mean queue depth by cell", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3, linestyle="--")
    ax_b.grid(axis="x", visible=False)

    fig.suptitle(
        "Real-time queue depth  (Quantity B — producer/consumer queue, maxsize = 4)",
        fontsize=11,
    )
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
    summary_a, rho_arrays, summary_b, queue_arrays = load_data(data_dir)
    print(f"  {len(summary_a)} cells, "
          f"{sum(len(v) for v in rho_arrays.values()):,} blocks total")
    if queue_arrays:
        print(f"  {len(queue_arrays)} Quantity B cells with queue depth data")

    figures: list[tuple] = [
        (make_fig_p1, "fig_p1_rho_over_time",  (summary_a, rho_arrays)),
        (make_fig_p2, "fig_p2_rho_summary",    (summary_a, rho_arrays)),
        (make_fig_p3, "fig_p3_queue_depth",    (summary_b, queue_arrays)),
    ]
    for fn, name, fargs in figures:
        print(f"\n[{name}]")
        try:
            fig = fn(*fargs)
            save(fig, name)
        except Exception as e:
            import traceback
            print(f"  WARNING: skipped — {e}")
            traceback.print_exc()

    print("\nDone. Figures saved to plots/")


if __name__ == "__main__":
    main()
