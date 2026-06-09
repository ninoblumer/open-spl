"""
Generate plots from realtime_benchmark.py benchmark results.

Expects an output directory containing benchmark_48k/ and benchmark_96k/
subdirectories, each with quantity_a.csv and quantity_b.csv.

Usage
-----
    python tests/benchmark/plot_benchmark_results.py [--data output/] [--save thesis_figures/]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_quantity_a(base: Path) -> list[dict]:
    rows = _read_csv(base / "quantity_a.csv")
    for r in rows:
        r["blocksize"] = int(r["blocksize"])
        r["samplerate"] = int(r["samplerate"])
        r["rho_median"] = float(r["rho_median"])
        r["rho_p99"] = float(r["rho_p99"])
        r["rho_max"] = float(r["rho_max"])
    return rows


def _load_quantity_b(base: Path) -> list[dict]:
    rows = _read_csv(base / "quantity_b.csv")
    for r in rows:
        r["blocksize"] = int(r["blocksize"])
        r["samplerate"] = int(r["samplerate"])
        r["overruns"] = int(r["overruns"])
        r["overrun_rate"] = float(r["overrun_rate"])
        r["mean_queue_depth"] = float(r["mean_queue_depth"])
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_LOADOUT_COLORS = {"K0": "#2196F3", "K1": "#FF9800", "K2": "#F44336"}
_LOADOUT_LABELS = {
    "K0": "K0 — minimum (LAeq + LAF)",
    "K1": "K1 — typical Class-1 (A/C/Z, F/S, eq/max)",
    "K2": "K2 — K1 + full 1/3-oct filter bank",
}


def _plot_quantity_a(
    rows_48k: list[dict],
    rows_96k: list[dict],
    save_dir: Path | None,
) -> None:
    """Figure 1: processing utilisation (rho) — median and p99 per loadout."""
    loadouts = ["K0", "K1", "K2"]
    datasets = [(48_000, rows_48k), (96_000, rows_96k)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        r"Median processing utilisation  $\rho = t_\mathrm{proc}\,/\,t_\mathrm{budget}$",
        fontsize=11,
    )

    for ax, (sr, rows) in zip(axes, datasets):
        for lo in loadouts:
            subset = sorted(
                [r for r in rows if r["loadout"] == lo],
                key=lambda r: r["blocksize"],
            )
            if not subset:
                continue
            xs = [r["blocksize"] for r in subset]
            medians = [r["rho_median"] * 100 for r in subset]
            color = _LOADOUT_COLORS[lo]
            ax.plot(xs, medians, "o-", color=color, label=_LOADOUT_LABELS[lo], linewidth=1.8)

        # deadline line
        ax.axhline(100, color="black", linewidth=1.0, linestyle=":",
                   label=r"$\rho = 100\,\%$")

        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024, 4096])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("Block size")
        ax.set_title(rf"$f_s$ = {sr // 1000} kHz")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    axes[0].set_ylabel(r"$\rho_\mathrm{med}$ (%)")

    # legend once, outside right panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=8, loc="upper right")

    fig.tight_layout()
    if save_dir:
        path = save_dir / "quantity_a_rho.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close()


def _plot_quantity_a_heatmap(
    rows_48k: list[dict],
    rows_96k: list[dict],
    save_dir: Path | None,
) -> None:
    """Figure 2: heatmap of rho_median — loadout × blocksize, side-by-side for each fs."""
    loadouts = ["K0", "K1", "K2"]
    blocksizes = sorted({r["blocksize"] for r in rows_48k})

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.suptitle(
        r"Median processing utilisation $\rho_\mathrm{med}$",
        fontsize=11,
    )

    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]

    for ax, (sr, rows) in zip(axes, [(48_000, rows_48k), (96_000, rows_96k)]):
        data = np.full((len(loadouts), len(blocksizes)), np.nan)
        for r in rows:
            i = loadouts.index(r["loadout"])
            j = blocksizes.index(r["blocksize"])
            data[i, j] = r["rho_median"] * 100

        im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=150, aspect="auto")
        ax.set_xticks(range(len(blocksizes)))
        ax.set_xticklabels([str(b) for b in blocksizes])
        ax.set_yticks(range(len(loadouts)))
        ax.set_yticklabels(loadouts)
        ax.set_xlabel("Block size")
        ax.set_title(rf"$f_s$ = {sr // 1000} kHz")

        for i in range(len(loadouts)):
            for j in range(len(blocksizes)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f} %", ha="center", va="center",
                            fontsize=8, color="black" if v < 100 else "white")

        fig.colorbar(im, ax=ax, label=r"$\rho_\mathrm{med}$ (%)")

    fig.tight_layout()
    if save_dir:
        path = save_dir / "quantity_a_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close()


def _plot_k2_combined(
    base_48k: Path,
    base_96k: Path,
    rows_b_48k: list[dict],
    rows_b_96k: list[dict],
    save_dir: Path | None,
) -> None:
    """Figure 3: K2 per-block distributions — 2 rows × 4 columns.

    Row 0: ρ = t_proc / t_budget histograms (48 kHz and 96 kHz overlaid).
    Row 1: queue-depth histograms (48 kHz and 96 kHz overlaid).
    Columns: block sizes 256 / 512 / 1024 / 4096.
    Missing cells (e.g. no 96 kHz queue data at bs=256) are left blank.
    """
    blocksizes = [256, 512, 1024, 4096]
    sr_color = {48_000: "#2196F3", 96_000: "#F44336"}
    sr_label = {48_000: "48 kHz", 96_000: "96 kHz"}

    b_lookup: dict[tuple[int, int], dict] = {}
    for rows, sr in [(rows_b_48k, 48_000), (rows_b_96k, 96_000)]:
        for r in rows:
            if r["loadout"] == "K2":
                b_lookup[(sr, r["blocksize"])] = r

    # x-limit for queue-depth plots: queue capacity inferred from data
    xlim_q = 1
    for bs in blocksizes:
        for sr, base in [(48_000, base_48k), (96_000, base_96k)]:
            p = base / "queue_depth" / f"K2_{bs}.csv"
            if p.exists():
                xlim_q = max(xlim_q, int(np.loadtxt(p, skiprows=1, dtype=int).max()) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    fig.suptitle(
        "K2 loadout - processing load distribution (1h real-time run)\n"
        r"Top: processing utilisation $\rho = t_\mathrm{proc}\,/\,t_\mathrm{budget}$"
        r"  ·  Bottom: queue depth",
        fontsize=11,
    )

    for ci, bs in enumerate(blocksizes):
        # ── top row: rho ──────────────────────────────────────────────────
        ax_rho = axes[0][ci]
        ax_rho.set_title(f"bs = {bs}", fontsize=10)
        legend_handles = []

        for sr, base in [(48_000, base_48k), (96_000, base_96k)]:
            rho_path = base / "rho" / f"K2_{bs}.csv"
            if not rho_path.exists():
                continue
            rho = np.loadtxt(rho_path, delimiter=",", skiprows=1)
            color = sr_color[sr]
            ax_rho.hist(
                np.clip(rho * 100, 0, 300), bins=np.arange(0, 301, 5),
                weights=np.ones(len(rho)) / len(rho),
                color=color, alpha=0.5, edgecolor="none",
            )
            med = float(np.median(rho)) * 100
            ax_rho.axvline(med, color=color, linewidth=1.5, linestyle="-",
                           label=f"{sr_label[sr]}  med={med:.0f} %")
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.6,
                               label=sr_label[sr])
            )

        ax_rho.axvline(100, color="black", linewidth=1.0, linestyle=":")
        ax_rho.set_xlabel(r"$\rho$ (%)", fontsize=8)
        ax_rho.set_xlim(0, 300)
        ax_rho.set_yscale('log')
        ax_rho.set_ylim(1e-5, 1e0)
        ax_rho.set_ylabel("Relative frequency", fontsize=8)
        ax_rho.tick_params(labelsize=7)
        ax_rho.grid(True, axis="y", which="major", linestyle=":", alpha=0.4)
        ax_rho.grid(True, axis="y", which="minor", linestyle=":", alpha=0.2)
        if ci == 3:
            deadline_handle = mlines.Line2D(
                [], [], color="black", linewidth=1.0, linestyle=":",
                label=r"$\rho = 100\,\%$",
            )
            ax_rho.legend(handles=legend_handles + [deadline_handle], fontsize=7)

        # ── bottom row: queue depth ───────────────────────────────────────
        ax_q = axes[1][ci]
        any_queue = False

        # stagger tail heights so arrows don't overlap when both srs share max_depth
        arrow_extra = {48_000: 0.18, 96_000: 0.10}  # fraction of y_top above bar
        pending_arrows: list[tuple[int, int, str]] = []  # (max_depth, sr, color)

        for sr, base in [(48_000, base_48k), (96_000, base_96k)]:
            q_path = base / "queue_depth" / f"K2_{bs}.csv"
            if not q_path.exists():
                continue
            any_queue = True
            data = np.loadtxt(q_path, skiprows=1, dtype=int)
            color = sr_color[sr]
            max_depth = int(data.max())
            bins = np.arange(0, max_depth + 2) - 0.5
            ax_q.hist(data, bins=bins, color=color, alpha=0.5, edgecolor="none",
                      weights=np.ones(len(data)) / len(data), label=sr_label[sr])
            pending_arrows.append((max_depth, sr, color))

            row_b = b_lookup.get((sr, bs))
            if row_b:
                ax_q.text(
                    0.97, 0.95 if sr == 48_000 else 0.82,
                    f"{sr_label[sr]}: {row_b['overruns']} overruns",
                    transform=ax_q.transAxes, ha="right", va="top",
                    fontsize=7, color=color,
                )

        if not any_queue:
            ax_q.set_visible(False)
            continue

        ax_q.set_xlabel("Queue depth", fontsize=8)
        ax_q.set_xlim(-0.5, xlim_q)
        ax_q.set_yscale('log')
        ax_q.set_ylim(1e-5, 1e0)
        ax_q.set_ylabel("Relative frequency", fontsize=8)
        ax_q.tick_params(labelsize=7)
        ax_q.grid(True, axis="y", which="major", linestyle=":", alpha=0.4)
        ax_q.grid(True, axis="y", which="minor", linestyle=":", alpha=0.2)

        # Draw short arrows anchored just above the bar at max_depth
        y_top = ax_q.get_ylim()[1]
        for max_depth, sr, color in pending_arrows:
            tail_y = arrow_extra[sr] * y_top
            ax_q.annotate(
                "max",
                xy=(max_depth, 0),
                xytext=(max_depth, tail_y),
                ha="center", va="bottom",
                fontsize=7, color=color,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
            )

    fig.tight_layout()
    if save_dir:
        path = save_dir / "quantity_k2_combined.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tests/benchmark/plot_benchmark_results.py",
        description="Plot realtime_benchmark.py benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", default="output", metavar="DIR",
        help="Directory containing benchmark_48k/ and benchmark_96k/ subdirs.",
    )
    p.add_argument(
        "--save", default="thesis_figures", metavar="DIR",
        help="Directory to save PNG files (default: thesis_figures/).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out = Path(args.data)
    base_48k = out / "benchmark_48k"
    base_96k = out / "benchmark_96k"

    for d in (base_48k, base_96k):
        if not d.exists():
            raise SystemExit(f"Directory not found: {d}")

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows_a_48k = _load_quantity_a(base_48k)
    rows_a_96k = _load_quantity_a(base_96k)

    rows_b_48k = _load_quantity_b(base_48k) if (base_48k / "quantity_b.csv").exists() else []
    rows_b_96k = _load_quantity_b(base_96k) if (base_96k / "quantity_b.csv").exists() else []

    _plot_quantity_a(rows_a_48k, rows_a_96k, save_dir)
    _plot_quantity_a_heatmap(rows_a_48k, rows_a_96k, save_dir)
    _plot_k2_combined(base_48k, base_96k, rows_b_48k, rows_b_96k, save_dir)


if __name__ == "__main__":
    main()
