"""Publication-quality theory figures for thesis — signal illustrations.

Run from repository root:
    python scripts/plots_theory.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from pathlib import Path

from scipy import signal as sig
from phonometry import WeightingFilter

# ── global style ─────────────────────────────────────────────────────────────
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

# ── colors ────────────────────────────────────────────────────────────────────
C_BLUE  = "#2166ac"
C_RED   = "#d6604d"
C_GREEN = "#4dac26"
C_GREY  = "#555555"
C_FILL  = "#d4d4d4"

# ── output ────────────────────────────────────────────────────────────────────
OUT = Path("plots")
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG T.1 — Toneburst Waveforms: how IEC 61672-1 Table 4 bursts are cut
# ═══════════════════════════════════════════════════════════════════════════════

# IEC 61672-1 §5.9 standard parameters
_TONEBURST_FREQ_HZ  = 4000   # Hz
_TONEBURST_SR       = 192000  # Hz

# Three representative burst durations (ms) where individual cycles are visible
_BURST_MS = [0.25, 1.0, 5.0]


def _toneburst_panel(ax, burst_ms: float) -> None:
    """Draw one toneburst panel on *ax*."""
    sr       = _TONEBURST_SR
    f0       = _TONEBURST_FREQ_HZ
    T_period = 1.0 / f0           # s, one sine period (0.25 ms at 4 kHz)

    T_burst  = burst_ms * 1e-3    # s
    n_burst  = int(round(T_burst * sr))

    # Context window: at least 3 sine periods on each side, up to 5 ms max
    T_ctx    = min(max(3.0 * T_period, 0.05 * T_burst), 5e-3)
    n_ctx    = int(round(T_ctx * sr))

    # Full signal: silence | burst | silence
    n_total  = n_ctx + n_burst + n_ctx
    # Sample indices relative to burst onset (t=0)
    i_all    = np.arange(n_total) - n_ctx
    t_ms     = i_all / sr * 1e3             # time in ms

    # Continuous sine (the "source" waveform)
    sine     = np.sin(2.0 * np.pi * f0 * i_all / sr)

    # Gated burst: sine inside window, zero outside
    burst    = np.zeros(n_total)
    burst[n_ctx : n_ctx + n_burst] = sine[n_ctx : n_ctx + n_burst]

    # ── draw ─────────────────────────────────────────────────────────────────

    # 1. Burst window shading
    ax.axvspan(0.0, burst_ms, color=C_FILL, alpha=0.7, zorder=1, label="Burst window")

    # 2. Continuous sine (ghost — shows what's being cut from)
    ax.plot(t_ms, sine, color=C_GREY, lw=0.9, ls="--", alpha=0.45, zorder=2,
            label="Continuous sine")

    # 3. Gated burst waveform
    ax.plot(t_ms, burst, color=C_BLUE, lw=1.2, zorder=3, label="Toneburst")

    # 4. Gate edges
    ax.axvline(0.0,     color="black", lw=0.9, ls=":", zorder=4)
    ax.axvline(burst_ms, color="black", lw=0.9, ls=":", zorder=4)

    # ── axes / annotation ────────────────────────────────────────────────────
    n_cycles = T_burst * f0
    if n_cycles < 2:
        cyc_str = f"{n_cycles:.2g} cycle"
    elif n_cycles < 100:
        cyc_str = f"{n_cycles:.3g} cycles"
    else:
        cyc_str = f"{int(n_cycles)} cycles"

    ax.set_title(f"$T_b$ = {burst_ms:g} ms  ({cyc_str})", fontsize=8)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

    # Format x-axis labels: fewer ticks for wide panels
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=False))

    # Duration bracket annotation below the waveform
    y_ann = -1.22
    ax.annotate("",
                xy=(burst_ms, y_ann), xytext=(0.0, y_ann),
                arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.0))
    ax.text(burst_ms / 2.0, y_ann - 0.04, f"{burst_ms:g} ms",
            ha="center", va="top", fontsize=6.5, color=C_BLUE)


def make_fig_t1():
    """Toneburst shapes: how IEC 61672-1 §5.9 bursts are cut from a sine.

    Three panels (0.25 ms, 1 ms, 5 ms) where individual sine cycles are visible,
    illustrating the rectangular gating of a continuous 4 kHz sine.
    """
    n_rows, n_cols = 1, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2),
                             gridspec_kw={"wspace": 0.35})

    for ax, burst_ms in zip(axes.flat, _BURST_MS):
        _toneburst_panel(ax, burst_ms)

    # Shared legend from first panel
    handles = [
        mpatches.Patch(facecolor=C_FILL, alpha=0.7, label="Burst window"),
        plt.Line2D([0], [0], color=C_GREY, lw=1.0, ls="--", alpha=0.6,
                   label="Continuous sine"),
        plt.Line2D([0], [0], color=C_BLUE, lw=1.4,
                   label="Toneburst signal"),
        plt.Line2D([0], [0], color="black", lw=0.9, ls=":",
                   label="Gate edges"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Toneburst signals (4 kHz) gated from a continuous sine",
        fontsize=10,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG T.2 — Frequency-weighting gain and group delay (A and C)
# ═══════════════════════════════════════════════════════════════════════════════

# Digital weighting filters are designed at this sample rate (bilinear transform),
# so the phase / group delay are those of the actual IIR implementation.
_WEIGHTING_FS = 48000


def make_fig_t2():
    """Gain and group delay of the A- and C-weighting filters.

    Top panel: magnitude response (dB). Bottom panel: group delay (ms), computed
    as -dφ/dω from the unwrapped phase. Both weightings are the digital SOS filters
    used by the SLM (``phonometry.WeightingFilter`` at fs = 48 kHz), so the
    group delay is that of the causal IIR implementation — large near the
    low-frequency high-pass corner, essentially flat above a few hundred Hz.
    """
    fs = _WEIGHTING_FS
    f = np.logspace(np.log10(10), np.log10(20000), 4000)
    w = 2.0 * np.pi * f / fs                       # rad/sample

    curves = {
        # high_accuracy=False matches the native-rate SOS the SLM actually applies
        # (PluginFrequencyWeighting); phonometry otherwise returns a 2x-oversampled design.
        "A": (C_BLUE, WeightingFilter(fs=fs, curve="A", high_accuracy=False).sos),
        "C": (C_RED,  WeightingFilter(fs=fs, curve="C", high_accuracy=False).sos),
    }

    fig, (ax_gain, ax_gd) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    for name, (color, sos) in curves.items():
        _, h = sig.sosfreqz(sos, worN=f, fs=fs)
        gain = 20.0 * np.log10(np.abs(h) + 1e-30)
        phase = np.unwrap(np.angle(h))
        gd_ms = -np.gradient(phase, w) / fs * 1e3
        ax_gain.semilogx(f, gain, color=color, label=f"{name}-weighting")
        ax_gd.semilogx(f, gd_ms, color=color, label=f"{name}-weighting")

    ax_gain.axhline(0.0, color="black", lw=0.6)
    ax_gain.set_ylabel("Gain (dB)")
    ax_gain.set_ylim(-60, 5)
    ax_gain.set_title("Frequency-weighting gain", fontsize=9)
    ax_gain.legend(loc="lower center", fontsize=8)

    ax_gd.axhline(0.0, color="black", lw=0.6)
    ax_gd.set_ylabel("Group delay (ms)")
    ax_gd.set_xlabel("Frequency (Hz)")
    ax_gd.set_xlim(10, 20000)
    ax_gd.set_title("Frequency-weighting group delay", fontsize=9)
    ax_gd.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"A- and C-weighting filters (fs = {fs // 1000} kHz)",
                 fontsize=10)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating theory figures…")
    figures = [
        (make_fig_t1, "fig_t1_toneburst_shapes"),
        (make_fig_t2, "fig_t2_weighting_gain_groupdelay"),
    ]
    for fn, name in figures:
        print(f"\n[{name}]")
        try:
            fig = fn()
            save(fig, name)
        except Exception as e:
            print(f"  WARNING: skipped — {e}")
    print("\nDone. Figures saved to plots/")
