"""Publication-quality figures for thesis chapter on SLM verification and validation.

Run from repository root:
    python scripts/generate_thesis_plots.py
"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import signal as sig
from scipy.interpolate import PchipInterpolator

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

# ── paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
sys.path.insert(0, "tests")
sys.path.insert(0, "tests/iec61672")
sys.path.insert(0, "tests/iec61260")
sys.path.insert(0, "tests/xl2")

# ── SLM imports ───────────────────────────────────────────────────────────────
from slm.frequency_weighting import (
    PluginAWeighting, PluginCWeighting, PluginZWeighting,
)
from slm.time_weighting import PluginFastTimeWeighting, PluginSlowTimeWeighting

# ── test-suite helpers ────────────────────────────────────────────────────────
from test_61672_frequency_weightings import _measure_gain_db, _TABLE3
from test_61260_1_filters import (
    _filterbank, _nominal_labels, _gain_db, _omega_for_bandwidth,
    _PASSBAND_CL1, _STOPBAND_CL1,
    FilterConfig, OCTAVE, THIRD_OCTAVE,
    G as G_IEC, SAMPLERATE,
)
from test_61672_time_weightings import _mock_bus, _process_steady
from test_61672_toneburst import _toneburst_response, _TABLE4, _TABLE4_SMAX
from test_61672_cpeak import _cpeak_minus_lc, _TABLE5
from test_61672_level_linearity import _get_sweep

# ── colors ────────────────────────────────────────────────────────────────────
C_BLUE    = "#2166ac"
C_RED     = "#d6604d"
C_ORANGE  = "#ff7f0e"   # matplotlib default 2nd color (tab:orange)
C_GREEN   = "#4dac26"
C_GREY    = "#555555"
C_FILL    = "#d4d4d4"
C_WARN    = "#fee090"
C_OUT     = "#d73027"

# ── measurement mode ─────────────────────────────────────────────────────────
# Pass --fast for quick preview (1 s integration, no skip).
# Default "exact" mode: 10 s + 0.5 s skip, satisfying the IEC ≥10 s requirement.
# These govern only the 34-point compliance dot measurements (_measure_gain_db).
# Dense filter-response curves always use freqz (exact, instantaneous).
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--fast", action="store_true")
_args, _ = _p.parse_known_args()

if _args.fast:
    MEAS_DURATION_S = 1.0
    MEAS_SKIP_S     = 0.0
    print("Mode: FAST (1 s, no skip)")
else:
    MEAS_DURATION_S = 10.0
    MEAS_SKIP_S     = 0.5
    print("Mode: EXACT (10 s + 0.5 s skip)")

# ── output ────────────────────────────────────────────────────────────────────
OUT = Path("plots")
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png")


def _freqz_gain_db(plugin_cls, freqs, samplerate=48000):
    """Exact filter frequency response via sosfreqz, normalised to 0 dB at 1 kHz.

    Uses the plugin's SOS coefficients directly — no simulation, instantaneous.
    For PluginZWeighting (_wf is None) returns 0 dB at all frequencies.
    """
    bus = _mock_bus(samplerate=samplerate, blocksize=4096)
    plugin = plugin_cls(input=bus)
    if plugin._wf is None:
        return np.zeros(len(np.atleast_1d(freqs)))
    worN = np.concatenate([[1000.0], np.atleast_1d(freqs)])
    _, h = sig.sosfreqz(plugin._wf, worN=worN, fs=samplerate)
    gain_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-300))
    return gain_db[1:] - gain_db[0]   # normalise to 0 dB at 1 kHz


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.1 — Frequency Weightings A, C, Z  (3-panel, 8:3 aspect)
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_1():
    print("  computing A/C/Z gains at 34 Table-3 frequencies (48 kHz)…")
    freqs_tbl = np.array([r[0] for r in _TABLE3])
    a_goals   = np.array([r[1] for r in _TABLE3])
    c_goals   = np.array([r[2] for r in _TABLE3])
    cl1_lo    = np.array([r[3] if r[3] is not None else np.nan for r in _TABLE3])
    cl1_hi    = np.array([r[4] for r in _TABLE3])

    z_meas    = np.array([_measure_gain_db(PluginZWeighting, f, duration_s=MEAS_DURATION_S, skip_s=MEAS_SKIP_S)
                          for f in freqs_tbl])

    print("  computing dense response curves via freqz (48 + 96 kHz)…")
    freqs_dense = np.geomspace(10, 20000, 500)
    a_dense    = _freqz_gain_db(PluginAWeighting, freqs_dense, samplerate=48000)
    c_dense    = _freqz_gain_db(PluginCWeighting, freqs_dense, samplerate=48000)
    a_dense_96 = _freqz_gain_db(PluginAWeighting, freqs_dense, samplerate=96000)
    c_dense_96 = _freqz_gain_db(PluginCWeighting, freqs_dense, samplerate=96000)
    z_meas_96  = np.array([_measure_gain_db(PluginZWeighting, f, duration_s=MEAS_DURATION_S, skip_s=MEAS_SKIP_S,
                                             samplerate=96000) for f in freqs_tbl])

    # Smooth acceptance-limit bands: linearly interpolate TABLE3 limits on log-freq grid
    log_f_tbl = np.log10(freqs_tbl)
    log_f_d   = np.log10(freqs_dense)
    cl1_lo_fill = np.where(np.isfinite(cl1_lo), cl1_lo, -200.0)
    a_tol_hi = np.interp(log_f_d, log_f_tbl, a_goals + cl1_hi)
    a_tol_lo = np.interp(log_f_d, log_f_tbl, a_goals + cl1_lo_fill)
    c_tol_hi = np.interp(log_f_d, log_f_tbl, c_goals + cl1_hi)
    c_tol_lo = np.interp(log_f_d, log_f_tbl, c_goals + cl1_lo_fill)
    # Z-weighting shares the same Table 3 class-1 limits (design goal 0 dB).
    z_tol_hi = np.interp(log_f_d, log_f_tbl, cl1_hi)
    z_tol_lo = np.interp(log_f_d, log_f_tbl, cl1_lo_fill)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    def _setup_xaxis(ax):
        ax.set_xscale("log")
        ax.set_xlim(10, 20000)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1.0], numticks=5))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: {10:"10", 100:"100", 1000:"1k", 10000:"10k"}.get(int(round(x)), "")
        ))
        ax.set_xlabel("Frequency (Hz)", fontsize=8)

    LEG_TOL = "Class 1 acceptance limits"
    LEG_48  = "Implementation (48 kHz)"
    LEG_96  = "Implementation (96 kHz)"

    # ── Panel A ──────────────────────────────────────────────────────────────
    ax = axes[0]
    YLIM_A = (-80, 6)
    ax.fill_between(freqs_dense, np.maximum(a_tol_lo, YLIM_A[0]), a_tol_hi,
                    color=C_FILL, alpha=0.5, label=LEG_TOL)
    ax.plot(freqs_dense, a_dense,     color=C_BLUE,  lw=1.5, label=LEG_48)
    ax.plot(freqs_dense, a_dense_96,  color=C_BLUE,  lw=1.5, ls="--", label=LEG_96)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylim(*YLIM_A)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.set_ylabel("Relative level (dB)", fontsize=8)
    ax.set_title("A-weighting", fontsize=9)
    ax.legend(fontsize=6, loc="lower center")
    _setup_xaxis(ax)

    # ── Panel C ──────────────────────────────────────────────────────────────
    ax = axes[1]
    YLIM_C = (-16, 3)
    ax.fill_between(freqs_dense, np.maximum(c_tol_lo, YLIM_C[0]), c_tol_hi,
                    color=C_FILL, alpha=0.5, label=LEG_TOL)
    ax.plot(freqs_dense, c_dense,     color=C_RED,   lw=1.5, label=LEG_48)
    ax.plot(freqs_dense, c_dense_96,  color=C_RED,   lw=1.5, ls="--", label=LEG_96)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylim(*YLIM_C)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.set_ylabel("Relative level (dB)", fontsize=8)
    ax.set_title("C-weighting", fontsize=9)
    ax.legend(fontsize=6, loc="lower center")
    _setup_xaxis(ax)

    # ── Panel Z ──────────────────────────────────────────────────────────────
    ax = axes[2]
    YLIM_Z = (-0.15, 0.15)
    # Z shares the Table 3 class-1 limits (same column as A/C, goal 0 dB).  They
    # are far wider than this y-zoom, so the band fills the panel — the measured
    # Z response sits well within the limits.
    ax.fill_between(freqs_dense, np.maximum(z_tol_lo, YLIM_Z[0]), z_tol_hi,
                    color=C_FILL, alpha=0.5, label=LEG_TOL)
    ax.plot(freqs_tbl, z_meas,    "-",  color=C_GREEN, lw=1.5, label=LEG_48)
    ax.plot(freqs_tbl, z_meas_96, "--", color=C_GREEN, lw=1.5, label=LEG_96)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylim(*YLIM_Z)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Relative level (dB)", fontsize=8)
    ax.set_title("Z-weighting", fontsize=9)
    ax.legend(fontsize=6, loc="lower center")
    _setup_xaxis(ax)

    fig.suptitle("IEC 61672-1:2013 Table 3 — Frequency Weightings A, C, Z", fontsize=9)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.2 — Frequency Weighting Deviation from Table 3
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_2():
    freqs_tbl = np.array([r[0] for r in _TABLE3])
    a_goals   = np.array([r[1] for r in _TABLE3])
    c_goals   = np.array([r[2] for r in _TABLE3])
    cl1_lo    = np.array([r[3] if r[3] is not None else -np.inf for r in _TABLE3])
    cl1_hi    = np.array([r[4] for r in _TABLE3])

    # Analytical A/C-weighting formulas as the smooth dense-grid reference.
    # Table 3 values (rounded to 0.1 dB) are used only for the 34 compliance dots.
    _f1, _f2, _f3, _f4 = 20.598997, 107.65265, 737.86223, 12194.217

    def _a_weight_db(f):
        f2 = np.asarray(f, float) ** 2
        ra = (_f4**2 * f2**2) / (
            (f2 + _f1**2) * np.sqrt((f2 + _f2**2) * (f2 + _f3**2)) * (f2 + _f4**2)
        )
        ra_1k = (_f4**2 * 1e12) / (   # 1e12 = 1000^4 (f^4 at 1 kHz)
            (1e6 + _f1**2) * np.sqrt((1e6 + _f2**2) * (1e6 + _f3**2)) * (1e6 + _f4**2)
        )
        return 20.0 * np.log10(ra / ra_1k)

    def _c_weight_db(f):
        f2 = np.asarray(f, float) ** 2
        rc = (_f4**2 * f2) / ((f2 + _f1**2) * (f2 + _f4**2))
        rc_1k = (_f4**2 * 1e6) / ((1e6 + _f1**2) * (1e6 + _f4**2))
        return 20.0 * np.log10(rc / rc_1k)

    freqs_dense = np.geomspace(10, 20000, 500)
    log_f_tbl   = np.log10(freqs_tbl)
    log_f_d     = np.log10(freqs_dense)
    a_goals_d   = _a_weight_db(freqs_dense)
    c_goals_d   = _c_weight_db(freqs_dense)
    cl1_lo_d    = np.interp(log_f_d, log_f_tbl,
                            np.where(np.isfinite(cl1_lo), cl1_lo, -200.0))
    cl1_hi_d    = np.interp(log_f_d, log_f_tbl, cl1_hi)

    print("  computing dense A/C frequency response via freqz (instant)…")
    a_dense    = _freqz_gain_db(PluginAWeighting, freqs_dense, samplerate=48000)
    c_dense    = _freqz_gain_db(PluginCWeighting, freqs_dense, samplerate=48000)
    a_dense_96 = _freqz_gain_db(PluginAWeighting, freqs_dense, samplerate=96000)
    c_dense_96 = _freqz_gain_db(PluginCWeighting, freqs_dense, samplerate=96000)

    # 34 Table 3 compliance points (exact goals, not interpolated)
    print("  computing A/C deviations (34 Table 3 points, 48 kHz)…")
    a_meas    = np.array([_measure_gain_db(PluginAWeighting, f, duration_s=MEAS_DURATION_S, skip_s=MEAS_SKIP_S)
                          for f in freqs_tbl])
    c_meas    = np.array([_measure_gain_db(PluginCWeighting, f, duration_s=MEAS_DURATION_S, skip_s=MEAS_SKIP_S)
                          for f in freqs_tbl])
    print("  computing A/C deviations (34 Table 3 points, 96 kHz)…")
    a_meas_96 = np.array([_measure_gain_db(PluginAWeighting, f, duration_s=MEAS_DURATION_S, skip_s=MEAS_SKIP_S,
                                           samplerate=96000) for f in freqs_tbl])
    c_meas_96 = np.array([_measure_gain_db(PluginCWeighting, f, duration_s=MEAS_DURATION_S, skip_s=MEAS_SKIP_S,
                                           samplerate=96000) for f in freqs_tbl])

    a_dev_d    = a_dense    - a_goals_d
    c_dev_d    = c_dense    - c_goals_d
    a_dev_d96  = a_dense_96 - a_goals_d
    c_dev_d96  = c_dense_96 - c_goals_d
    a_dev      = a_meas     - a_goals
    c_dev      = c_meas     - c_goals
    a_dev_96   = a_meas_96  - a_goals
    c_dev_96   = c_meas_96  - c_goals

    LEG_TOL = "Class 1 acceptance limits"
    LEG_48  = "Implementation (48 kHz)"
    LEG_96  = "Implementation (96 kHz)"

    fig, (ax_a, ax_c) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                      gridspec_kw={"hspace": 0.25})

    def _residual_panel(ax, dev_d, dev_d96, dev_pts, dev_pts96, color, title, ylim):
        ax.fill_between(freqs_dense, cl1_lo_d, cl1_hi_d,
                        color=C_FILL, alpha=0.45, label=LEG_TOL)

        ax.plot(freqs_dense, dev_d,   color=color, lw=1.2, label=LEG_48)
        ax.plot(freqs_dense, dev_d96, color=color, lw=1.2, ls="--", label=LEG_96)
        ax.scatter(freqs_tbl, dev_pts,   color=color, s=14, zorder=4, linewidths=0)
        ax.scatter(freqs_tbl, dev_pts96, color=color, s=14, zorder=4,
                   marker="^", linewidths=0)

        ax.set_xscale("log")
        ax.set_xlim(10, 20000)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.set_ylabel("Deviation (dB)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="lower center")

    _residual_panel(ax_a, a_dev_d, a_dev_d96, a_dev, a_dev_96, C_BLUE, "A-weighting", (-3.0, 3.0))
    _residual_panel(ax_c, c_dev_d, c_dev_d96, c_dev, c_dev_96, C_RED,  "C-weighting", (-3.0, 3.0))

    ax_c.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: {10: "10", 100: "100", 1000: "1 000", 10000: "10 000"}.get(int(round(x)), "")
    ))
    ax_c.set_xlabel("Frequency (Hz)")
    fig.suptitle("IEC 61672-1:2013 Table 3 — Weighting Deviation from Reference")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.3 — Time Weighting Decay: Fast and Slow
# ═══════════════════════════════════════════════════════════════════════════════

def _decay_curve(plugin_cls, samplerate=48000, blocksize=4096, freq_hz=4000):
    """Return (t, level_db, tau) for exponential decay after cessation.

    Settles for 10 s (IEC requirement), then records 3.6 s of silence.
    """
    bus    = _mock_bus(samplerate=samplerate, blocksize=blocksize)
    plugin = plugin_cls(input=bus)
    tau    = plugin.tau

    n_settle = max(1, math.ceil(10.0 * samplerate / blocksize))   # 10 s fixed
    _process_steady(plugin, freq_hz, n_settle, samplerate, blocksize)

    n_decay = max(1, math.ceil(3.6 * samplerate / blocksize))
    zeros   = np.zeros((1, blocksize))
    chunks  = []
    for _ in range(n_decay):
        plugin.process(zeros)
        chunks.append(plugin.output[0, :].copy())
    decay = np.concatenate(chunks)

    y0       = max(float(decay[0]), 1e-40)
    level_db = 10.0 * np.log10(np.maximum(decay, 1e-40) / y0)
    t        = np.arange(len(decay)) / samplerate
    return t, level_db, tau


def make_fig_4_3():
    import matplotlib.patches as mpatches

    print("  computing Fast time-weighting decay…")
    t_F, l_F, tau_F = _decay_curve(PluginFastTimeWeighting)
    print("  computing Slow time-weighting decay…")
    t_S, l_S, tau_S = _decay_curve(PluginSlowTimeWeighting)

    n_fit_F = int(2 * tau_F * 48000)
    n_fit_S = int(2 * tau_S * 48000)
    rate_F  = abs(np.polyfit(t_F[:n_fit_F], l_F[:n_fit_F], 1)[0])
    rate_S  = abs(np.polyfit(t_S[:n_fit_S], l_S[:n_fit_S], 1)[0])

    F_LO, F_HI = 31.0, 38.5
    S_LO, S_HI = 3.6,  5.1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.5),
                                  gridspec_kw={"width_ratios": [1.9, 1]})

    # ── left panel: decay curves ──────────────────────────────────────────────
    # acceptance-limit wedges (gray)
    t_w = np.array([0.0, 3.5])
    ax.fill_between(t_w, -F_HI * t_w, -F_LO * t_w,
                    color=C_FILL, alpha=0.9, zorder=1,
                    label="Class 1 acceptance limits")
    ax.fill_between(t_w, -S_HI * t_w, -S_LO * t_w,
                    color=C_FILL, alpha=0.9, zorder=1)

    # Raw curves — set_ylim crops the view (no data clipping, which would leave a
    # false knee where Fast flattens against the clip floor).
    ax.plot(t_F, l_F, color=C_BLUE, lw=2,
            label=r"Fast (F), $\tau$ = 125 ms", zorder=3)
    ax.plot(t_S, l_S, color=C_RED,  lw=2,
            label=r"Slow (S), $\tau$ = 1 s",   zorder=3)

    ax.axhline(0, color="black", lw=0.8)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(-50, 3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel("Time after cessation (s)")
    ax.set_ylabel("Level relative to steady state (dB)")
    ax.set_title("Time-weighting decay")
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.95), fontsize=8)

    # ── right panel: decay-rate deviation from the ideal exponential rate ──────
    # The one-pole detector decays as exp(-t/τ), i.e. a level slope of
    # 10/ln(10)/τ dB/s; the §5.8 class 1 limits bracket that ideal rate. Plotting
    # (measured − ideal) puts F and S on one axis despite their ~8x rate ratio.
    ideal_F = 10.0 / np.log(10.0) / tau_F
    ideal_S = 10.0 / np.log(10.0) / tau_S
    dev_F,  dev_S  = rate_F - ideal_F, rate_S - ideal_S
    band_F = (F_LO - ideal_F, F_HI - ideal_F)
    band_S = (S_LO - ideal_S, S_HI - ideal_S)
    in_F = F_LO <= rate_F <= F_HI
    in_S = S_LO <= rate_S <= S_HI

    x_F, x_S = 0, 1
    w = 0.6

    # Acceptance-limit bars (reference ± class 1 limit), centred on the ideal rate.
    for xi, (lo, hi) in [(x_F, band_F), (x_S, band_S)]:
        ax2.add_patch(mpatches.Rectangle((xi - w / 2, lo), w, hi - lo,
                                         color=C_FILL, alpha=0.7, zorder=1))
    ax2.axhline(0, color=C_GREY, lw=1.5, ls="--", zorder=2)

    # Measured deviation markers, coloured by pass/fail (as in fig_4_5b).
    ax2.scatter([x_F], [dev_F], c=[C_BLUE if in_F else C_OUT], s=55, zorder=5)
    ax2.scatter([x_S], [dev_S], c=[C_BLUE if in_S else C_OUT], s=55, zorder=5)

    band_proxy = mpatches.Patch(color=C_FILL, alpha=0.7, label="Class 1 acceptance limits")
    ref_proxy  = plt.Line2D([0], [0], color=C_GREY, lw=1.5, ls="--",
                            label="Ideal exponential rate")
    meas_proxy = plt.Line2D([0], [0], marker="o", color=C_BLUE, lw=0,
                            label="Measured decay rate")
    ax2.legend(handles=[band_proxy, ref_proxy, meas_proxy], fontsize=7, loc="upper right")

    ax2.set_xticks([x_F, x_S])
    ax2.set_xticklabels(["Fast (F)", "Slow (S)"], fontsize=8)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(-5, 5)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax2.set_ylabel("Decay-rate deviation from ideal (dB/s)")
    ax2.set_title("Decay rate vs Class 1 limits")

    fig.suptitle("IEC 61672-1:2013 §5.8 — Time Weighting Decay")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.4 — Toneburst Response vs IEC 61672-1:2013 Table 4
# ═══════════════════════════════════════════════════════════════════════════════

def _stepped_limits(dense_ms, table_ms, lo, hi):
    """Class 1 limit offsets on a dense duration grid, per IEC 61672-1 §5.9.4.

    For a duration between two Table 4 entries the applicable acceptance limits
    are those of the next *shorter* tabulated duration, so the limits step at
    each tabulated duration instead of interpolating.  Returns (lo, hi) arrays,
    NaN below the shortest tabulated duration where no limits are defined.
    """
    order  = np.argsort(table_ms)
    tm     = np.asarray(table_ms)[order]
    lo_s   = np.asarray(lo)[order]
    hi_s   = np.asarray(hi)[order]
    idx    = np.searchsorted(tm, dense_ms, side="right") - 1   # largest tm <= T
    out_lo = np.full(np.shape(dense_ms), np.nan)
    out_hi = np.full(np.shape(dense_ms), np.nan)
    ok     = idx >= 0
    out_lo[ok] = lo_s[idx[ok]]
    out_hi[ok] = hi_s[idx[ok]]
    return out_lo, out_hi


def make_fig_4_4():
    burst_ms   = np.array([r[0] for r in _TABLE4])
    ref_fmax   = np.array([r[1] for r in _TABLE4])
    ref_sel    = np.array([r[2] for r in _TABLE4])
    cl1_lo     = np.array([r[3] for r in _TABLE4])
    cl1_hi     = np.array([r[4] for r in _TABLE4])

    # S-max sub-block of Table 4 (Eq. 7, τ = 1 s) — own durations and limits.
    burst_ms_s = np.array([r[0] for r in _TABLE4_SMAX])
    ref_smax   = np.array([r[1] for r in _TABLE4_SMAX])
    cl1_lo_s   = np.array([r[2] for r in _TABLE4_SMAX])
    cl1_hi_s   = np.array([r[3] for r in _TABLE4_SMAX])

    print(f"  computing toneburst response for {len(_TABLE4)} burst lengths…")
    results   = [_toneburst_response(ms / 1000.0) for ms in burst_ms]
    meas_fmax = np.array([r[0] for r in results])
    meas_sel  = np.array([r[1] for r in results])
    smax_by_ms = {ms: r[2] for ms, r in zip(burst_ms, results)}
    meas_smax = np.array([smax_by_ms[ms] for ms in burst_ms_s])
    dev_fmax  = meas_fmax - ref_fmax
    dev_sel   = meas_sel  - ref_sel
    dev_smax  = meas_smax - ref_smax

    # Dense reference curves
    t_dense        = np.geomspace(0.0001, 1.5, 500)
    ref_fmax_dense = 10 * np.log10(np.maximum(1 - np.exp(-t_dense / 0.125), 1e-30))
    ref_sel_dense  = 10 * np.log10(t_dense / 1.0)
    ref_smax_dense = 10 * np.log10(np.maximum(1 - np.exp(-t_dense / 1.0), 1e-30))

    # Fixed x-ticks
    tick_ms  = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    tick_lbl = ["0.25", "0.5", "1", "2", "5", "10", "20", "50",
                "100", "200", "500", "1000"]

    fig, (ax_f, ax_smax, ax_sel) = plt.subplots(3, 1, sharex=True, figsize=(8, 9),
                                                gridspec_kw={"hspace": 0.28})

    dense_ms = t_dense * 1000.0

    def _burst_panel(ax, bms, ref_dense, meas_vals, dev_vals,
                     lo, hi, ylabel, title):
        # Acceptance-limit band: continuous reference response (Eq. 7/8) plus the
        # class 1 limit of the next-shorter tabulated duration (§5.9.4), so the
        # band steps at each Table 4 duration rather than interpolating.
        lo_step, hi_step = _stepped_limits(dense_ms, bms, lo, hi)
        ax.fill_between(dense_ms, ref_dense + lo_step, ref_dense + hi_step,
                        color=C_FILL, alpha=0.5,
                        label="Class 1 acceptance limits")

        # Reference curve
        ax.plot(t_dense * 1000, ref_dense, color=C_GREY, lw=1.5, ls="--",
                label="IEC 61672-1 reference")

        # Measured values
        in_tol = (dev_vals >= lo) & (dev_vals <= hi)
        colors = [C_BLUE if t else C_OUT for t in in_tol]
        ax.scatter(bms, meas_vals, c=colors, s=30, zorder=5, label="Measured")

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 1500)
        ax.xaxis.set_major_locator(ticker.FixedLocator(tick_ms))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_lbl))
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    _burst_panel(ax_f, burst_ms, ref_fmax_dense, meas_fmax, dev_fmax,
                 cl1_lo, cl1_hi,
                 r"$\delta_{F_\mathrm{max}}$ (dB)", "4 kHz tone burst, F time weighting")
    _burst_panel(ax_smax, burst_ms_s, ref_smax_dense, meas_smax, dev_smax,
                 cl1_lo_s, cl1_hi_s,
                 r"$\delta_{S_\mathrm{max}}$ (dB)", "4 kHz tone burst, S time weighting")
    _burst_panel(ax_sel, burst_ms, ref_sel_dense, meas_sel, dev_sel,
                 cl1_lo, cl1_hi,
                 r"$\delta_\mathrm{SEL}$ (dB)", "4 kHz tone burst, SEL response")

    ax_f.set_ylim(-40, 3)
    ax_smax.set_ylim(-40, 3)
    ax_sel.set_ylim(-40, 3)
    ax_sel.set_xlabel("Burst duration (ms)")
    ax_f.legend(loc="lower right", fontsize=8)
    fig.suptitle("IEC 61672-1:2013 §5.9 Table 4 — Toneburst Response (Class 1)")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.4b — Toneburst Response Deviation from Reference
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_4b():
    burst_ms   = np.array([r[0] for r in _TABLE4])
    ref_fmax   = np.array([r[1] for r in _TABLE4])
    ref_sel    = np.array([r[2] for r in _TABLE4])
    cl1_lo     = np.array([r[3] for r in _TABLE4])
    cl1_hi     = np.array([r[4] for r in _TABLE4])

    burst_ms_s = np.array([r[0] for r in _TABLE4_SMAX])
    ref_smax   = np.array([r[1] for r in _TABLE4_SMAX])
    cl1_lo_s   = np.array([r[2] for r in _TABLE4_SMAX])
    cl1_hi_s   = np.array([r[3] for r in _TABLE4_SMAX])

    print(f"  computing toneburst response for {len(_TABLE4)} burst lengths…")
    results    = [_toneburst_response(ms / 1000.0) for ms in burst_ms]
    meas_fmax  = np.array([r[0] for r in results])
    meas_sel   = np.array([r[1] for r in results])
    smax_by_ms = {ms: r[2] for ms, r in zip(burst_ms, results)}
    meas_smax  = np.array([smax_by_ms[ms] for ms in burst_ms_s])
    dev_fmax   = meas_fmax - ref_fmax
    dev_sel    = meas_sel  - ref_sel
    dev_smax   = meas_smax - ref_smax

    # Dense duration grid for the stepped acceptance band.
    dense_ms = np.geomspace(0.2, 1500, 600)

    tick_ms  = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    tick_lbl = ["0.25", "0.5", "1", "2", "5", "10", "20", "50",
                "100", "200", "500", "1000"]

    fig, (ax_f, ax_smax, ax_sel) = plt.subplots(3, 1, sharex=True, figsize=(8, 9),
                                                gridspec_kw={"hspace": 0.28})

    def _dev_panel(ax, bms, dev_vals, lo, hi, ylabel, title):
        # Stepped class 1 band around zero (§5.9.4: limits snap to the next-shorter
        # tabulated duration).
        lo_step, hi_step = _stepped_limits(dense_ms, bms, lo, hi)
        ax.fill_between(dense_ms, lo_step, hi_step, color=C_FILL, alpha=0.5,
                        label="Class 1 acceptance limits")

        ax.axhline(0, color=C_GREY, lw=1.5, ls="--", label="IEC 61672-1 reference")

        in_tol = (dev_vals >= lo) & (dev_vals <= hi)
        colors = [C_BLUE if t else C_OUT for t in in_tol]
        ax.plot(bms, dev_vals, color=C_BLUE, lw=0.8, alpha=0.4, zorder=4)
        ax.scatter(bms, dev_vals, c=colors, s=30, zorder=5, label="Measured deviation")

        ax.set_xscale("log")
        ax.set_xlim(0.2, 1500)
        ax.set_ylim(-3.5, 1.6)
        ax.xaxis.set_major_locator(ticker.FixedLocator(tick_ms))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_lbl))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    _dev_panel(ax_f, burst_ms, dev_fmax, cl1_lo, cl1_hi,
               "Deviation (dB)", "4 kHz tone burst, F time weighting")
    _dev_panel(ax_smax, burst_ms_s, dev_smax, cl1_lo_s, cl1_hi_s,
               "Deviation (dB)", "4 kHz tone burst, S time weighting")
    _dev_panel(ax_sel, burst_ms, dev_sel, cl1_lo, cl1_hi,
               "Deviation (dB)", "4 kHz tone burst, SEL response")

    ax_sel.set_xlabel("Burst duration (ms)")
    ax_f.legend(loc="lower right", fontsize=8)
    fig.suptitle("IEC 61672-1:2013 §5.9 Table 4 — Toneburst Response Deviation (Class 1)")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.5 — Level Linearity: Measured vs. Input + Residuals
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_5():
    print("  computing level linearity sweep (121 levels)…")
    l_in, l_meas = _get_sweep()
    slope, intercept = np.polyfit(l_in, l_meas, 1)
    fit_line  = slope * l_in + intercept
    residuals = l_meas - fit_line

    fig, (ax_t, ax_b) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                      gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    # Top — measured vs input
    ax_t.plot([-12, 114], [-12, 114], color="#aaaaaa", lw=1.0, ls="--", label="Ideal (1:1)")
    ax_t.plot(l_in, l_meas, "o", color=C_BLUE, ms=2.5, alpha=0.7, label="Measured")
    ax_t.set_xlim(-12, 114)
    ax_t.set_ylim(-12, 114)
    ax_t.set_ylabel(r"Measured $L_{Aeq}$ (dB)")
    ax_t.set_title("IEC 61672-1:2013 §5.6 — Level Linearity (Class 1)")

    # Bottom — residuals
    ax_b.axhspan(-0.8, 0.8, color=C_FILL, alpha=0.4, label="Class 1 acceptance limits")
    ax_b.axhline(0, color="black", lw=0.8)

    in_tol = np.abs(residuals) <= 0.8
    ax_b.scatter(l_in[in_tol],  residuals[in_tol],  color=C_BLUE, s=8, zorder=4)
    ax_b.scatter(l_in[~in_tol], residuals[~in_tol], color=C_OUT,  s=8, zorder=4, marker="x")
    ax_b.plot(l_in, residuals, color=C_BLUE, lw=0.8, alpha=0.5)

    ax_b.text(0.98, 0.92,
              "Linear range: ≥120 dB\n(Class 1 minimum: 60 dB)",
              transform=ax_b.transAxes, fontsize=7, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax_b.set_xlim(-12, 114)
    ax_b.set_ylim(-1.2, 1.2)
    ax_b.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_b.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax_b.set_xlabel("Input level (dB SPL)")
    ax_b.set_ylabel("Deviation (dB)")

    # Unified legend: acceptance limits first, then top-panel entries
    handles_t, labels_t = ax_t.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()   # only "Class 1 acceptance limits"
    ax_t.legend(handles_b + handles_t, labels_b + labels_t,
                fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.5b — C-weighted Peak Sound Level vs IEC 61672-1:2013 Table 5
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_5b():
    import matplotlib.patches as mpatches

    print(f"  computing C-peak response for {len(_TABLE5)} Table 5 signals…")
    cyc_lbl = {"1cycle": "1 cycle", "pos_half": "+½ cycle", "neg_half": "−½ cycle"}
    f_lbl   = lambda f: "8 kHz" if f >= 1000 else f"{f:g} Hz"

    labels, refs, los, his, meas = [], [], [], [], []
    for sig_type, freq, ref, lo, hi in _TABLE5:
        labels.append(f"{cyc_lbl[sig_type]}\n{f_lbl(freq)}")
        refs.append(ref)
        los.append(lo)
        his.append(hi)
        meas.append(_cpeak_minus_lc(freq, sig_type))

    refs = np.array(refs); los = np.array(los); his = np.array(his); meas = np.array(meas)
    devs   = meas - refs
    in_tol = (devs >= los) & (devs <= his)

    x = np.arange(len(_TABLE5))
    w = 0.6
    fig, ax = plt.subplots(figsize=(8, 5))

    # Acceptance band (reference ± class 1 limit) and Table 5 reference per signal.
    for xi, ref, lo, hi in zip(x, refs, los, his):
        ax.add_patch(mpatches.Rectangle((xi - w / 2, ref + lo), w, hi - lo,
                                        color=C_FILL, alpha=0.7, zorder=1))
        ax.plot([xi - w / 2, xi + w / 2], [ref, ref],
                color=C_GREY, lw=1.5, ls="--", zorder=2)

    colors = [C_BLUE if t else C_OUT for t in in_tol]
    ax.scatter(x, meas, c=colors, s=55, zorder=5)

    band_proxy = mpatches.Patch(color=C_FILL, alpha=0.7, label="Class 1 acceptance limits")
    ref_proxy  = plt.Line2D([0], [0], color=C_GREY, lw=1.5, ls="--",
                            label="Table 5 reference")
    meas_proxy = plt.Line2D([0], [0], marker="o", color=C_BLUE, lw=0,
                            label=r"Measured $L_{C\mathrm{peak}} - L_C$")
    ax.legend(handles=[band_proxy, ref_proxy, meas_proxy], fontsize=8, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.6, len(_TABLE5) - 0.4)
    ax.set_ylim(0, 6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.set_ylabel(r"$L_{C\mathrm{peak}} - L_C$ (dB)")
    ax.set_title("IEC 61672-1:2013 §5.13 Table 5 — C-weighted Peak Sound Level (Class 1)")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.6 — Octave Band Filter Transfer Function with IEC 61260-1 Acceptance-limit Mask
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_hz(f):
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"


def _fmt_label(nominal: str) -> str:
    """Render a bank nominal label ('8k', '1.25k', '500') as '8 kHz' / '500 Hz'."""
    return f"{nominal[:-1]} kHz" if nominal.endswith("k") else f"{nominal} Hz"


def make_fig_4_6():
    return _make_filter_mask_fig(1000)


def make_fig_4_6d():
    # 16 kHz is a standard centre for both bandwidths, but the default banks stop
    # at 8 kHz (octave) / 10 kHz (1/3-octave); extend them so the 16 kHz band
    # exists.  At 48 kHz this band sits close to Nyquist (24 kHz), so it shows the
    # high-frequency limit of the digital filter design.
    return _make_filter_mask_fig(
        16000,
        oct_cfg=FilterConfig(b=1, limits=(63, 16000)),
        t3_cfg=FilterConfig(b=3, limits=(50, 16000)),
    )


def _make_filter_mask_fig(target_hz, oct_cfg=OCTAVE, t3_cfg=THIRD_OCTAVE):
    # Compare the octave and one-third-octave filters nearest *target_hz* against
    # the IEC 61260-1 acceptance mask.  Curves are plotted against the OCTAVE-band
    # breakpoint coordinate x = G^e: for the octave filter x = Ω = f/f_m, while
    # for the 1/3-octave filter the test frequency is the Formula (9)/(10) remap
    # Ω = remap(G^e, b).  In this coordinate Table 1 and Table F.1 coincide, so a
    # single mask applies to both bandwidths (cf. IEC 61260-1 Annex F).
    def _band_near(cfg):
        centers, sos_list = _filterbank(cfg)
        nominal = _nominal_labels(cfg)
        i = int(np.argmin([abs(c - target_hz) for c in centers]))
        return sos_list[i], centers[i], nominal[i]

    sos_oct, fm_oct, nom_oct = _band_near(oct_cfg)     # octave (1/1) bank
    sos_t3,  fm_t3,  nom_t3  = _band_near(t3_cfg)       # one-third-octave bank

    def atten_at(sos, fm, b, exp_range, n=2000):
        """Relative attenuation vs octave-breakpoint coordinate x = G^e.

        The actual test frequency is f = remap(G^e, b)·fm (identity for b=1).
        Points above the Nyquist frequency are returned as NaN (not plotted).
        """
        e     = np.linspace(-exp_range, exp_range, n)
        x     = G_IEC ** e                                 # display coordinate
        omega = np.array([_omega_for_bandwidth(xi, b) for xi in x])
        freqs = omega * fm
        valid = freqs < SAMPLERATE / 2 * 0.999
        g_ctr = _gain_db(sos, [fm], SAMPLERATE)[0]
        da    = np.full_like(x, np.nan)
        da[valid] = g_ctr - _gain_db(sos, freqs[valid], SAMPLERATE)
        return x, da

    omega_pb,    atten_pb    = atten_at(sos_oct, fm_oct, 1, 1.05)
    omega_full,  atten_full  = atten_at(sos_oct, fm_oct, 1, 4.6)
    omega_pb3,   atten_pb3   = atten_at(sos_t3,  fm_t3, t3_cfg.b, 1.05)
    omega_full3, atten_full3 = atten_at(sos_t3,  fm_t3, t3_cfg.b, 4.6)
    log_G = np.log(G_IEC)

    # ── acceptance-limit mask with correct discontinuity at band edges ───
    # IEC 61260-1 §5.10.7: limits change discontinuously at band-edge G^{±0.5}.
    # Implementation: piecewise function; two adjacent Ω values at G^{0.5}
    # produce a near-vertical step in fill_between.
    eps_exp = 1e-7   # tiny offset in G-exponent units to represent the jump

    def _mask(omega_arr):
        ae = np.abs(np.log(omega_arr) / log_G)
        lo = np.empty_like(ae)
        hi = np.empty_like(ae)

        # ── passband (|ae| <= 3/8): Table 1 breakpoints ──────────────
        pb = ae <= 3/8
        pb_e  = np.array([0,    1/8,  1/4,  3/8 ])
        pb_lo = np.array([-0.4, -0.4, -0.4, -0.4])
        pb_hi = np.array([+0.4, +0.5, +0.7, +1.4])
        lo[pb] = np.interp(ae[pb], pb_e, pb_lo)
        hi[pb] = np.interp(ae[pb], pb_e, pb_hi)

        # ── transition (3/8 < |ae| < 0.5):
        #    lo stays flat at -0.4 (discontinuous jump at G^0.5)
        #    hi interpolates linearly from 1.4 (at G^3/8) to 5.3 (at G^0.5)
        tr = (ae > 3/8) & (ae < 0.5)
        lo[tr] = -0.4
        hi[tr] = np.interp(ae[tr], [3/8, 0.5], [1.4, 5.3])

        # ── stop-band (|ae| >= 0.5): discontinuous jump; no upper limit ─
        sb = ae >= 0.5
        sb_e  = np.array([0.5,  1,    2,    3,    4   ])
        sb_lo = np.array([1.2,  16.6, 40.5, 60.0, 70.0])
        lo[sb] = np.interp(ae[sb], sb_e, sb_lo, right=70.0)
        hi[sb] = 200.0          # no upper limit; matplotlib clips to ylim

        return lo, hi

    # Explicit omega values just before/at the band edges to render the
    # discontinuity as a near-vertical step in fill_between.
    _be    = G_IEC**0.5
    _extra = np.array([1/_be*(1-eps_exp), 1/_be, _be*(1-eps_exp), _be])
    omega_m_pb   = np.sort(np.unique(np.concatenate([
        np.geomspace(G_IEC**(-1.05), G_IEC**(1.05), 2000), _extra])))
    omega_m_full = np.sort(np.unique(np.concatenate([
        np.geomspace(G_IEC**(-4.6),  G_IEC**(4.6),  4000), _extra])))

    lo_pb,   hi_pb   = _mask(omega_m_pb)    # matplotlib clips to ylim(-2,10)
    lo_full, hi_full = _mask(omega_m_full)  # matplotlib clips to ylim(-2,82)

    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(8, 6),
                                      gridspec_kw={"hspace": 0.35})

    LEG_OCT = f"Octave (1/1, {_fmt_label(nom_oct)})"
    LEG_T3  = f"Third-octave (1/3, {_fmt_label(nom_t3)})"

    # ── TOP: passband zoom (G⁻¹ to G¹), y-axis inverted ─────────────────
    ax_t.fill_between(omega_m_pb, lo_pb, hi_pb,
                      color=C_FILL, alpha=0.8, label="Class 1 acceptance limits")
    ax_t.plot(omega_pb,  atten_pb,  color=C_BLUE,   lw=1.5, label=LEG_OCT)
    ax_t.plot(omega_pb3, atten_pb3, color=C_ORANGE, lw=1.5, label=LEG_T3)
    ax_t.axhline(0, color="black", lw=0.8)
    ax_t.axvline(1.0,           color="black", lw=0.6, ls=":")
    ax_t.axvline(G_IEC**( 0.5), color="grey",  lw=0.6, ls=":")
    ax_t.axvline(G_IEC**(-0.5), color="grey",  lw=0.6, ls=":")
    ax_t.set_xscale("log")
    ax_t.set_xlim(G_IEC**(-1.05), G_IEC**(1.05))
    ax_t.set_ylim(-2, 10)
    ax_t.invert_yaxis()
    ax_t.xaxis.set_major_locator(ticker.FixedLocator([G_IEC**k for k in range(-1, 2)]))
    ax_t.xaxis.set_major_formatter(ticker.FixedFormatter(["G⁻¹", "1", "G¹"]))
    ax_t.xaxis.set_minor_locator(ticker.NullLocator())
    ax_t.set_ylabel("ΔA (dB)")
    ax_t.set_title("Passband detail", fontsize=9)

    # ── BOTTOM: full range (G⁻⁴ to G⁴), y-axis inverted ─────────────────
    ax_b.fill_between(omega_m_full, lo_full, hi_full,
                      color=C_FILL, alpha=0.8, label="Class 1 acceptance limits")
    ax_b.plot(omega_full,  atten_full,  color=C_BLUE,   lw=1.5, label=LEG_OCT)
    ax_b.plot(omega_full3, atten_full3, color=C_ORANGE, lw=1.5, label=LEG_T3)
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.axvline(1.0,           color="black", lw=0.6, ls=":")
    ax_b.axvline(G_IEC**( 0.5), color="grey",  lw=0.6, ls=":")
    ax_b.axvline(G_IEC**(-0.5), color="grey",  lw=0.6, ls=":")
    ax_b.set_xscale("log")
    ax_b.set_xlim(G_IEC**(-4.6), G_IEC**(4.6))
    ax_b.set_ylim(-2, 82)
    ax_b.invert_yaxis()
    g_ticks = [G_IEC**k for k in range(-4, 5)]
    g_lbls  = ["G⁻⁴", "G⁻³", "G⁻²", "G⁻¹", "1", "G¹", "G²", "G³", "G⁴"]
    ax_b.xaxis.set_major_locator(ticker.FixedLocator(g_ticks))
    ax_b.xaxis.set_major_formatter(ticker.FixedFormatter(g_lbls))
    ax_b.set_xlabel(r"$\Omega = f / f_m$")
    ax_b.set_ylabel("ΔA (dB)")
    ax_b.set_title("Full range", fontsize=9)
    ax_b.legend(fontsize=8, loc="lower right")   # lower right = visual top-right after inversion

    fig.suptitle(
        f"IEC 61260-1:2014 §5.10 — Octave & Third-Octave Filters "
        f"($f_m$ = {_fmt_hz(target_hz)}, Class 1)", fontsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.6b — Octave Band Filter Phase and Group Delay
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_6b():
    # Compare the octave-band and one-third-octave-band filters, both centred at
    # 1 kHz.  A single Ω range (G⁻² … G²) is shown — the response is smooth, so a
    # separate passband zoom adds nothing.
    def _band_at_1k(cfg=None):
        centers, sos_list = _filterbank() if cfg is None else _filterbank(cfg)
        i = int(np.argmin([abs(c - 1000) for c in centers]))
        return sos_list[i], centers[i]

    sos_oct, f_m = _band_at_1k()                 # octave (1/1) bank, 1 kHz band
    sos_t3,  _   = _band_at_1k(THIRD_OCTAVE)     # one-third-octave bank, 1 kHz band

    def _phase_and_gd(sos):
        freqs_hz = np.clip(np.geomspace(f_m * G_IEC**(-2), f_m * G_IEC**(2), 2000),
                           1.0, SAMPLERATE / 2 * 0.999)
        _, h = sig.sosfreqz(sos, worN=freqs_hz, fs=SAMPLERATE)
        phase   = np.unwrap(np.angle(h))
        w       = 2.0 * np.pi * freqs_hz / SAMPLERATE   # rad/sample
        gd_samp = -np.gradient(phase, w)                 # samples
        return freqs_hz / f_m, phase * 180.0 / np.pi, gd_samp / SAMPLERATE * 1000.0

    omega, phase_oct, gd_oct = _phase_and_gd(sos_oct)
    _,     phase_t3,  gd_t3  = _phase_and_gd(sos_t3)

    g_ticks = [G_IEC**k for k in range(-2, 3)]
    g_lbls  = ["G⁻²", "G⁻¹", "1", "G¹", "G²"]

    def _setup(ax):
        ax.set_xscale("log")
        ax.set_xlim(G_IEC**(-2), G_IEC**(2))
        ax.xaxis.set_major_locator(ticker.FixedLocator(g_ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(g_lbls))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel(r"$\Omega = f / f_m$", fontsize=8)
        ax.axvline(1.0, color="black", lw=0.6, ls=":")

    LEG_OCT = "Octave (1/1)"
    LEG_T3  = "Third-octave (1/3)"

    fig, (ax_ph, ax_gd) = plt.subplots(1, 2, figsize=(10, 4),
                                       gridspec_kw={"wspace": 0.3})

    # ── Phase ──────────────────────────────────────────────────────────────────
    ax_ph.plot(omega, phase_oct, color=C_BLUE, lw=1.5, label=LEG_OCT)
    ax_ph.plot(omega, phase_t3,  color=C_ORANGE, lw=1.5, label=LEG_T3)
    ax_ph.set_ylabel("Phase (°)")
    ax_ph.set_title("Phase", fontsize=9)
    _setup(ax_ph)
    ax_ph.legend(fontsize=8, loc="best")

    # ── Group delay ──────────────────────────────────────────────────────────
    ax_gd.plot(omega, gd_oct, color=C_BLUE, lw=1.5, label=LEG_OCT)
    ax_gd.plot(omega, gd_t3,  color=C_ORANGE, lw=1.5, label=LEG_T3)
    ax_gd.set_ylabel("Group delay (ms)")
    ax_gd.set_title("Group delay", fontsize=9)
    _setup(ax_gd)
    ax_gd.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Octave vs Third-Octave Filter Phase & Group Delay "
        f"($f_m$ = {int(round(f_m))} Hz, Class 1)", fontsize=10)
    fig.tight_layout()
    return fig



# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.6c — Octave Filter Summation of Output Signals (§5.16)
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_6c():
    """§5.16 — deviation of adjacent-filter energy sum from reference level.

    For each adjacent octave-band pair and five test frequencies between their
    mid-bands, measures (L_in − A_ref) − L_sum and checks against the class 1
    acceptance limits [−1.8, +0.8] dB.
    """
    import math
    from scipy.signal import sosfilt as _sosfilt

    TEST_EXPONENTS = [1/4, 3/8, 1/2, 5/8, 3/4]
    LO_CL1 = -1.8
    HI_CL1 = +0.8
    DURATION_S = 1.0

    def _summation(centers, sos_list, b):
        """Return (freqs, devs) for all adjacent-pair test frequencies.

        Test frequencies span one band spacing, i.e. G^(exp/b) from the lower
        mid-band (the crossover sits at G^(1/2b)).
        """
        n_bands = len(centers)
        n    = int(round(DURATION_S * SAMPLERATE))
        t    = np.arange(n) / SAMPLERATE
        skip = n // 2
        freqs, devs = [], []
        for pair_idx in range(n_bands - 1):
            f_lo   = centers[pair_idx]
            sos_lo = sos_list[pair_idx]
            sos_hi = sos_list[pair_idx + 1]
            a_ref  = -20.0 * np.log10(
                abs(sig.sosfreqz(sos_lo, worN=[f_lo], fs=SAMPLERATE)[1][0]))
            l_in   = 10.0 * math.log10(0.5)
            for exp in TEST_EXPONENTS:
                f_test = f_lo * G_IEC ** (exp / b)
                xs     = np.sin(2.0 * np.pi * f_test * t)
                p_lo   = float(np.mean(_sosfilt(sos_lo, xs)[skip:] ** 2))
                p_hi   = float(np.mean(_sosfilt(sos_hi, xs)[skip:] ** 2))
                l_sum  = 10.0 * math.log10(max(p_lo + p_hi, 1e-300))
                freqs.append(f_test)
                devs.append((l_in - a_ref) - l_sum)
        return np.array(freqs), np.array(devs)

    print("  computing §5.16 summation deviations (octave + third-octave)…")
    f_oct, d_oct = _summation(*_filterbank(), b=1)
    f_t3,  d_t3  = _summation(*_filterbank(THIRD_OCTAVE), b=THIRD_OCTAVE.b)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axhspan(LO_CL1, HI_CL1, color=C_FILL, alpha=0.5, label="Class 1 acceptance limits")
    ax.axhline(0, color="black", lw=0.8)

    def _plot_band(freqs, devs, color, label):
        ax.scatter(freqs, devs, color=color, s=30, marker="x", linewidths=1.2,
                   zorder=5, label=label)
        out = (devs < LO_CL1) | (devs > HI_CL1)
        if np.any(out):
            ax.scatter(freqs[out], devs[out], color=C_OUT, s=55, marker="x",
                       linewidths=1.8, zorder=6)

    _plot_band(f_oct, d_oct, C_BLUE,   "Octave (1/1)")
    _plot_band(f_t3,  d_t3,  C_ORANGE, "Third-octave (1/3)")

    ax.set_xscale("log")
    ax.set_xlim(50, 15000)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1.0], numticks=5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: {10: "10", 100: "100", 1000: "1k", 10000: "10k"}.get(int(round(x)), "")
    ))
    ax.set_ylim(-2.5, 1.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel("Test frequency (Hz)")
    ax.set_ylabel(r"$(L_\mathrm{in} - A_\mathrm{ref}) - L_\mathrm{sum}$ (dB)")
    ax.set_title("IEC 61260-1:2014 §5.16 — Summation of Adjacent Filter Outputs (Class 1)",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig



# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating thesis figures…")
    figures = [
        (make_fig_4_1, "fig_4_1_frequency_weightings"),
        (make_fig_4_2, "fig_4_2_frequency_weighting_residuals"),
        (make_fig_4_3, "fig_4_3_time_weighting_decay"),
        (make_fig_4_4, "fig_4_4_toneburst_response"),
        (make_fig_4_4b, "fig_4_4b_toneburst_response_deviation"),
        (make_fig_4_5, "fig_4_5_level_linearity"),
        (make_fig_4_5b, "fig_4_5b_cpeak"),
        (make_fig_4_6, "fig_4_6_octave_filter_mask"),
        (make_fig_4_6d, "fig_4_6d_octave_filter_mask_16kHz"),
        (make_fig_4_6b, "fig_4_6b_octave_filter_phase_groupdelay"),
        (make_fig_4_6c, "fig_4_6c_octave_summation"),
    ]
    for fn, name in figures:
        print(f"\n[{name}]")
        try:
            fig = fn()
            save(fig, name)
        except Exception as e:
            print(f"  WARNING: skipped — {e}")
    print("\nDone. Figures saved to plots/")
