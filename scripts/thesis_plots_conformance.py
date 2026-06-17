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
import soundfile as sf

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
from slm.frequency_weighting import PluginAWeighting, PluginCWeighting, PluginZWeighting
from slm.time_weighting import PluginFastTimeWeighting, PluginSlowTimeWeighting

# ── test-suite helpers ────────────────────────────────────────────────────────
from test_61672_frequency_weightings import _measure_gain_db, _TABLE3
from test_61260_1_filters import (
    _get_filterbank, _gain_db,
    _PASSBAND_CL1, _STOPBAND_CL1,
    G as G_IEC, SAMPLERATE,
)
from test_61672_time_weightings import _mock_bus, _process_steady
from test_61672_toneburst import _toneburst_response, _TABLE4, _TABLE4_SMAX
from test_61672_level_linearity import _get_sweep
from test_xl2_broadband import (
    compute_leq, compute_lmax, compute_lpeak,
    compute_interval_leq,
    _PluginXL2Z, TOLERANCE_DB,
)
from test_xl2_rta import compute_octave_leq
from conftest import XL2Measurement
from slm.engine import Engine
from slm.io.file_controller import FileController
from slm.octave_band import PluginOctaveBand

# ── colors ────────────────────────────────────────────────────────────────────
C_BLUE    = "#2166ac"
C_RED     = "#d6604d"
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
OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


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


_offset_cache: dict = {}
_audio96_cache: dict = {}


# ── 96 kHz compute helpers ────────────────────────────────────────────────────

def _audio_96k(meas):
    """Load WAV, upsample 48 → 96 kHz (result cached per file)."""
    from scipy.signal import resample_poly
    key = str(meas.wav_path)
    if key not in _audio96_cache:
        audio, sr = sf.read(str(meas.wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        _audio96_cache[key] = resample_poly(audio, 96000 // int(sr), 1) \
                              if int(sr) != 96000 else audio
    return _audio96_cache[key]


def _fw_sos_96k(weighting_cls):
    """Return SOS coefficients for weighting_cls at 96 kHz."""
    bus    = _mock_bus(samplerate=96000)
    plugin = weighting_cls(input=bus)
    sos    = getattr(plugin, '_wf', None)
    if sos is None:
        sos = getattr(plugin, '_sos', None)
    return sos


def compute_leq_96k(meas, weighting_cls):
    from scipy.signal import sosfilt
    aw = sosfilt(_fw_sos_96k(weighting_cls), _audio_96k(meas))
    return 10.0 * np.log10(np.mean(aw ** 2) / (20e-6 * meas.sensitivity) ** 2)


def compute_lmax_96k(meas, weighting_cls, tw_cls):
    from scipy.signal import sosfilt
    sr = 96000
    aw = sosfilt(_fw_sos_96k(weighting_cls), _audio_96k(meas))
    # Build EMA SOS for the time-weighting filter at 96 kHz
    alpha = 1.0 - np.exp(-1.0 / (sr * tw_cls(input=_mock_bus(samplerate=sr)).tau))
    tw_sos = np.array([[alpha, 0.0, 0.0, 1.0, -(1.0 - alpha), 0.0]])
    y = sosfilt(tw_sos, aw ** 2)
    return 10.0 * np.log10(np.max(y) / (20e-6 * meas.sensitivity) ** 2)


def compute_lpeak_96k(meas, weighting_cls):
    from scipy.signal import sosfilt
    aw = sosfilt(_fw_sos_96k(weighting_cls), _audio_96k(meas))
    return 10.0 * np.log10(np.max(aw ** 2) / (20e-6 * meas.sensitivity) ** 2)


def _find_slm001_offset(meas, weighting_cls=None,
                        xl2_col: str = "LAeq_dt",
                        coarse_step_ms: float = 10.0,
                        fine_step_ms: float = 1.0):
    """Find WAV start offset that best aligns 1-s LAeq windows with XL2 reference.

    Key fix: for any non-zero offset fewer than n complete windows fit in the
    recording.  RMSE is computed only over the windows that are available and
    have a matching non-boundary XL2 reference second.

    Two-pass grid search over offsets in [0, 1) s:
      1. Coarse pass at coarse_step_ms resolution (full error curve returned).
      2. Fine pass at fine_step_ms resolution around the coarse minimum.

    Returns (best_offset_samples, best_offset_seconds,
             coarse_offsets_ms, coarse_rmse_array).
    """
    if weighting_cls is None:
        weighting_cls = PluginAWeighting
    key = (str(meas.wav_path), xl2_col, weighting_cls.__name__
           if hasattr(weighting_cls, '__name__') else str(weighting_cls))
    if key in _offset_cache:
        return _offset_cache[key]

    audio, sr = sf.read(str(meas.wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]

    from scipy.signal import sosfilt as _sosfilt
    bus    = _mock_bus(samplerate=int(sr))
    plugin = weighting_cls(input=bus)
    # PluginFrequencyWeighting → _wf; PluginHPF → _sos; passthrough → None
    sos = getattr(plugin, '_wf', None)
    if sos is None:
        sos = getattr(plugin, '_sos', None)
    audio_w = _sosfilt(sos, audio) if sos is not None else audio.copy()

    xl2_ref  = meas.log_series(xl2_col)
    n        = len(xl2_ref)
    window   = int(sr)
    p_ref_sq = (20e-6 * meas.sensitivity) ** 2

    cum = np.concatenate([[0.0], np.cumsum(audio_w ** 2)])

    boundary = {9, 10, 19, 20, 29}
    stable   = np.array([i for i in range(n) if i not in boundary])

    def _rmse(offset: int) -> float:
        # Use only windows that fit within the recording
        n_avail = (len(audio_w) - offset) // window
        use_n   = min(n, n_avail)
        if use_n < 5:
            return np.inf
        idx     = np.arange(use_n, dtype=np.intp)
        starts  = offset + idx * window
        ends    = starts + window
        mean_sq = (cum[ends] - cum[starts]) / window
        leq     = 10.0 * np.log10(np.maximum(mean_sq, 1e-300) / p_ref_sq)
        st      = stable[stable < use_n]
        if len(st) < 3:
            return np.inf
        return float(np.sqrt(np.mean((leq[st] - xl2_ref[st]) ** 2)))

    # Coarse pass
    cs             = max(1, int(coarse_step_ms * sr / 1000))
    coarse_offsets = np.arange(0, window, cs)
    coarse_errors  = np.array([_rmse(int(o)) for o in coarse_offsets])
    best_c         = int(coarse_offsets[np.argmin(coarse_errors)])

    # Fine pass ±cs around coarse minimum
    fs           = max(1, int(fine_step_ms * sr / 1000))
    fine_offsets = np.arange(max(0, best_c - cs), min(window, best_c + cs), fs)
    fine_errors  = np.array([_rmse(int(o)) for o in fine_offsets])
    best         = int(fine_offsets[np.argmin(fine_errors)])

    result = (best, best / sr,
              coarse_offsets / sr * 1000.0,   # ms, for plotting
              coarse_errors)
    _offset_cache[key] = result
    return result


def _load_meas(num: int) -> XL2Measurement:
    name = f"2026-02-06_SLM_{num:03d}"
    try:
        return XL2Measurement(name)
    except FileNotFoundError:
        raise FileNotFoundError(f"XL2 data not found for {name} — skipping figure")


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
    ax.axhspan(-0.1, 0.1, color=C_FILL, alpha=0.5, label=LEG_TOL)
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

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── acceptance-limit wedges (gray) ────────────────────────────────────────
    t_w = np.array([0.0, 3.5])
    tol_fill = ax.fill_between(t_w, -F_HI * t_w, -F_LO * t_w,
                               color=C_FILL, alpha=0.9, zorder=1,
                               label="Class 1 acceptance limits")
    ax.fill_between(t_w, -S_HI * t_w, -S_LO * t_w,
                    color=C_FILL, alpha=0.9, zorder=1)

    # ── decay curves ─────────────────────────────────────────────────────────
    ax.plot(t_F, np.clip(l_F, -50, 0), color=C_BLUE, lw=2,
            label=r"Fast (F), $\tau$ = 125 ms", zorder=3)
    ax.plot(t_S, np.clip(l_S, -50, 0), color=C_RED,  lw=2,
            label=r"Slow (S), $\tau$ = 1 s",   zorder=3)

    ax.axhline(0, color="black", lw=0.8)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(-50, 3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel("Time after cessation (s)")
    ax.set_ylabel("Level relative to steady state (dB)")
    ax.set_title("IEC 61672-1:2013 §5.8 — Time Weighting Decay")
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.95), fontsize=8)

    # ── inset: bar chart of decay rates with acceptance limits ────────────────
    ax_in = ax.inset_axes([0.4575, 0.13, 0.5125, 0.45])
    ax_in.set_facecolor("white")

    w = 0.55
    x_F, x_S = 0, 1

    # Acceptance-limit rectangles
    for x, lo, hi in [(x_F, F_LO, F_HI), (x_S, S_LO, S_HI)]:
        ax_in.add_patch(mpatches.Rectangle(
            (x - w/2, lo), w, hi - lo, color=C_FILL, zorder=2))

    # Bars
    ax_in.bar([x_F], [rate_F], width=w * 0.45, color=C_BLUE, zorder=4)
    ax_in.bar([x_S], [rate_S], width=w * 0.45, color=C_RED,  zorder=4)

    # Value labels above bars
    ax_in.text(x_F, rate_F + 0.8, f"{rate_F:.1f}", ha="center", va="bottom",
               fontsize=7, color=C_BLUE, fontweight="bold")
    ax_in.text(x_S, rate_S + 0.1, f"{rate_S:.1f}", ha="center", va="bottom",
               fontsize=7, color=C_RED,  fontweight="bold")

    ax_in.set_xticks([x_F, x_S])
    ax_in.set_xticklabels(["Fast (F)", "Slow (S)"], fontsize=7)
    ax_in.set_ylabel("dB/s", fontsize=7)
    ax_in.set_xlim(-0.55, 1.55)
    ax_in.set_ylim(0, max(rate_F * 1.18, F_HI * 1.1))
    ax_in.tick_params(labelsize=7)
    ax_in.set_title("Decay rate", fontsize=8)
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)

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
# FIG 4.6 — Octave Band Filter Transfer Function with IEC 61260-1 Acceptance-limit Mask
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_6():
    # Use the 1000 Hz band (best-case for digital filter accuracy)
    centers, sos_list = _get_filterbank()
    band_idx = int(np.argmin([abs(c - 1000) for c in centers]))
    sos = sos_list[band_idx]
    f_m = centers[band_idx]

    log_G = np.log(G_IEC)
    g_ctr = _gain_db(sos, [f_m], SAMPLERATE)[0]

    def atten_at(exp_range):
        f_lo = max(f_m * G_IEC**(-exp_range), 1.0)
        f_hi = min(f_m * G_IEC**(+exp_range), SAMPLERATE / 2 * 0.999)
        freqs = np.geomspace(f_lo, f_hi, 2000)
        return freqs / f_m, g_ctr - _gain_db(sos, freqs, SAMPLERATE)

    omega_pb,   atten_pb   = atten_at(1.05)
    omega_full, atten_full = atten_at(4.6)

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

    # ── TOP: passband zoom (G⁻¹ to G¹), y-axis inverted ─────────────────
    ax_t.fill_between(omega_m_pb, lo_pb, hi_pb,
                      color=C_FILL, alpha=0.8, label="Class 1 acceptance limits")
    ax_t.plot(omega_pb, atten_pb, color=C_BLUE, lw=1.5, label="Implemented filter")
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
    ax_b.plot(omega_full, atten_full,
              color=C_BLUE, lw=1.5, label="Implemented filter")
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
        f"IEC 61260-1:2014 §5.10 — Octave-Band Filter "
        f"($f_m$ = {int(round(f_m))} Hz, Class 1)", fontsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.6b — Octave Band Filter Phase and Group Delay
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_6b():
    centers, sos_list = _get_filterbank()
    band_idx = int(np.argmin([abs(c - 1000) for c in centers]))
    sos = sos_list[band_idx]
    f_m = centers[band_idx]

    def _phase_and_gd(freqs_hz):
        freqs_hz = np.clip(freqs_hz, 1.0, SAMPLERATE / 2 * 0.999)
        _, h = sig.sosfreqz(sos, worN=freqs_hz, fs=SAMPLERATE)
        phase   = np.unwrap(np.angle(h))
        w       = 2.0 * np.pi * freqs_hz / SAMPLERATE   # rad/sample
        gd_samp = -np.gradient(phase, w)                 # samples
        return freqs_hz / f_m, phase * 180.0 / np.pi, gd_samp / SAMPLERATE * 1000.0

    omega_pb,   phase_pb,   gd_pb   = _phase_and_gd(
        np.geomspace(f_m * G_IEC**(-1.05), f_m * G_IEC**(1.05), 2000))
    omega_full, phase_full, gd_full = _phase_and_gd(
        np.geomspace(f_m * G_IEC**(-4.6),  f_m * G_IEC**(4.6),  4000))

    g_ticks_pb   = [G_IEC**k for k in range(-1, 2)]
    g_lbls_pb    = ["G⁻¹", "1", "G¹"]
    g_ticks_full = [G_IEC**k for k in range(-4, 5)]
    g_lbls_full  = ["G⁻⁴", "G⁻³", "G⁻²", "G⁻¹", "1", "G¹", "G²", "G³", "G⁴"]

    def _setup(ax, ticks, lbls, xlim):
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(lbls))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel(r"$\Omega = f / f_m$", fontsize=8)
        ax.axvline(1.0,           color="black", lw=0.6, ls=":")
        ax.axvline(G_IEC**( 0.5), color="grey",  lw=0.6, ls=":")
        ax.axvline(G_IEC**(-0.5), color="grey",  lw=0.6, ls=":")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.35})

    # ── Phase passband ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(omega_pb, phase_pb, color=C_BLUE, lw=1.5)
    ax.set_ylabel("Phase (°)")
    ax.set_title("Phase — passband detail", fontsize=9)
    _setup(ax, g_ticks_pb, g_lbls_pb, (G_IEC**(-1.05), G_IEC**(1.05)))

    # ── Group delay passband ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(omega_pb, gd_pb, color=C_RED, lw=1.5)
    ax.set_ylabel("Group delay (ms)")
    ax.set_title("Group delay — passband detail", fontsize=9)
    _setup(ax, g_ticks_pb, g_lbls_pb, (G_IEC**(-1.05), G_IEC**(1.05)))

    # ── Phase full range ──────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(omega_full, phase_full, color=C_BLUE, lw=1.5)
    ax.set_ylabel("Phase (°)")
    ax.set_title("Phase — full range", fontsize=9)
    _setup(ax, g_ticks_full, g_lbls_full, (G_IEC**(-4.6), G_IEC**(4.6)))

    # ── Group delay full range ────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(omega_full, gd_full, color=C_RED, lw=1.5)
    ax.set_ylabel("Group delay (ms)")
    ax.set_title("Group delay — full range", fontsize=9)
    _setup(ax, g_ticks_full, g_lbls_full, (G_IEC**(-4.6), G_IEC**(4.6)))

    fig.suptitle(
        f"Octave-Band Filter Phase & Group Delay "
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

    centers, sos_list = _get_filterbank()
    n_bands = len(centers)

    print(f"  computing §5.16 summation deviations "
          f"({n_bands - 1} pairs × {len(TEST_EXPONENTS)} frequencies)…")

    freqs_all = []
    devs_all  = []

    n    = int(round(DURATION_S * SAMPLERATE))
    t    = np.arange(n) / SAMPLERATE
    skip = n // 2

    for pair_idx in range(n_bands - 1):
        f_lo   = centers[pair_idx]
        sos_lo = sos_list[pair_idx]
        sos_hi = sos_list[pair_idx + 1]

        gain_mid = 20.0 * np.log10(
            abs(sig.sosfreqz(sos_lo, worN=[f_lo], fs=SAMPLERATE)[1][0]))
        a_ref = -gain_mid
        l_in  = 10.0 * math.log10(0.5)

        for exp in TEST_EXPONENTS:
            f_test = f_lo * G_IEC ** exp
            x      = np.sin(2.0 * np.pi * f_test * t)
            p_lo   = float(np.mean(_sosfilt(sos_lo, x)[skip:] ** 2))
            p_hi   = float(np.mean(_sosfilt(sos_hi, x)[skip:] ** 2))
            l_sum  = 10.0 * math.log10(max(p_lo + p_hi, 1e-300))
            freqs_all.append(f_test)
            devs_all.append((l_in - a_ref) - l_sum)

    freqs_all = np.array(freqs_all)
    devs_all  = np.array(devs_all)
    in_tol    = (devs_all >= LO_CL1) & (devs_all <= HI_CL1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axhspan(LO_CL1, HI_CL1, color=C_FILL, alpha=0.5, label="Class 1 acceptance limits")
    ax.axhline(0, color="black", lw=0.8)

    for f_m in centers:
        ax.axvline(f_m, color="grey", lw=0.6, ls=":", alpha=0.5)

    # Connect points within each pair with thin lines
    for i in range(n_bands - 1):
        sl = slice(i * len(TEST_EXPONENTS), (i + 1) * len(TEST_EXPONENTS))
        ax.plot(freqs_all[sl], devs_all[sl], color=C_BLUE, lw=0.8, alpha=0.45)

    ax.scatter(freqs_all[in_tol], devs_all[in_tol],
               color=C_BLUE, s=28, zorder=5, label="Measured")
    if not np.all(in_tol):
        ax.scatter(freqs_all[~in_tol], devs_all[~in_tol],
                   color=C_OUT, s=28, zorder=5, marker="x", label="Outside acceptance limits")

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
# FIG 4.7 — XL2 Broadband Validation: Metric Differences
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_7():
    meas_000 = _load_meas(0)
    meas_003 = _load_meas(3)
    meas_004 = _load_meas(4)

    metrics = ["LAeq", "LCeq", "LZeq", "LAFmax", "LASmax", "LCpeak"]

    def _slm_val(meas, m):
        if m == "LAeq":   return compute_leq(meas, PluginAWeighting)
        if m == "LCeq":   return compute_leq(meas, PluginCWeighting)
        if m == "LZeq":   return compute_leq(meas, _PluginXL2Z)
        if m == "LAFmax": return compute_lmax(meas, PluginAWeighting, PluginFastTimeWeighting)
        if m == "LASmax": return compute_lmax(meas, PluginAWeighting, PluginSlowTimeWeighting)
        if m == "LCpeak": return compute_lpeak(meas, PluginCWeighting)
        raise ValueError(m)

    def _ref_val(meas, m):
        col_map = {
            "LAeq":  "LAeq",
            "LCeq":  "LCeq",
            "LZeq":  "LZeq",
            "LAFmax":"LAFmax",
            "LASmax":"LASmax",
            "LCpeak":"LCPKmax",
        }
        return meas.report_value(col_map[m])

    meas_001 = _load_meas(1)
    datasets = [
        (meas_000, "SLM_000 (1 kHz cal., 94 dB)", C_BLUE,    "o", metrics),
        (meas_001, "SLM_001 (freq. sweep, 30 s)",  "#e6821e", "D", metrics),
        (meas_003, "SLM_003 (pink noise, high)",   C_RED,     "s", metrics),
        (meas_004, "SLM_004 (pink noise, low)",    C_GREEN,   "^", metrics),
    ]

    diffs = {}
    for meas, label, color, marker, mets in datasets:
        print(f"  computing {label}…")
        row = {}
        for m in mets:
            try:
                row[m] = _slm_val(meas, m) - _ref_val(meas, m)
            except Exception as e:
                print(f"    skipping {m}: {e}")
        diffs[label] = (row, color, marker)

    n_ds = len(diffs)
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(metrics))

    ax.axhline(0, color="black", lw=0.8)

    off_vals = np.linspace(-0.25, 0.25, n_ds)
    for (label, (row, color, marker)), offset in zip(diffs.items(), off_vals):
        xd, yd, ec, lw = [], [], [], []
        for i, m in enumerate(metrics):
            if m in row:
                xd.append(x_pos[i] + offset)
                yd.append(row[m])
                outside = abs(row[m]) > TOLERANCE_DB
                ec.append(C_OUT if outside else color)
                lw.append(2.0 if outside else 0.5)
        if xd:
            for xi, yi in zip(xd, yd):
                ax.plot(xi, yi, marker=marker, ms=8, color=color,
                        markeredgecolor="none", zorder=5)
            ax.plot([], [], marker=marker, ms=8, color=color,
                    markeredgecolor="none", label=label, linestyle="none")

    for xi in x_pos:
        ax.axvline(xi, color="grey", lw=0.5, alpha=0.3)

    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_ylabel("Difference: soundlevelmeter − XL2 (dB)")
    ax.set_title("XL2 Broadband Validation")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.8 — XL2 Per-Second LAeq Time Series (SLM_001)
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_4_8():
    meas_001 = _load_meas(1)
    xl2_vals = meas_001.log_series("LZeq_dt")
    n        = len(xl2_vals)

    # Compute per-second LZeq (offset = 0)
    print("  computing per-second LZeq for SLM_001…")
    slm_vals = compute_interval_leq(meas_001, _PluginXL2Z, dt=1.0)
    n = min(len(slm_vals), n)
    slm_vals = slm_vals[:n]
    xl2_vals = xl2_vals[:n]
    t    = np.arange(n)
    diff = slm_vals - xl2_vals

    fig, (ax_t, ax_b) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                      gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    # Top panel
    ax_t.plot(t, xl2_vals, color=C_GREY, lw=1.5, ls="--", label="XL2 reference")
    ax_t.plot(t, slm_vals, color=C_BLUE, lw=1.5, label="soundlevelmeter")
    ax_t.axvline(10, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax_t.axvline(20, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax_t.set_ylabel(r"$L_{Zeq,1s}$ (dB)")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.set_title(r"XL2 Per-Second $L_{Zeq}$ Comparison — SLM_001")

    # Bottom panel
    ax_b.axhline(0, color="black", lw=0.8)

    in_tol  = np.abs(diff) <= TOLERANCE_DB
    ax_b.scatter(t[in_tol],  diff[in_tol],  color=C_BLUE, s=20, zorder=4)
    ax_b.scatter(t[~in_tol], diff[~in_tol], color=C_OUT,  s=20, zorder=4, marker="x")

    ax_t.set_ylim(90, 100)
    ax_b.set_xlim(-0.5, 30.5)
    ax_b.set_ylim(-0.5, 0.5)
    ax_b.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_b.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Difference (dB)")
    ax_b.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4.9 — XL2 Octave-Band RTA Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_hz(s: str) -> float:
    """Parse XL2 column label to frequency in Hz ("6.3" → 6.3, "1k" → 1000)."""
    s = str(s).strip()
    if s.lower().endswith('k'):
        return float(s[:-1]) * 1000.0
    return float(s)


def _compute_rta_diff(meas):
    """Return (diff_array, band_labels) aligned by frequency.

    Detects whether the XL2 RTA is 1/1-octave or 1/3-octave from the column
    spacing, computes matching per-band LZeq from the WAV, then aligns by
    frequency value so position mismatches are impossible.
    """
    df        = meas.rta_report.sections["RTA Results"].df
    xl2_row   = df.loc["LZeq"].astype(float)
    xl2_freqs = {_parse_hz(c): float(v) for c, v in xl2_row.items()}

    # Detect bands_per_oct from column spacing
    fvals = sorted(xl2_freqs.keys())
    ratio = fvals[1] / fvals[0] if len(fvals) >= 2 else 2.0
    # 1/1-octave ≈ 2.0, 1/3-octave ≈ 1.26
    bands_per_oct = 3 if ratio < 1.6 else 1

    # Compute matching LZeq from WAV
    blocksize  = 1024
    controller = FileController(str(meas.wav_path), blocksize=blocksize)
    controller.set_sensitivity(meas.sensitivity, unit="V")
    engine     = Engine(controller, dt=1e9)
    bus        = engine.add_bus("bus", PluginZWeighting)
    freq_w     = bus.frequency_weighting
    octave     = bus.add_plugin(PluginOctaveBand(
        limits=(fvals[0], fvals[-1]),
        bands_per_oct=float(bands_per_oct),
        input=freq_w,
        zero_zi=True,
    ))

    sum_sq = np.zeros(octave.n_bands, dtype=np.float64)
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)
        sum_sq += np.sum(octave.output ** 2, axis=1)

    p_ref_sq  = (20e-6 * meas.sensitivity) ** 2
    slm_leq   = 10.0 * np.log10(sum_sq / meas.n_frames / p_ref_sq)
    slm_freqs = {_parse_hz(c): (str(c), v)
                 for c, v in zip(octave.center_frequencies, slm_leq)}

    # Align by frequency (match within 5 %)
    pairs = []
    for f_slm, (lbl, slm_val) in sorted(slm_freqs.items()):
        best = min(xl2_freqs, key=lambda x: abs(x - f_slm))
        if abs(best - f_slm) / f_slm < 0.05:
            pairs.append((lbl, slm_val - xl2_freqs[best]))

    labels = [p[0] for p in pairs]
    diffs  = np.array([p[1] for p in pairs])
    return diffs, labels


def make_fig_4_9():
    TOLERANCE_RTA = 0.2
    _COLORS  = [C_BLUE, C_RED, C_GREEN, "#e6821e", C_GREY]
    _MARKERS = ["o", "s", "^", "D", "v"]

    # Collect all measurements that have an RTA report
    datasets = []
    for num, label in [(3, "SLM_003"), (4, "SLM_004"), (5, "SLM_005")]:
        try:
            m = _load_meas(num)
            if m.rta_report is None:
                continue
            print(f"  computing octave-band LZeq for {label}…")
            diff, lbl = _compute_rta_diff(m)
            datasets.append((label, diff, lbl, len(diff)))
        except Exception as e:
            print(f"  skipping {label}: {e}")

    if not datasets:
        raise FileNotFoundError("No RTA measurements found")

    # Build a common frequency axis from the union of all datasets' bands
    all_freqs = sorted({_parse_hz(lbl)
                        for _, _, lbl_list, _ in datasets
                        for lbl in lbl_list})
    freq_to_x = {f: i for i, f in enumerate(all_freqs)}
    n_all     = len(all_freqs)
    x_all     = np.arange(n_all)

    # 1/1-octave standard centre frequencies — only these get tick labels
    OCT_CENTERS = {8, 16, 31.5, 63, 125, 250, 500, 1000,
                   2000, 4000, 8000, 16000}

    def _oct_label(f):
        for oc in OCT_CENTERS:
            if abs(f - oc) / oc < 0.05:
                return f"{int(oc/1000)}k" if oc >= 1000 else (
                    str(int(oc)) if oc == int(oc) else str(oc))
        return ""

    tick_labels = [_oct_label(f) for f in all_freqs]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhline(0, color="black", lw=0.8)

    n_ds    = len(datasets)
    offsets = np.linspace(-0.25, 0.25, n_ds) if n_ds > 1 else [0.0]

    for (label, diff, lbl_list, nb), color, marker, xoff in zip(
            datasets, _COLORS, _MARKERS, offsets):
        for d, lbl in zip(diff, lbl_list):
            xi = freq_to_x[_parse_hz(lbl)]
            ax.plot(xi + xoff, d, marker=marker, ms=7, color=color,
                    markeredgecolor="none", zorder=5)
        ax.plot([], [], marker=marker, ms=7, color=color,
                markeredgecolor="none", label=label, linestyle="none")

    for xi in x_all:
        ax.axvline(xi, color="grey", lw=0.5, alpha=0.3)

    ax.set_xlim(-0.5, n_all - 0.5)
    ax.set_ylim(-1, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.set_xticks(x_all)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Octave band centre frequency (Hz)")
    ax.set_ylabel("Difference: soundlevelmeter − XL2 (dB)")
    ax.set_title("XL2 Octave-Band RTA Validation")
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
        (make_fig_4_6, "fig_4_6_octave_filter_mask"),
        (make_fig_4_6b, "fig_4_6b_octave_filter_phase_groupdelay"),
        (make_fig_4_6c, "fig_4_6c_octave_summation"),
        (make_fig_4_7, "fig_4_7_xl2_broadband_comparison"),
        (make_fig_4_8, "fig_4_8_xl2_interval_leq"),
        (make_fig_4_9, "fig_4_9_xl2_rta_comparison"),
    ]
    for fn, name in figures:
        print(f"\n[{name}]")
        try:
            fig = fn()
            save(fig, name)
        except Exception as e:
            print(f"  WARNING: skipped — {e}")
    print("\nDone. Figures saved to thesis_figures/")
