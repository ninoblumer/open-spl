"""Publication-quality figures for the slm-test-02 XL2 validation set.

Companion to ``thesis_plots_conformance.py`` (same visual style), but instead of
the IEC reference-signal conformance tests this visualises the field validation
against the NTi XL2 on the *slm-test-02* recordings.

The numbers come straight from ``scripts/compare_xl2_slm02.py`` so the figures and
the textual comparison report can never drift apart: the calibration pinning
(SLM_000, 94 dB @ 1 kHz), the 2 s filter warm-up, and the flat Z weighting (no
XL2 modelling) are all inherited from that module.

Four figures are produced:

  fig_s1_broadband   — broadband metric differences (measured - reference), one panel per
                       weighting (A/C/Z), markers per recording.
  fig_s2_interval_*  — per-second difference (measured - reference) vs time, 2x2 recording
                       grid, A/C/Z overlaid. One figure per metric: Leq at 1 s,
                       5", 10", 15", then LFmax_dt, LSmax_dt, LFmin_dt, LSmin_dt,
                       Lpeak_dt. The XL2 logs no Z for the 5"/10"/15" windows, so
                       those three show A and C only.
  fig_s3_rta_spectra — whole-file 1/3-octave L_Zeq spectra, SLM vs XL2, 2x2 grid.
  fig_s4_rta_dev     — 1/3-octave L_Zeq deviation vs band, all recordings.

Run from repository root:
    venv/Scripts/python scripts/thesis_plots_slm02.py
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

# ── global style (matches thesis_plots_conformance.py) ────────────────────────
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

# ── colors (shared with the conformance figures) ──────────────────────────────
C_BLUE   = "#2166ac"
C_RED    = "#d6604d"
C_ORANGE = "#e6821e"
C_GREEN  = "#4dac26"
C_GREY   = "#555555"

# ── comparison machinery (single source of truth for the numbers) ─────────────
from scripts.compare_xl2_slm02 import (
    DATA_DIR, WEIGHTINGS,
    discover, calibrate, display_label,
    compute_broadband, compute_interval_metrics, compute_moving_leq,
    compute_octave_lzeq,
    xl2_report_scalar, xl2_log_series, xl2_rta_lzeq,
)

# ── output ────────────────────────────────────────────────────────────────────
OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png")


# ── recording presentation order, colours and markers ─────────────────────────
# SLM_000 is the calibrator tone (used only to pin sensitivity); not plotted.
PLOT_KEYS = ["SLM_001", "SLM_002", "SLM_003", "SLM_004"]
REC_COLOR  = {"SLM_001": C_BLUE, "SLM_002": C_ORANGE,
              "SLM_003": C_RED,  "SLM_004": C_GREEN}
REC_MARKER = {"SLM_001": "o", "SLM_002": "D",
              "SLM_003": "s", "SLM_004": "^"}

_octave_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _parse_hz(s) -> float:
    """Parse a band centre label to Hz ('6.3' -> 6.3, '1k' -> 1000)."""
    s = str(s).strip()
    if s.lower().endswith("k"):
        return float(s[:-1]) * 1000.0
    return float(s)


def _octave(rec):
    """Cached whole-file 1/3-octave L_Zeq spectrum for *rec* (freqs as floats)."""
    if rec.name not in _octave_cache:
        leq, labels = compute_octave_lzeq(rec)
        freqs = np.array([_parse_hz(s) for s in labels])
        _octave_cache[rec.name] = (leq, freqs)
    return _octave_cache[rec.name]


def _fmt_hz(f: float) -> str:
    """'63.0' -> '63', '1000.0' -> '1k', '12500.0' -> '12.5k'."""
    if f >= 1000:
        v = f / 1000.0
        return f"{v:g}k"
    return f"{f:g}"


# ═══════════════════════════════════════════════════════════════════════════════
# FIG S.1 — Broadband metric differences (measured - reference), per weighting
# ═══════════════════════════════════════════════════════════════════════════════

# Metric "base" names (weighting letter stripped) in display order.
# Impulse (I) time weighting is intentionally excluded.
_BASES = ["eq", "E", "PKmax", "Smax", "Smin", "Fmax", "Fmin"]
_BASE_LABEL = {
    "eq": r"$L_{eq}$", "E": "$L_E$", "PKmax": "$L_{peak}$",
    "Smax": r"$L_{S,max}$", "Smin": r"$L_{S,min}$",
    "Fmax": r"$L_{F,max}$", "Fmin": r"$L_{F,min}$",
}


def make_fig_s1(recordings) -> plt.Figure:
    """Broadband metric difference SLM - XL2, one panel per weighting.

    Each panel's x-axis is the metric (weighting letter stripped); one marker
    per recording.
    """
    # slm[(key, weighting)][base] = diff
    diffs: dict[tuple[str, str], dict[str, float]] = {}
    for key in PLOT_KEYS:
        rec = recordings[key]
        print(f"  broadband {key}…")
        slm_bb = compute_broadband(rec)
        for metric, slm_val in slm_bb.items():
            xl2_val = xl2_report_scalar(rec, metric)
            if xl2_val is None:
                continue
            w, base = metric[1], metric[2:]
            diffs.setdefault((key, w), {})[base] = slm_val - xl2_val

    weightings = [w for w, _ in WEIGHTINGS]
    fig, axes = plt.subplots(1, len(weightings), figsize=(12, 4.5), sharey=True,
                             gridspec_kw={"wspace": 0.08})

    n_rec = len(PLOT_KEYS)
    offsets = np.linspace(-0.26, 0.26, n_rec)
    x_pos = np.arange(len(_BASES))

    for ax, w in zip(axes, weightings):
        ax.axhline(0, color="black", lw=0.8)
        for key, xoff in zip(PLOT_KEYS, offsets):
            row = diffs.get((key, w), {})
            color, marker = REC_COLOR[key], REC_MARKER[key]
            for i, base in enumerate(_BASES):
                if base not in row:
                    continue
                ax.plot(x_pos[i] + xoff, row[base], marker=marker, ms=7,
                        color=color, markeredgecolor="none", zorder=5)
        for xi in x_pos:
            ax.axvline(xi, color="grey", lw=0.5, alpha=0.25)
        ax.set_xlim(-0.5, len(_BASES) - 0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_BASE_LABEL[b] for b in _BASES], fontsize=8,
                           rotation=45, ha="right")
        ax.set_title(f"{w}-weighting", fontsize=10)

    axes[0].set_ylim(-1.5, 1.5)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes[0].set_ylabel("Difference: measured - reference (dB)")

    # Shared recording legend (markers).
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=REC_MARKER[k], color=REC_COLOR[k], lw=0,
                      ms=7, markeredgecolor="none",
                      label=f"{display_label(k)}  ({recordings[k].label})")
               for k in PLOT_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Broadband Validation",
                 fontsize=11)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIG S.2 — Per-second metric differences vs time (one figure per metric)
# ═══════════════════════════════════════════════════════════════════════════════

# Weighting trace colours within each per-second panel.
W_COLOR = {"A": C_BLUE, "C": C_RED, "Z": C_GREEN}

# Per-interval metric series, computed once per (recording, weighting) and reused
# across every fig_s2 figure.
_metrics_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def _interval_metrics(rec, w: str, w_cls) -> dict[str, np.ndarray]:
    key = (rec.name, w)
    if key not in _metrics_cache:
        print(f"  interval metrics {rec.name} [{w}]…")
        _metrics_cache[key] = compute_interval_metrics(rec, w_cls)
    return _metrics_cache[key]


# Per-second trailing moving Leq (LeqMovingMeter), computed once per
# (recording, weighting) via the real SLM chain and reused across figures.
_moving_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}


def _moving_metrics(rec, w: str, w_cls) -> dict[int, np.ndarray]:
    key = (rec.name, w)
    if key not in _moving_cache:
        print(f"  moving Leq {rec.name} [{w}]…")
        _moving_cache[key] = compute_moving_leq(rec, w_cls)
    return _moving_cache[key]


def _interval_diff_figure(recordings, *, slm_fn, xl2_col, title,
                          mask: int = 0, ylim=(-0.6, 0.6)) -> plt.Figure:
    """Per-second difference (measured - reference) vs time, 2x2 recording grid, A/C/Z overlaid.

    *slm_fn(rec, w, w_cls)* returns the SLM series for one weighting; *xl2_col*
    maps a weighting letter to the matching XL2 log column. The first *mask*
    intervals are blanked (filter cold-start for the time-weighted detectors).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.18})

    for ax, key in zip(axes.flat, PLOT_KEYS):
        rec = recordings[key]
        for w, w_cls in WEIGHTINGS:
            try:
                ref = xl2_log_series(rec, xl2_col(w))
            except (KeyError, ValueError, TypeError):
                ref = None
            if ref is None:
                continue
            slm = slm_fn(rec, w, w_cls)
            n = min(len(slm), len(ref))
            if n == 0:
                continue
            diff = slm[:n] - ref[:n]
            if mask:
                diff = diff.copy()
                diff[:mask] = np.nan
            ax.plot(np.arange(n), diff, color=W_COLOR[w], lw=1.0, marker="o",
                    ms=2, alpha=0.85, label=w)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_title(f"{display_label(key)} - {rec.label}", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(r"$\Delta$ (dB)", fontsize=8)
        if key == PLOT_KEYS[0] and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best", title="weighting", ncol=3)

    fig.suptitle(title, fontsize=11)
    return fig


# Leq at several integration windows, all built in the same style. The 1 s value
# is the XL2's own per-interval LWeq_dt (A/C/Z all logged); the 5/10/15 s values
# come from the SLM's moving-Leq meters (NaN until the window fills). The XL2 logs
# no LZeq5"/10"/15", so the trailing-window figures show A and C only.
# (seconds, filename suffix, title)
_LEQ_WINDOW_SPECS = [
    (1, "leq_1s", r"Per-second (1 s) $L_{eq}$ Difference (measured - reference)"),
    (5, "leq_5s", r'Trailing 5" $L_{eq}$ Difference (measured - reference)'),
    (10, "leq_10s", r'Trailing 10" $L_{eq}$ Difference (measured - reference)'),
    (15, "leq_15s", r'Trailing 15" $L_{eq}$ Difference (measured - reference)'),
]


def _leq_slm_fn(secs: int):
    """SLM series for the *secs*-second Leq (1 s = per-interval; else moving meter)."""
    if secs == 1:
        return lambda rec, w, wc: _interval_metrics(rec, w, wc)["eq"]
    return lambda rec, w, wc, s=secs: _moving_metrics(rec, w, wc)[s]


def _leq_col(secs: int):
    """XL2 column for the *secs*-second Leq (1 s = LWeq_dt; else LWeqT\")."""
    if secs == 1:
        return lambda w: f"L{w}eq_dt"
    return lambda w, s=secs: f'L{w}eq{s}"'


# Per-second (dt) metrics other than Leq. F/S detectors start cold; the Slow
# detector (tau = 1 s) needs ~5 s to settle, so the first 5 intervals of the
# time-weighted max/min are masked.
# (filename suffix, metric key, XL2 column fn, title, mask, ylim)
_INTERVAL_SPECS = [
    ("lfmax_dt", "Fmax", lambda w: f"L{w}Fmax_dt",
     r"Per-second $L_{F,max,dt}$ Difference (measured - reference)", 5, (-1.5, 1.5)),
    ("lsmax_dt", "Smax", lambda w: f"L{w}Smax_dt",
     r"Per-second $L_{S,max,dt}$ Difference (measured - reference)", 5, (-1.5, 1.5)),
    ("lfmin_dt", "Fmin", lambda w: f"L{w}Fmin_dt",
     r"Per-second $L_{F,min,dt}$ Difference (measured - reference)", 5, (-1.5, 1.5)),
    ("lsmin_dt", "Smin", lambda w: f"L{w}Smin_dt",
     r"Per-second $L_{S,min,dt}$ Difference (measured - reference)", 5, (-1.5, 1.5)),
    ("lpeak_dt", "peak", lambda w: f"L{w}PKmax_dt",
     r"Per-second $L_{peak,dt}$ Difference (measured - reference)", 0, (-1.5, 1.5)),
]


def _interval_figures():
    """Build the (callable, filename) list for every fig_s2 per-second figure."""
    figs = []
    for secs, suffix, title in _LEQ_WINDOW_SPECS:
        figs.append((partial(_interval_diff_figure, slm_fn=_leq_slm_fn(secs),
                             xl2_col=_leq_col(secs), title=title, mask=0,
                             ylim=(-0.6, 0.6)),
                     f"fig_s2_slm02_interval_{suffix}"))
    for suffix, key, col, title, mask, ylim in _INTERVAL_SPECS:
        slm_fn = lambda rec, w, wc, k=key: _interval_metrics(rec, w, wc)[k]
        figs.append((partial(_interval_diff_figure, slm_fn=slm_fn, xl2_col=col,
                             title=title, mask=mask, ylim=ylim),
                     f"fig_s2_slm02_interval_{suffix}"))
    return figs


# ═══════════════════════════════════════════════════════════════════════════════
# FIG S.3 — 1/3-octave L_Zeq spectra, SLM vs XL2
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_s3(recordings) -> plt.Figure:
    """Whole-file 1/3-octave L_Zeq spectra, SLM vs XL2, 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.18})

    for ax, key in zip(axes.flat, PLOT_KEYS):
        rec = recordings[key]
        try:
            ref = xl2_rta_lzeq(rec)
        except (KeyError, AttributeError):
            ax.set_visible(False)
            continue
        print(f"  RTA spectrum {key}…")
        slm, freqs = _octave(rec)
        k = min(len(slm), len(ref))
        x = np.arange(k)

        ax.plot(x, ref[:k], color=C_GREY, lw=1.5, ls="--", marker="s", ms=3,
                label="reference")
        ax.plot(x, slm[:k], color=REC_COLOR[key], lw=1.5, marker="o", ms=3,
                label="measured")

        ax.set_title(f"{display_label(key)} - {rec.label}", fontsize=9)
        ax.set_ylabel(r"$L_{Zeq}$ (dB)", fontsize=8)
        ax.set_xlabel("1/3-octave band centre (Hz)", fontsize=8)
        _set_band_xticks(ax, freqs[:k], x)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("1/3-octave $L_{Zeq}$ Spectra",
                 fontsize=11)
    return fig


def _set_band_xticks(ax, freqs, x) -> None:
    """Label only the decade/standard 1/1-octave centres to avoid clutter."""
    oct_centers = [6.3, 12.5, 25, 50, 100, 200, 400, 800, 1600,
                   3150, 6300, 12500]
    ticks, labels = [], []
    for xi, f in zip(x, freqs):
        if any(abs(f - oc) / oc < 0.04 for oc in oct_centers):
            ticks.append(xi)
            labels.append(_fmt_hz(f))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG S.4 — 1/3-octave L_Zeq deviation vs band, all recordings
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_s4(recordings) -> plt.Figure:
    """1/3-octave L_Zeq deviation (measured - reference) vs band, all recordings overlaid."""
    # Use the band grid of the first available recording as the common axis.
    series: list[tuple[str, np.ndarray, np.ndarray]] = []  # key, diff, freqs
    ref_freqs = None
    for key in PLOT_KEYS:
        rec = recordings[key]
        try:
            ref = xl2_rta_lzeq(rec)
        except (KeyError, AttributeError):
            continue
        print(f"  RTA deviation {key}…")
        slm, freqs = _octave(rec)
        k = min(len(slm), len(ref))
        series.append((key, slm[:k] - ref[:k], np.asarray(freqs[:k])))
        if ref_freqs is None or len(freqs[:k]) > len(ref_freqs):
            ref_freqs = np.asarray(freqs[:k])

    if not series:
        raise FileNotFoundError("No RTA reports found in slm-test-02")

    n = len(ref_freqs)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.axhline(0, color="black", lw=0.8)

    n_ds = len(series)
    offsets = np.linspace(-0.22, 0.22, n_ds) if n_ds > 1 else [0.0]
    for (key, diff, freqs), xoff in zip(series, offsets):
        color, marker = REC_COLOR[key], REC_MARKER[key]
        xi = x[:len(diff)] + xoff
        ax.plot(xi, diff, color=color, lw=0.6, alpha=0.4, zorder=3)
        ax.scatter(xi, diff, color=color, s=22, marker=marker,
                   linewidths=0, zorder=5,
                   label=f"{display_label(key)} ({recordings[key].label})")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-2.0, 2.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    _set_band_xticks(ax, ref_freqs, x)
    ax.set_xlabel("1/3-octave band centre frequency (Hz)")
    ax.set_ylabel("Difference: measured - reference (dB)")
    ax.set_title("1/3-octave $L_{Zeq}$ Deviation",
                 fontsize=10)
    ax.legend(loc="upper center", fontsize=8, ncol=3)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Generating slm-test-02 validation figures…")
    recordings = discover(DATA_DIR)
    calibrate(recordings)

    figures = [
        (make_fig_s1, "fig_s1_slm02_broadband"),
        *_interval_figures(),
        (make_fig_s3, "fig_s3_slm02_rta_spectra"),
        (make_fig_s4, "fig_s4_slm02_rta_deviation"),
    ]
    for fn, name in figures:
        print(f"\n[{name}]")
        try:
            fig = fn(recordings)
            save(fig, name)
        except Exception as e:
            import traceback
            print(f"  WARNING: skipped — {e}")
            traceback.print_exc()

    print("\nDone. Figures saved to thesis_figures/")


if __name__ == "__main__":
    main()
