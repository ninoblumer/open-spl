"""Publication-quality figures for the slm-test-02 XL2 validation set.

Companion to ``thesis_plots_conformance.py`` (same visual style), but instead of
the IEC reference-signal conformance tests this visualises the field validation
against the NTi XL2 on the *slm-test-02* recordings.

The SLM-vs-XL2 computation machinery (calibration pinning on SLM_000, 94 dB @
1 kHz; the 2 s filter warm-up; the per-metric ``compute_*`` passes) is inlined
below — this script is self-contained. The XL2 input filter on the broadband
chains is toggled by the ``USE_XL2_INPUT_FILTER`` switch below (the RTA /
octave-band spectra are always flat Z, unaffected by the switch).

Four figures are produced:

  fig_s1_broadband   — broadband metric differences (measured - reference), one panel per
                       weighting (A/C/Z), markers per recording.
  fig_s2_interval_*  — per-second difference (measured - reference) vs time, one
                       panel per recording, A/C/Z overlaid. One figure per metric:
                       Leq at 1 s, 5", 10", 15", then LFmax_dt, LSmax_dt,
                       Lpeak_dt. The XL2 logs no Z for the 5"/10"/15" windows, so
                       those three show A and C only.
  fig_s3_rta_spectra — whole-file 1/3-octave L_Zeq spectra, SLM vs XL2, one panel
                       per recording.
  fig_s4_rta_dev     — 1/3-octave L_Zeq deviation vs band, all recordings.

The 94 dB / 1 kHz calibrator recording (SLM_000 / M00) still pins sensitivity but
is not plotted; only the four field recordings are shown.

Run from repository root:
    venv/Scripts/python scripts/thesis_plots_validation.py
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
C_PURPLE = "#762a83"
C_GREY   = "#555555"

# ── comparison machinery (inlined; formerly scripts/compare_xl2_slm02.py) ─────
# Self-contained SLM-vs-XL2 computation for the slm-test-02 set. The broadband
# metering chains carry the XL2 analog input filter (PluginXL2InputFilter) when
# USE_XL2_INPUT_FILTER is True (set in the switch section below); the RTA /
# octave-band path is always flat Z.
import re
from dataclasses import dataclass

from slm.constants import REFERENCE_PRESSURE
from slm.engine import Engine
from slm.frequency_weighting import (
    PluginAWeighting, PluginCWeighting, PluginZWeighting, PluginXL2InputFilter,
)
from slm.io.file_controller import FileController
from slm.meter import (
    LeqAccumulator, MaxAccumulator, MinAccumulator, LEAccumulator, LeqMovingMeter,
)
from slm.octave_band import PluginOctaveBand
from slm.time_weighting import (
    PluginFastTimeWeighting, PluginSlowTimeWeighting, PluginImpulseTimeWeighting,
    PluginSquare,
)
from util.xl2 import XL2_SLM_File

DATA_DIR = Path("data/slm-test-02")
WAV_SKIP_SAMPLES = 0
WEIGHTINGS = (("A", PluginAWeighting), ("C", PluginCWeighting), ("Z", PluginZWeighting))
_TW_CLS = {
    "F": PluginFastTimeWeighting,
    "S": PluginSlowTimeWeighting,
    "I": PluginImpulseTimeWeighting,
}


def parse_fs_db(wav_path: Path) -> float:
    """Full-scale peak level (dB) from an XL2 WAV filename: '..._FS129.1dB(PK)_..'."""
    m = re.search(r"FS([\d.]+)dB\(PK\)", wav_path.name)
    if not m:
        raise ValueError(f"No FS annotation in {wav_path.name}")
    return float(m.group(1))


def sensitivity_from_fs(fs_db: float) -> float:
    """Controller sensitivity (V/Pa) implied by the nominal full-scale annotation."""
    return 1.0 / (10 ** (fs_db / 20) * REFERENCE_PRESSURE)


def display_label(name: str) -> str:
    """Short human-facing recording label: 'SLM_002' or '..._SLM_002' -> 'M02'."""
    m = re.search(r"SLM_0*(\d+)", name)
    return f"M{int(m.group(1)):02d}" if m else name


@dataclass
class Recording:
    """One XL2 dataset: the WAV plus its parsed reference files."""

    name: str            # e.g. '2026-06-29_SLM_002'
    label: str           # human description
    wav: Path
    fs_db: float
    sensitivity: float = 0.0
    report: XL2_SLM_File | None = None
    log: XL2_SLM_File | None = None
    rta_report: XL2_SLM_File | None = None

    @property
    def samplerate(self) -> int:
        import soundfile as sf
        return sf.info(str(self.wav)).samplerate


def discover(data_dir: Path) -> dict[str, Recording]:
    """Find every SLM_NNN recording in *data_dir* and parse its reference files."""
    labels = {
        "SLM_000": "94 dB / 1 kHz calibration tone",
        "SLM_001": "Pink noise",
        "SLM_002": "Road traffic noise",
        "SLM_003": "Tapping machine",
        "SLM_004": "Manual hammering",
    }
    recordings: dict[str, Recording] = {}
    for wav in sorted(data_dir.glob("*_Audio_*.wav")):
        m = re.search(r"(\d{4}-\d{2}-\d{2}_SLM_\d{3})", wav.name)
        if not m:
            continue
        name = m.group(1)
        key = name.split("_", 1)[1]  # 'SLM_002'

        def first(pattern: str) -> XL2_SLM_File | None:
            hits = list(data_dir.glob(pattern))
            return XL2_SLM_File(hits[0]) if hits else None

        recordings[key] = Recording(
            name=name,
            label=labels.get(key, "?"),
            wav=wav,
            fs_db=parse_fs_db(wav),
            report=first(f"{name}_123_Report.txt"),
            log=first(f"{name}_123_Log.txt"),
            rta_report=first(f"{name}_RTA_*_Report.txt"),
        )
    return recordings


def calibrate(recordings: dict[str, Recording]) -> None:
    """Set every recording's sensitivity from its own FS annotation."""
    print("Calibration (FS annotation)")
    for rec in recordings.values():
        rec.sensitivity = sensitivity_from_fs(rec.fs_db)
        print(f"  {display_label(rec.name)}: FS{rec.fs_db} -> {rec.sensitivity:.6g} V/Pa")
    print()


def _controller(rec: Recording, blocksize: int) -> FileController:
    c = FileController(str(rec.wav), blocksize=blocksize)
    c.set_sensitivity(rec.sensitivity, unit="V")
    if WAV_SKIP_SAMPLES:
        c._sf.seek(WAV_SKIP_SAMPLES)
        c._stream = c._sf.blocks(blocksize=blocksize, overlap=0,
                                 fill_value=0.0, always_2d=True)
    return c


def _metering_source(bus):
    """Frequency-weighting output, with the XL2 input filter inserted if enabled.

    Used by the broadband metering chains (report, per-second log, moving Leq) so
    the XL2's band-limiting input filter (4.4 Hz HPF + 23 kHz LPF, both 4th-order
    Butterworth) is applied there — unless ``USE_XL2_INPUT_FILTER`` is False, in
    which case the flat frequency-weighting output is metered directly. The RTA /
    octave path always bypasses this and reads ``bus.frequency_weighting``.
    """
    if not USE_XL2_INPUT_FILTER:
        return bus.frequency_weighting
    return bus.add_plugin(PluginXL2InputFilter(
        input=bus.frequency_weighting, zero_zi=True,
    ))


def compute_broadband(rec: Recording, blocksize: int = 1024,
                      warmup_s: float = 2.0) -> dict[str, float]:
    """All broadband scalar metrics in a single pass: eq / S,F,I max,min / E / peak.

    The time-weighting filters start from zero, so their first samples are a
    settling transient. We let *warmup_s* seconds of signal settle the filters,
    then reset the time-weighted max/min meters (and the peak, which the input
    filter's cold-start ring would overshoot). ``eq``/``E`` accumulate over the
    whole file and are unaffected.
    """
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)            # never logs; we read at the end

    results: dict[str, float] = {}
    reads: list[tuple[str, object, str]] = []      # (metric, plugin, meter_name)
    warmup_meters: list[object] = []               # reset after warm-up

    for w, w_cls in WEIGHTINGS:
        bus = engine.add_bus(w, w_cls)
        fw = _metering_source(bus)

        # eq / SEL / peak from the squared (Pa²) pressure.
        sq = bus.add_plugin(PluginSquare(input=fw, zero_zi=True))
        sq.create_meter(LeqAccumulator, name="eq")
        sq.create_meter(LEAccumulator, name="E")
        sq.create_meter(MaxAccumulator, name="peak")
        warmup_meters.append(sq.meters["peak"])   # filter startup overshoots the peak
        reads += [(f"L{w}eq", sq, "eq"), (f"L{w}E", sq, "E"), (f"L{w}PKmax", sq, "peak")]

        # max / min for each time weighting (S/F/I).
        for tw in ("S", "F", "I"):
            twp = bus.add_plugin(_TW_CLS[tw](input=fw, zero_zi=True))
            twp.create_meter(MaxAccumulator, name="max")
            twp.create_meter(MinAccumulator, name="min")
            warmup_meters += [twp.meters["max"], twp.meters["min"]]
            reads += [(f"L{w}{tw}max", twp, "max"), (f"L{w}{tw}min", twp, "min")]

    busses = list(engine._busses.values())
    warmup_samples = int(round(warmup_s * controller.samplerate))
    warmed = False
    n = 0
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        for bus in busses:
            bus.process(block.T)
        n += block.shape[0]
        if not warmed and n >= warmup_samples:
            for m in warmup_meters:
                m.reset()
            warmed = True

    for metric, plugin, meter in reads:
        results[metric] = float(plugin.read_db(meter)[0])
    return results


def compute_interval_metrics(rec: Recording, w_cls, dt: float = 1.0,
                             blocksize: int = 4800) -> dict[str, np.ndarray]:
    """Per-interval metric series for one weighting, all in a single pass.

    Returns a dict of equal-length arrays (one value per *dt* interval):
    ``eq`` (LWeq_dt), ``peak`` (LWPKmax_dt), ``Fmax``/``Fmin`` (LWF max/min_dt),
    ``Smax``/``Smin`` (LWS max/min_dt). Only the meter accumulators reset at each
    interval boundary; the time-weighting filter stays warm across intervals, so
    the first interval is a cold-start transient callers should discard.
    """
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)
    bus = engine.add_bus("bus", w_cls)
    fw = _metering_source(bus)

    sq = bus.add_plugin(PluginSquare(input=fw, zero_zi=True))
    sq.create_meter(LeqAccumulator, name="eq")
    sq.create_meter(MaxAccumulator, name="peak")
    fast = bus.add_plugin(PluginFastTimeWeighting(input=fw, zero_zi=True))
    fast.create_meter(MaxAccumulator, name="max")
    fast.create_meter(MinAccumulator, name="min")
    slow = bus.add_plugin(PluginSlowTimeWeighting(input=fw, zero_zi=True))
    slow.create_meter(MaxAccumulator, name="max")
    slow.create_meter(MinAccumulator, name="min")

    reads = [("eq", sq, "eq"), ("peak", sq, "peak"),
             ("Fmax", fast, "max"), ("Fmin", fast, "min"),
             ("Smax", slow, "max"), ("Smin", slow, "min")]
    meters = [sq.meters["eq"], sq.meters["peak"], fast.meters["max"],
              fast.meters["min"], slow.meters["max"], slow.meters["min"]]

    interval = int(round(dt * controller.samplerate))
    out: dict[str, list[float]] = {name: [] for name, _, _ in reads}
    n_acc = 0
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)
        n_acc += blocksize
        if n_acc >= interval:
            for name, plugin, meter in reads:
                out[name].append(float(plugin.read_db(meter)[0]))
            for m in meters:
                m.reset()
            n_acc -= interval
    return {name: np.array(vals) for name, vals in out.items()}


def compute_moving_leq(rec: Recording, w_cls, windows=(5, 10, 15),
                       dt: float = 1.0, blocksize: int = 4800) -> dict[int, np.ndarray]:
    """Per-second trailing moving Leq for each window length, via the SLM's own
    moving-Leq meters (``LeqMovingMeter``). Each series is NaN for the first
    *window* seconds and the full-window trailing Leq thereafter."""
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)
    bus = engine.add_bus("bus", w_cls)
    sq = bus.add_plugin(PluginSquare(input=_metering_source(bus), zero_zi=True))
    for w in windows:
        sq.create_meter(LeqMovingMeter, name=f"m{w}", t=float(w))

    interval = int(round(dt * controller.samplerate))
    out: dict[int, list[float]] = {w: [] for w in windows}
    n_acc = 0
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)
        n_acc += blocksize
        if n_acc >= interval:
            for w in windows:
                out[w].append(float(sq.read_db(f"m{w}")[0]))
            n_acc -= interval
    return {w: np.array(vals) for w, vals in out.items()}


def compute_octave_lzeq(rec: Recording, blocksize: int = 1024) -> tuple[np.ndarray, list[str]]:
    """Whole-file 1/3-octave LZeq spectrum (flat Z), 6.3 Hz – 20 kHz (36 bands)."""
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)
    bus = engine.add_bus("Z", PluginZWeighting)
    # No XL2 input filter here: the RTA band edges (6.3 Hz–20 kHz) already sit
    # inside the filter passband (4.4 Hz–23 kHz).
    octave = bus.add_plugin(PluginOctaveBand(
        limits=(6.3, 20000), bands_per_oct=3.0,
        input=bus.frequency_weighting, zero_zi=True,
    ))
    sum_sq = np.zeros(octave.n_bands, dtype=np.float64)
    n = 0
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)
        sum_sq += np.sum(octave.output ** 2, axis=1)
        n += block.shape[0]
    p_ref_sq = (REFERENCE_PRESSURE * rec.sensitivity) ** 2
    leq = 10 * np.log10(sum_sq / n / p_ref_sq)
    return leq, octave.center_frequencies


def xl2_report_scalar(rec: Recording, col: str) -> float | None:
    df = rec.report.sections["Broadband Results"].df
    if col not in df.columns:
        return None
    val = df[col].iloc[0]
    return None if val is None else float(val)


def xl2_log_series(rec: Recording, col: str) -> np.ndarray | None:
    df = rec.log.sections["Broadband LOG Results"].df
    if col not in df.columns:
        return None
    return df[col].astype(float).values


def xl2_rta_lzeq(rec: Recording) -> np.ndarray:
    df = rec.rta_report.sections["RTA Results"].df
    return df.loc["LZeq"].astype(float).values

# ── XL2 input-filter switch ───────────────────────────────────────────────────
# When True, the XL2 analog input filter (4.4 Hz HPF + 23 kHz LPF, both 4th-order
# Butterworth) is applied to the BROADBAND metering chains (figs s1, s2),
# reproducing the XL2's broadband roll-off. Set False to meter flat A/C/Z with no
# input filter. The RTA / octave-band spectra (figs s3, s4) are always flat Z and
# are unaffected by this switch.
USE_XL2_INPUT_FILTER = True

# ── output ────────────────────────────────────────────────────────────────────
OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png")


# ── recording presentation order, colours and markers ─────────────────────────
# SLM_000 (the 94 dB / 1 kHz calibrator tone) still pins sensitivity via
# calibrate(), but is not plotted — only the field recordings are shown.
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


def _grid_shape(n: int, ncol: int = 3) -> tuple[int, int]:
    """(rows, cols) for an *n*-panel recording grid, at most *ncol* wide."""
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


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
_BASES = ["eq", "E", "PKmax", "Smax", "Fmax"]
_BASE_LABEL = {
    "eq": r"$L_{eq}$", "E": "$L_E$", "PKmax": "$L_{peak}$",
    "Smax": r"$L_{S,max}$", "Fmax": r"$L_{F,max}$",
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
    """Per-second difference (measured - reference) vs time, one panel per recording, A/C/Z overlaid.

    *slm_fn(rec, w, w_cls)* returns the SLM series for one weighting; *xl2_col*
    maps a weighting letter to the matching XL2 log column. The first *mask*
    intervals are blanked (filter cold-start for the time-weighted detectors).
    """
    nrow, ncol = _grid_shape(len(PLOT_KEYS), ncol=2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.25 * nrow),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.18})
    axes_flat = np.asarray(axes).flatten()

    for ax, key in zip(axes_flat, PLOT_KEYS):
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

    for ax in axes_flat[len(PLOT_KEYS):]:
        ax.set_visible(False)

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
# time-weighted max are masked.
# (filename suffix, metric key, XL2 column fn, title, mask, ylim)
_INTERVAL_SPECS = [
    ("lfmax_dt", "Fmax", lambda w: f"L{w}Fmax_dt",
     r"Per-second $L_{F,max,dt}$ Difference (measured - reference)", 5, (-1.5, 1.5)),
    ("lsmax_dt", "Smax", lambda w: f"L{w}Smax_dt",
     r"Per-second $L_{S,max,dt}$ Difference (measured - reference)", 5, (-1.5, 1.5)),
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
    """Whole-file 1/3-octave L_Zeq spectra, SLM vs XL2, recording grid."""
    nrow, ncol = _grid_shape(len(PLOT_KEYS), ncol=2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.5 * nrow),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.18})
    axes_flat = np.asarray(axes).flatten()

    for ax in axes_flat[len(PLOT_KEYS):]:
        ax.set_visible(False)

    for ax, key in zip(axes_flat, PLOT_KEYS):
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
