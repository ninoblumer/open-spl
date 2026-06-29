"""Compare the soundlevelmeter pipeline against the NTi XL2 on the slm-test-02 set.

The XL2 is the reference (type-approved Class-1 SLM). For each recording it logged
its own metrics *and* recorded the audio WAV; here we feed the *same* WAV through
the SLM offline and diff every metric the SLM implements against the XL2's parsed
report / log / RTA files.

Calibration
-----------
SLM_000 is a 94 dB / 1 kHz calibrator tone. We derive the controller sensitivity
from it (``calibrate_from_file``) and pin every other recording to it. Because each
WAV is normalised to its own full-scale range (the ``FSxxx.xdB(PK)`` filename
annotation), the cal tone only fixes one range directly; the others are scaled by
the *same* device-level calibration offset:

    sensitivity_i = sensitivity_from_fs(fs_db_i) * (sens_cal / sensitivity_from_fs(fs_db_000))

The ratio is the residual error of the nominal FS annotation (rounding, mic
sensitivity) and is range-independent on a calibrated Class-1 meter.

What is compared
----------------
* Broadband Report (whole measurement, scalar): Leq, S/F/I max & min, SEL (E),
  peak — for A, C and Z weighting.
* Broadband Log (per-second series): LAeq_dt / LCeq_dt / LZeq_dt — reported as
  robust error statistics (the WAV vs XL2-log start offset makes per-sample
  matching unreliable at sharp transients; see slm-test-01 notes).
* RTA Report (whole measurement): 1/3-octave LZeq spectrum.

Not compared (not implemented in the SLM / out of scope): percentiles (LAFn%),
the impulse-equivalent family (LAIeq, measure "eq" forbids a time-weighting
letter), and the XL2's derived columns (LCeq-LAeq etc.).

No corrections are applied: Z weighting is flat PluginZWeighting (the SLM's
actual implementation, correct per IEC 61672-1 Annex E.5), broadband and per-band
alike. The SLM output is reported as-is and any disagreement with the XL2 is
shown raw.

Usage::

    venv/Scripts/python scripts/compare_xl2_slm02.py
    venv/Scripts/python scripts/compare_xl2_slm02.py --csv out.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root → 'util'

from slm.app.cli import calibrate_from_file
from slm.constants import REFERENCE_PRESSURE
from slm.engine import Engine
from slm.frequency_weighting import (
    PluginAWeighting, PluginCWeighting, PluginZWeighting,
)
from slm.io.file_controller import FileController
from slm.meter import LeqAccumulator, MaxAccumulator, MinAccumulator, LEAccumulator
from slm.octave_band import PluginOctaveBand
from slm.time_weighting import (
    PluginFastTimeWeighting, PluginSlowTimeWeighting, PluginImpulseTimeWeighting,
    PluginSquare,
)
from util.xl2 import XL2_SLM_File

DATA_DIR = Path("data/slm-test-02")

# Weightings: A, C, and flat Z (no XL2 modeling).
WEIGHTINGS = (("A", PluginAWeighting), ("C", PluginCWeighting), ("Z", PluginZWeighting))

# Tolerances (dB). Errors are always printed; these only colour the PASS/FAIL flag.
TOL_EQ = 0.3        # Leq / SEL
TOL_MAXMIN = 0.5    # time-weighted max / min (transient-sensitive on impulsive scenes)
TOL_PEAK = 1.0      # instantaneous peak (single-sample, most sensitive to alignment)
TOL_RTA = 0.5       # per 1/3-oct band

_TW_CLS = {
    "F": PluginFastTimeWeighting,
    "S": PluginSlowTimeWeighting,
    "I": PluginImpulseTimeWeighting,
}


# --------------------------------------------------------------------------- #
# Dataset / sensitivity                                                        #
# --------------------------------------------------------------------------- #

def parse_fs_db(wav_path: Path) -> float:
    """Full-scale peak level (dB) from an XL2 WAV filename: '..._FS129.1dB(PK)_..'."""
    m = re.search(r"FS([\d.]+)dB\(PK\)", wav_path.name)
    if not m:
        raise ValueError(f"No FS annotation in {wav_path.name}")
    return float(m.group(1))


def sensitivity_from_fs(fs_db: float) -> float:
    """Controller sensitivity (V/Pa) implied by the nominal full-scale annotation."""
    return 1.0 / (10 ** (fs_db / 20) * REFERENCE_PRESSURE)


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
        "SLM_000": "94 dB / 1 kHz calibrator tone",
        "SLM_001": "Pink noise, moving source + moving mic",
        "SLM_002": "Background noise (room)",
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
    """Pin every recording's sensitivity to the SLM_000 cal tone (see module docstring)."""
    cal = recordings["SLM_000"]
    sens_cal = calibrate_from_file(str(cal.wav), cal_freq=1000.0, cal_level=94.0)
    ratio = sens_cal / sensitivity_from_fs(cal.fs_db)
    offset_db = 20 * np.log10(ratio)
    print("Calibration (from SLM_000, 94 dB @ 1 kHz)")
    print(f"  sensitivity (cal tone)     : {sens_cal:.6g} V/Pa")
    print(f"  sensitivity (FS annotation): {sensitivity_from_fs(cal.fs_db):.6g} V/Pa")
    print(f"  residual FS offset         : {offset_db:+.3f} dB "
          f"(applied to all recordings)\n")
    for rec in recordings.values():
        rec.sensitivity = sensitivity_from_fs(rec.fs_db) * ratio


# --------------------------------------------------------------------------- #
# SLM computation                                                              #
# --------------------------------------------------------------------------- #

def _controller(rec: Recording, blocksize: int) -> FileController:
    c = FileController(str(rec.wav), blocksize=blocksize)
    c.set_sensitivity(rec.sensitivity, unit="V")
    return c


def _run(controller, bus_processors) -> None:
    """Drive *controller* to EOF, feeding each block to every bus."""
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        for proc in bus_processors:
            proc(block.T)


def compute_broadband(rec: Recording, blocksize: int = 1024,
                      warmup_s: float = 2.0) -> dict[str, float]:
    """All broadband scalar metrics in a single pass: eq / S,F,I max,min / E / peak.

    The time-weighting filters start from zero, so their first samples are a
    settling transient (near-silence → spuriously low ``min``; a step into the
    impulse detector → overshooting ``max``). We let *warmup_s* seconds of signal
    settle the filters, then reset the time-weighted max/min meters. ``eq``/``E``/
    ``peak`` are unaffected by the transient and accumulate over the whole file.
    """
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)            # never logs; we read at the end

    results: dict[str, float] = {}
    reads: list[tuple[str, object, str]] = []      # (metric, plugin, meter_name)
    warmup_meters: list[object] = []               # reset after warm-up

    for w, w_cls in WEIGHTINGS:
        bus = engine.add_bus(w, w_cls)
        fw = bus.frequency_weighting

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


def compute_interval_eq(rec: Recording, w_cls, dt: float = 1.0,
                        blocksize: int = 4800) -> np.ndarray:
    """Per-interval Leq series (LWeq_dt). blocksize=4800 → 10 blocks per 1 s @ 48 kHz."""
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)
    bus = engine.add_bus("bus", w_cls)
    sq = bus.add_plugin(PluginSquare(input=bus.frequency_weighting, zero_zi=True))
    meter = sq.create_meter(LeqAccumulator, name="eq")

    interval = int(round(dt * controller.samplerate))
    out: list[float] = []
    n_acc = 0
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)
        n_acc += blocksize
        if n_acc >= interval:
            out.append(float(sq.read_db("eq")[0]))
            meter.reset()
            n_acc -= interval
    return np.array(out)


def compute_octave_lzeq(rec: Recording, blocksize: int = 1024) -> tuple[np.ndarray, list[str]]:
    """Whole-file 1/3-octave LZeq spectrum (flat Z), 6.3 Hz – 20 kHz (36 bands)."""
    controller = _controller(rec, blocksize)
    engine = Engine(controller, dt=1e9)
    bus = engine.add_bus("Z", PluginZWeighting)
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


# --------------------------------------------------------------------------- #
# XL2 reference accessors                                                       #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Comparison + reporting                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    recording: str
    kind: str
    metric: str
    slm: float
    xl2: float
    tol: float

    @property
    def err(self) -> float:
        return self.slm - self.xl2

    @property
    def ok(self) -> bool:
        return abs(self.err) <= self.tol


def compare_recording(rec: Recording) -> list[Row]:
    rows: list[Row] = []

    # --- broadband report scalars ---
    slm_bb = compute_broadband(rec)
    eq_set = {f"L{w}eq" for w in "ACZ"} | {f"L{w}E" for w in "ACZ"}
    peak_set = {f"L{w}PKmax" for w in "ACZ"}
    for metric, slm_val in slm_bb.items():
        xl2_val = xl2_report_scalar(rec, metric)
        if xl2_val is None:
            continue
        tol = TOL_EQ if metric in eq_set else TOL_PEAK if metric in peak_set else TOL_MAXMIN
        rows.append(Row(rec.name, "report", metric, slm_val, xl2_val, tol))

    # --- per-second Leq_dt series (robust stats only) ---
    for w, w_cls in WEIGHTINGS:
        ref = xl2_log_series(rec, f"L{w}eq_dt")
        if ref is None:
            continue
        slm_series = compute_interval_eq(rec, w_cls)
        k = min(len(slm_series), len(ref))
        if k == 0:
            continue
        err = slm_series[:k] - ref[:k]
        rows.append(Row(rec.name, "log-series", f"L{w}eq_dt[median|abs]",
                        float(np.median(np.abs(err))), 0.0, TOL_EQ))
        rows.append(Row(rec.name, "log-series", f"L{w}eq_dt[p95|abs]",
                        float(np.percentile(np.abs(err), 95)), 0.0, TOL_MAXMIN))

    # --- 1/3-oct LZeq spectrum (RTA report) ---
    try:
        ref = xl2_rta_lzeq(rec)
        slm_bands, labels = compute_octave_lzeq(rec)
        k = min(len(slm_bands), len(ref))
        for i in range(k):
            rows.append(Row(rec.name, "rta-LZeq", f"{labels[i]}Hz",
                            float(slm_bands[i]), float(ref[i]), TOL_RTA))
    except (KeyError, AttributeError):
        pass

    return rows


def print_recording(rec: Recording, rows: list[Row]) -> None:
    print(f"\n{'='*72}\n{rec.name}  -  {rec.label}")
    print(f"  FS={rec.fs_db} dB(PK)   sensitivity={rec.sensitivity:.6g} V/Pa")
    by_kind: dict[str, list[Row]] = {}
    for r in rows:
        by_kind.setdefault(r.kind, []).append(r)

    for kind, group in by_kind.items():
        print(f"\n  [{kind}]")
        if kind == "log-series":
            print(f"    {'metric':<22}{'SLM-XL2 stat':>14}{'tol':>8}  flag")
            for r in group:
                flag = "ok" if r.slm <= r.tol else "HIGH"
                print(f"    {r.metric:<22}{r.slm:>13.3f} {r.tol:>7.2f}  {flag}")
            continue
        print(f"    {'metric':<12}{'SLM':>9}{'XL2':>9}{'err':>9}{'tol':>7}  flag")
        for r in group:
            print(f"    {r.metric:<12}{r.slm:>9.2f}{r.xl2:>9.2f}{r.err:>+9.2f}"
                  f"{r.tol:>7.2f}  {'PASS' if r.ok else 'FAIL'}")
        worst = max((r for r in group), key=lambda r: abs(r.err), default=None)
        if worst is not None:
            n_fail = sum(1 for r in group if not r.ok)
            print(f"    -> {len(group)} metrics, {n_fail} outside tol, "
                  f"max |err| = {abs(worst.err):.2f} dB ({worst.metric})")


def write_csv(path: Path, all_rows: list[Row]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["recording", "kind", "metric", "slm_db", "xl2_db", "err_db",
                     "tol_db", "pass"])
        for r in all_rows:
            if r.kind == "log-series":
                wr.writerow([r.recording, r.kind, r.metric, f"{r.slm:.3f}", "",
                             "", f"{r.tol:.2f}", ""])
            else:
                wr.writerow([r.recording, r.kind, r.metric, f"{r.slm:.2f}",
                             f"{r.xl2:.2f}", f"{r.err:+.2f}", f"{r.tol:.2f}",
                             "pass" if r.ok else "fail"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "comparison_summary.csv")
    ap.add_argument("--only", nargs="*", help="restrict to these keys, e.g. SLM_002")
    args = ap.parse_args()

    recordings = discover(args.data_dir)
    if "SLM_000" not in recordings:
        raise SystemExit("SLM_000 (calibration tone) not found — cannot calibrate.")
    calibrate(recordings)

    keys = [k for k in recordings if k != "SLM_000"]
    if args.only:
        keys = [k for k in keys if k in args.only]

    all_rows: list[Row] = []
    for key in keys:
        rec = recordings[key]
        rows = compare_recording(rec)
        print_recording(rec, rows)
        all_rows += rows

    # Overall summary
    scalar_rows = [r for r in all_rows if r.kind in ("report", "rta-LZeq")]
    n_fail = sum(1 for r in scalar_rows if not r.ok)
    print(f"\n{'='*72}\nOverall: {len(scalar_rows)} scalar comparisons, "
          f"{n_fail} outside tolerance.")
    if scalar_rows:
        worst = max(scalar_rows, key=lambda r: abs(r.err))
        print(f"  Worst: {worst.recording} {worst.metric} "
              f"err={worst.err:+.2f} dB (tol {worst.tol})")

    write_csv(args.csv, all_rows)
    print(f"  Summary written to {args.csv}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
