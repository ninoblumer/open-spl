"""End-to-end XL2 validation: run the product CLI, diff its CSVs against the XL2.

Unlike a unit test, this drives the *actual measurement entry point*
(:func:`slm.app.cli.run_measurement`, the function behind ``python -m slm --file
…``) on every reference recording, reads back the CSV report/log the SLM itself
writes, and compares those numbers against the NTi XL2's parsed report / log /
RTA files. The XL2 (a type-approved Class-1 SLM) is the reference.

Both reference sets are covered (see :func:`tests.conftest.discover_xl2_recordings`):

* ``slm-test-01`` — *electrical*: a signal generator fed straight into the XL2,
  which logged its metrics and recorded the WAV.
* ``slm-test-02`` — *acoustic*: real microphone recordings.

SLM_002 is excluded in both sets (see conftest). Calibration is purely from each
recording's ``FSxxx.xdB(PK)`` filename annotation — no per-recording tuning.

Signal path
-----------
Broadband metrics are measured with ``signal_conditioning="xl2"``, which inserts
:class:`~slm.frequency_weighting.PluginXL2InputFilter` (a 4.4 Hz Butterworth HPF
+ 23 kHz Butterworth LPF, per NTi technical support) at the head of every bus —
the XL2's analog input filter, in its correct physical position in front of the
frequency weighting. The RTA spectrum is measured *unconditioned*: the XL2's RTA
path has no such filter and its band edges (6.3 Hz–20 kHz) sit inside the
passband, so conditioning would only bias the edge bands.

No other corrections are applied. In particular ``warmup`` is 0 (exactly what the
plain CLI does), so comparisons target the metrics that are robust to a cold
filter start — whole-file Leq / SEL, per-interval Leq, and the RTA spectrum.
Time-weighted max/min and peak are out of scope: the XL2 reset its detectors
mid-stream while the SLM's start from zero, and the single ``warmup`` knob cannot
settle them without also shortening the Leq window.

All tests are marked ``slow`` (each recording is run through the engine twice);
pass ``--slow`` to execute them.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from slm.app.cli import run_measurement
from slm.app.config import SLMConfig
from tests.conftest import XL2Measurement, discover_xl2_recordings

pytestmark = pytest.mark.slow

# Recordings across both datasets → one parametrization each.
_RECORDINGS = discover_xl2_recordings()
_IDS = [f"{data_dir.name}:{key}" for data_dir, _name, key in _RECORDINGS]

# Metric groups measured in the single broadband pass.
EQ_METRICS = ("LAeq", "LCeq", "LZeq")
SEL_METRICS = ("LAE", "LCE", "LZE")
SERIES_METRICS = ("LAeq_dt", "LCeq_dt", "LZeq_dt")

# Tolerances (dB). The dominant residual is the SLM's IIR frequency-weighting
# filter (~0.1–0.15 dB at band edges); the XL2 input-filter model removes the
# broadband-Z infrasound offset.
TOL_EQ = 0.2            # whole-file Leq        (observed worst ≈ 0.2)
TOL_SEL = 0.2          # sound exposure level  (observed worst ≈ 0.2)

# Per-interval Leq is only compared where the reference is stationary enough that
# the ~sub-second XL2-log/WAV start offset does not dominate. Stationarity is read
# from the XL2 series itself (per metric): a log sweep's A level swings second to
# second while its flat-Z level barely moves, so the gate is applied per metric.
# The measure is the 90th percentile of |consecutive-second differences|, which
# catches both continuous drift and occasional level jumps; recordings with
# p90(|Δ/s|) ≤ 0.52 dB compare cleanly, those ≥ 0.72 do not (sweep / moving source
# / hammering), so 0.6 dB separates them with margin.
SERIES_MAX_VOLATILITY = 0.6   # dB, p90 of |consecutive-second Δ| in the XL2 series
TOL_SERIES_MEDIAN = 0.35      # per-interval Leq, median |error|  (observed worst ≈ 0.22)
TOL_SERIES_P90 = 0.6          # per-interval Leq, 90th-percentile |error|  (observed worst ≈ 0.34)

# RTA is compared over the bands that carry the signal — those within
# RTA_STRONG_BAND_RANGE_DB of the spectral peak. Weaker bands are dominated by the
# noise floor and by filter-stopband/leakage differences that legitimately diverge
# between the XL2 and an IIR filter bank (and are meaningless on a pure tone).
RTA_STRONG_BAND_RANGE_DB = 20.0
TOL_RTA = 1.0          # per strong 1/3- or 1/1-octave band  (observed worst ≈ 0.77)


# --------------------------------------------------------------------------- #
# CSV readers                                                                  #
# --------------------------------------------------------------------------- #

def _read_report_csv(path: str) -> dict[str, float]:
    """Single-row ``*_report.csv`` → ``{column: value}``."""
    with open(path, newline="", encoding="utf-8") as f:
        row = next(iter(csv.DictReader(f)))
    return {k: float(v) for k, v in row.items() if v != ""}


def _read_log_csv(path: str) -> dict[str, np.ndarray]:
    """Per-interval ``*_log.csv`` → ``{column: array}`` (the ``timestamp`` column
    is dropped; only numeric metric columns are returned)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols: dict[str, list[float]] = {k: [] for k in rows[0] if k != "timestamp"}
    for row in rows:
        for k in cols:
            cols[k].append(float(row[k]))
    return {k: np.asarray(v) for k, v in cols.items()}


def _read_rta_report_csv(path: str) -> np.ndarray:
    """Single-row ``*_rta_report.csv`` → band values in column order."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                       # header (one column per band)
        values = next(reader)
    return np.asarray([float(v) for v in values])


# --------------------------------------------------------------------------- #
# One measurement per recording, cached for the whole module                  #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module", params=_RECORDINGS, ids=_IDS)
def comparison(request, tmp_path_factory):
    """Run the SLM CLI on one recording (broadband + RTA passes) and collect both
    the SLM CSV outputs and the XL2 reference object."""
    data_dir, name, key = request.param
    meas = XL2Measurement(name, data_dir=data_dir)
    tmp = tmp_path_factory.mktemp(f"{data_dir.name}_{key}")

    # Broadband pass — with the XL2 analog input filter in the chain.
    bb_stem = str(tmp / "bb")
    run_measurement(
        str(meas.wav_path), meas.sensitivity,
        SLMConfig(metrics=list(EQ_METRICS + SEL_METRICS + SERIES_METRICS),
                  dt=1.0, output=bb_stem, signal_conditioning="xl2"),
    )

    # RTA pass — unconditioned, band set matched to the XL2 RTA resolution.
    rta_stem = str(tmp / "rta")
    run_measurement(
        str(meas.wav_path), meas.sensitivity,
        SLMConfig(metrics=[meas.rta_metric()], dt=1.0, output=rta_stem),
    )

    return {
        "meas": meas,
        "key": key,
        "report": _read_report_csv(bb_stem + "_report.csv"),
        "log": _read_log_csv(bb_stem + "_log.csv"),
        "rta": _read_rta_report_csv(rta_stem + "_rta_report.csv"),
    }


# --------------------------------------------------------------------------- #
# Comparisons                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("metric", EQ_METRICS)
def test_broadband_leq(comparison, metric):
    """Whole-file A/C/Z Leq from the report CSV matches the XL2 Broadband report."""
    slm = comparison["report"][metric]
    ref = comparison["meas"].report_value(metric)
    assert abs(slm - ref) <= TOL_EQ, (
        f"{comparison['key']} {metric}: SLM {slm:.2f} vs XL2 {ref:.2f} dB "
        f"(Δ {slm - ref:+.2f}, tol {TOL_EQ})"
    )


@pytest.mark.parametrize("metric", SEL_METRICS)
def test_broadband_sel(comparison, metric):
    """Whole-file A/C/Z sound exposure level matches the XL2 Broadband report."""
    slm = comparison["report"][metric]
    ref = comparison["meas"].report_value(metric)
    assert abs(slm - ref) <= TOL_SEL, (
        f"{comparison['key']} {metric}: SLM {slm:.2f} vs XL2 {ref:.2f} dB "
        f"(Δ {slm - ref:+.2f}, tol {TOL_SEL})"
    )


@pytest.mark.parametrize("metric", SERIES_METRICS)
def test_per_interval_leq(comparison, metric):
    """Per-second LWeq_dt series matches the XL2 log (robust error stats).

    A ~sub-second offset between the XL2 log grid and the WAV start makes exact
    per-sample matching unreliable where the level changes abruptly (see the
    SLM_001 boundary-second note in memory). We therefore (a) skip the comparison
    when the XL2 reference itself is too non-stationary for a time-aligned diff to
    be meaningful, and (b) assert on the median and 90th percentile of the absolute
    error rather than every sample.
    """
    meas = comparison["meas"]
    ref = meas.log_series(metric)
    volatility = float(np.percentile(np.abs(np.diff(ref)), 90))
    if volatility > SERIES_MAX_VOLATILITY:
        pytest.skip(
            f"{comparison['key']} {metric}: XL2 series too non-stationary "
            f"(p90 |Δ/s| {volatility:.2f} dB) for a time-aligned per-second "
            "comparison; whole-file Leq still validates the energy"
        )

    slm = comparison["log"][metric]
    k = min(len(slm), len(ref))
    err = np.abs(slm[:k] - ref[:k])
    err = err[np.isfinite(err)]        # drop the leading partial-window NaN sample
    assert err.size, f"{comparison['key']} {metric}: no comparable samples"
    median = float(np.median(err))
    p90 = float(np.percentile(err, 90))
    assert median <= TOL_SERIES_MEDIAN and p90 <= TOL_SERIES_P90, (
        f"{comparison['key']} {metric}: median |Δ| {median:.2f} "
        f"(tol {TOL_SERIES_MEDIAN}), p90 {p90:.2f} (tol {TOL_SERIES_P90})"
    )


def test_rta_lzeq_spectrum(comparison):
    """Per-band LZeq spectrum matches the XL2 RTA report over the signal-carrying
    bands (unconditioned Z path)."""
    slm = comparison["rta"]
    ref = comparison["meas"].rta_lzeq()
    assert len(slm) == len(ref), (
        f"{comparison['key']}: {len(slm)} SLM bands vs {len(ref)} XL2 bands"
    )
    strong = ref >= ref.max() - RTA_STRONG_BAND_RANGE_DB
    err = np.abs(slm - ref)
    err[~strong] = 0.0                 # ignore weak / noise-floor bands
    worst = int(np.argmax(err))
    assert err[worst] <= TOL_RTA, (
        f"{comparison['key']}: worst strong band {worst} |Δ| {err[worst]:.2f} dB "
        f"(SLM {slm[worst]:.2f} vs XL2 {ref[worst]:.2f}, tol {TOL_RTA})"
    )
