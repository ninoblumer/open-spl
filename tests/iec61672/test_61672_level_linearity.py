"""IEC 61672-1:2013 §5.6 — level linearity (class 1).

Tests PluginAWeighting + LeqAccumulator at 1 kHz across a wide amplitude range.

§5.6 specifications:
  - Linear operating range: ≥ 60 dB at 1 kHz (on reference level range).
  - Level linearity deviation across total linear range: ≤ ±0.8 dB (class 1).
  - Any 1–10 dB change in input must produce the same change in output:
    deviation ≤ ±0.3 dB (class 1).

Approach:
  - Sweep input level from −10 dB SPL to +110 dB SPL in 1 dB steps (120 dB range).
  - Fit a best-fit line (linear regression) to measured L_Aeq vs L_input.
  - Residuals from the fit = linearity deviations.
  - Verify (a) all residuals ≤ ±0.8 dB, (b) incremental changes ≤ ±0.3 dB,
    and (c) the range over which residuals ≤ ±0.8 dB spans ≥ 60 dB.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginAWeighting
from slm.meter import LeqAccumulator
from slm.constants import REFERENCE_PRESSURE

p0         = REFERENCE_PRESSURE  # 20 µPa
SAMPLERATE = 48_000
BLOCKSIZE  = 4_096
FREQ_HZ    = 1_000               # §5.6 specifies 1 kHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_bus(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, sensitivity=1.0, dt=1.0):
    mock = types.SimpleNamespace(
        samplerate=samplerate, blocksize=blocksize,
        sensitivity=sensitivity, dt=dt,
        width=1, get_chain=lambda: [],
    )
    mock.bus = mock
    return mock


def _measure_leq(amplitude: float, duration_s: float = 1.0) -> float:
    """Return A-weighted L_Aeq (dB) for a 1 kHz sine of given *amplitude* (Pa)."""
    n      = (int(duration_s * SAMPLERATE) // BLOCKSIZE) * BLOCKSIZE
    t      = np.arange(n) / SAMPLERATE
    signal = amplitude * np.sin(2.0 * np.pi * FREQ_HZ * t)

    bus    = _mock_bus()
    plugin = PluginAWeighting(input=bus)
    leq_m  = plugin.create_meter(LeqAccumulator, name="leq")

    for start in range(0, n, BLOCKSIZE):
        plugin.process(signal[start : start + BLOCKSIZE][np.newaxis, :])

    return float(plugin.read_db("leq")[0])


def _input_level_db(amplitude: float) -> float:
    """True RMS level of a sine with given peak amplitude (Pa) in dB SPL."""
    return 10.0 * np.log10(amplitude ** 2 / 2.0 / p0 ** 2)


# ---------------------------------------------------------------------------
# Build sweep once at module level so all test classes share the data.
# Amplitudes span −10 to +110 dB SPL (120 dB range) in 1 dB steps.
# ---------------------------------------------------------------------------

_L_INPUT_DB = np.arange(-10.0, 111.0, 1.0)           # 121 levels
_AMPLITUDES  = p0 * np.sqrt(2) * 10 ** (_L_INPUT_DB / 20.0)  # peak Pa

# Measure all levels once (module-level, cached).
_L_MEASURED: list[float] | None = None


def _get_sweep():
    global _L_MEASURED
    if _L_MEASURED is None:
        _L_MEASURED = [_measure_leq(A) for A in _AMPLITUDES]
    return np.array(_L_INPUT_DB), np.array(_L_MEASURED)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLevelLinearityTotalRange:
    """IEC 61672-1 §5.6 — total-range linearity deviation ≤ ±0.8 dB, class 1."""

    def test_residuals_within_08dB(self, report: bool = False):
        l_in, l_meas = _get_sweep()

        # Best-fit line (slope should be ~1.0, intercept ~ A-weighting gain at 1 kHz).
        slope, intercept = np.polyfit(l_in, l_meas, 1)
        residuals = l_meas - (slope * l_in + intercept)

        worst = float(np.max(np.abs(residuals)))
        margin = 0.8 - worst
        if report:
            worst_level = float(l_in[np.argmax(np.abs(residuals))])
            return {"label": "max residual", "value": worst, "limit": 0.8,
                    "margin": margin, "note": f"worst @ {worst_level:.0f} dB SPL"}
        assert worst <= 0.8, (
            f"Max linearity deviation = {worst:.4f} dB (class 1 limit: ±0.8 dB)\n"
            f"Worst level: {l_in[np.argmax(np.abs(residuals))]:.0f} dB SPL"
        )

    def test_slope_is_unity(self, report: bool = False):
        """Measured level must change 1 dB per 1 dB input change (slope = 1)."""
        l_in, l_meas = _get_sweep()
        slope, _ = np.polyfit(l_in, l_meas, 1)
        margin = 0.01 - abs(slope - 1.0)
        if report:
            return {"label": "slope deviation", "value": abs(slope - 1.0), "limit": 0.01,
                    "margin": margin, "note": f"slope = {slope:.6f}"}
        assert abs(slope - 1.0) <= 0.01, (
            f"Regression slope = {slope:.6f} (expected 1.0 ± 0.01)"
        )


class TestLevelLinearityIncremental:
    """IEC 61672-1 §5.6 — any 1–10 dB input step → same output change ± 0.3 dB."""

    def test_1dB_steps(self, report: bool = False):
        """Consecutive 1 dB input increments must produce 1 dB output increments ± 0.3 dB."""
        l_in, l_meas = _get_sweep()
        delta_in   = np.diff(l_in)    # all 1.0 dB
        delta_meas = np.diff(l_meas)
        deviations = delta_meas - delta_in

        worst = float(np.max(np.abs(deviations)))
        margin = 0.3 - worst
        if report:
            worst_idx = int(np.argmax(np.abs(deviations)))
            return {"label": "1 dB step deviation", "value": worst, "limit": 0.3,
                    "margin": margin,
                    "note": f"worst at {l_in[worst_idx]:.0f}->{l_in[worst_idx+1]:.0f} dB SPL"}
        assert worst <= 0.3, (
            f"Max 1 dB-step deviation = {worst:.4f} dB (class 1 limit: ±0.3 dB)\n"
            f"Worst at input {l_in[np.argmax(np.abs(deviations))]:.0f} → "
            f"{l_in[np.argmax(np.abs(deviations))+1]:.0f} dB SPL"
        )

    def test_10dB_steps(self):
        """10 dB input increments must produce 10 dB output increments ± 0.3 dB."""
        l_in, l_meas = _get_sweep()
        # Sample every 10th point for 10 dB steps.
        idx  = np.arange(0, len(l_in), 10)
        l_in_10   = l_in[idx]
        l_meas_10 = l_meas[idx]
        delta_in   = np.diff(l_in_10)    # all 10.0 dB
        delta_meas = np.diff(l_meas_10)
        deviations = delta_meas - delta_in

        worst = np.max(np.abs(deviations))
        assert worst <= 0.3, (
            f"Max 10 dB-step deviation = {worst:.4f} dB (class 1 limit: ±0.3 dB)"
        )


class TestLinearRangeWidth:
    """IEC 61672-1 §5.6 — linear operating range ≥ 60 dB at 1 kHz."""

    def test_linear_range_at_least_60dB(self):
        l_in, l_meas = _get_sweep()
        slope, intercept = np.polyfit(l_in, l_meas, 1)
        residuals = np.abs(l_meas - (slope * l_in + intercept))

        in_range = l_in[residuals <= 0.8]
        range_db = float(in_range[-1] - in_range[0]) if len(in_range) >= 2 else 0.0
        assert range_db >= 60.0, (
            f"Linear operating range = {range_db:.1f} dB (minimum: 60 dB)"
        )
