"""IEC 61672-1:2013 §5.5, Table 3 — frequency weighting conformance (class 1).

Tests PluginAWeighting, PluginCWeighting, and PluginZWeighting against the
Table 3 reference values and class 1 tolerance limits.

Approach: feed a 3 s pure sine through each plugin, skip the first 0.5 s
transient, and compare the steady-state RMS gain to the IEC Table 3 goal
± class 1 tolerance.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginAWeighting, PluginCWeighting, PluginZWeighting


# ---------------------------------------------------------------------------
# Shared mock bus (no real Bus or Engine needed for plugin-level tests)
# ---------------------------------------------------------------------------

def _mock_bus(samplerate=48000, blocksize=4096, sensitivity=1.0, dt=1.0):
    mock = types.SimpleNamespace(
        samplerate=samplerate, blocksize=blocksize,
        sensitivity=sensitivity, dt=dt,
        width=1, get_chain=lambda: [],
    )
    mock.bus = mock  # Plugin.__init__ does `self.bus = input.bus` for non-Bus inputs
    return mock


# ---------------------------------------------------------------------------
# Gain measurement helper
# ---------------------------------------------------------------------------

def _measure_gain_db(
    plugin_cls,
    freq_hz: float,
    *,
    duration_s: float = 3.0,
    skip_s: float = 0.5,
    samplerate: int = 48000,
    blocksize: int = 4096,
) -> float:
    """Feed a pure sine through *plugin_cls* and return the steady-state gain in dB."""
    bus = _mock_bus(samplerate=samplerate, blocksize=blocksize)
    plugin = plugin_cls(input=bus)

    n_total = int(duration_s * samplerate)
    n_skip = int(skip_s * samplerate)
    t = np.arange(n_total) / samplerate
    signal = np.sin(2.0 * np.pi * freq_hz * t)

    # Process in fixed-size blocks; collect only valid (non-padded) samples.
    out_chunks = []
    for start in range(0, n_total, blocksize):
        end = min(start + blocksize, n_total)
        block = signal[start:end]
        if len(block) < blocksize:
            block = np.pad(block, (0, blocksize - len(block)))
        plugin.process(block[np.newaxis, :])  # shape (1, blocksize)
        out_chunks.append(plugin.output[0, : end - start].copy())

    output = np.concatenate(out_chunks)

    # Skip transient; compare steady-state RMS.
    rms_out = np.sqrt(np.mean(output[n_skip:] ** 2))
    rms_in = np.sqrt(np.mean(signal[n_skip:] ** 2))
    return 20.0 * np.log10(rms_out / rms_in)


# ---------------------------------------------------------------------------
# IEC 61672-1:2013 Table 3 — weighting reference levels and class 1 limits
# (freq_hz, A_goal_dB, C_goal_dB, cl1_lo_dB, cl1_hi_dB)
# cl1_lo = None  →  no lower bound (tolerance is one-sided upward)
# ---------------------------------------------------------------------------

_TABLE3 = [
    (    10, -70.4, -14.3,   None, +3.0),
    (  12.5, -63.4, -11.2,   None, +2.5),
    (    16, -56.7,  -8.5,   -4.0, +2.0),
    (    20, -50.5,  -6.2,   -2.0, +2.0),
    (    25, -44.7,  -4.4,   -1.5, +2.0),
    (  31.5, -39.4,  -3.0,   -1.5, +1.5),
    (    40, -34.6,  -2.0,   -1.0, +1.0),
    (    50, -30.2,  -1.3,   -1.0, +1.0),
    (    63, -26.2,  -0.8,   -1.0, +1.0),
    (    80, -22.5,  -0.5,   -1.0, +1.0),
    (   100, -19.1,  -0.3,   -1.0, +1.0),
    (   125, -16.1,  -0.2,   -1.0, +1.0),
    (   160, -13.4,  -0.1,   -1.0, +1.0),
    (   200, -10.9,   0.0,   -1.0, +1.0),
    (   250,  -8.6,   0.0,   -1.0, +1.0),
    (   315,  -6.6,   0.0,   -1.0, +1.0),
    (   400,  -4.8,   0.0,   -1.0, +1.0),
    (   500,  -3.2,   0.0,   -1.0, +1.0),
    (   630,  -1.9,   0.0,   -1.0, +1.0),
    (   800,  -0.8,   0.0,   -1.0, +1.0),
    (  1000,   0.0,   0.0,   -0.7, +0.7),
    (  1250,  +0.6,   0.0,   -1.0, +1.0),
    (  1600,  +1.0,  -0.1,   -1.0, +1.0),
    (  2000,  +1.2,  -0.2,   -1.0, +1.0),
    (  2500,  +1.3,  -0.3,   -1.0, +1.0),
    (  3150,  +1.2,  -0.5,   -1.0, +1.0),
    (  4000,  +1.0,  -0.8,   -1.0, +1.0),
    (  5000,  +0.5,  -1.3,   -1.5, +1.5),
    (  6300,  -0.1,  -2.0,   -2.0, +1.5),
    (  8000,  -1.1,  -3.0,   -2.5, +1.5),
    ( 10000,  -2.5,  -4.4,   -3.0, +2.0),
    ( 12500,  -4.3,  -6.2,   -5.0, +2.0),
    ( 16000,  -6.6,  -8.5,  -16.0, +2.5),
    ( 20000,  -9.3, -11.2,   None, +3.0),
]
_TABLE3_IDS = [str(row[0]) for row in _TABLE3]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAWeightingClass1:
    """IEC 61672-1 §5.5 — A-weighting frequency response, class 1 limits."""

    @pytest.mark.parametrize("row", _TABLE3, ids=_TABLE3_IDS)
    def test_gain_within_class1(self, row):
        freq_hz, a_goal, _c, cl1_lo, cl1_hi = row
        gain = _measure_gain_db(PluginAWeighting, freq_hz)
        dev = gain - a_goal
        if cl1_lo is not None:
            assert dev >= cl1_lo, (
                f"A @ {freq_hz} Hz: gain={gain:.3f} dB, goal={a_goal:.1f} dB, "
                f"lower limit={a_goal + cl1_lo:.2f} dB"
            )
        assert dev <= cl1_hi, (
            f"A @ {freq_hz} Hz: gain={gain:.3f} dB, goal={a_goal:.1f} dB, "
            f"upper limit={a_goal + cl1_hi:.2f} dB"
        )


class TestCWeightingClass1:
    """IEC 61672-1 §5.5 — C-weighting frequency response, class 1 limits."""

    @pytest.mark.parametrize("row", _TABLE3, ids=_TABLE3_IDS)
    def test_gain_within_class1(self, row):
        freq_hz, _a, c_goal, cl1_lo, cl1_hi = row
        gain = _measure_gain_db(PluginCWeighting, freq_hz)
        dev = gain - c_goal
        if cl1_lo is not None:
            assert dev >= cl1_lo, (
                f"C @ {freq_hz} Hz: gain={gain:.3f} dB, goal={c_goal:.1f} dB, "
                f"lower limit={c_goal + cl1_lo:.2f} dB"
            )
        assert dev <= cl1_hi, (
            f"C @ {freq_hz} Hz: gain={gain:.3f} dB, goal={c_goal:.1f} dB, "
            f"upper limit={c_goal + cl1_hi:.2f} dB"
        )


class TestZWeightingFlat:
    """IEC 61672-1 Annex E.5 — Z-weighting is a flat passthrough (0 dB)."""

    @pytest.mark.parametrize("row", _TABLE3, ids=_TABLE3_IDS)
    def test_gain_is_zero(self, row):
        freq_hz = row[0]
        gain = _measure_gain_db(PluginZWeighting, freq_hz)
        assert abs(gain) <= 0.1, (
            f"Z @ {freq_hz} Hz: gain={gain:.4f} dB (expected 0.0 ± 0.1 dB)"
        )


class TestNormalisationAt1kHz:
    """IEC 61672-1 §5.5 — all three weightings are normalised to 0 dB at 1 kHz."""

    def test_a_weighting(self):
        gain = _measure_gain_db(PluginAWeighting, 1000)
        assert abs(gain) <= 0.05, f"A @ 1 kHz: gain={gain:.4f} dB"

    def test_c_weighting(self):
        gain = _measure_gain_db(PluginCWeighting, 1000)
        assert abs(gain) <= 0.05, f"C @ 1 kHz: gain={gain:.4f} dB"

    def test_z_weighting(self):
        gain = _measure_gain_db(PluginZWeighting, 1000)
        assert abs(gain) <= 0.05, f"Z @ 1 kHz: gain={gain:.4f} dB"


class TestWeightingDifferencesAt1kHz:
    """IEC 61672-1 §5.5.9 — C, A, Z agree within 0.2 dB at 1 kHz."""

    def test_c_minus_a(self):
        gain_a = _measure_gain_db(PluginAWeighting, 1000)
        gain_c = _measure_gain_db(PluginCWeighting, 1000)
        assert abs(gain_c - gain_a) <= 0.2, (
            f"|L_C − L_A| = {abs(gain_c - gain_a):.4f} dB at 1 kHz (limit: 0.2 dB)"
        )

    def test_z_minus_a(self):
        gain_a = _measure_gain_db(PluginAWeighting, 1000)
        gain_z = _measure_gain_db(PluginZWeighting, 1000)
        assert abs(gain_z - gain_a) <= 0.2, (
            f"|L_Z − L_A| = {abs(gain_z - gain_a):.4f} dB at 1 kHz (limit: 0.2 dB)"
        )
