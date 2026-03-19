"""IEC 61672-1:2013 §5.13 Table 5 — C-weighted peak level (class 1).

Reference differences L_Cpeak − L_C and class 1 limits:

  Signal              Freq (Hz)   Ref diff (dB)   Class 1 limits
  ──────────────────  ─────────   ─────────────   ───────────────
  1 cycle             31.5        2.5             ±2.0
  1 cycle             500         3.5             ±1.0
  1 cycle             8 000       3.4             ±2.0
  positive half-cycle 500         2.4             ±1.0
  negative half-cycle 500         2.4             ±1.0

Definitions:
  L_Cpeak = 20·log₁₀(max|p_C(t)| / p₀)   — peak of C-weighted output
  L_C     = 10·log₁₀(⟨p_C²⟩ / p₀²)      — C-weighted steady-state RMS level

Since p₀ cancels in the difference L_Cpeak − L_C, the computation only depends
on the ratio of the burst peak to the steady-state RMS gain of the C-weighting
filter at the test frequency.

Signal chain:  signal → PluginCWeighting → track max|output| (burst)
                                         → compute mean_sq   (steady state)
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginCWeighting
from slm.constants import REFERENCE_PRESSURE

p0         = REFERENCE_PRESSURE   # 20 µPa
SAMPLERATE = 48_000
BLOCKSIZE  = 4_096


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


def _process_signal(signal: np.ndarray, samplerate: int = SAMPLERATE,
                    blocksize: int = BLOCKSIZE) -> np.ndarray:
    """Feed *signal* through PluginCWeighting; return concatenated output."""
    bus    = _mock_bus(samplerate=samplerate, blocksize=blocksize)
    plugin = PluginCWeighting(input=bus)
    n      = len(signal)
    chunks = []
    for start in range(0, n, blocksize):
        end   = min(start + blocksize, n)
        block = signal[start:end]
        if len(block) < blocksize:
            block = np.pad(block, (0, blocksize - len(block)))
        plugin.process(block[np.newaxis, :])
        chunks.append(plugin.output[0, : end - start].copy())
    return np.concatenate(chunks)


def _cpeak_minus_lc(freq_hz: float, signal_type: str,
                    amplitude: float = 1.0) -> float:
    """Return L_Cpeak − L_C (dB) for the specified signal.

    signal_type: "1cycle" | "pos_half" | "neg_half"

    L_Cpeak is measured on the burst output (plus trailing silence to catch
    post-burst filter ring-down).  L_C is the C-weighted steady-state level
    at the same amplitude, measured from a 3 s sine (transient skipped).
    """
    n_period = int(round(SAMPLERATE / freq_hz))

    # Build burst signal
    if signal_type == "1cycle":
        n_burst = n_period
    else:
        n_burst = n_period // 2

    t_burst = np.arange(n_burst) / SAMPLERATE
    raw     = amplitude * np.sin(2.0 * np.pi * freq_hz * t_burst)
    if signal_type == "neg_half":
        raw = -raw

    # Pad with silence after burst (2 × blocksize) to capture filter ring-down.
    burst_signal = np.concatenate([raw, np.zeros(2 * BLOCKSIZE)])
    output_burst = _process_signal(burst_signal)

    # L_Cpeak: maximum absolute value of C-weighted output over the whole window.
    peak     = float(np.max(np.abs(output_burst)))
    l_cpeak  = 20.0 * np.log10(peak / p0)

    # L_C: steady-state C-weighted level from a 3 s sine, transient skipped.
    n_settle  = int(0.5 * SAMPLERATE)   # skip first 0.5 s
    n_total_s = int(3.0 * SAMPLERATE)
    n_ss      = (n_total_s // BLOCKSIZE) * BLOCKSIZE
    t_ss      = np.arange(n_ss) / SAMPLERATE
    ss_signal = amplitude * np.sin(2.0 * np.pi * freq_hz * t_ss)
    output_ss = _process_signal(ss_signal)
    mean_sq   = float(np.mean(output_ss[n_settle:] ** 2))
    l_c       = 10.0 * np.log10(mean_sq / p0 ** 2)

    return l_cpeak - l_c


# ---------------------------------------------------------------------------
# IEC 61672-1:2013 Table 5 — parametrised test data
# (signal_type, freq_hz, ref_diff_dB, cl1_lo, cl1_hi)
# ---------------------------------------------------------------------------

_TABLE5 = [
    ("1cycle",   31.5, 2.5, -2.0, +2.0),
    ("1cycle",  500,   3.5, -1.0, +1.0),
    ("1cycle",  8000,  3.4, -2.0, +2.0),
    ("pos_half", 500,  2.4, -1.0, +1.0),
    ("neg_half", 500,  2.4, -1.0, +1.0),
]
_TABLE5_IDS = [
    f"{sig}_{int(freq)}Hz" for sig, freq, *_ in _TABLE5
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCWeightedPeak:
    """IEC 61672-1 §5.13 — L_Cpeak − L_C vs Table 5, class 1."""

    @pytest.mark.parametrize("row", _TABLE5, ids=_TABLE5_IDS)
    def test_cpeak_minus_lc(self, row, report: bool = False):
        sig_type, freq_hz, ref_diff, cl1_lo, cl1_hi = row
        diff = _cpeak_minus_lc(freq_hz, sig_type)
        dev  = diff - ref_diff
        margin = min(dev - cl1_lo, cl1_hi - dev)
        if report:
            return {"label": f"{sig_type} @ {freq_hz} Hz", "deviation": dev,
                    "limit_lo": cl1_lo, "limit_hi": cl1_hi, "margin": margin}
        assert cl1_lo <= dev <= cl1_hi, (
            f"{sig_type} @ {freq_hz} Hz: "
            f"L_Cpeak − L_C = {diff:.3f} dB, ref = {ref_diff:.1f} dB, "
            f"dev = {dev:+.3f} dB (class 1: [{cl1_lo:+.1f}, {cl1_hi:+.1f}])"
        )


class TestCPeakHalfCycleSymmetry:
    """IEC 61672-1 §5.13 — positive and negative half-cycles give equal L_Cpeak."""

    def test_half_cycle_symmetry_500hz(self):
        """Difference between positive and negative half-cycle peaks ≤ 1.5 dB."""
        diff_pos = _cpeak_minus_lc(500, "pos_half")
        diff_neg = _cpeak_minus_lc(500, "neg_half")
        asymmetry = abs(diff_pos - diff_neg)
        assert asymmetry <= 1.5, (
            f"|L_Cpeak(pos) − L_Cpeak(neg)| = {asymmetry:.4f} dB "
            f"(limit: 1.5 dB)"
        )
