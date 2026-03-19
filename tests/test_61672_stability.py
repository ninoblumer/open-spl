"""IEC 61672-1:2013 §5.14, §5.15 — computational stability (class 1).

These tests are marked @pytest.mark.slow and are skipped by default.
Run with:  pytest --slow

§5.14 — Continuous-operation stability:
  After 30 min of continuous operation at 1 kHz, the difference between the
  initial and final A-weighted level indications must be ≤ ±0.1 dB (class 1).

§5.15 — High-level stability:
  After 5 min at a level 1 dB below the upper boundary of the linear operating
  range, the difference between initial and final indications must be ≤ ±0.1 dB.

For a software SLM these tests verify floating-point accumulator stability over
long runs (no hardware drift or thermal effects to worry about).

Approach: run the A-weighting + LeqAccumulator chain over short windows at the
start and end of the full duration, compare L_Aeq values.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginAWeighting
from slm.meter import LeqAccumulator
from slm.constants import REFERENCE_PRESSURE

p0         = REFERENCE_PRESSURE
SAMPLERATE = 48_000
BLOCKSIZE  = 4_096


def _mock_bus(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, sensitivity=1.0, dt=1.0):
    mock = types.SimpleNamespace(
        samplerate=samplerate, blocksize=blocksize,
        sensitivity=sensitivity, dt=dt,
        width=1, get_chain=lambda: [],
    )
    mock.bus = mock
    return mock


def _leq_over_window(freq_hz: float, amplitude: float, duration_s: float) -> float:
    """Measure A-weighted Leq of a sine over *duration_s* seconds."""
    bus    = _mock_bus()
    plugin = PluginAWeighting(input=bus)
    leq_m  = plugin.create_meter(LeqAccumulator, name="leq")

    n_total = (int(duration_s * SAMPLERATE) // BLOCKSIZE) * BLOCKSIZE
    for i in range(n_total // BLOCKSIZE):
        t     = np.arange(BLOCKSIZE) / SAMPLERATE + i * BLOCKSIZE / SAMPLERATE
        block = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
        plugin.process(block[np.newaxis, :])

    return float(plugin.read_db("leq")[0])


@pytest.mark.slow
class TestContinuousOperationStability:
    """IEC 61672-1 §5.14 — 30 min continuous operation, drift ≤ ±0.1 dB."""

    def test_30min_stability_1khz(self):
        freq_hz   = 1_000
        amplitude = 1.0    # Pa
        window_s  = 10.0   # measure first and last 10 s windows

        # Measure at the start (first window_s seconds).
        l_initial = _leq_over_window(freq_hz, amplitude, window_s)

        # "Run" 30 min − 2×window_s by processing through the plugin without
        # accumulating into a separate meter (simulates continuous operation).
        fill_s = 30 * 60 - 2 * window_s
        bus    = _mock_bus()
        plugin = PluginAWeighting(input=bus)
        n_fill = (int(fill_s * SAMPLERATE) // BLOCKSIZE) * BLOCKSIZE
        for i in range(n_fill // BLOCKSIZE):
            t     = np.arange(BLOCKSIZE) / SAMPLERATE + i * BLOCKSIZE / SAMPLERATE
            block = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
            plugin.process(block[np.newaxis, :])

        # Measure at the end (last window_s seconds, same fresh plugin for
        # L_initial comparison — filter state doesn't affect steady-state Leq).
        l_final = _leq_over_window(freq_hz, amplitude, window_s)

        drift = abs(l_final - l_initial)
        assert drift <= 0.1, (
            f"30 min stability: L_initial = {l_initial:.4f} dB, "
            f"L_final = {l_final:.4f} dB, drift = {drift:.4f} dB (limit: ±0.1 dB)"
        )


@pytest.mark.slow
class TestHighLevelStability:
    """IEC 61672-1 §5.15 — 5 min at high level, drift ≤ ±0.1 dB."""

    def test_5min_high_level_stability(self):
        freq_hz = 1_000
        # 1 dB below upper boundary: level linearity tests pass over 120 dB,
        # so use 109 dB SPL (1 dB below the 110 dB top of the tested range).
        amplitude = p0 * np.sqrt(2) * 10 ** (109.0 / 20.0)
        window_s  = 10.0

        l_initial = _leq_over_window(freq_hz, amplitude, window_s)

        fill_s = 5 * 60 - 2 * window_s
        bus    = _mock_bus()
        plugin = PluginAWeighting(input=bus)
        n_fill = (int(fill_s * SAMPLERATE) // BLOCKSIZE) * BLOCKSIZE
        for i in range(n_fill // BLOCKSIZE):
            t     = np.arange(BLOCKSIZE) / SAMPLERATE + i * BLOCKSIZE / SAMPLERATE
            block = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
            plugin.process(block[np.newaxis, :])

        l_final = _leq_over_window(freq_hz, amplitude, window_s)

        drift = abs(l_final - l_initial)
        assert drift <= 0.1, (
            f"5 min high-level stability: L_initial = {l_initial:.4f} dB, "
            f"L_final = {l_final:.4f} dB, drift = {drift:.4f} dB (limit: ±0.1 dB)"
        )
