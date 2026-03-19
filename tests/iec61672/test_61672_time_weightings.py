"""IEC 61672-1:2013 §5.8 — time weighting conformance (F and S).

Tests PluginFastTimeWeighting and PluginSlowTimeWeighting against:
  - §5.8.2: exponential decay rate after cessation of a steady 4 kHz signal
      F: design goal 34.7 dB/s, class 1 limits [31.0, 38.5] dB/s
      S: design goal 4.3 dB/s,  class 1 limits [3.6,  5.1] dB/s
  - §5.8.3: |L_S − L_F| ≤ 0.1 dB for a steady 1 kHz signal

Approach: process complete blocks of a pure sine until steady state (10 τ),
then feed silence and fit the exponential decay slope in dB/s.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.time_weighting import PluginFastTimeWeighting, PluginSlowTimeWeighting


# ---------------------------------------------------------------------------
# Shared mock bus
# ---------------------------------------------------------------------------

def _mock_bus(samplerate=48000, blocksize=4096, sensitivity=1.0, dt=1.0):
    mock = types.SimpleNamespace(
        samplerate=samplerate, blocksize=blocksize,
        sensitivity=sensitivity, dt=dt,
        width=1, get_chain=lambda: [],
    )
    mock.bus = mock
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_steady(plugin, freq_hz: float, n_blocks: int,
                    samplerate: int, blocksize: int) -> None:
    """Feed *n_blocks* complete blocks of a sine wave through *plugin*."""
    for i in range(n_blocks):
        t = np.arange(blocksize) / samplerate + i * blocksize / samplerate
        block = np.sin(2.0 * np.pi * freq_hz * t)
        plugin.process(block[np.newaxis, :])


def _measure_decay_rate_dbs(
    plugin_cls,
    freq_hz: float = 4000,
    samplerate: int = 48000,
    blocksize: int = 4096,
) -> float:
    """Return the exponential decay rate (dB/s) after cessation of a steady sine.

    Procedure (IEC 61672-1:2013 §5.8):
      1. Settle with ≥ 10 τ complete blocks of sine.
      2. Feed silence; collect ≥ 3 τ of output.
      3. Fit a linear slope of 10·log₁₀(output/output[0]) vs time (s).
    """
    bus = _mock_bus(samplerate=samplerate, blocksize=blocksize)
    plugin = plugin_cls(input=bus)
    tau = plugin.tau  # valid for PluginSymmetricTimeWeighting

    # Round up to whole blocks so the settle phase ends cleanly (no zero-padding).
    n_blocks_settle = max(1, int(np.ceil(10 * tau * samplerate / blocksize)))
    n_blocks_decay  = max(1, int(np.ceil(3  * tau * samplerate / blocksize)))

    _process_steady(plugin, freq_hz, n_blocks_settle, samplerate, blocksize)

    # Collect decay output (feed complete zero blocks).
    zeros = np.zeros((1, blocksize))
    decay_chunks = []
    for _ in range(n_blocks_decay):
        plugin.process(zeros)
        decay_chunks.append(plugin.output[0, :].copy())
    decay = np.concatenate(decay_chunks)

    # Fit over 2 τ samples (well above numerical noise floor).
    n_fit = int(2 * tau * samplerate)
    y = np.maximum(decay[:n_fit], 1e-40)
    level_db = 10.0 * np.log10(y / y[0])
    t_fit = np.arange(n_fit) / samplerate
    slope, _ = np.polyfit(t_fit, level_db, 1)
    return abs(slope)  # magnitude in dB/s (positive)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFastTimeWeightingDecayRate:
    """IEC 61672-1 §5.8.2 — F time weighting decay rate, class 1."""

    # Design goal 34.7 dB/s; class 1 limits: +3.8 / −3.7 dB/s → [31.0, 38.5]
    GOAL_DBS = 34.7
    CL1_LO   = 31.0
    CL1_HI   = 38.5

    def test_decay_rate_4khz(self, report: bool = False):
        rate = _measure_decay_rate_dbs(PluginFastTimeWeighting, freq_hz=4000)
        margin = min(rate - self.CL1_LO, self.CL1_HI - rate)
        if report:
            return {"label": "F @ 4 kHz", "rate": rate,
                    "limit_lo": self.CL1_LO, "limit_hi": self.CL1_HI, "margin": margin}
        assert self.CL1_LO <= rate <= self.CL1_HI, (
            f"F decay rate = {rate:.2f} dB/s "
            f"(goal {self.GOAL_DBS}, class 1: [{self.CL1_LO}, {self.CL1_HI}] dB/s)"
        )


class TestSlowTimeWeightingDecayRate:
    """IEC 61672-1 §5.8.2 — S time weighting decay rate, class 1."""

    # Design goal 4.3 dB/s; class 1 limits: +0.8 / −0.7 dB/s → [3.6, 5.1]
    GOAL_DBS = 4.3
    CL1_LO   = 3.6
    CL1_HI   = 5.1

    def test_decay_rate_4khz(self, report: bool = False):
        rate = _measure_decay_rate_dbs(PluginSlowTimeWeighting, freq_hz=4000)
        margin = min(rate - self.CL1_LO, self.CL1_HI - rate)
        if report:
            return {"label": "S @ 4 kHz", "rate": rate,
                    "limit_lo": self.CL1_LO, "limit_hi": self.CL1_HI, "margin": margin}
        assert self.CL1_LO <= rate <= self.CL1_HI, (
            f"S decay rate = {rate:.2f} dB/s "
            f"(goal {self.GOAL_DBS}, class 1: [{self.CL1_LO}, {self.CL1_HI}] dB/s)"
        )


class TestFvsSteadyState:
    """IEC 61672-1 §5.8.3 — F and S outputs agree ≤ 0.1 dB for steady 1 kHz."""

    def test_steady_1khz(self):
        samplerate = 48000
        blocksize  = 4096
        freq_hz    = 1000

        # Settle for 10 × τ_S = 10 s, which also covers 80 × τ_F.
        n_blocks = int(np.ceil(10.0 * samplerate / blocksize))

        bus      = _mock_bus(samplerate=samplerate, blocksize=blocksize)
        plugin_f = PluginFastTimeWeighting(input=bus)
        plugin_s = PluginSlowTimeWeighting(input=bus)

        _process_steady(plugin_f, freq_hz, n_blocks, samplerate, blocksize)
        _process_steady(plugin_s, freq_hz, n_blocks, samplerate, blocksize)

        # Average the last block to suppress any residual high-frequency ripple.
        y_f = float(np.mean(plugin_f.output[0, :]))
        y_s = float(np.mean(plugin_s.output[0, :]))

        diff_db = abs(10.0 * np.log10(y_f / y_s))
        assert diff_db <= 0.1, (
            f"|L_F − L_S| = {diff_db:.4f} dB at 1 kHz (class 1 limit: 0.1 dB)"
        )
