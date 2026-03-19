"""Unit tests for PluginImpulseTimeWeighting.

IEC 61672-1:2013 §5.8 covers only F and S time weightings — Impulse was
deliberately excluded from the 2013 revision.  PluginImpulseTimeWeighting
implements the legacy IEC 60651/IEC 60804 I-weighting:
    τ_rise = 35 ms   (attack)
    τ_fall = 1500 ms (decay)

These tests verify that the implemented time constants are correct:
  1. Fall rate: decay after signal cutoff ≈ 10/(τ_fall·ln 10) ≈ 2.90 dB/s
  2. Rise time constant: output reaches 63 % of steady state within τ_rise
  3. Asymmetry: rise is faster than fall (τ_rise << τ_fall)

Tolerance: ±15 % on each time constant (in the absence of a normative limit,
this matches the IEC 61260-1 Annex B guidance of 10 % for filter decay time
with some margin for the asymmetric filter approximation).
"""
from __future__ import annotations

import types

import numpy as np

from slm.time_weighting import PluginImpulseTimeWeighting

SAMPLERATE = 48_000
BLOCKSIZE  = 4_096
FREQ_HZ    = 4_000   # test frequency, same as IEC 61672-1 §5.8 decay-rate tests


# ---------------------------------------------------------------------------
# Helper — mock bus
# ---------------------------------------------------------------------------

def _mock_bus(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, sensitivity=1.0, dt=1.0):
    mock = types.SimpleNamespace(
        samplerate=samplerate, blocksize=blocksize,
        sensitivity=sensitivity, dt=dt,
        width=1, get_chain=lambda: [],
    )
    mock.bus = mock
    return mock


# ---------------------------------------------------------------------------
# Helper — process complete blocks
# ---------------------------------------------------------------------------

def _process_steady(plugin, freq_hz: float, n_blocks: int) -> None:
    """Feed *n_blocks* complete sine blocks through *plugin*."""
    for i in range(n_blocks):
        t     = np.arange(BLOCKSIZE) / SAMPLERATE + i * BLOCKSIZE / SAMPLERATE
        block = np.sin(2.0 * np.pi * freq_hz * t)
        plugin.process(block[np.newaxis, :])


def _process_zeros(plugin, n_blocks: int) -> np.ndarray:
    """Feed *n_blocks* of silence; return concatenated output."""
    zeros  = np.zeros((1, BLOCKSIZE))
    chunks = []
    for _ in range(n_blocks):
        plugin.process(zeros)
        chunks.append(plugin.output[0, :].copy())
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImpulseFallRate:
    """Fall time constant τ_fall = 1500 ms → decay rate ≈ 2.90 dB/s."""

    TAU_FALL     = 1.500          # s
    RATE_GOAL    = 10.0 / (TAU_FALL * np.log(10))   # ≈ 2.895 dB/s
    TOLERANCE    = 0.15           # ±15 %

    def test_fall_rate_4khz(self):
        bus    = _mock_bus()
        plugin = PluginImpulseTimeWeighting(input=bus)

        # Settle for 10 × τ_fall (all via complete blocks).
        n_settle = max(1, int(np.ceil(10 * self.TAU_FALL * SAMPLERATE / BLOCKSIZE)))
        _process_steady(plugin, FREQ_HZ, n_settle)

        # Collect 3 × τ_fall of decay.
        n_decay  = max(1, int(np.ceil(3 * self.TAU_FALL * SAMPLERATE / BLOCKSIZE)))
        decay    = _process_zeros(plugin, n_decay)

        # Fit slope over 2 × τ_fall.
        n_fit    = int(2 * self.TAU_FALL * SAMPLERATE)
        y        = np.maximum(decay[:n_fit], 1e-40)
        level_db = 10.0 * np.log10(y / y[0])
        t_fit    = np.arange(n_fit) / SAMPLERATE
        slope, _ = np.polyfit(t_fit, level_db, 1)
        rate     = abs(slope)   # dB/s

        lo = self.RATE_GOAL * (1 - self.TOLERANCE)
        hi = self.RATE_GOAL * (1 + self.TOLERANCE)
        assert lo <= rate <= hi, (
            f"Impulse fall rate = {rate:.3f} dB/s "
            f"(goal {self.RATE_GOAL:.3f} dB/s, "
            f"tolerance ±{self.TOLERANCE*100:.0f} %: [{lo:.3f}, {hi:.3f}])"
        )


class TestImpulseRiseTimeConstant:
    """Rise time constant τ_rise = 35 ms tested with a DC step input.

    A constant (DC) input ensures x²[n] > y[n-1] every sample, so α_rise is
    applied unconditionally and the output follows the ideal exponential:
        y(n) = A² · (1 − (1 − α_rise)ⁿ)
    The 63 % crossing therefore occurs at n = τ_rise · f_s, i.e. t = τ_rise.

    A sine input would interleave α_rise and α_fall (rise during peaks,
    fall during troughs), making the effective rise ~2× slower — that is
    intentional filter behaviour, not a deviation.
    """

    TAU_RISE  = 0.035   # s
    TOLERANCE = 0.15    # ±15 %

    def test_rise_tau_dc_step(self):
        A        = 1.0   # Pa — arbitrary amplitude (cancels in ratio)
        bus      = _mock_bus()
        plugin   = PluginImpulseTimeWeighting(input=bus)

        y_ss_dc   = A ** 2                          # DC squared steady state
        threshold = y_ss_dc * (1.0 - 1.0 / np.e)   # ≈ 0.632 × A²

        n_max    = int(5 * self.TAU_RISE * SAMPLERATE)
        dc_block = np.full((1, BLOCKSIZE), A)        # constant input

        rise_output = []
        for _ in range(int(np.ceil(n_max / BLOCKSIZE))):
            plugin.process(dc_block)
            rise_output.append(plugin.output[0, :].copy())
        rise_output = np.concatenate(rise_output)[:n_max]

        crossings = np.where(rise_output >= threshold)[0]
        assert len(crossings) > 0, (
            f"Output never reached 63 % of DC steady state in "
            f"{n_max / SAMPLERATE * 1000:.0f} ms"
        )
        t_63pct = crossings[0] / SAMPLERATE

        lo = self.TAU_RISE * (1 - self.TOLERANCE)
        hi = self.TAU_RISE * (1 + self.TOLERANCE)
        assert lo <= t_63pct <= hi, (
            f"Time to 63 % of DC steady state = {t_63pct * 1000:.2f} ms "
            f"(τ_rise = {self.TAU_RISE * 1000:.0f} ms, "
            f"tolerance ±{self.TOLERANCE * 100:.0f} %: "
            f"[{lo * 1000:.1f}, {hi * 1000:.1f}] ms)"
        )


class TestImpulseAsymmetry:
    """Rise must be faster than fall: τ_rise << τ_fall."""

    def test_rise_faster_than_fall(self):
        """Time to reach 63 % of steady state must be < time to fall to 37 %."""
        bus    = _mock_bus()
        plugin = PluginImpulseTimeWeighting(input=bus)

        # Settle fully.
        n_settle = max(1, int(np.ceil(10 * 1.500 * SAMPLERATE / BLOCKSIZE)))
        _process_steady(plugin, FREQ_HZ, n_settle)
        y_ss = float(plugin.output[0, -1])

        # Measure fall time (time for output to drop to 37 % of y_ss = 1 e-fold).
        n_fall_collect = int(2 * 1.500 * SAMPLERATE)
        fall_out = []
        zeros    = np.zeros((1, BLOCKSIZE))
        collected = 0
        while collected < n_fall_collect:
            plugin.process(zeros)
            chunk = plugin.output[0, :].copy()
            fall_out.append(chunk)
            collected += BLOCKSIZE
        fall_out = np.concatenate(fall_out)[:n_fall_collect]
        fall_crossings = np.where(fall_out <= y_ss / np.e)[0]
        assert len(fall_crossings) > 0, "Fall did not reach 37 % within 2 × τ_fall"
        t_fall_1e = fall_crossings[0] / SAMPLERATE

        # Measure rise time (from zero to 63 % of y_ss).
        bus2    = _mock_bus()
        plugin2 = PluginImpulseTimeWeighting(input=bus2)
        n_rise_collect = int(5 * 0.035 * SAMPLERATE)
        t_rise = np.arange(n_rise_collect) / SAMPLERATE
        signal = np.sin(2.0 * np.pi * FREQ_HZ * t_rise)
        rise_out = []
        for start in range(0, n_rise_collect, BLOCKSIZE):
            end   = min(start + BLOCKSIZE, n_rise_collect)
            block = signal[start:end]
            if len(block) < BLOCKSIZE:
                block = np.pad(block, (0, BLOCKSIZE - len(block)))
            plugin2.process(block[np.newaxis, :])
            rise_out.append(plugin2.output[0, : end - start].copy())
        rise_out = np.concatenate(rise_out)
        rise_crossings = np.where(rise_out >= y_ss * (1.0 - 1.0 / np.e))[0]
        assert len(rise_crossings) > 0, "Rise did not reach 63 % of steady state"
        t_rise_1e = rise_crossings[0] / SAMPLERATE

        assert t_rise_1e < t_fall_1e, (
            f"Rise to 63 % ({t_rise_1e*1000:.1f} ms) is not faster than "
            f"fall to 37 % ({t_fall_1e*1000:.1f} ms)"
        )
        # Quantitative: rise should be at least 10× faster (35 ms vs 1500 ms).
        ratio = t_fall_1e / t_rise_1e
        assert ratio >= 10.0, (
            f"Fall/rise ratio = {ratio:.1f}× (expected ≥ 10×, "
            f"from τ_fall/τ_rise = {1500/35:.0f}×)"
        )
