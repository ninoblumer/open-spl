"""IEC 61672-1:2013 §5.9 Table 4 — toneburst response (class 1).

Signal: 4 kHz pure-tone burst of duration T_b fed through PluginAWeighting
into PluginFastTimeWeighting and PluginSlowTimeWeighting.  Every result is
expressed relative to the A-weighted steady-state level of the same tone, so
the A-weighting gain cancels exactly in the differences (the weighting is kept
in the chain for consistency with the other §5.x tests, not because it changes
the reported deltas).

Reference formulas (§5.9):
  δ_ref(F-max) = 10 · log₁₀(1 − exp(−T_b / τ))   τ = 0.125 s  (Formula 7)
  δ_ref(SEL)   = 10 · log₁₀(T_b / T₀)             T₀ = 1 s     (Formula 8)
  δ_ref(S-max) = 10 · log₁₀(1 − exp(−T_b / τ))   τ = 1 s      (Formula 7)

All are expressed relative to the steady-state level L_A (long-term RMS of
the same sine signal), so frequency-weighting gain cancels exactly.  Table 4
gives the three reference responses (F-max and SEL over 1000…0.25 ms; S-max
over 1000…2 ms) each with their own class-1 acceptance limits.

Notes on numerical behaviour at 48 kHz / 4 kHz:
  - One period = 12 samples; all Table 4 burst lengths are whole numbers of
    periods, so the unweighted burst energy would be exactly n_burst/2.
  - The discrete-time EMA converges analytically: y(n) = y_ss·(1−(1−α)^n)
    which equals y_ss·(1−exp(−T_b/τ)) to floating-point precision.
  - Deviations from the Table 4 rounded references are a few mdB for the
    longer bursts, rising to ~0.15 dB for the shortest (sub-ms) bursts, whose
    wide spectra sample the A-weighting across a band rather than at 4 kHz —
    all well within the (widening) class 1 limits.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginAWeighting
from slm.time_weighting import PluginFastTimeWeighting, PluginSlowTimeWeighting


TAU_F   = 0.125  # s — F time constant (IEC 61672-1 §5.8)
T0      = 1.0    # s — SEL reference duration (IEC 61672-1 §3.12)
FREQ_HZ = 4000   # Hz — standard toneburst frequency (§5.9)


# ---------------------------------------------------------------------------
# Mock bus
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
# Measurement helper
# ---------------------------------------------------------------------------

def _a_weighted_steady_ms(amplitude: float, samplerate: int, blocksize: int) -> float:
    """Steady-state time-weighted output (Pa²) of a *continuous* A-weighted 4 kHz
    sine — the reference level L_A the toneburst deltas are measured against.

    Measured through the same A-weighting as the burst, so the weighting gain
    (|H_A(4 kHz)|²) cancels exactly in every delta.  The one-pole detector
    converges to the true mean square, so this equals |H_A(4 kHz)|²·amplitude²/2
    without any period-averaging error.
    """
    bus = _mock_bus(samplerate=samplerate, blocksize=blocksize)
    w   = PluginAWeighting(input=bus)
    f   = PluginFastTimeWeighting(input=w)          # subscribes to w
    n_settle = max(1, int(np.ceil(2.0 * samplerate / blocksize)))   # 2 s >> τ_F
    for i in range(n_settle):
        t = (np.arange(blocksize) + i * blocksize) / samplerate
        w.process((amplitude * np.sin(2.0 * np.pi * FREQ_HZ * t))[np.newaxis, :])
    return float(f.output[0, -1])


def _toneburst_response(
    burst_s: float,
    amplitude: float = 1.0,
    samplerate: int = 48000,
    blocksize: int = 4096,
) -> tuple[float, float, float]:
    """Apply a 4 kHz burst then silence; return (delta_Fmax, delta_SEL, delta_Smax) in dB.

    delta_Fmax_dB = max F-time-weighted output (dB) − steady-state level (dB)
    delta_SEL_dB  = SEL (dB) − steady-state level (dB)
    delta_Smax_dB = max S-time-weighted output (dB) − steady-state level (dB)

    The signal is A-weighted before the F/S time weighting.  Because every delta
    is taken relative to the A-weighted steady-state level y_ss (and the SEL uses
    the A-weighted event energy), the A-weighting gain cancels: results match the
    unweighted computation to a few mdB, growing to ~0.15 dB only on the shortest
    (sub-ms) bursts, whose wide spectra sample the A-weighting across a band
    rather than at 4 kHz alone.  All deviations stay well inside the class 1
    limits, which widen to ±3 dB there.
    """
    bus       = _mock_bus(samplerate=samplerate, blocksize=blocksize)
    weighting = PluginAWeighting(input=bus)
    plugin_f  = PluginFastTimeWeighting(input=weighting)   # subscribe to weighting
    plugin_s  = PluginSlowTimeWeighting(input=weighting)

    # Steady-state A-weighted level of the same sine (see helper).
    y_ss = _a_weighted_steady_ms(amplitude, samplerate, blocksize)

    n_burst = int(round(burst_s * samplerate))
    # Two seconds of silence after the burst to capture the full decay peak.
    n_after = 2 * samplerate
    n_total = n_burst + n_after

    t_burst      = np.arange(n_burst) / samplerate
    burst_signal = amplitude * np.sin(2.0 * np.pi * FREQ_HZ * t_burst)
    full_signal  = np.concatenate([burst_signal, np.zeros(n_after)])

    # Process in blocksize chunks; weighting.process drives the F/S subscribers.
    # Collect the A-weighted signal (for SEL) and both time-weighted outputs.
    aw_chunks, f_chunks, s_chunks = [], [], []
    for start in range(0, n_total, blocksize):
        end   = min(start + blocksize, n_total)
        block = full_signal[start:end]
        if len(block) < blocksize:
            block = np.pad(block, (0, blocksize - len(block)))
        weighting.process(block[np.newaxis, :])
        aw_chunks.append(weighting.output[0, : end - start].copy())
        f_chunks.append(plugin_f.output[0, : end - start].copy())
        s_chunks.append(plugin_s.output[0, : end - start].copy())
    aw_signal = np.concatenate(aw_chunks)
    f_output  = np.concatenate(f_chunks)
    s_output  = np.concatenate(s_chunks)

    # F-max / S-max: maximum of the time-weighted output over the whole window.
    delta_fmax = 10.0 * np.log10(float(np.max(f_output)) / y_ss)
    delta_smax = 10.0 * np.log10(float(np.max(s_output)) / y_ss)

    # SEL: total A-weighted sound exposure = integral of the A-weighted signal
    # squared over the whole event (burst + filter ring-down), not just the burst
    # window — the A-weighting's group delay/ring-out carries part of the burst
    # energy past n_burst.  Same weighting as y_ss, so the gain cancels.
    e_burst    = float(np.sum(aw_signal ** 2)) / samplerate  # Pa² · s
    delta_sel  = 10.0 * np.log10(e_burst / (y_ss * T0))

    return delta_fmax, delta_sel, delta_smax


# ---------------------------------------------------------------------------
# IEC 61672-1:2013 Table 4 — reference deltas and class 1 limits
# (burst_ms, delta_Fmax_ref, delta_SEL_ref, cl1_lo, cl1_hi)
# The class 1 limits apply to the deviation (measured − reference) for both
# the F-max and SEL measurements.
# ---------------------------------------------------------------------------

_TABLE4 = [
    #  ms    Fmax    SEL   lo     hi
    (1000,   0.0,   0.0, -0.5, +0.5),
    ( 500,  -0.1,  -3.0, -0.5, +0.5),
    ( 200,  -1.0,  -7.0, -0.5, +0.5),
    ( 100,  -2.6, -10.0, -1.0, +1.0),
    (  50,  -4.8, -13.0, -1.0, +1.0),
    (  20,  -8.3, -17.0, -1.0, +1.0),
    (  10, -11.1, -20.0, -1.0, +1.0),
    (   5, -14.1, -23.0, -1.0, +1.0),
    (   2, -18.0, -27.0, -1.5, +1.0),
    (   1, -21.0, -30.0, -2.0, +1.0),
    ( 0.5, -24.0, -33.0, -2.5, +1.0),
    (0.25, -27.0, -36.0, -3.0, +1.0),
]
_TABLE4_IDS = [f"{row[0]}ms" for row in _TABLE4]


# ---------------------------------------------------------------------------
# IEC 61672-1:2013 Table 4 — S-time-weighted maximum sub-block.
# Reference δ from Eq. (7) with τ = 1 s; durations 1000 → 2 ms only.
# (burst_ms, delta_Smax_ref, cl1_lo, cl1_hi) — the S-max block has its own
# acceptance limits, distinct from the shared F-max / SEL limits above.
# ---------------------------------------------------------------------------

_TABLE4_SMAX = [
    #  ms    Smax    lo     hi
    (1000,  -2.0, -0.5, +0.5),
    ( 500,  -4.1, -0.5, +0.5),
    ( 200,  -7.4, -0.5, +0.5),
    ( 100, -10.2, -1.0, +1.0),
    (  50, -13.1, -1.0, +1.0),
    (  20, -17.0, -1.5, +1.0),
    (  10, -20.0, -2.0, +1.0),
    (   5, -23.0, -2.5, +1.0),
    (   2, -27.0, -3.0, +1.0),
]
_TABLE4_SMAX_IDS = [f"{row[0]}ms" for row in _TABLE4_SMAX]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFmaxToneburst:
    """IEC 61672-1 §5.9 — F-time-weighted maximum toneburst response, class 1."""

    @pytest.mark.parametrize("row", _TABLE4, ids=_TABLE4_IDS)
    def test_fmax_vs_table4(self, row, report: bool = False):
        burst_ms, ref_fmax, _ref_sel, cl1_lo, cl1_hi = row
        delta_fmax, _, _ = _toneburst_response(burst_ms / 1000.0)
        dev = delta_fmax - ref_fmax
        margin = min(dev - cl1_lo, cl1_hi - dev)
        if report:
            return {"label": f"{burst_ms} ms", "deviation": dev,
                    "limit_lo": cl1_lo, "limit_hi": cl1_hi, "margin": margin}
        assert cl1_lo <= dev <= cl1_hi, (
            f"F-max @ {burst_ms} ms: δ = {delta_fmax:.3f} dB, "
            f"ref = {ref_fmax:.1f} dB, dev = {dev:+.3f} dB "
            f"(class 1: [{cl1_lo:+.1f}, {cl1_hi:+.1f}])"
        )


class TestSELToneburst:
    """IEC 61672-1 §5.9 — sound exposure level toneburst response, class 1."""

    @pytest.mark.parametrize("row", _TABLE4, ids=_TABLE4_IDS)
    def test_sel_vs_table4(self, row, report: bool = False):
        burst_ms, _ref_fmax, ref_sel, cl1_lo, cl1_hi = row
        _, delta_sel, _ = _toneburst_response(burst_ms / 1000.0)
        dev = delta_sel - ref_sel
        margin = min(dev - cl1_lo, cl1_hi - dev)
        if report:
            return {"label": f"{burst_ms} ms", "deviation": dev,
                    "limit_lo": cl1_lo, "limit_hi": cl1_hi, "margin": margin}
        assert cl1_lo <= dev <= cl1_hi, (
            f"SEL @ {burst_ms} ms: δ = {delta_sel:.3f} dB, "
            f"ref = {ref_sel:.1f} dB, dev = {dev:+.3f} dB "
            f"(class 1: [{cl1_lo:+.1f}, {cl1_hi:+.1f}])"
        )


class TestSmaxToneburst:
    """IEC 61672-1 §5.9 — S-time-weighted maximum toneburst response, class 1."""

    @pytest.mark.parametrize("row", _TABLE4_SMAX, ids=_TABLE4_SMAX_IDS)
    def test_smax_vs_table4(self, row, report: bool = False):
        burst_ms, ref_smax, cl1_lo, cl1_hi = row
        _, _, delta_smax = _toneburst_response(burst_ms / 1000.0)
        dev = delta_smax - ref_smax
        margin = min(dev - cl1_lo, cl1_hi - dev)
        if report:
            return {"label": f"{burst_ms} ms", "deviation": dev,
                    "limit_lo": cl1_lo, "limit_hi": cl1_hi, "margin": margin}
        assert cl1_lo <= dev <= cl1_hi, (
            f"S-max @ {burst_ms} ms: δ = {delta_smax:.3f} dB, "
            f"ref = {ref_smax:.1f} dB, dev = {dev:+.3f} dB "
            f"(class 1: [{cl1_lo:+.1f}, {cl1_hi:+.1f}])"
        )
