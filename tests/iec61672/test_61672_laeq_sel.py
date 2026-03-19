"""IEC 61672-1:2013 §3.9, §3.12, §5.10 — L_Aeq formula, SEL formula, repeated tonebursts.

§3.9  L_Aeq,T = 10·log₁₀((1/T)·∫p_A²(t)dt / p₀²)
§3.12 L_AE,T  = L_Aeq,T + 10·log₁₀(T/T₀)   T₀ = 1 s
§5.10 For n equal-amplitude 4 kHz tonebursts of duration T_b in window T_m:
        δ_ref = 10·log₁₀(n·T_b / T_m)
      Deviations must meet Table 4 SEL acceptance limits for the effective
      on-time n·T_b (class 1).

Signal chain used here:  1 kHz / 4 kHz sine → PluginAWeighting → LeqAccumulator
                                                                 → LEAccumulator

All signals are trimmed to an exact multiple of blocksize before processing so
that no zero-padding enters the accumulator sample count.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginAWeighting
from slm.meter import LeqAccumulator, LEAccumulator
from slm.constants import REFERENCE_PRESSURE

p0        = REFERENCE_PRESSURE   # 20 µPa
T0        = 1.0                  # SEL reference duration (s)
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


def _make_sine(freq_hz: float, amplitude: float, n_samples: int) -> np.ndarray:
    """Return a pure sine of exactly *n_samples* samples (no trimming here)."""
    t = np.arange(n_samples) / SAMPLERATE
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _trim(signal: np.ndarray) -> np.ndarray:
    """Trim *signal* to the largest multiple of BLOCKSIZE that fits."""
    n = (len(signal) // BLOCKSIZE) * BLOCKSIZE
    return signal[:n]


def _measure(signal: np.ndarray) -> tuple[float, float, int]:
    """Feed *signal* (pre-trimmed to multiple of BLOCKSIZE) through A-weighting.

    Returns (L_Aeq_dB, L_AE_dB, n_samples_processed).
    """
    assert len(signal) % BLOCKSIZE == 0, "signal must be a multiple of BLOCKSIZE"
    bus    = _mock_bus()
    plugin = PluginAWeighting(input=bus)
    leq_m  = plugin.create_meter(LeqAccumulator, name="leq")
    le_m   = plugin.create_meter(LEAccumulator,  name="le")

    for start in range(0, len(signal), BLOCKSIZE):
        plugin.process(signal[start : start + BLOCKSIZE][np.newaxis, :])

    l_aeq = float(plugin.read_db("leq")[0])
    l_ae  = float(plugin.read_db("le")[0])
    return l_aeq, l_ae, len(signal)


# ---------------------------------------------------------------------------
# §3.9 — L_Aeq formula
# ---------------------------------------------------------------------------

class TestLeqFormula:
    """IEC 61672-1 §3.9 — L_Aeq,T = 10·log₁₀((1/T)·∫p_A²dt / p₀²)."""

    def test_laeq_1khz_unit_amplitude(self):
        """A-weighted Leq of 1 kHz unit-amplitude sine matches analytic value."""
        A = 1.0  # Pa
        # Analytic: L_Aeq = 10·log₁₀(A²/2 / p₀²); A-weighting at 1 kHz = 0 dB.
        l_expected = 10.0 * np.log10(A ** 2 / 2.0 / p0 ** 2)

        n_samples = _trim(_make_sine(1000, A, int(5.0 * SAMPLERATE)))
        signal    = _trim(_make_sine(1000, A, int(5.0 * SAMPLERATE)))
        l_aeq, _, _ = _measure(signal)

        assert abs(l_aeq - l_expected) <= 0.05, (
            f"L_Aeq = {l_aeq:.4f} dB, expected {l_expected:.4f} dB, "
            f"deviation = {l_aeq - l_expected:+.4f} dB (limit: ±0.05 dB)"
        )

    def test_laeq_consistent_across_durations(self):
        """L_Aeq of a stationary 1 kHz sine is independent of integration time."""
        A = 1.0
        durations_s = [1.0, 3.0, 5.0, 10.0]
        leq_values = []
        for dur in durations_s:
            signal = _trim(_make_sine(1000, A, int(dur * SAMPLERATE)))
            l_aeq, _, _ = _measure(signal)
            leq_values.append(l_aeq)

        spread = max(leq_values) - min(leq_values)
        assert spread <= 0.1, (
            f"L_Aeq spread over {durations_s} s: {spread:.4f} dB "
            f"(limit: ±0.05 dB each → spread ≤ 0.1 dB)\n"
            f"Values: {[f'{v:.4f}' for v in leq_values]}"
        )


# ---------------------------------------------------------------------------
# §3.12 — SEL formula
# ---------------------------------------------------------------------------

class TestSELFormula:
    """IEC 61672-1 §3.12 — L_AE,T = L_Aeq,T + 10·log₁₀(T/T₀)."""

    def test_sel_equals_leq_plus_duration_term(self):
        """L_AE − L_Aeq = 10·log₁₀(T/T₀) for a stationary 1 kHz sine."""
        A       = 1.0
        dur_s   = 5.0
        signal  = _trim(_make_sine(1000, A, int(dur_s * SAMPLERATE)))
        T_actual = len(signal) / SAMPLERATE  # trimmed duration

        l_aeq, l_ae, _ = _measure(signal)

        expected_diff = 10.0 * np.log10(T_actual / T0)
        measured_diff = l_ae - l_aeq
        assert abs(measured_diff - expected_diff) <= 0.05, (
            f"L_AE − L_Aeq = {measured_diff:.4f} dB, "
            f"expected 10·log₁₀({T_actual:.3f}) = {expected_diff:.4f} dB, "
            f"deviation = {measured_diff - expected_diff:+.4f} dB (limit: ±0.05 dB)"
        )

    def test_sel_reference_exposure(self):
        """E₀ = p₀²·T₀ = (20 µPa)²·1 s = 400×10⁻¹² Pa²·s (§3.12 note).

        A 1 kHz sine at 0 dB SPL (RMS = p₀) over T₀ = 1 s should give L_AE = 0 dB.
        """
        A_rms   = p0              # 0 dB SPL RMS
        A_peak  = A_rms * np.sqrt(2)
        signal  = _trim(_make_sine(1000, A_peak, SAMPLERATE))  # ≈ 1 s
        T_actual = len(signal) / SAMPLERATE

        l_aeq, l_ae, _ = _measure(signal)

        # L_Aeq ≈ 0 dB SPL; L_AE ≈ 0 + 10·log₁₀(T_actual) ≈ 10·log₁₀(T_actual) dB
        expected_l_ae = 10.0 * np.log10(T_actual / T0)  # ≈ 0 dB when T≈T₀
        assert abs(l_ae - expected_l_ae) <= 0.05, (
            f"L_AE = {l_ae:.4f} dB, expected {expected_l_ae:.4f} dB "
            f"(deviation {l_ae - expected_l_ae:+.4f} dB, limit: ±0.05 dB)"
        )


# ---------------------------------------------------------------------------
# §5.10 — Repeated tonebursts
# ---------------------------------------------------------------------------

# Table 4 SEL acceptance limits for effective on-time n·T_b:
# (effective_ms, delta_SEL_ref, cl1_lo, cl1_hi)
_TABLE4_SEL = {
    1000: ( 0.0, -0.5, +0.5),
     500: (-3.0, -0.5, +0.5),
     200: (-7.0, -0.5, +0.5),
     100: (-10.0, -1.0, +1.0),
      50: (-13.0, -1.0, +1.0),
      20: (-17.0, -1.0, +1.0),
      10: (-20.0, -1.0, +1.0),
}

# (n_bursts, T_b_ms, T_m_s, effective_ms)
_REPEATED = [
    (5,  100, 1.0,  500),
    (5,   40, 1.0,  200),
    (10,  10, 1.0,  100),
    (10,   5, 1.0,   50),
    (20,   1, 1.0,   20),
    (10,   1, 1.0,   10),
]
_REPEATED_IDS = [f"n={r[0]}_Tb={r[1]}ms" for r in _REPEATED]


def _repeated_burst_leq(n_bursts: int, T_b_ms: float, T_m_s: float) -> float:
    """
    Generate n equal-amplitude 4 kHz tonebursts of T_b_ms each inside a T_m_s
    window, measure A-weighted L_Aeq,T_m, and return it.
    """
    samplerate = SAMPLERATE
    freq_hz    = 4000
    amplitude  = 1.0  # Pa

    T_b_s  = T_b_ms / 1000.0
    n_b    = int(round(T_b_s * samplerate))       # samples per burst
    n_m    = (int(T_m_s * samplerate) // BLOCKSIZE) * BLOCKSIZE  # trimmed window

    # Build signal: n bursts spread evenly across T_m, padded with silence.
    full = np.zeros(n_m)
    stride = n_m // n_bursts  # samples between burst starts
    for k in range(n_bursts):
        start = k * stride
        end   = min(start + n_b, n_m)
        t_b   = np.arange(end - start) / samplerate
        full[start:end] = amplitude * np.sin(2.0 * np.pi * freq_hz * t_b)

    l_aeq, _, _ = _measure(full)
    return l_aeq


class TestRepeatedTonebursts:
    """IEC 61672-1 §5.10 — repeated toneburst formula, class 1.

    δ_ref = 10·log₁₀(n·T_b / T_m); limits per Table 4 SEL column for n·T_b.
    """

    @pytest.mark.parametrize("row", _REPEATED, ids=_REPEATED_IDS)
    def test_repeated_toneburst(self, row):
        n_bursts, T_b_ms, T_m_s, eff_ms = row
        ref_sel, cl1_lo, cl1_hi = _TABLE4_SEL[eff_ms]

        # Steady-state A-weighted level at 1 kHz for unit sine ≈ A-wt at 4 kHz.
        # Use analytic: L_A = 10·log₁₀(A²/2 / p₀²) + A_weighting_at_4kHz_dB.
        # Since frequency weighting cancels in the difference, we measure L_A
        # directly by running a full-window burst.
        signal_ss = _trim(_make_sine(4000, 1.0, int(T_m_s * SAMPLERATE)))
        l_a, _, _ = _measure(signal_ss)

        l_aeq = _repeated_burst_leq(n_bursts, T_b_ms, T_m_s)
        delta  = l_aeq - l_a
        dev    = delta - ref_sel

        assert cl1_lo <= dev <= cl1_hi, (
            f"n={n_bursts}, T_b={T_b_ms} ms, T_m={T_m_s} s: "
            f"δ = {delta:.3f} dB, ref = {ref_sel:.1f} dB, "
            f"dev = {dev:+.3f} dB (class 1: [{cl1_lo:+.1f}, {cl1_hi:+.1f}])"
        )
