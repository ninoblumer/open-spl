"""IEC 61672-1:2013 §3.9, §3.12, §5.10 — L_Aeq formula, SEL formula, repeated tonebursts.

§3.9  L_Aeq,T = 10·log₁₀((1/T)·∫p_A²(t)dt / p₀²)
§3.12 L_AE,T  = L_Aeq,T + 10·log₁₀(T/T₀)   T₀ = 1 s
§5.10 For n equal-amplitude 4 kHz tonebursts of duration T_b in window T_m:
        δ_ref = 10·log₁₀(n·T_b / T_m)                                  Eq. (9)
      The reference response is computed from Eq. (9); the deviation must meet
      the Table 4 SEL acceptance limit for the *individual* toneburst duration
      T_b (§5.10.1, §5.10.2), not for the effective on-time n·T_b (class 1).

Signal chain used here:  1 kHz / 4 kHz sine → PluginAWeighting → LeqAccumulator
                                                                 → LEAccumulator

The §3.9/§3.12 signals are trimmed to a multiple of blocksize before processing;
the §5.10 measurements use a block size that divides the exact 1 s window (see
_measure_leq_window) so the averaging time equals T_m exactly.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from slm.frequency_weighting import PluginAWeighting
from slm.time_weighting import PluginSquare
from slm.meter import LeqAccumulator, LEAccumulator
from slm.constants import REFERENCE_PRESSURE

p0        = REFERENCE_PRESSURE   # 20 µPa
T0        = 1.0                  # SEL reference duration (s)
SAMPLERATE = 48_000
BLOCKSIZE  = 4_096
# Divisor of the 1 s @ 48 kHz repeated-toneburst window (48000 = 10 × 4800), so
# the §5.10 averaging window is exactly T_m = 1 s with no block-size rounding.
WINDOW_BLOCKSIZE = 4_800


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
    sq     = PluginSquare(input=plugin)      # Leq/LE meters consume Pa²
    leq_m  = sq.create_meter(LeqAccumulator, name="leq")
    le_m   = sq.create_meter(LEAccumulator,  name="le")

    for start in range(0, len(signal), BLOCKSIZE):
        plugin.process(signal[start : start + BLOCKSIZE][np.newaxis, :])

    l_aeq = float(sq.read_db("leq")[0])
    l_ae  = float(sq.read_db("le")[0])
    return l_aeq, l_ae, len(signal)


def _measure_leq_window(signal: np.ndarray) -> float:
    """A-weighted L_Aeq of *signal*, integrated over its exact length.

    Processes the signal in WINDOW_BLOCKSIZE-sample blocks (a divisor of the
    1 s @ 48 kHz window), so the averaging window equals len(signal)/SAMPLERATE
    exactly with no rounding to a multiple of BLOCKSIZE.
    """
    assert len(signal) % WINDOW_BLOCKSIZE == 0, (
        "signal length must be a multiple of WINDOW_BLOCKSIZE"
    )
    bus    = _mock_bus(blocksize=WINDOW_BLOCKSIZE)
    plugin = PluginAWeighting(input=bus)
    sq     = PluginSquare(input=plugin)      # Leq meter consumes Pa²
    sq.create_meter(LeqAccumulator, name="leq")
    for start in range(0, len(signal), WINDOW_BLOCKSIZE):
        plugin.process(signal[start : start + WINDOW_BLOCKSIZE][np.newaxis, :])
    return float(sq.read_db("leq")[0])


# ---------------------------------------------------------------------------
# §3.9 — L_Aeq formula
# ---------------------------------------------------------------------------

class TestLeqFormula:
    """IEC 61672-1 §3.9 — L_Aeq,T = 10·log₁₀((1/T)·∫p_A²dt / p₀²).

    NOTE: §3.9 / §3.12 are *definitions*, not performance specifications — the
    standard sets no acceptance limit on how exactly an implementation must
    reproduce these formulae.  The ±0.05 dB / 0.1 dB bounds below are
    SELF-IMPOSED implementation-accuracy checks, so they are not shown in the
    conformance report.
    """

    def test_laeq_1khz_unit_amplitude(self, report: bool = False):
        """A-weighted Leq of 1 kHz unit-amplitude sine matches analytic value."""
        A = 1.0  # Pa
        # Analytic: L_Aeq = 10·log₁₀(A²/2 / p₀²); A-weighting at 1 kHz = 0 dB.
        l_expected = 10.0 * np.log10(A ** 2 / 2.0 / p0 ** 2)

        signal    = _trim(_make_sine(1000, A, int(5.0 * SAMPLERATE)))
        l_aeq, _, _ = _measure(signal)

        dev = l_aeq - l_expected
        if report:
            return {"label": "L_Aeq 1 kHz unit", "deviation": dev,
                    "limit_lo": -0.05, "limit_hi": 0.05,
                    "margin": min(dev + 0.05, 0.05 - dev)}
        assert abs(l_aeq - l_expected) <= 0.05, (
            f"L_Aeq = {l_aeq:.4f} dB, expected {l_expected:.4f} dB, "
            f"deviation = {l_aeq - l_expected:+.4f} dB (limit: ±0.05 dB)"
        )

    def test_laeq_consistent_across_durations(self, report: bool = False):
        """L_Aeq of a stationary 1 kHz sine is independent of integration time."""
        A = 1.0
        durations_s = [1.0, 3.0, 5.0, 10.0]
        leq_values = []
        for dur in durations_s:
            signal = _trim(_make_sine(1000, A, int(dur * SAMPLERATE)))
            l_aeq, _, _ = _measure(signal)
            leq_values.append(l_aeq)

        spread = max(leq_values) - min(leq_values)
        if report:
            return {"label": "L_Aeq duration spread", "value": spread,
                    "limit": 0.1, "margin": 0.1 - spread}
        assert spread <= 0.1, (
            f"L_Aeq spread over {durations_s} s: {spread:.4f} dB "
            f"(limit: ±0.05 dB each → spread ≤ 0.1 dB)\n"
            f"Values: {[f'{v:.4f}' for v in leq_values]}"
        )


# ---------------------------------------------------------------------------
# §3.12 — SEL formula
# ---------------------------------------------------------------------------

class TestSELFormula:
    """IEC 61672-1 §3.12 — L_AE,T = L_Aeq,T + 10·log₁₀(T/T₀).

    NOTE: §3.12 is a *definition*; the ±0.05 dB bounds below are SELF-IMPOSED
    implementation-accuracy checks, not IEC acceptance limits, and are not shown
    in the conformance report.
    """

    def test_sel_equals_leq_plus_duration_term(self, report: bool = False):
        """L_AE − L_Aeq = 10·log₁₀(T/T₀) for a stationary 1 kHz sine."""
        A       = 1.0
        dur_s   = 5.0
        signal  = _trim(_make_sine(1000, A, int(dur_s * SAMPLERATE)))
        T_actual = len(signal) / SAMPLERATE  # trimmed duration

        l_aeq, l_ae, _ = _measure(signal)

        expected_diff = 10.0 * np.log10(T_actual / T0)
        measured_diff = l_ae - l_aeq
        dev = measured_diff - expected_diff
        if report:
            return {"label": "L_AE - L_Aeq = 10log(T/T0)", "deviation": dev,
                    "limit_lo": -0.05, "limit_hi": 0.05,
                    "margin": min(dev + 0.05, 0.05 - dev)}
        assert abs(measured_diff - expected_diff) <= 0.05, (
            f"L_AE − L_Aeq = {measured_diff:.4f} dB, "
            f"expected 10·log₁₀({T_actual:.3f}) = {expected_diff:.4f} dB, "
            f"deviation = {measured_diff - expected_diff:+.4f} dB (limit: ±0.05 dB)"
        )

    def test_sel_reference_exposure(self, report: bool = False):
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
        dev = l_ae - expected_l_ae
        if report:
            return {"label": "SEL reference exposure", "deviation": dev,
                    "limit_lo": -0.05, "limit_hi": 0.05,
                    "margin": min(dev + 0.05, 0.05 - dev)}
        assert abs(l_ae - expected_l_ae) <= 0.05, (
            f"L_AE = {l_ae:.4f} dB, expected {expected_l_ae:.4f} dB "
            f"(deviation {l_ae - expected_l_ae:+.4f} dB, limit: ±0.05 dB)"
        )


# ---------------------------------------------------------------------------
# §5.10 — Repeated tonebursts
# ---------------------------------------------------------------------------

# IEC 61672-1:2013 Table 4 — class 1 acceptance limits for the SEL toneburst
# response, keyed by the *individual* toneburst duration T_b (§5.10.1 applies the
# "applicable acceptance limits of Table 4 for the SEL toneburst response", and
# §5.10.2 frames the relevant duration as the individual T_b).  The reference
# response itself is not taken from this table; it is computed from Eq. (9).
# (T_b_ms: (cl1_lo, cl1_hi))
_TABLE4_SEL_LIMITS = {
    1000: (-0.5, +0.5),
     500: (-0.5, +0.5),
     200: (-0.5, +0.5),
     100: (-1.0, +1.0),
      50: (-1.0, +1.0),
      20: (-1.0, +1.0),
      10: (-1.0, +1.0),
       5: (-1.0, +1.0),
       2: (-1.5, +1.0),
       1: (-2.0, +1.0),
     0.5: (-2.5, +1.0),
    0.25: (-3.0, +1.0),
}

# (n_bursts, T_b_ms, T_m_s) — repeated 4 kHz toneburst sequences in a 1 s window.
# Each T_b is a Table 4 duration (so its acceptance band is defined directly) and
# a whole number of 4 kHz periods at 48 kHz (so ∑sin² is exact).  n ≥ 2 and
# n·T_b < T_m so the bursts form a genuine sequence with gaps; the 1000 ms and
# 500 ms rows are omitted because they cannot be repeated within 1 s (that steady
# / single-burst regime is covered by the §5.9 SEL test).  0.25 ms is one period,
# the minimum duration of §5.10.2.
_REPEATED = [
    ( 2,  200,   1.0),
    ( 3,  100,   1.0),
    ( 5,   50,   1.0),
    (10,   20,   1.0),
    (10,   10,   1.0),
    (10,    5,   1.0),
    (10,    2,   1.0),
    (10,    1,   1.0),
    (10,    0.5, 1.0),
    (10,    0.25, 1.0),
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
    n_m    = int(round(T_m_s * samplerate))       # exact window (T_m = 1 s)

    # Build signal: n bursts spread evenly across T_m, padded with silence.
    full = np.zeros(n_m)
    stride = n_m // n_bursts  # samples between burst starts
    for k in range(n_bursts):
        start = k * stride
        end   = min(start + n_b, n_m)
        t_b   = np.arange(end - start) / samplerate
        full[start:end] = amplitude * np.sin(2.0 * np.pi * freq_hz * t_b)

    return _measure_leq_window(full)


class TestRepeatedTonebursts:
    """IEC 61672-1 §5.10 — response to repeated tonebursts, class 1.

    The reference response is δ_ref = 10·log₁₀(n·T_b / T_m) [Eq. (9)]; the
    acceptance band is the Table 4 SEL limit for the individual toneburst
    duration T_b (§5.10.1, §5.10.2).
    """

    @pytest.mark.parametrize("row", _REPEATED, ids=_REPEATED_IDS)
    def test_repeated_toneburst(self, row, report: bool = False):
        n_bursts, T_b_ms, T_m_s = row
        cl1_lo, cl1_hi = _TABLE4_SEL_LIMITS[T_b_ms]

        # Reference response of the sequence, Eq. (9): δ_ref = 10·log₁₀(n·T_b/T_m).
        ref_sel = 10.0 * np.log10(n_bursts * (T_b_ms / 1000.0) / T_m_s)

        # Steady-state A-weighted level of the corresponding 4 kHz sine, averaged
        # over the same exact T_m window.  The frequency weighting cancels in the
        # difference, so its absolute gain is irrelevant.
        signal_ss = _make_sine(4000, 1.0, int(round(T_m_s * SAMPLERATE)))
        l_a = _measure_leq_window(signal_ss)

        l_aeq = _repeated_burst_leq(n_bursts, T_b_ms, T_m_s)
        delta  = l_aeq - l_a
        dev    = delta - ref_sel

        margin = min(dev - cl1_lo, cl1_hi - dev)
        if report:
            return {"label": f"n={n_bursts}, T_b={T_b_ms} ms", "deviation": dev,
                    "limit_lo": cl1_lo, "limit_hi": cl1_hi, "margin": margin}
        assert cl1_lo <= dev <= cl1_hi, (
            f"n={n_bursts}, T_b={T_b_ms} ms, T_m={T_m_s} s: "
            f"δ = {delta:.3f} dB, ref = {ref_sel:.2f} dB, "
            f"dev = {dev:+.3f} dB (class 1: [{cl1_lo:+.1f}, {cl1_hi:+.1f}])"
        )
