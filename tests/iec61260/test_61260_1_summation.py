"""IEC 61260-1:2014 §5.16 — Summation of output signals (class 1).

For a sinusoidal input at any frequency between two consecutive octave mid-band
frequencies, the difference:

    dev = (L_in − A_ref) − L_sum

must lie within [−1.8, +0.8] dB (class 1), where:

    L_in   = 10·log₁₀(mean_sq_input)          — time-averaged input level
    A_ref  = reference attenuation at mid-band  — nominally 0 dB for our filters
    L_sum  = 10·log₁₀(P_lower + P_upper)       — sum of adjacent filter mean-sq outputs

Negative dev means more energy in the sum than expected (filter overlap).
Positive dev means less energy (gap between adjacent filters).

The five test frequencies per adjacent pair are evenly spaced (in G-exponent) between
the two mid-bands, including the geometric-mean crossover at G^{+1/2}.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal as sig

from test_61260_1_filters import G, SAMPLERATE, _get_filterbank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMATION_LO_CL1 = -1.8   # dB, §5.16 class 1 lower limit
SUMMATION_HI_CL1 = +0.8   # dB, §5.16 class 1 upper limit

# G-exponents from the lower mid-band frequency in each adjacent pair.
# Span the full inter-band range; crossover is at G^{+0.5}.
_TEST_EXPONENTS = [1/4, 3/8, 1/2, 5/8, 3/4]

# Sine duration and averaging window.
_DURATION_S = 1.0   # total signal duration
_SKIP_FRAC  = 0.5   # skip first half to avoid filter startup transient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a_ref(sos: np.ndarray, f_m: float) -> float:
    """Reference attenuation at mid-band frequency f_m (positive dB)."""
    gain_db = 20.0 * np.log10(
        abs(sig.sosfreqz(sos, worN=[f_m], fs=SAMPLERATE)[1][0]))
    return -gain_db


def _filter_mean_sq(sos: np.ndarray, freq_hz: float) -> float:
    """Steady-state mean square of filter output for a unit-amplitude sine.

    Computes over the second half of the signal to skip the startup transient.
    """
    n    = int(round(_DURATION_S * SAMPLERATE))
    t    = np.arange(n) / SAMPLERATE
    x    = np.sin(2.0 * math.pi * freq_hz * t)
    y    = sig.sosfilt(sos, x)
    skip = int(n * _SKIP_FRAC)
    return float(np.mean(y[skip:] ** 2))


# ---------------------------------------------------------------------------
# Build parametrize list at collection time
# ---------------------------------------------------------------------------

_centers, _sos_list = _get_filterbank()
_n_bands = len(_centers)

_PARAMS: list[tuple[int, float]] = []
_IDS:   list[str]                = []
for _pi in range(_n_bands - 1):
    _f_lo = _centers[_pi]
    _f_hi = _centers[_pi + 1]
    for _exp in _TEST_EXPONENTS:
        _PARAMS.append((_pi, _f_lo * G ** _exp))
        _IDS.append(
            f"pair{int(round(_f_lo))}-{int(round(_f_hi))}Hz_G^{_exp:+.3f}")


# ---------------------------------------------------------------------------
# §5.16: Summation of output signals
# ---------------------------------------------------------------------------

class TestSummationOfOutputSignals:
    """§5.16 — sum of adjacent filter outputs lies within [−1.8, +0.8] dB.

    Seven adjacent pairs (63–8000 Hz), five test frequencies per pair.
    Class 1 limits: −1.8 dB (max overlap) to +0.8 dB (max gap).
    """

    @pytest.mark.parametrize("row", _PARAMS, ids=_IDS)
    def test_summation(self, row: tuple, report: bool = False):
        pair_idx, f_test = row
        sos_lo = _sos_list[pair_idx]
        sos_hi = _sos_list[pair_idx + 1]
        f_m_lo = _centers[pair_idx]

        a_ref_db = _a_ref(sos_lo, f_m_lo)          # reference attenuation (~0 dB)
        l_in     = 10.0 * math.log10(0.5)          # unit-amplitude sine: mean_sq = 0.5

        p_lo   = _filter_mean_sq(sos_lo, f_test)
        p_hi   = _filter_mean_sq(sos_hi, f_test)
        l_sum  = 10.0 * math.log10(max(p_lo + p_hi, 1e-300))

        dev    = (l_in - a_ref_db) - l_sum
        margin = min(dev - SUMMATION_LO_CL1, SUMMATION_HI_CL1 - dev)

        if report:
            return {
                "label":    f"{int(round(_centers[pair_idx]))}/{int(round(_centers[pair_idx+1]))} Hz @ {f_test:.1f} Hz",
                "deviation": dev,
                "limit_lo":  SUMMATION_LO_CL1,
                "limit_hi":  SUMMATION_HI_CL1,
                "margin":    margin,
            }

        assert SUMMATION_LO_CL1 <= dev <= SUMMATION_HI_CL1, (
            f"Pair {_centers[pair_idx]:.1f}/{_centers[pair_idx + 1]:.1f} Hz "
            f"@ {f_test:.2f} Hz: "
            f"L_sum = {l_sum:.3f} dB, (L_in−A_ref) = {l_in - a_ref_db:.3f} dB, "
            f"dev = {dev:+.3f} dB "
            f"(class 1: [{SUMMATION_LO_CL1:+.1f}, {SUMMATION_HI_CL1:+.1f}] dB)"
        )
