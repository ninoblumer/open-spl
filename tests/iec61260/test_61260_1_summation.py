"""IEC 61260-1:2014 §5.16 — Summation of output signals (class 1).

For a sinusoidal input at any frequency between two consecutive mid-band
frequencies, the difference:

    dev = (L_in − A_ref) − L_sum

must lie within [−1.8, +0.8] dB (class 1), where:

    L_in   = 10·log₁₀(mean_sq_input)          — time-averaged input level
    A_ref  = reference attenuation at mid-band  — nominally 0 dB for our filters
    L_sum  = 10·log₁₀(P_lower + P_upper)       — sum of adjacent filter mean-sq outputs

Negative dev means more energy in the sum than expected (filter overlap).
Positive dev means less energy (gap between adjacent filters).

This test is parametric over filter bandwidth (see ``BANDWIDTHS`` in
test_61260_1_filters); the five test frequencies per adjacent pair are evenly
spaced (in G-exponent) between the two mid-bands, scaled by the bandwidth
designator so the crossover at the geometric mean is always covered.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest
from scipy import signal as sig

from test_61260_1_filters import G, BANDWIDTHS, FilterConfig, _filterbank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMATION_LO_CL1 = -1.8   # dB, §5.16 class 1 lower limit
SUMMATION_HI_CL1 = +0.8   # dB, §5.16 class 1 upper limit

# Octave G-exponents from the lower mid-band frequency in an adjacent pair.
# Span the full inter-band range; the octave crossover is at G^{+0.5}.  These
# are divided by the bandwidth designator b so that, for any bandwidth, the
# test frequencies span one band spacing and include the crossover.
_OCTAVE_EXPONENTS = [1/4, 3/8, 1/2, 5/8, 3/4]

# Sine duration and averaging window.
_DURATION_S = 1.0   # total signal duration
_SKIP_FRAC  = 0.5   # skip first half to avoid filter startup transient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a_ref(sos: np.ndarray, f_m: float, samplerate: int) -> float:
    """Reference attenuation at mid-band frequency f_m (positive dB)."""
    gain_db = 20.0 * np.log10(
        abs(sig.sosfreqz(sos, worN=[f_m], fs=samplerate)[1][0]))
    return -gain_db


def _filter_mean_sq(sos: np.ndarray, freq_hz: float, samplerate: int) -> float:
    """Steady-state mean square of filter output for a unit-amplitude sine.

    Computes over the second half of the signal to skip the startup transient.
    """
    n    = int(round(_DURATION_S * samplerate))
    t    = np.arange(n) / samplerate
    x    = np.sin(2.0 * math.pi * freq_hz * t)
    y    = sig.sosfilt(sos, x)
    skip = int(n * _SKIP_FRAC)
    return float(np.mean(y[skip:] ** 2))


# ---------------------------------------------------------------------------
# Parametric test cases
# ---------------------------------------------------------------------------

class SummationCase(NamedTuple):
    cfg: FilterConfig
    pair_idx: int     # index of the lower band in the adjacent pair
    f_test: float
    label: str


def summation_cases(cfg: FilterConfig) -> list[SummationCase]:
    centers, _ = _filterbank(cfg)
    cases = []
    for pair_idx in range(len(centers) - 1):
        f_lo, f_hi = centers[pair_idx], centers[pair_idx + 1]
        for exp in _OCTAVE_EXPONENTS:
            f_test = f_lo * G ** (exp / cfg.b)
            cases.append(SummationCase(
                cfg, pair_idx, f_test,
                f"{int(round(f_lo))}/{int(round(f_hi))} Hz @ {f_test:.1f} Hz"))
    return cases


_PARAMS: list[SummationCase] = []
_IDS:    list[str]           = []
for _cfg in BANDWIDTHS:
    for _case in summation_cases(_cfg):
        _PARAMS.append(_case)
        _IDS.append(f"{_cfg.name}-{_case.label.replace(' ', '')}")


# ---------------------------------------------------------------------------
# §5.16: Summation of output signals
# ---------------------------------------------------------------------------

class TestSummationOfOutputSignals:
    """§5.16 — sum of adjacent filter outputs lies within [−1.8, +0.8] dB.

    Adjacent band pairs across each bandwidth's range, five test frequencies
    per pair.  Class 1 limits: −1.8 dB (max overlap) to +0.8 dB (max gap).
    """

    @pytest.mark.parametrize("case", _PARAMS, ids=_IDS)
    def test_summation(self, case: SummationCase, report: bool = False):
        centers, sos_list = _filterbank(case.cfg)
        sr     = case.cfg.samplerate
        sos_lo = sos_list[case.pair_idx]
        sos_hi = sos_list[case.pair_idx + 1]
        f_m_lo = centers[case.pair_idx]

        a_ref_db = _a_ref(sos_lo, f_m_lo, sr)      # reference attenuation (~0 dB)
        l_in     = 10.0 * math.log10(0.5)          # unit-amplitude sine: mean_sq = 0.5

        p_lo   = _filter_mean_sq(sos_lo, case.f_test, sr)
        p_hi   = _filter_mean_sq(sos_hi, case.f_test, sr)
        l_sum  = 10.0 * math.log10(max(p_lo + p_hi, 1e-300))

        dev    = (l_in - a_ref_db) - l_sum
        margin = min(dev - SUMMATION_LO_CL1, SUMMATION_HI_CL1 - dev)

        if report:
            return {
                "label":     case.label,
                "deviation": dev,
                "limit_lo":  SUMMATION_LO_CL1,
                "limit_hi":  SUMMATION_HI_CL1,
                "margin":    margin,
            }

        assert SUMMATION_LO_CL1 <= dev <= SUMMATION_HI_CL1, (
            f"{case.cfg.name} pair {centers[case.pair_idx]:.1f}/"
            f"{centers[case.pair_idx + 1]:.1f} Hz @ {case.f_test:.2f} Hz: "
            f"L_sum = {l_sum:.3f} dB, (L_in−A_ref) = {l_in - a_ref_db:.3f} dB, "
            f"dev = {dev:+.3f} dB "
            f"(class 1: [{SUMMATION_LO_CL1:+.1f}, {SUMMATION_HI_CL1:+.1f}] dB)"
        )
