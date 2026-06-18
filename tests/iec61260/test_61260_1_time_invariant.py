"""IEC 61260-1:2014 §5.14 — Time-invariant operation (class 1).

An exponential frequency sweep (1 decade in 2–5 s) is applied to the filter
bank.  For each band the time-averaged output level L_out must agree with the
theoretical level L_c from Formula (17) within ±0.4 dB (class 1).  This test is
parametric over filter bandwidth (see ``BANDWIDTHS`` in test_61260_1_filters).

Formula (17) with T_avg = T_sweep (average over the full sweep duration):

    L_c = L_in − A_ref + 10·lg( lg(f₂/f₁) / lg(f_end/f_start) )

where:
    L_in   = time-averaged input level  (= 10·log₁₀(0.5) dB for amplitude-1 sine)
    A_ref  = reference attenuation at mid-band (gain of filter at f_m, negated)
    f₁/f₂  = lower/upper band-edge frequencies (§5.6: f_m · G^{∓1/2b})
    f_start, f_end = sweep start/end frequencies (span all filter bands)

NOTE §5.14 Note 1: lg(f₂/f₁) = 3/(10·b) for bandwidth designator 1/b.
For octave-band filters (b=1): lg(f₂/f₁) = 0.3; for 1/3-octave (b=3): 0.1.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest
from scipy import signal as sig

from test_61260_1_filters import (
    G, SAMPLERATE, BANDWIDTHS, FilterConfig, _filterbank, _nominal_hz,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_INVARIANT_LIMIT_CL1 = 0.4   # ±dB, §5.14.3 class 1

# Sweep rate: 1 decade in T_DECADE seconds — must be in [2, 5] s per §5.14.3.
T_DECADE = 3.0   # s per decade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exponential_sweep(f_start: float, f_end: float, t_decade: float,
                       samplerate: int = SAMPLERATE,
                       amplitude: float = 1.0) -> tuple[np.ndarray, float]:
    """Return (samples, t_sweep) for an exponential sine sweep.

    Instantaneous frequency: f(t) = f_start · (f_end/f_start)^(t/t_sweep)
    Phase:  φ(t) = 2π · f_start · t_sweep/ln(f_end/f_start) · [(f_end/f_start)^(t/t_sweep) − 1]
    """
    n_decades = math.log10(f_end / f_start)
    t_sweep   = n_decades * t_decade
    n         = int(round(t_sweep * samplerate))
    t         = np.arange(n) / samplerate
    k         = (f_end / f_start) ** (1.0 / t_sweep)
    phase     = 2.0 * math.pi * f_start * (k ** t - 1.0) / math.log(k)
    return amplitude * np.sin(phase), t_sweep


def _sweep_bounds(cfg: FilterConfig) -> tuple[float, float]:
    """Sweep start/end frequencies: one octave beyond the outermost mid-bands,
    capped just below Nyquist (§5.14.2 — span all bands and their edges)."""
    centers, _ = _filterbank(cfg)
    f_start = centers[0] / G
    f_end   = min(centers[-1] * G, cfg.samplerate * 0.499)
    return f_start, f_end


def _l_c_theory(f_m: float, f_start: float, f_end: float, b: int,
                sos: np.ndarray, samplerate: int) -> float:
    """Theoretical time-averaged output level for a unit-amplitude sweep (Formula 17).

    Uses T_avg = T_sweep so the ratio cancels and only band-width and sweep
    span matter.
    """
    f1 = f_m * G ** (-1 / (2 * b))     # lower band-edge (§5.6)
    f2 = f_m * G ** (+1 / (2 * b))     # upper band-edge
    # Reference attenuation: actual filter gain at mid-band (negated → positive dB)
    gain_mid_db = 20.0 * np.log10(
        abs(sig.sosfreqz(sos, worN=[f_m], fs=samplerate)[1][0]))
    a_ref = -gain_mid_db
    l_in  = 10.0 * math.log10(0.5)   # -3.01 dB: time-avg level of amplitude-1 sine
    return l_in - a_ref + 10.0 * math.log10(
        math.log10(f2 / f1) / math.log10(f_end / f_start))


def _measure_sweep_level(sweep: np.ndarray, sos: np.ndarray) -> float:
    """Apply sweep through a filter and return time-averaged output level (dB)."""
    filtered = sig.sosfilt(sos, sweep)
    mean_sq  = float(np.mean(filtered ** 2))
    return 10.0 * math.log10(max(mean_sq, 1e-300))


# ---------------------------------------------------------------------------
# Per-config sweep cache and parametric test cases
# ---------------------------------------------------------------------------

_sweep_cache: dict = {}


def _sweep_for(cfg: FilterConfig) -> tuple[np.ndarray, float, float, float]:
    """Return (sweep, t_sweep, f_start, f_end) for *cfg* (cached)."""
    key = (cfg.b, cfg.limits, cfg.samplerate)
    if key not in _sweep_cache:
        f_start, f_end = _sweep_bounds(cfg)
        sweep, t_sweep = _exponential_sweep(f_start, f_end, T_DECADE,
                                            cfg.samplerate)
        _sweep_cache[key] = (sweep, t_sweep, f_start, f_end)
    return _sweep_cache[key]


class BandCase(NamedTuple):
    cfg: FilterConfig
    band_idx: int
    f_m: float
    label: str


def time_invariant_cases(cfg: FilterConfig) -> list[BandCase]:
    centers, _ = _filterbank(cfg)
    nom = _nominal_hz(cfg)
    return [BandCase(cfg, i, f_m, f"{nom[i]} Hz")
            for i, f_m in enumerate(centers)]


_PARAMS: list[BandCase] = []
_IDS:    list[str]      = []
for _cfg in BANDWIDTHS:
    for _case in time_invariant_cases(_cfg):
        _PARAMS.append(_case)
        _IDS.append(f"{_cfg.name}-{_case.label.replace(' ', '')}")


# ---------------------------------------------------------------------------
# §5.14: Time-invariant operation
# ---------------------------------------------------------------------------

class TestTimeInvariantOperation:
    """§5.14 — exponential-sweep response matches Formula (17) within ±0.4 dB.

    Sweep spans one octave beyond the outermost bands of each bandwidth, at
    {} s/decade.  Class 1 limit: ±{} dB.
    """.format(T_DECADE, TIME_INVARIANT_LIMIT_CL1)

    @pytest.mark.parametrize("case", _PARAMS, ids=_IDS)
    def test_time_invariant(self, case: BandCase, report: bool = False):
        _, sos_list = _filterbank(case.cfg)
        sos = sos_list[case.band_idx]
        sweep, _t_sweep, f_start, f_end = _sweep_for(case.cfg)

        l_out  = _measure_sweep_level(sweep, sos)
        l_c    = _l_c_theory(case.f_m, f_start, f_end, case.cfg.b, sos,
                             case.cfg.samplerate)
        dev    = l_out - l_c
        margin = TIME_INVARIANT_LIMIT_CL1 - abs(dev)
        if report:
            return {"label": case.label, "deviation": dev,
                    "limit_lo": -TIME_INVARIANT_LIMIT_CL1,
                    "limit_hi": +TIME_INVARIANT_LIMIT_CL1,
                    "margin": margin}
        assert abs(dev) <= TIME_INVARIANT_LIMIT_CL1, (
            f"{case.cfg.name} band {case.f_m:.1f} Hz: L_out = {l_out:.3f} dB, "
            f"L_c = {l_c:.3f} dB, dev = {dev:+.3f} dB "
            f"(class 1: ±{TIME_INVARIANT_LIMIT_CL1} dB)"
        )
