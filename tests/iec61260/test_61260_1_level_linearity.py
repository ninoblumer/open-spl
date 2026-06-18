"""IEC 61260-1:2014 §5.13 — Linear operating range (class 1).

At the exact mid-band frequency of each filter, the level linearity deviation
shall be:

    §5.13.3: ±0.5 dB for the upper 40 dB of the linear operating range
    §5.13.4: ±0.7 dB for the remaining lower zone

over a minimum linear operating range of 60 dB (§5.13.1, class 1).  §5.13.1
states these limits apply "for all filter bandwidths", so this test is
parametric over bandwidth (see ``BANDWIDTHS`` in test_61260_1_filters).

Level linearity deviation is defined as:

    dev(A) = L_out(A) − [L_out(A_ref) + 20·log₁₀(A / A_ref)]

where A_ref = 1.0 (reference amplitude, 0 dB relative) and A is the test amplitude.

For a linear digital IIR filter (sosfilt), deviations are bounded by
floating-point precision (≪ 1e-9 dB) and the 60 dB range is satisfied by
construction.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest
from scipy import signal as sig

from test_61260_1_filters import BANDWIDTHS, FilterConfig, _filterbank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINEAR_RANGE_DB = 60.0   # minimum linear range, class 1 (§5.13.1)
UPPER_ZONE_DB   = 40.0   # upper zone: 0 dB down to −40 dB (§5.13.3)
LIMIT_UPPER_CL1 = 0.5    # ±dB class 1 upper zone (§5.13.3)
LIMIT_LOWER_CL1 = 0.7    # ±dB class 1 lower zone (§5.13.4)

# Test levels: 0 to −60 dB in 5 dB steps (reference at 0 dB, 12 test points)
_LEVEL_DB = list(range(0, -int(LINEAR_RANGE_DB) - 1, -5))   # [0, -5, …, -60]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_mean_sq(sos: np.ndarray, f_m: float, amplitude: float,
                    samplerate: int, duration: float = 0.5) -> float:
    """Steady-state mean-square output for a sine at *f_m* scaled by *amplitude*.

    Uses the second half of the signal to skip the startup transient.
    """
    n    = int(round(duration * samplerate))
    t    = np.arange(n) / samplerate
    skip = n // 2
    x    = amplitude * np.sin(2.0 * math.pi * f_m * t)
    y    = sig.sosfilt(sos, x)
    return float(np.mean(y[skip:] ** 2))


# ---------------------------------------------------------------------------
# Parametric test cases
# ---------------------------------------------------------------------------

class BandCase(NamedTuple):
    cfg: FilterConfig
    band_idx: int
    f_m: float
    label: str


def linearity_cases(cfg: FilterConfig) -> list[BandCase]:
    centers, _ = _filterbank(cfg)
    return [BandCase(cfg, i, f_m, f"{int(round(f_m))} Hz")
            for i, f_m in enumerate(centers)]


_PARAMS: list[BandCase] = []
_IDS:    list[str]      = []
for _cfg in BANDWIDTHS:
    for _case in linearity_cases(_cfg):
        _PARAMS.append(_case)
        _IDS.append(f"{_cfg.name}-{_case.label.replace(' ', '')}")


# ---------------------------------------------------------------------------
# §5.13: Linear operating range
# ---------------------------------------------------------------------------

class TestLinearOperatingRange:
    """§5.13 — level linearity over 60 dB amplitude range, class 1.

    Each test covers one band filter at its exact mid-band frequency.
    Amplitudes span 0 dB to −60 dB relative in 5 dB steps.
    Zone limits: ±0.5 dB (upper 40 dB) / ±0.7 dB (lower 20 dB).
    """

    @pytest.mark.parametrize("case", _PARAMS, ids=_IDS)
    def test_level_linearity(self, case: BandCase, report: bool = False):
        _, sos_list = _filterbank(case.cfg)
        f_m = case.f_m
        sos = sos_list[case.band_idx]
        sr  = case.cfg.samplerate

        # Reference output at amplitude = 1.0
        ms_ref = _filter_mean_sq(sos, f_m, 1.0, sr)
        l_ref  = 10.0 * math.log10(max(ms_ref, 1e-300))

        rows     = []
        failures = []

        for level_db in _LEVEL_DB[1:]:   # skip 0 dB (reference; dev = 0 by definition)
            amplitude = 10.0 ** (level_db / 20.0)
            ms_meas   = _filter_mean_sq(sos, f_m, amplitude, sr)
            l_meas    = 10.0 * math.log10(max(ms_meas, 1e-300))
            dev       = l_meas - (l_ref + level_db)
            limit     = LIMIT_UPPER_CL1 if level_db >= -UPPER_ZONE_DB else LIMIT_LOWER_CL1
            margin    = limit - abs(dev)

            if report:
                rows.append({
                    "label":    f"{int(round(f_m))} Hz @ {level_db:+d} dB",
                    "deviation": dev,
                    "limit_lo":  -limit,
                    "limit_hi":  +limit,
                    "margin":    margin,
                })
            elif abs(dev) > limit:
                failures.append(
                    f"  {level_db:+d} dB: dev = {dev:+.6f} dB > ±{limit} dB")

        if report:
            return rows
        assert not failures, (
            f"{case.cfg.name} band {f_m:.1f} Hz linear operating range failures:\n"
            + "\n".join(failures)
        )

    @pytest.mark.parametrize("case", _PARAMS, ids=_IDS)
    def test_linear_range_width(self, case: BandCase):
        """§5.13.1 — linear operating range at mid-band is at least 60 dB."""
        _, sos_list = _filterbank(case.cfg)
        f_m = case.f_m
        sos = sos_list[case.band_idx]
        sr  = case.cfg.samplerate

        ms_ref = _filter_mean_sq(sos, f_m, 1.0, sr)
        l_ref  = 10.0 * math.log10(max(ms_ref, 1e-300))

        # Walk down in 1 dB steps until deviation exceeds ±0.7 dB (most relaxed limit)
        linear_range_db = 0.0
        for level_db in range(0, -121, -1):
            amplitude = 10.0 ** (level_db / 20.0)
            ms_meas   = _filter_mean_sq(sos, f_m, amplitude, sr)
            l_meas    = 10.0 * math.log10(max(ms_meas, 1e-300))
            dev       = abs(l_meas - (l_ref + level_db))
            if dev > LIMIT_LOWER_CL1:
                break
            linear_range_db = abs(level_db)

        assert linear_range_db >= LINEAR_RANGE_DB, (
            f"{case.cfg.name} band {f_m:.1f} Hz: linear range = {linear_range_db:.0f} dB "
            f"(class 1 minimum: {LINEAR_RANGE_DB:.0f} dB)"
        )
