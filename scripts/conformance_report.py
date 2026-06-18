#!/usr/bin/env python
"""IEC 61672-1 / IEC 61260-1 Conformance Margin Report.

Calls conformance test methods directly with report=True and displays a
formatted table of measured values, limits, and margins from each limit.

Usage:
    python scripts/conformance_report.py
    python scripts/conformance_report.py --no-color
    python scripts/conformance_report.py --precision 4
    python scripts/conformance_report.py --slow   # include §5.14/§5.15 stability
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
# Some IEC 61260 test modules import their siblings by bare name (relying on
# pytest's per-directory sys.path insertion); replicate that for standalone use.
sys.path.insert(0, str(_ROOT / "tests" / "iec61260"))

from tests.iec61672.test_61672_frequency_weightings import (
    TestAWeightingClass1, TestCWeightingClass1,
    TestWeightingDifferencesAt1kHz, _TABLE3,
)
from tests.iec61672.test_61672_time_weightings import (
    TestFastTimeWeightingDecayRate, TestSlowTimeWeightingDecayRate,
    TestFvsSteadyState,
)
from tests.iec61672.test_61672_toneburst import (
    TestFmaxToneburst, TestSELToneburst, TestSmaxToneburst, _TABLE4, _TABLE4_SMAX,
)
from tests.iec61672.test_61672_cpeak import (
    TestCWeightedPeak, _TABLE5,
)
from tests.iec61672.test_61672_level_linearity import (
    TestLevelLinearityTotalRange, TestLevelLinearityIncremental,
    TestLinearRangeWidth,
)
from tests.iec61672.test_61672_laeq_sel import (
    TestRepeatedTonebursts, _REPEATED,
)
from tests.iec61672.test_61672_stability import (
    TestContinuousOperationStability, TestHighLevelStability,
)
from tests.iec61260.test_61260_1_filters import (
    TestOctaveRelativeAttenuation, TestOctaveEffectiveBandwidth,
    BANDWIDTHS, passband_cases, stopband_cases, bandwidth_cases,
)
from tests.iec61260.test_61260_1_summation import (
    TestSummationOfOutputSignals, summation_cases,
)
from tests.iec61260.test_61260_1_level_linearity import (
    TestLinearOperatingRange, linearity_cases,
)
from tests.iec61260.test_61260_1_time_invariant import (
    TestTimeInvariantOperation, time_invariant_cases,
)

# ---------------------------------------------------------------------------
# ANSI colour
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and "--no-color" not in sys.argv

_R = "\033[31m" if USE_COLOR else ""   # red
_Y = "\033[33m" if USE_COLOR else ""   # yellow
_G = "\033[32m" if USE_COLOR else ""   # green
_B = "\033[1m"  if USE_COLOR else ""   # bold
_X = "\033[0m"  if USE_COLOR else ""   # reset


def _color_margin(margin: float, half_width: float, p: int = 3) -> str:
    pct = margin / half_width * 100 if half_width > 0 else 100.0
    c = _G if pct > 50 else (_Y if pct > 20 else _R)
    return f"{c}{margin:+7.{p}f}{_X}"


def _pass_fail(ok: bool) -> str:
    return f"{_G}PASS{_X}" if ok else f"{_R}FAIL{_X}"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    width = 74
    print()
    print("=" * width)
    print(f"{_B}{title}{_X}")
    print("-" * width)


def _sub(title: str) -> None:
    print(f"\n  {_B}{title}{_X}")
    print("  " + "-" * 70)


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def _print_weighting_section(title: str, rows: list[dict], p: int = 3) -> None:
    """Print a frequency-weighting or toneburst/cpeak margin table."""
    _hdr(title)
    print(f"  {'Label':>14}  {'Dev (dB)':>9}  {'Lo':>7}  {'Hi':>7}  {'Margin':>8}  {'':4}")
    print("  " + "-" * 58)
    any_fail = False
    for r in rows:
        lo  = r["limit_lo"]
        hi  = r["limit_hi"]
        dev = r["deviation"]
        margin = r["margin"]
        ok = (lo is None or dev >= lo) and (hi is None or dev <= hi)
        any_fail = any_fail or not ok
        half = (hi if lo is None else (hi - lo) / 2) if hi is not None else abs(lo)
        lo_str = f"{lo:+.1f}" if lo is not None else "  n/a"
        hi_str = f"{hi:+.1f}" if hi is not None else "  n/a"
        print(f"  {r['label']:>14}  {dev:>+9.{p}f}  {lo_str:>7}  {hi_str:>7}  "
              f"{_color_margin(margin, half, p)}  {_pass_fail(ok)}")
    worst = min(rows, key=lambda r: r["margin"])
    print(f"\n  Worst margin: {worst['margin']:+.{p}f} dB @ {worst['label']}"
          f"  --  {_pass_fail(not any_fail)}")


def _print_rate_section(title: str, rows: list[dict], p: int = 3) -> None:
    """Print a time-weighting decay-rate table."""
    _hdr(title)
    print(f"  {'Label':>10}  {'Rate (dB/s)':>12}  {'Lo':>7}  {'Hi':>7}  {'Margin':>8}  {'':4}")
    print("  " + "-" * 58)
    any_fail = False
    for r in rows:
        lo, hi = r["limit_lo"], r["limit_hi"]
        rate   = r["rate"]
        margin = r["margin"]
        ok = lo <= rate <= hi
        any_fail = any_fail or not ok
        half = (hi - lo) / 2
        print(f"  {r['label']:>10}  {rate:>12.{p}f}  {lo:>7.1f}  {hi:>7.1f}  "
              f"{_color_margin(margin, half, p)}  {_pass_fail(ok)}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


def _print_linearity_section(title: str, rows: list[dict], p: int = 3) -> None:
    """Print a level-linearity table (value vs upper limit)."""
    _hdr(title)
    print(f"  {'Metric':<22}  {'Value':>8}  {'Limit':>7}  {'Margin':>8}  {'Note'}")
    print("  " + "-" * 68)
    any_fail = False
    for r in rows:
        val    = r["value"]
        lim    = r["limit"]
        margin = r["margin"]
        ok = val <= lim
        any_fail = any_fail or not ok
        note = f"  ({r['note']})" if r.get("note") else ""
        print(f"  {r['label']:<22}  {val:>8.{p}f}  {lim:>7.3f}  "
              f"{_color_margin(margin, lim, p)}  {_pass_fail(ok)}{note}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


def _print_filter_section(title: str, rows: list[dict], p: int = 3) -> None:
    """Print a pass-band or stop-band filter margin table (worst per band)."""
    _hdr(title)
    # Group by band label prefix (e.g. "63 Hz …") — use worst margin per band
    from collections import defaultdict
    bands: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        band_key = r["label"].split(" Hz")[0] + " Hz"
        bands[band_key].append(r)

    print(f"  {'Band':>8}  {'Dev (dB)':>9}  {'Lo':>7}  {'Hi':>7}  {'Margin':>8}  {'':4}")
    print("  " + "-" * 56)
    any_fail = False
    for band_key, band_rows in bands.items():
        worst_row = min(band_rows, key=lambda r: r["margin"])
        lo  = worst_row["limit_lo"]
        hi  = worst_row["limit_hi"]
        dev = worst_row["deviation"]
        margin = worst_row["margin"]
        ok = (lo is None or dev >= lo) and (hi is None or dev <= hi)
        any_fail = any_fail or not ok
        if hi is None:
            half = abs(lo) if lo is not None else 1.0
        elif lo is None:
            half = hi
        else:
            half = (hi - lo) / 2
        lo_str = f"{lo:+.1f}" if lo is not None else "  n/a"
        hi_str = f"{hi:+.1f}" if hi is not None else "  n/a"
        print(f"  {band_key:>8}  {dev:>+9.{p}f}  {lo_str:>7}  {hi_str:>7}  "
              f"{_color_margin(margin, half, p)}  {_pass_fail(ok)}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


def _print_bw_section(title: str, rows: list[dict], p: int = 3) -> None:
    """Print an effective-bandwidth deviation table."""
    _hdr(title)
    print(f"  {'Band':>8}  {'DB (dB)':>9}  {'|DB|':>7}  {'Limit':>7}  {'Margin':>8}  {'':4}")
    print("  " + "-" * 58)
    any_fail = False
    for r in rows:
        db  = r["deviation"]
        lim = r["limit_hi"]
        margin = r["margin"]
        ok = abs(db) <= lim
        any_fail = any_fail or not ok
        print(f"  {r['label']:>8}  {db:>+9.{p}f}  {abs(db):>7.{p}f}  "
              f"±{lim:>5.3f}  {_color_margin(margin, lim, p)}  {_pass_fail(ok)}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(precision: int = 3, slow: bool = False) -> None:
    p = precision
    print(f"\n{_B}IEC 61672-1:2013 / IEC 61260-1:2014 Conformance Margin Report{_X}")

    # --- IEC 61672-1 §5.5 Frequency weightings ---
    a_test = TestAWeightingClass1()
    _print_weighting_section(
        "IEC 61672-1 §5.5 A-weighting (class 1)",
        [a_test.test_gain_within_class1(row, report=True) for row in _TABLE3], p,
    )

    c_test = TestCWeightingClass1()
    _print_weighting_section(
        "IEC 61672-1 §5.5 C-weighting (class 1)",
        [c_test.test_gain_within_class1(row, report=True) for row in _TABLE3], p,
    )

    diff_test = TestWeightingDifferencesAt1kHz()
    _print_weighting_section(
        "IEC 61672-1 §5.5.9 Weighting differences at 1 kHz (± 0.2 dB)",
        [
            diff_test.test_c_minus_a(report=True),
            diff_test.test_z_minus_a(report=True),
        ], p,
    )

    # --- IEC 61672-1 §5.8 Time-weighting decay rates ---
    _print_rate_section(
        "IEC 61672-1 §5.8 Time-weighting decay rates (class 1)",
        [
            TestFastTimeWeightingDecayRate().test_decay_rate_4khz(report=True),
            TestSlowTimeWeightingDecayRate().test_decay_rate_4khz(report=True),
        ], p,
    )

    # --- IEC 61672-1 §5.8.3 F / S / Leq agreement at 1 kHz ---
    _print_weighting_section(
        "IEC 61672-1 §5.8.3 F / S / Leq agreement at 1 kHz (± 0.1 dB)",
        TestFvsSteadyState().test_steady_1khz(report=True), p,
    )

    # --- IEC 61672-1 §5.9 Toneburst response ---
    fmax_test = TestFmaxToneburst()
    _print_weighting_section(
        "IEC 61672-1 §5.9 F-max toneburst (class 1)",
        [fmax_test.test_fmax_vs_table4(row, report=True) for row in _TABLE4], p,
    )

    smax_test = TestSmaxToneburst()
    _print_weighting_section(
        "IEC 61672-1 §5.9 S-max toneburst (class 1)",
        [smax_test.test_smax_vs_table4(row, report=True) for row in _TABLE4_SMAX], p,
    )

    sel_test = TestSELToneburst()
    _print_weighting_section(
        "IEC 61672-1 §5.9 SEL toneburst (class 1)",
        [sel_test.test_sel_vs_table4(row, report=True) for row in _TABLE4], p,
    )

    # --- IEC 61672-1 §5.10 Repeated tonebursts ---
    rep_test = TestRepeatedTonebursts()
    _print_weighting_section(
        "IEC 61672-1 §5.10 Repeated tonebursts (class 1)",
        [rep_test.test_repeated_toneburst(row, report=True) for row in _REPEATED], p,
    )

    # --- IEC 61672-1 §5.13 C-weighted peak ---
    cpeak_test = TestCWeightedPeak()
    _print_weighting_section(
        "IEC 61672-1 §5.13 C-weighted peak L_Cpeak - L_C (class 1)",
        [cpeak_test.test_cpeak_minus_lc(row, report=True) for row in _TABLE5], p,
    )

    # --- IEC 61672-1 §5.6 Level linearity ---
    # (§5.6.5 residuals ±0.8 dB and §5.6.6 1 dB-step deviation ±0.3 dB are the
    #  standard-defined acceptance limits; the regression-slope check is a
    #  self-imposed sanity test and is intentionally not reported here.)
    print("\n", end="", flush=True)
    lin_test  = TestLevelLinearityTotalRange()
    inc_test  = TestLevelLinearityIncremental()
    _print_linearity_section(
        "IEC 61672-1 §5.6 Level linearity (class 1, 1 kHz)",
        [
            lin_test.test_residuals_within_08dB(report=True),
            inc_test.test_1dB_steps(report=True),
        ], p,
    )

    _print_weighting_section(
        "IEC 61672-1 §5.6 Linear operating range (>= 60 dB at 1 kHz)",
        [TestLinearRangeWidth().test_linear_range_at_least_60dB(report=True)], p,
    )

    # --- IEC 61672-1 §5.14 / §5.15 Long-time stability (slow) ---
    if slow:
        print("\n  (running long-time stability tests, this takes a while...)",
              flush=True)
        _print_linearity_section(
            "IEC 61672-1 §5.14 / §5.15 Long-time stability (drift <= 0.1 dB)",
            [
                TestContinuousOperationStability().test_30min_stability_1khz(report=True),
                TestHighLevelStability().test_5min_high_level_stability(report=True),
            ], p,
        )

    # --- IEC 61260-1 filter conformance, once per bandwidth in BANDWIDTHS ---
    # Mid-band frequencies (§5.4) and band edges (§5.6) are omitted: the standard
    # defines f_m exactly by formula (no acceptance tolerance) and the band-edge
    # value sits in a Table-1 discontinuity rather than at a defined limit, so the
    # tolerances those tests use are self-imposed (see the test files).
    pb_test   = TestOctaveRelativeAttenuation()
    bw_test   = TestOctaveEffectiveBandwidth()
    lin260_test = TestLinearOperatingRange()
    ti_test     = TestTimeInvariantOperation()
    sum_test    = TestSummationOfOutputSignals()

    for cfg in BANDWIDTHS:
        bw = cfg.name

        # --- §5.10 Pass-band ---
        _print_filter_section(
            f"IEC 61260-1 §5.10 Pass-band attenuation ({bw}, class 1, worst per band)",
            [pb_test.test_passband(c, report=True) for c in passband_cases(cfg)], p,
        )

        # --- §5.10 Stop-band ---
        _print_filter_section(
            f"IEC 61260-1 §5.10 Stop-band attenuation ({bw}, class 1, worst per band)",
            [pb_test.test_stopband(c, report=True) for c in stopband_cases(cfg)], p,
        )

        # --- §5.12 Effective bandwidth ---
        _print_bw_section(
            f"IEC 61260-1 §5.12 Effective bandwidth deviation DB ({bw}, class 1)",
            [bw_test.test_bandwidth_deviation(c, report=True) for c in bandwidth_cases(cfg)], p,
        )

        # --- §5.13 Level linearity ---
        lin260_rows = [
            row
            for c in linearity_cases(cfg)
            for row in lin260_test.test_level_linearity(c, report=True)
        ]
        _print_filter_section(
            f"IEC 61260-1 §5.13 Level linearity ({bw}, class 1, worst per band)",
            lin260_rows, p,
        )

        # --- §5.14 Time-invariant operation ---
        _print_filter_section(
            f"IEC 61260-1 §5.14 Time-invariant operation ({bw}, class 1)",
            [ti_test.test_time_invariant(c, report=True) for c in time_invariant_cases(cfg)], p,
        )

        # --- §5.16 Summation of output signals ---
        _print_filter_section(
            f"IEC 61260-1 §5.16 Summation ({bw}, class 1, worst per band pair)",
            [sum_test.test_summation(c, report=True) for c in summation_cases(cfg)], p,
        )

    print()
    print("=" * 74)
    print("Done.")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-color", action="store_true",
                        help="disable ANSI colour output")
    parser.add_argument("--precision", type=int, default=3, metavar="N",
                        help="decimal places for reported values and margins (default: 3)")
    parser.add_argument("--slow", action="store_true",
                        help="include the long-time stability tests (§5.14/§5.15); "
                             "these take several minutes to run")
    args = parser.parse_args()
    main(precision=args.precision, slow=args.slow)
