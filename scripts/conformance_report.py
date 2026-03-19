#!/usr/bin/env python
"""IEC 61672-1 / IEC 61260-1 Conformance Margin Report.

Calls conformance test methods directly with report=True and displays a
formatted table of measured values, limits, and margins from each limit.

Usage:
    python scripts/conformance_report.py
    python scripts/conformance_report.py --no-color
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.iec61672.test_61672_frequency_weightings import (
    TestAWeightingClass1, TestCWeightingClass1, TestZWeightingFlat, _TABLE3,
)
from tests.iec61672.test_61672_time_weightings import (
    TestFastTimeWeightingDecayRate, TestSlowTimeWeightingDecayRate,
)
from tests.iec61672.test_61672_toneburst import (
    TestFmaxToneburst, TestSELToneburst, _TABLE4,
)
from tests.iec61672.test_61672_cpeak import TestCWeightedPeak, _TABLE5
from tests.iec61672.test_61672_level_linearity import (
    TestLevelLinearityTotalRange, TestLevelLinearityIncremental,
)
from tests.iec61260.test_61260_1_filters import (
    TestOctaveRelativeAttenuation, TestOctaveEffectiveBandwidth,
    _PB_PARAMS, _SB_PARAMS, _BAND_IDS,
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


def _color_margin(margin: float, half_width: float) -> str:
    pct = margin / half_width * 100 if half_width > 0 else 100.0
    c = _G if pct > 50 else (_Y if pct > 20 else _R)
    return f"{c}{margin:+7.3f}{_X}"


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

def _print_weighting_section(title: str, rows: list[dict]) -> None:
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
        print(f"  {r['label']:>14}  {dev:>+9.4f}  {lo_str:>7}  {hi_str:>7}  "
              f"{_color_margin(margin, half)}  {_pass_fail(ok)}")
    worst = min(rows, key=lambda r: r["margin"])
    print(f"\n  Worst margin: {worst['margin']:+.4f} dB @ {worst['label']}"
          f"  --  {_pass_fail(not any_fail)}")


def _print_rate_section(title: str, rows: list[dict]) -> None:
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
        print(f"  {r['label']:>10}  {rate:>12.4f}  {lo:>7.1f}  {hi:>7.1f}  "
              f"{_color_margin(margin, half)}  {_pass_fail(ok)}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


def _print_linearity_section(title: str, rows: list[dict]) -> None:
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
        print(f"  {r['label']:<22}  {val:>8.4f}  {lim:>7.3f}  "
              f"{_color_margin(margin, lim)}  {_pass_fail(ok)}{note}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


def _print_filter_section(title: str, rows: list[dict]) -> None:
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
        print(f"  {band_key:>8}  {dev:>+9.4f}  {lo_str:>7}  {hi_str:>7}  "
              f"{_color_margin(margin, half)}  {_pass_fail(ok)}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


def _print_bw_section(title: str, rows: list[dict]) -> None:
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
        print(f"  {r['label']:>8}  {db:>+9.4f}  {abs(db):>7.4f}  "
              f"±{lim:>5.3f}  {_color_margin(margin, lim)}  {_pass_fail(ok)}")
    print(f"\n  Overall: {_pass_fail(not any_fail)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{_B}IEC 61672-1:2013 / IEC 61260-1:2014 Conformance Margin Report{_X}")

    # --- IEC 61672-1 §5.5 Frequency weightings ---
    a_test = TestAWeightingClass1()
    _print_weighting_section(
        "IEC 61672-1 §5.5 A-weighting (class 1)",
        [a_test.test_gain_within_class1(row, report=True) for row in _TABLE3],
    )

    c_test = TestCWeightingClass1()
    _print_weighting_section(
        "IEC 61672-1 §5.5 C-weighting (class 1)",
        [c_test.test_gain_within_class1(row, report=True) for row in _TABLE3],
    )

    z_test = TestZWeightingFlat()
    _print_weighting_section(
        "IEC 61672-1 Annex E.5 Z-weighting (flat, ±0.1 dB)",
        [z_test.test_gain_is_zero(row, report=True) for row in _TABLE3],
    )

    # --- IEC 61672-1 §5.8 Time-weighting decay rates ---
    _print_rate_section(
        "IEC 61672-1 §5.8 Time-weighting decay rates (class 1)",
        [
            TestFastTimeWeightingDecayRate().test_decay_rate_4khz(report=True),
            TestSlowTimeWeightingDecayRate().test_decay_rate_4khz(report=True),
        ],
    )

    # --- IEC 61672-1 §5.9 Toneburst response ---
    fmax_test = TestFmaxToneburst()
    _print_weighting_section(
        "IEC 61672-1 §5.9 F-max toneburst (class 1)",
        [fmax_test.test_fmax_vs_table4(row, report=True) for row in _TABLE4],
    )

    sel_test = TestSELToneburst()
    _print_weighting_section(
        "IEC 61672-1 §5.9 SEL toneburst (class 1)",
        [sel_test.test_sel_vs_table4(row, report=True) for row in _TABLE4],
    )

    # --- IEC 61672-1 §5.13 C-weighted peak ---
    cpeak_test = TestCWeightedPeak()
    _print_weighting_section(
        "IEC 61672-1 §5.13 C-weighted peak L_Cpeak - L_C (class 1)",
        [cpeak_test.test_cpeak_minus_lc(row, report=True) for row in _TABLE5],
    )

    # --- IEC 61672-1 §5.6 Level linearity ---
    print("\n", end="", flush=True)
    lin_test  = TestLevelLinearityTotalRange()
    inc_test  = TestLevelLinearityIncremental()
    _print_linearity_section(
        "IEC 61672-1 §5.6 Level linearity (class 1, 1 kHz)",
        [
            lin_test.test_residuals_within_08dB(report=True),
            lin_test.test_slope_is_unity(report=True),
            inc_test.test_1dB_steps(report=True),
        ],
    )

    # --- IEC 61260-1 §5.10 Pass-band ---
    pb_test = TestOctaveRelativeAttenuation()
    _print_filter_section(
        "IEC 61260-1 §5.10 Octave pass-band attenuation (class 1, worst per band)",
        [pb_test.test_passband(row, report=True) for row in _PB_PARAMS],
    )

    # --- IEC 61260-1 §5.10 Stop-band ---
    _print_filter_section(
        "IEC 61260-1 §5.10 Octave stop-band attenuation (class 1, worst per band)",
        [pb_test.test_stopband(row, report=True) for row in _SB_PARAMS],
    )

    # --- IEC 61260-1 §5.12 Effective bandwidth ---
    bw_test = TestOctaveEffectiveBandwidth()
    _print_bw_section(
        "IEC 61260-1 §5.12 Effective bandwidth deviation DB (class 1)",
        [bw_test.test_bandwidth_deviation(i, report=True) for i in range(8)],
    )

    print()
    print("=" * 74)
    print("Done.")
    print()


if __name__ == "__main__":
    main()
