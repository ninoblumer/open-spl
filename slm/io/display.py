"""Console display functions for Reporter callbacks."""
from __future__ import annotations

import shutil
import sys
from datetime import timedelta
from typing import Callable

from slm.io.reporter import _fmt_timestamp


def make_display_fn(mode: str, db_min: float = 40.0, db_max: float = 120.0,
                    precision: int = 1) -> Callable:
    """Return a display callback for Reporter.

    The callback signature is ``fn(timestamp, broadband_row, band_row)`` where:
    - *timestamp* is a :class:`datetime.timedelta`
    - *broadband_row* is ``{label: float}`` (timestamp key excluded)
    - *band_row* is ``{label: np.ndarray}`` (timestamp key excluded)
    """
    if mode == "bars" and sys.stdout.isatty():
        return _BarDisplay(db_min, db_max, precision)
    return _PlainDisplay(precision)


class _PlainDisplay:
    """Scrolling plain-text display (same as Reporter's built-in plain mode)."""

    def __init__(self, precision: int = 1) -> None:
        self._fmt = f"{{:.{precision}f}}"

    def __call__(self, timestamp: timedelta, broadband_row: dict,
                 band_row: dict) -> None:
        ts_str = _fmt_timestamp(timestamp)
        fmt = self._fmt
        if broadband_row:
            parts = [ts_str]
            for label, val in broadband_row.items():
                parts.append(f"{label}: {fmt.format(val)}")
            print("  ".join(parts))
        for label, arr in band_row.items():
            arr_str = "[" + ", ".join(fmt.format(v) for v in arr) + "]"
            print(f"{ts_str}  {label}: {arr_str}")


class _BarDisplay:
    """Live-updating bar-graph console display."""

    _GREEN  = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED    = "\x1b[31m"
    _RESET  = "\x1b[0m"

    def __init__(self, db_min: float = 40.0, db_max: float = 120.0,
                 precision: int = 1, threshold_lo: float = 85.0,
                 threshold_hi: float = 95.0) -> None:
        self._db_min = db_min
        self._db_max = db_max
        self._precision = precision
        self._threshold_lo = threshold_lo
        self._threshold_hi = threshold_hi
        self._lines_printed = 0

    def __call__(self, timestamp: timedelta, broadband_row: dict,
                 band_row: dict) -> None:
        ts_str = _fmt_timestamp(timestamp)
        fmt = f"{{:.{self._precision}f}}"
        cols = shutil.get_terminal_size().columns

        label_w = max((len(k) for k in broadband_row), default=6) + 2
        db_label_w = self._precision + 8   # e.g. "120.0 dB"
        bar_w = max(cols - label_w - db_label_w - 5, 10)

        lines: list[str] = [ts_str]
        for label, val in broadband_row.items():
            clamped = max(self._db_min, min(self._db_max, val))
            fraction = (clamped - self._db_min) / (self._db_max - self._db_min)
            filled = int(fraction * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            if val < self._threshold_lo:
                color = self._GREEN
            elif val < self._threshold_hi:
                color = self._YELLOW
            else:
                color = self._RED
            db_str = fmt.format(val) + " dB"
            lines.append(f"{label:<{label_w}} [{color}{bar}{self._RESET}]  {db_str}")

        # Band rows printed in plain style below the bars (too wide for bars)
        for label, arr in band_row.items():
            arr_str = "[" + ", ".join(fmt.format(v) for v in arr) + "]"
            lines.append(f"{ts_str}  {label}: {arr_str}")

        if self._lines_printed > 0:
            sys.stdout.write(f"\x1b[{self._lines_printed}A")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._lines_printed = len(lines)
