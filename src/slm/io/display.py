"""Console display functions for Reporter callbacks."""
from __future__ import annotations

import shutil
import sys
from datetime import timedelta
from typing import TYPE_CHECKING, Callable

from slm.io.reporter import _fmt_timestamp

if TYPE_CHECKING:
    from slm.io.controller import Controller


def make_display_fn(mode: str, db_min: float = 40.0, db_max: float = 120.0,
                    precision: int = 1,
                    controller: "Controller | None" = None) -> Callable:
    """Return a display callback for Reporter.

    The callback signature is ``fn(timestamp, broadband_row, band_row)`` where:
    - *timestamp* is a :class:`datetime.timedelta`
    - *broadband_row* is ``{label: float}`` (timestamp key excluded)
    - *band_row* is ``{label: np.ndarray}`` (timestamp key excluded)

    If *controller* is provided, a status line showing rolling process load (ρ)
    and queue depth is appended to each output line/frame.
    """
    if mode == "bars" and sys.stdout.isatty():
        return _BarDisplay(db_min, db_max, precision, controller=controller)
    return _PlainDisplay(precision, controller=controller)


def _fmt_status(controller: "Controller | None") -> str:
    """Return the controller's live load/telemetry line, or '' if it has none."""
    if controller is None:
        return ""
    return controller.load_status() or ""


class _PlainDisplay:
    """Scrolling plain-text display."""

    def __init__(self, precision: int = 1,
                 controller: "Controller | None" = None) -> None:
        self._fmt = f"{{:.{precision}f}}"
        self._controller = controller

    def __call__(self, timestamp: timedelta, broadband_row: dict,
                 band_row: dict) -> None:
        ts_str = _fmt_timestamp(timestamp)
        fmt = self._fmt
        status = _fmt_status(self._controller)

        if broadband_row:
            parts = [ts_str]
            for label, val in broadband_row.items():
                parts.append(f"{label}: {fmt.format(val)}")
            if status:
                parts.append(f"[{status}]")
            print("  ".join(parts))

        for label, arr in band_row.items():
            arr_str = "[" + ", ".join(fmt.format(v) for v in arr) + "]"
            line = f"{ts_str}  {label}: {arr_str}"
            if status and not broadband_row:
                line += f"  [{status}]"
            print(line)


class _BarDisplay:
    """Live-updating bar-graph console display."""

    _GREEN  = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED    = "\x1b[31m"
    _RESET  = "\x1b[0m"
    _DIM    = "\x1b[2m"

    def __init__(self, db_min: float = 40.0, db_max: float = 120.0,
                 precision: int = 1, threshold_lo: float = 85.0,
                 threshold_hi: float = 95.0,
                 controller: "Controller | None" = None) -> None:
        self._db_min = db_min
        self._db_max = db_max
        self._precision = precision
        self._threshold_lo = threshold_lo
        self._threshold_hi = threshold_hi
        self._controller = controller
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
            bar = "█" * filled + "░" * (bar_w - filled)
            if val < self._threshold_lo:
                color = self._GREEN
            elif val < self._threshold_hi:
                color = self._YELLOW
            else:
                color = self._RED
            db_str = fmt.format(val) + " dB"
            lines.append(f"{label:<{label_w}} [{color}{bar}{self._RESET}]  {db_str}")

        # Band rows in plain style (too wide for bars)
        for label, arr in band_row.items():
            arr_str = "[" + ", ".join(fmt.format(v) for v in arr) + "]"
            lines.append(f"{ts_str}  {label}: {arr_str}")

        # Status line
        status = _fmt_status(self._controller)
        if status:
            lines.append(f"{self._DIM}{status}{self._RESET}")

        if self._lines_printed > 0:
            sys.stdout.write(f"\x1b[{self._lines_printed}A")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._lines_printed = len(lines)
