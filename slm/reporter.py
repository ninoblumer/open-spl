from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from slm.plugin_meter import PluginMeter


class Reporter:
    def __init__(self, precision: int = 1, print_to_console: bool = False):
        self._broadband_columns: list[tuple[str, PluginMeter, str]] = []
        self._band_columns: list[tuple[str, PluginMeter, str, list[float]]] = []
        self._broadband_rows: list[dict] = []
        self._band_rows: list[dict] = []
        self._last_log: timedelta = timedelta(0)
        self._precision = precision
        self._print_to_console = print_to_console

    def _fmt_timestamp(self, td: timedelta) -> str:
        total = td.total_seconds()
        h = int(total) // 3600
        m = (int(total) % 3600) // 60
        s = total % 60
        return "{:02}:{:02}:{:06.3f}".format(h, m, s)

    def add_column(self, label: str, plugin: PluginMeter, meter_name: str,
                   center_frequencies: list[float] | None = None) -> None:
        """Register a meter output as a column.

        Single-channel plugins go to broadband; multi-channel plugins go to band-split.
        For multi-channel plugins, center_frequencies is required.
        """
        if plugin.width == 1:
            self._broadband_columns.append((label, plugin, meter_name))
        else:
            if center_frequencies is None:
                raise ValueError(
                    f"center_frequencies is required for multi-channel plugin '{label}' (width={plugin.width})"
                )
            self._band_columns.append((label, plugin, meter_name, center_frequencies))

    def record(self, timestamp: timedelta, dt: float) -> None:
        """Sample all registered meters and append rows if dt has elapsed since last log."""
        if (timestamp - self._last_log).total_seconds() < dt:
            return

        fmt = f"{{:.{self._precision}f}}"
        ts_str = self._fmt_timestamp(timestamp)

        broadband_row: dict = {"timestamp": timestamp}
        for label, plugin, meter_name in self._broadband_columns:
            broadband_row[label] = float(plugin.read_db(meter_name)[0])
        self._broadband_rows.append(broadband_row)

        band_row: dict = {"timestamp": timestamp}
        for label, plugin, meter_name, _ in self._band_columns:
            band_row[label] = plugin.read_db(meter_name).copy()
        self._band_rows.append(band_row)

        if self._print_to_console:
            if self._broadband_columns:
                parts = [ts_str]
                for label, _, _ in self._broadband_columns:
                    parts.append(f"{label}: {fmt.format(broadband_row[label])}")
                print("  ".join(parts))
            for label, _, _, _ in self._band_columns:
                arr = band_row[label]
                arr_str = "[" + ", ".join(fmt.format(v) for v in arr) + "]"
                print(f"{ts_str}  {label}: {arr_str}")

        self._last_log = timestamp

    def write(self, path: str | Path) -> None:
        """Write _log.csv, _report.csv, and optionally _rta_log.csv, _rta_report.csv."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fmt = f"{{:.{self._precision}f}}"

        def _format_value(v) -> str:
            if isinstance(v, timedelta):
                return self._fmt_timestamp(v)
            return fmt.format(v)

        # --- Broadband ---
        if self._broadband_rows:
            fieldnames = list(self._broadband_rows[0].keys())

            log_path = path.parent / (path.name + "_log.csv")
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._broadband_rows:
                    writer.writerow({k: _format_value(v) for k, v in row.items()})

            report_fieldnames = [k for k in fieldnames if k != "timestamp"]
            last_row = {k: v for k, v in self._broadband_rows[-1].items() if k != "timestamp"}
            report_path = path.parent / (path.name + "_report.csv")
            with open(report_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=report_fieldnames)
                writer.writeheader()
                writer.writerow({k: _format_value(v) for k, v in last_row.items()})

        # --- Band-split (RTA) ---
        if self._band_columns and self._band_rows:
            # Build flat fieldnames: timestamp + label_freq per band column
            rta_fieldnames = ["timestamp"]
            for label, _, _, freqs in self._band_columns:
                for freq in freqs:
                    rta_fieldnames.append(f"{label}_{freq:.0f}")

            rta_log_path = path.parent / (path.name + "_rta_log.csv")
            with open(rta_log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rta_fieldnames)
                writer.writeheader()
                for row in self._band_rows:
                    flat: dict = {"timestamp": _format_value(row["timestamp"])}
                    for label, _, _, freqs in self._band_columns:
                        arr = row[label]
                        for freq, val in zip(freqs, arr):
                            flat[f"{label}_{freq:.0f}"] = fmt.format(val)
                    writer.writerow(flat)

            rta_report_fieldnames = [f for f in rta_fieldnames if f != "timestamp"]
            last_band_row = self._band_rows[-1]
            rta_report_path = path.parent / (path.name + "_rta_report.csv")
            with open(rta_report_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rta_report_fieldnames)
                writer.writeheader()
                flat_last: dict = {}
                for label, _, _, freqs in self._band_columns:
                    arr = last_band_row[label]
                    for freq, val in zip(freqs, arr):
                        flat_last[f"{label}_{freq:.0f}"] = fmt.format(val)
                writer.writerow(flat_last)
