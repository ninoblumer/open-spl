"""Unit tests for slm/reporter.py."""
import csv
import io
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from slm.reporter import Reporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plugin(width: int, db_values: np.ndarray):
    """Minimal PluginMeter stub: fixed read_db return value."""
    return types.SimpleNamespace(
        width=width,
        read_db=lambda name: db_values.copy(),
    )


def _td(seconds: float) -> timedelta:
    return timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# _fmt_timestamp
# ---------------------------------------------------------------------------

class TestFmtTimestamp:

    def test_zero(self):
        r = Reporter()
        assert r._fmt_timestamp(_td(0)) == "00:00:00.000"

    def test_sub_minute(self):
        r = Reporter()
        assert r._fmt_timestamp(_td(5.5)) == "00:00:05.500"

    def test_over_one_minute(self):
        r = Reporter()
        assert r._fmt_timestamp(_td(75.25)) == "00:01:15.250"

    def test_over_one_hour(self):
        r = Reporter()
        assert r._fmt_timestamp(_td(3661.0)) == "01:01:01.000"


# ---------------------------------------------------------------------------
# add_column routing
# ---------------------------------------------------------------------------

class TestAddColumn:

    def test_width1_goes_to_broadband(self):
        r = Reporter()
        p = _plugin(1, np.array([1.0]))
        r.add_column("LAF", p, "LAF")
        assert len(r._broadband_columns) == 1
        assert len(r._band_columns) == 0

    def test_widthN_goes_to_band(self):
        r = Reporter()
        p = _plugin(3, np.array([1.0, 2.0, 3.0]))
        r.add_column("LZeq", p, "LZeq", center_frequencies=["63", "125", "250"])
        assert len(r._band_columns) == 1
        assert len(r._broadband_columns) == 0

    def test_widthN_without_frequencies_raises(self):
        r = Reporter()
        p = _plugin(3, np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="center_frequencies"):
            r.add_column("LZeq", p, "LZeq")

    def test_width1_ignores_center_frequencies(self):
        """Passing center_frequencies for a width=1 plugin is silently ignored (goes broadband)."""
        r = Reporter()
        p = _plugin(1, np.array([42.0]))
        r.add_column("LAF", p, "LAF", center_frequencies=[1000.0])
        assert len(r._broadband_columns) == 1
        assert len(r._band_columns) == 0


# ---------------------------------------------------------------------------
# record() throttling
# ---------------------------------------------------------------------------

class TestRecordThrottling:

    def test_first_call_records_when_elapsed(self):
        r = Reporter()
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)  # 1.0 - 0.0 = 1.0, not < 1.0 → records
        assert len(r._broadband_rows) == 1

    def test_first_call_skipped_when_below_dt(self):
        r = Reporter()
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(0.5), dt=1.0)  # 0.5 < 1.0 → skipped
        assert len(r._broadband_rows) == 0

    def test_second_call_before_dt_skipped(self):
        r = Reporter()
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        r.record(_td(1.5), dt=1.0)  # 0.5 elapsed since last → skipped
        assert len(r._broadband_rows) == 1

    def test_second_call_after_dt_recorded(self):
        r = Reporter()
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        r.record(_td(2.0), dt=1.0)  # 1.0 elapsed → records
        assert len(r._broadband_rows) == 2


# ---------------------------------------------------------------------------
# record() row content
# ---------------------------------------------------------------------------

class TestRecordContent:

    def test_broadband_row_has_scalar(self):
        r = Reporter()
        p = _plugin(1, np.array([94.3]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        row = r._broadband_rows[0]
        assert row["timestamp"] == _td(1.0)
        assert isinstance(row["LAF"], float)
        assert row["LAF"] == pytest.approx(94.3)

    def test_band_row_has_array(self):
        r = Reporter()
        vals = np.array([72.1, 81.4, 88.2])
        p = _plugin(3, vals)
        r.add_column("LZeq", p, "LZeq", center_frequencies=["63", "125", "250"])
        r.record(_td(1.0), dt=1.0)
        row = r._band_rows[0]
        np.testing.assert_array_equal(row["LZeq"], vals)

    def test_band_row_is_copy(self):
        """Stored array must be independent of future plugin output changes."""
        vals = np.array([72.1, 81.4, 88.2])
        r = Reporter()
        p = _plugin(3, vals)
        r.add_column("LZeq", p, "LZeq", center_frequencies=["63", "125", "250"])
        r.record(_td(1.0), dt=1.0)
        stored = r._band_rows[0]["LZeq"]
        vals[:] = 0.0  # mutate original
        assert stored[0] == pytest.approx(72.1)

    def test_multiple_broadband_columns(self):
        r = Reporter()
        p1 = _plugin(1, np.array([94.0]))
        p2 = _plugin(1, np.array([95.0]))
        r.add_column("LAF", p1, "LAF")
        r.add_column("LAFmax", p2, "LAFmax")
        r.record(_td(1.0), dt=1.0)
        row = r._broadband_rows[0]
        assert row["LAF"] == pytest.approx(94.0)
        assert row["LAFmax"] == pytest.approx(95.0)

    def test_broadband_and_band_rows_always_paired(self):
        """Both row lists grow together even when only one type is added."""
        r = Reporter()
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        r.record(_td(2.0), dt=1.0)
        assert len(r._broadband_rows) == len(r._band_rows) == 2


# ---------------------------------------------------------------------------
# write() — broadband CSVs
# ---------------------------------------------------------------------------

class TestWriteBroadband:

    def _make_reporter(self, tmp_path) -> tuple[Reporter, Path]:
        r = Reporter(precision=1)
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        r.record(_td(2.0), dt=1.0)
        base = tmp_path / "out"
        r.write(base)
        return r, base

    def test_log_csv_row_count(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_log.csv"))))
        assert len(rows) == 2

    def test_log_csv_has_timestamp(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_log.csv"))))
        assert rows[0]["timestamp"] == "00:00:01.000"

    def test_report_csv_single_row(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_report.csv"))))
        assert len(rows) == 1

    def test_report_csv_no_timestamp(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_report.csv"))))
        assert "timestamp" not in rows[0]

    def test_report_csv_last_row_values(self, tmp_path):
        r = Reporter(precision=1)
        values = [94.0, 95.0]
        call_count = [0]

        def read_db(name):
            v = values[call_count[0] % 2]
            call_count[0] += 1
            return np.array([v])

        p = types.SimpleNamespace(width=1, read_db=read_db)
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        r.record(_td(2.0), dt=1.0)
        base = tmp_path / "out"
        r.write(base)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_report.csv"))))
        assert rows[0]["LAF"] == "95.0"


# ---------------------------------------------------------------------------
# write() — RTA CSVs
# ---------------------------------------------------------------------------

class TestWriteRTA:

    def _make_reporter(self, tmp_path) -> tuple[Reporter, Path]:
        r = Reporter(precision=1)
        freqs = ["63", "125", "250"]
        p = _plugin(3, np.array([72.1, 81.4, 88.2]))
        r.add_column("LZeq", p, "LZeq", center_frequencies=freqs)
        r.record(_td(1.0), dt=1.0)
        r.record(_td(2.0), dt=1.0)
        base = tmp_path / "out"
        r.write(base)
        return r, base

    def test_rta_log_exists(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        assert (base.parent / (base.name + "_rta_log.csv")).exists()

    def test_rta_report_exists(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        assert (base.parent / (base.name + "_rta_report.csv")).exists()

    def test_rta_log_headers(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_rta_log.csv"))))
        assert list(rows[0].keys()) == ["timestamp", "LZeq_63", "LZeq_125", "LZeq_250"]  # strings used as-is

    def test_rta_log_row_count(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_rta_log.csv"))))
        assert len(rows) == 2

    def test_rta_log_values(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_rta_log.csv"))))
        assert rows[0]["LZeq_63"] == "72.1"
        assert rows[0]["LZeq_125"] == "81.4"
        assert rows[0]["LZeq_250"] == "88.2"

    def test_rta_report_single_row_no_timestamp(self, tmp_path):
        _, base = self._make_reporter(tmp_path)
        rows = list(csv.DictReader(open(base.parent / (base.name + "_rta_report.csv"))))
        assert len(rows) == 1
        assert "timestamp" not in rows[0]

    def test_no_rta_files_when_no_band_columns(self, tmp_path):
        r = Reporter()
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        base = tmp_path / "out"
        r.write(base)
        assert not (base.parent / (base.name + "_rta_log.csv")).exists()
        assert not (base.parent / (base.name + "_rta_report.csv")).exists()


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------

class TestConsoleOutput:

    def test_broadband_printed(self, capsys):
        r = Reporter(precision=1, print_to_console=True)
        p = _plugin(1, np.array([94.3]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        out = capsys.readouterr().out
        assert "LAF: 94.3" in out
        assert "00:00:01.000" in out

    def test_band_printed(self, capsys):
        r = Reporter(precision=1, print_to_console=True)
        p = _plugin(3, np.array([72.1, 81.4, 88.2]))
        r.add_column("LZeq", p, "LZeq", center_frequencies=["63", "125", "250"])
        r.record(_td(1.0), dt=1.0)
        out = capsys.readouterr().out
        assert "LZeq:" in out
        assert "72.1" in out

    def test_no_output_when_disabled(self, capsys):
        r = Reporter(print_to_console=False)
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(1.0), dt=1.0)
        out = capsys.readouterr().out
        assert out == ""

    def test_throttled_record_produces_no_output(self, capsys):
        r = Reporter(print_to_console=True)
        p = _plugin(1, np.array([94.0]))
        r.add_column("LAF", p, "LAF")
        r.record(_td(0.1), dt=1.0)
        capsys.readouterr()  # clear first record output
        r.record(_td(0.5), dt=1.0)  # too soon — should be skipped
        out = capsys.readouterr().out
        assert out == ""
