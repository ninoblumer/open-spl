"""Unit tests for XL2 file parser — Time section parsing."""
from datetime import datetime, timedelta

import pytest

from util.xl2 import _SectionTime


@pytest.mark.parametrize("file_attr,expected_duration_s", [
    ("log",        30),   # 123_Log
    ("rta_log",    30),   # RTA_3rd_Log
    ("rta_report", 30),   # RTA_3rd_Report
])
def test_time_section_meas_001(meas_001, file_attr, expected_duration_s):
    xl2 = getattr(meas_001, file_attr)
    sec = xl2.sections["Time"]
    assert isinstance(sec, _SectionTime)
    assert sec.start == datetime(2026, 2, 6, 11, 19, 44)
    assert sec.end == datetime(2026, 2, 6, 11, 20, 14)
    assert sec.end - sec.start == timedelta(seconds=expected_duration_s)


@pytest.mark.parametrize("file_attr,expected_duration_s", [
    ("rta_log",    10),   # RTA_Oct_Log
    ("rta_report", 10),   # RTA_Oct_Report
])
def test_time_section_meas_005(meas_005, file_attr, expected_duration_s):
    xl2 = getattr(meas_005, file_attr)
    sec = xl2.sections["Time"]
    assert isinstance(sec, _SectionTime)
    assert sec.start == datetime(2026, 2, 6, 16, 55, 2)
    assert sec.end == datetime(2026, 2, 6, 16, 55, 12)
    assert sec.end - sec.start == timedelta(seconds=expected_duration_s)


def test_123_report_has_no_time_section(meas_001):
    assert "Time" not in meas_001.report.sections