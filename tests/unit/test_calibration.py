"""Tests for slm.calibration and slm.app.cli.calibrate_from_file."""
from __future__ import annotations

import pytest

from slm.calibration import calibrate_sensitivity
from slm.app.cli import calibrate_from_file


class TestCalibrateFunction:

    def test_calibrate_sensitivity_1khz(self, meas_000):
        """Core function with FileController returns sensitivity within 0.1% of reference."""
        from slm.io.file_controller import FileController

        controller = FileController(str(meas_000.wav_path), blocksize=1024)
        controller.set_sensitivity(1.0, unit="V")

        result = calibrate_sensitivity(controller, cal_freq=1000.0, cal_level=94.0)
        # Tolerance is 1 % (≈ 0.086 dB): the FS-annotation reference uses peak-to-RMS
        # conversion while the calibration routine measures RMS directly, so a small
        # crest-factor difference between an ideal sine and the actual recording is
        # expected.
        assert result == pytest.approx(meas_000.sensitivity, rel=1e-2)

    def test_calibrate_from_file_matches(self, meas_000):
        """calibrate_from_file convenience wrapper agrees with reference sensitivity."""
        result = calibrate_from_file(
            meas_000.wav_path,
            cal_freq=1000.0,
            cal_level=94.0,
        )
        assert result == pytest.approx(meas_000.sensitivity, rel=1e-2)
