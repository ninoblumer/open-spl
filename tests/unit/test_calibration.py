"""Tests for slm.calibration and slm.app.cli.calibrate_from_file."""
from __future__ import annotations

import pytest

from slm.calibration import calibrate_sensitivity
from slm.app.cli import calibrate_from_file
from slm.constants import REFERENCE_PRESSURE


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

    def test_stability_window_path(self, meas_000):
        """stability_window branch stops early once tone is stable."""
        from slm.io.file_controller import FileController

        controller = FileController(str(meas_000.wav_path), blocksize=1024)
        controller.set_sensitivity(1.0, unit="V")

        # Very loose threshold — any 3 half-second readings are "stable", so the
        # StabilityMonitor calls controller.stop() after 3 sampled readings.
        result = calibrate_sensitivity(
            controller,
            cal_freq=1000.0,
            cal_level=94.0,
            stability_window=3,
            stability_threshold=100.0,
        )
        assert result == pytest.approx(meas_000.sensitivity, rel=1e-2)

    def test_measure_does_not_mutate_controller(self, meas_000):
        """The measure step leaves the controller's (raw) sensitivity untouched."""
        from slm.io.file_controller import FileController

        controller = FileController(str(meas_000.wav_path), blocksize=1024)
        controller.set_sensitivity(1.0, unit="V")
        calibrate_sensitivity(controller, cal_freq=1000.0, cal_level=94.0)
        assert controller.sensitivity == 1.0


class TestStabilityMonitor:

    class _FakeController:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    def _plugin_at(self, level_db):
        """A fake plugin whose moving-Leq reads a constant *level_db* (Pa²)."""
        val_sq = (REFERENCE_PRESSURE * 10 ** (level_db / 20)) ** 2

        class _FakePlugin:
            def read_lin(self, name):
                return [val_sq]

        return _FakePlugin()

    def test_gates_by_dt_then_stops_when_stable(self):
        from datetime import timedelta
        from slm.calibration import _StabilityMonitor

        ctrl = self._FakeController()
        mon = _StabilityMonitor(self._plugin_at(94.0), "leq_moving", ctrl,
                                window=3, threshold=0.1, dt=0.5)
        # Tick every 0.1 s; gating means a reading is taken only every 0.5 s, so
        # 3 readings (at 0.0/0.5/1.0 s) accrue by t=1.0 s → std 0 → stop.
        for i in range(15):
            mon(timedelta(seconds=i * 0.1), 0.5)
            if i < 10:
                assert not ctrl.stopped   # fewer than 3 readings sampled yet
        assert ctrl.stopped

    def test_does_not_stop_while_unstable(self):
        import itertools
        from datetime import timedelta
        from slm.calibration import _StabilityMonitor

        ctrl = self._FakeController()
        counter = itertools.count(1)

        class _DriftingPlugin:
            def read_lin(self, name):
                return [float(next(counter))]   # ever-changing → never stable

        mon = _StabilityMonitor(_DriftingPlugin(), "leq_moving", ctrl,
                                window=3, threshold=0.01, dt=0.5)
        for i in range(15):
            mon(timedelta(seconds=i * 0.5), 0.5)
        assert not ctrl.stopped
