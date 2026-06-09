"""Unit tests for FileController."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _write_sine(path, freq=1000.0, duration=1.0, samplerate=48000, amplitude=0.5):
    t = np.arange(int(duration * samplerate)) / samplerate
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, samplerate)


# ---------------------------------------------------------------------------
# Basic I/O and error paths
# ---------------------------------------------------------------------------

class TestFileControllerBasics:

    def test_path_object_accepted(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(wav, blocksize=1024)  # Path, not str
        block, idx = ctrl.read_block()
        assert block.shape[1] == 1

    def test_open_on_non_done_raises(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        with pytest.raises(RuntimeError, match="not been finished"):
            ctrl.open(str(wav), blocksize=1024)

    def test_calibrate_raises_not_implemented(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        with pytest.raises(NotImplementedError):
            ctrl.calibrate()

    def test_stop_sets_done(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        assert not ctrl.done
        ctrl.stop()
        assert ctrl.done

    def test_pull_source_telemetry_defaults(self, tmp_path):
        """A pull-based source uses the Controller base defaults: no dropped
        blocks and no live load telemetry."""
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        assert ctrl.overruns == 0
        assert ctrl.load_status() is None

    def test_context_manager(self, tmp_path):
        """Controller.__enter__/__exit__ start the (no-op) source and stop it."""
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        with FileController(str(wav), blocksize=1024) as ctrl:
            block, _ = ctrl.read_block()
            assert block.shape[1] == 1
        assert ctrl.done

    def test_overruns_property_zero(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        assert ctrl.overruns == 0

    def test_uniform_interface_defaults(self, tmp_path):
        """A pull-based source uses the inert uniform-interface defaults."""
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        assert ctrl.load_status() is None    # no live telemetry
        ctrl.start()                         # no-op — must not raise

    def test_context_manager_starts_and_stops(self, tmp_path):
        """``with controller:`` starts on enter and stops (closes file) on exit."""
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        with ctrl as c:
            assert c is ctrl
            assert not ctrl.done
        assert ctrl.done                     # __exit__ called stop()


# ---------------------------------------------------------------------------
# Controller.set_sensitivity unit paths
# ---------------------------------------------------------------------------

class TestSetSensitivityUnits:

    def test_db_unit(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        ctrl.set_sensitivity(0.0, unit="dB")  # 0 dBV re 1 V/Pa → 1.0 V/Pa
        assert ctrl.sensitivity == pytest.approx(1.0)

    def test_unknown_unit_raises(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=1024)
        with pytest.raises(ValueError, match="Unknown sensitivity unit"):
            ctrl.set_sensitivity(1.0, unit="Pa")


# ---------------------------------------------------------------------------
# Realtime mode
# ---------------------------------------------------------------------------

class TestFileControllerRealtime:

    def test_realtime_blocks_readable(self, tmp_path):
        wav = tmp_path / "sine.wav"
        _write_sine(wav, duration=0.5)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=128, realtime=True)
        ctrl.set_sensitivity(1.0, unit="V")
        # Read two blocks — first sets _next_block_time, second may sleep
        b0, _ = ctrl.read_block()
        b1, _ = ctrl.read_block()
        assert b0.shape[1] == 1
        assert b1.shape[1] == 1

    def test_realtime_first_block_increments_overruns(self, tmp_path):
        """The first block sets _next_block_time=now, so sleep_for=0 → overruns+=1."""
        wav = tmp_path / "sine.wav"
        _write_sine(wav, duration=1.0)
        from slm.io.file_controller import FileController
        ctrl = FileController(str(wav), blocksize=4096, realtime=True)
        ctrl.set_sensitivity(1.0, unit="V")
        ctrl.read_block()
        assert ctrl.overruns >= 1
