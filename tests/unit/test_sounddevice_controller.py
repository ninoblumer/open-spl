"""Unit tests for SounddeviceController using a mocked sounddevice stream."""
from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("sounddevice", reason="sounddevice not installed — skipping real-time audio tests")
from slm.io.sounddevice_controller import SounddeviceController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(**kwargs) -> SounddeviceController:
    kwargs.setdefault("samplerate", 48_000)
    kwargs.setdefault("blocksize", 1_024)
    kwargs.setdefault("channels", 1)
    return SounddeviceController(**kwargs)


class _FakeStream:
    """Minimal stand-in for sd.InputStream."""

    def __init__(self, callback, n_blocks: int, blocksize: int, channels: int,
                 on_done=None):
        self._callback = callback
        self._n_blocks = n_blocks
        self._blocksize = blocksize
        self._channels = channels
        self._on_done = on_done   # called after all blocks are delivered
        self._thread: threading.Thread | None = None
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        self._thread = threading.Thread(target=self._deliver, daemon=True)
        self._thread.start()

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def _deliver(self):
        """Push *n_blocks* blocks then invoke on_done (if provided)."""
        rng = np.random.default_rng(0)
        for _ in range(self._n_blocks):
            block = rng.standard_normal((self._blocksize, self._channels)).astype(np.float32)
            self._callback(block, self._blocksize, None, None)
            time.sleep(0)   # yield to main thread
        if self._on_done is not None:
            self._on_done()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSounddeviceControllerInterface:

    def test_properties(self):
        with pytest.warns(UserWarning, match="channel 0"):
            ctrl = _make_controller(samplerate=44_100, blocksize=512, channels=2)
        assert ctrl.samplerate == 44_100
        assert ctrl.blocksize == 512
        assert ctrl.sensitivity == 1.0

    def test_set_sensitivity_v(self):
        ctrl = _make_controller()
        ctrl.set_sensitivity(0.05, unit="V")
        assert ctrl.sensitivity == pytest.approx(0.05)

    def test_set_sensitivity_mv(self):
        ctrl = _make_controller()
        ctrl.set_sensitivity(50.0, unit="mV")
        assert ctrl.sensitivity == pytest.approx(0.05)

    def test_overruns_initially_zero(self):
        ctrl = _make_controller()
        assert ctrl.overruns == 0

    def test_rho_mean_none_before_blocks(self):
        ctrl = _make_controller()
        assert ctrl.rho_mean is None

    def test_queue_depth_max_zero_initially(self):
        ctrl = _make_controller()
        assert ctrl.queue_depth_max == 0


class TestReadBlock:

    def _run_with_fake_stream(self, n_blocks: int, blocksize: int = 1_024,
                               channels: int = 1):
        """Start a controller backed by _FakeStream and read all blocks."""
        ctrl = _make_controller(blocksize=blocksize, channels=channels,
                                queue_maxsize=n_blocks + 4)
        ctrl.set_sensitivity(1.0, unit="V")

        fake = _FakeStream(ctrl._callback, n_blocks, blocksize, ctrl._channels,
                           on_done=ctrl.stop)

        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake):
            ctrl.start()
            blocks = []
            indices = []
            try:
                while True:
                    block, idx = ctrl.read_block()
                    blocks.append(block)
                    indices.append(idx)
            except StopIteration:
                pass

        return ctrl, blocks, indices

    def test_block_count(self):
        _, blocks, _ = self._run_with_fake_stream(n_blocks=10)
        assert len(blocks) == 10

    def test_block_shape(self):
        blocksize, channels = 512, 1
        _, blocks, _ = self._run_with_fake_stream(n_blocks=5, blocksize=blocksize,
                                                   channels=channels)
        for b in blocks:
            assert b.shape == (blocksize, channels)

    def test_block_shape_stereo_clamped_to_mono(self):
        """channels=2 is clamped to 1 with a warning; blocks are always mono."""
        with pytest.warns(UserWarning, match="channel 0"):
            _, blocks, _ = self._run_with_fake_stream(n_blocks=3, blocksize=256, channels=2)
        for b in blocks:
            assert b.shape == (256, 1)

    def test_block_indices_sequential(self):
        _, _, indices = self._run_with_fake_stream(n_blocks=8)
        assert indices == list(range(8))

    def test_stop_raises_stop_iteration(self):
        ctrl = _make_controller(queue_maxsize=4)
        fake = _FakeStream(ctrl._callback, n_blocks=0, blocksize=1_024, channels=1)

        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake):
            ctrl.start()
            ctrl.stop()
            with pytest.raises(StopIteration):
                ctrl.read_block()

    def test_callback_copies_buffer(self):
        ctrl = _make_controller(blocksize=64, queue_maxsize=16)
        original = np.ones((64, 1), dtype=np.float32)
        ctrl._callback(original, 64, None, None)
        original[:] = 0.0
        queued = ctrl._queue.get_nowait()
        assert np.all(queued == 1.0)

    def test_overrun_on_full_queue(self):
        ctrl = _make_controller(blocksize=64, queue_maxsize=2)
        block = np.zeros((64, 1), dtype=np.float32)
        ctrl._callback(block, 64, None, None)
        ctrl._callback(block, 64, None, None)
        ctrl._callback(block, 64, None, None)
        assert ctrl.overruns == 1

    def test_overrun_on_callback_status(self):
        ctrl = _make_controller(blocksize=64, queue_maxsize=8)
        block = np.zeros((64, 1), dtype=np.float32)
        status = MagicMock()
        status.__bool__ = lambda s: True
        ctrl._callback(block, 64, None, status)
        assert ctrl.overruns == 1


class TestLoadMonitoring:

    def test_rho_mean_populated_after_blocks(self):
        ctrl = _make_controller(blocksize=1_024, queue_maxsize=20)
        fake = _FakeStream(ctrl._callback, n_blocks=10, blocksize=1_024, channels=1,
                           on_done=ctrl.stop)
        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake):
            ctrl.start()
            try:
                while True:
                    ctrl.read_block()
            except StopIteration:
                pass
        assert ctrl.rho_mean is not None
        assert ctrl.rho_mean >= 0.0

    def test_rho_mean_resets_on_restart(self):
        ctrl = _make_controller(blocksize=1_024, queue_maxsize=20)
        fake = _FakeStream(ctrl._callback, n_blocks=5, blocksize=1_024, channels=1,
                           on_done=ctrl.stop)
        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake):
            ctrl.start()
            try:
                while True:
                    ctrl.read_block()
            except StopIteration:
                pass
        assert ctrl.rho_mean is not None
        # start() resets monitoring state
        fake2 = _FakeStream(ctrl._callback, n_blocks=0, blocksize=1_024, channels=1)
        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake2):
            ctrl.start()
            ctrl.stop()
        assert ctrl.rho_mean is None

    def test_queue_depth_max_non_negative(self):
        ctrl = _make_controller(blocksize=1_024, queue_maxsize=20)
        fake = _FakeStream(ctrl._callback, n_blocks=10, blocksize=1_024, channels=1,
                           on_done=ctrl.stop)
        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake):
            ctrl.start()
            try:
                while True:
                    ctrl.read_block()
            except StopIteration:
                pass
        assert ctrl.queue_depth_max >= 0


class TestListDevices:

    def test_returns_list_of_dicts(self):
        fake_devices = [
            {"name": "Mic A", "max_input_channels": 1, "default_samplerate": 48_000.0,
             "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48_000.0,
             "max_output_channels": 2},
            {"name": "Mic B", "max_input_channels": 2, "default_samplerate": 44_100.0,
             "max_output_channels": 0},
        ]
        with patch("slm.io.sounddevice_controller.sd.query_devices",
                   return_value=fake_devices):
            result = SounddeviceController.list_devices()

        assert len(result) == 2
        assert result[0]["name"] == "Mic A"
        assert result[1]["name"] == "Mic B"

    def test_result_keys(self):
        fake_devices = [
            {"name": "X", "max_input_channels": 1, "default_samplerate": 48_000.0,
             "max_output_channels": 0},
        ]
        with patch("slm.io.sounddevice_controller.sd.query_devices",
                   return_value=fake_devices):
            result = SounddeviceController.list_devices()

        assert set(result[0].keys()) == {"index", "name", "max_input_channels",
                                          "default_samplerate"}


class TestEngineIntegration:

    def test_engine_processes_blocks(self):
        from slm.engine import Engine
        from slm.assembly import parse_metric, build_chain
        from slm.io.reporter import Reporter

        n_blocks = 20
        blocksize = 1_024
        samplerate = 48_000

        ctrl = _make_controller(samplerate=samplerate, blocksize=blocksize,
                                queue_maxsize=32)
        ctrl.set_sensitivity(1.0, unit="V")

        fake = _FakeStream(ctrl._callback, n_blocks, blocksize, channels=1,
                           on_done=ctrl.stop)

        with patch("slm.io.sounddevice_controller.sd.InputStream",
                   return_value=fake):
            ctrl.start()

            reporter = Reporter(precision=2)
            engine = Engine(ctrl, dt=0.1, reporter=reporter)
            build_chain([parse_metric("LAeq")], engine)

            engine.run()

        assert len(reporter._broadband_rows) >= 1
        last = reporter._broadband_rows[-1]["LAeq"]
        assert np.isfinite(last)
