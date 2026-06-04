"""Unit tests for NoiseController."""
from __future__ import annotations

import time

import numpy as np
import pytest

from slm.io.noise_controller import NoiseController


def _make_controller(**kwargs) -> NoiseController:
    kwargs.setdefault("samplerate", 48_000)
    kwargs.setdefault("blocksize", 1_024)
    kwargs.setdefault("channels", 1)
    return NoiseController(**kwargs)


class TestNoiseControllerInterface:

    def test_properties(self):
        with pytest.warns(UserWarning, match="channel 0"):
            ctrl = _make_controller(samplerate=44_100, blocksize=512, channels=2)
        assert ctrl.samplerate == 44_100
        assert ctrl.blocksize == 512
        assert ctrl.sensitivity == 1.0

    def test_list_devices_empty(self):
        assert NoiseController.list_devices() == []

    def test_overruns_initially_zero(self):
        assert _make_controller().overruns == 0

    def test_rho_mean_none_before_blocks(self):
        assert _make_controller().rho_mean is None

    def test_queue_depth_max_zero_initially(self):
        assert _make_controller().queue_depth_max == 0

    def test_calibrate_raises(self):
        with pytest.raises(NotImplementedError):
            _make_controller().calibrate()

    def test_set_sensitivity(self):
        ctrl = _make_controller()
        ctrl.set_sensitivity(50.0, unit="mV")
        assert ctrl.sensitivity == pytest.approx(0.05)


class TestReadBlock:

    def _read_n(self, ctrl: NoiseController, n: int):
        ctrl.start()
        blocks, indices = [], []
        try:
            for _ in range(n):
                block, idx = ctrl.read_block()
                blocks.append(block)
                indices.append(idx)
        finally:
            ctrl.stop()
        return blocks, indices

    def test_block_shape(self):
        ctrl = _make_controller(blocksize=256, channels=1)
        blocks, _ = self._read_n(ctrl, 3)
        for b in blocks:
            assert b.shape == (256, 1)

    def test_block_shape_stereo_clamped_to_mono(self):
        """channels=2 is clamped to 1 with a warning; blocks are always mono."""
        with pytest.warns(UserWarning, match="channel 0"):
            ctrl = _make_controller(blocksize=128, channels=2)
        blocks, _ = self._read_n(ctrl, 3)
        for b in blocks:
            assert b.shape == (128, 1)

    def test_block_indices_sequential(self):
        ctrl = _make_controller()
        _, indices = self._read_n(ctrl, 5)
        assert indices == list(range(5))

    def test_block_dtype(self):
        ctrl = _make_controller(blocksize=64)
        blocks, _ = self._read_n(ctrl, 2)
        for b in blocks:
            assert b.dtype == np.float32

    def test_stop_raises_stop_iteration(self):
        ctrl = _make_controller()
        ctrl.stop()  # sets stop event; no producer running so queue stays empty
        with pytest.raises(StopIteration):
            ctrl.read_block()

    def test_overrun_on_full_queue(self):
        ctrl = _make_controller(blocksize=64, queue_maxsize=2)
        # Force a full-queue drop by calling _produce logic directly
        block = np.zeros((64, 1), dtype=np.float32)
        ctrl._queue.put_nowait(block)
        ctrl._queue.put_nowait(block)
        import queue as _q
        try:
            ctrl._queue.put_nowait(block)
        except _q.Full:
            ctrl._overruns += 1
        assert ctrl.overruns == 1


class TestLoadMonitoring:

    def test_rho_mean_populated_after_blocks(self):
        ctrl = _make_controller(blocksize=1_024, queue_maxsize=16)
        ctrl.start()
        try:
            for _ in range(10):
                ctrl.read_block()
        finally:
            ctrl.stop()
        assert ctrl.rho_mean is not None
        assert ctrl.rho_mean >= 0.0

    def test_rho_mean_resets_on_restart(self):
        ctrl = _make_controller(blocksize=1_024, queue_maxsize=16)
        ctrl.start()
        for _ in range(5):
            ctrl.read_block()
        ctrl.stop()
        assert ctrl.rho_mean is not None
        # restart clears the buffer
        ctrl.start()
        ctrl.stop()
        assert ctrl.rho_mean is None

    def test_queue_depth_max_non_negative(self):
        ctrl = _make_controller(blocksize=1_024, queue_maxsize=16)
        ctrl.start()
        try:
            for _ in range(10):
                ctrl.read_block()
        finally:
            ctrl.stop()
        assert ctrl.queue_depth_max >= 0


class TestNonRealtimeMode:

    def test_non_realtime_produces_blocks(self):
        """realtime=False path: producer blocks on full queue; consumer drives pace."""
        ctrl = _make_controller(blocksize=256, realtime=False, queue_maxsize=4)
        ctrl.start()
        try:
            block, idx = ctrl.read_block()
            assert block.shape == (256, 1)
            assert idx == 0
        finally:
            ctrl.stop()

    def test_realtime_overrun_via_full_queue(self):
        """realtime=True: overruns increment when queue fills and producer can't put."""
        import time
        ctrl = _make_controller(blocksize=64, samplerate=48_000,
                                realtime=True, queue_maxsize=1)
        ctrl.start()
        time.sleep(0.05)  # 50ms >> several 1.3ms block periods; queue fills quickly
        ctrl.stop()
        assert ctrl.overruns > 0


class TestEngineIntegration:

    def test_engine_processes_blocks(self):
        from slm.engine import Engine
        from slm.assembly import parse_metric, build_chain

        ctrl = _make_controller(samplerate=48_000, blocksize=1_024, queue_maxsize=16)
        ctrl.set_sensitivity(1.0, unit="V")
        ctrl.start()

        engine = Engine(ctrl, dt=0.5)
        build_chain([parse_metric("LAeq")], engine)

        # Read 20 blocks then stop
        for _ in range(20):
            engine._process_block()
        ctrl.stop()

        assert ctrl.rho_mean is not None
        assert ctrl.rho_mean >= 0.0
