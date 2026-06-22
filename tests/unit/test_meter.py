"""Unit tests for Meter subclasses."""
import types

import numpy as np
import pytest

from slm.meter import (
    LeqAccumulator, MaxAccumulator, MinAccumulator, LastAccumulatingMeter,
    LeqMovingMeter, MaxMovingMeter, MinMovingMeter, LastMovingMeter,
)


def _parent(width=1, samplerate=48000, blocksize=4):
    return types.SimpleNamespace(width=width, samplerate=samplerate, blocksize=blocksize)


# ---------------------------------------------------------------------------
# LeqAccumulator
# ---------------------------------------------------------------------------

class TestLeqAccumulator:

    def test_single_block(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        block = np.array([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4), already Pa²
        m.process(block)
        expected = np.mean(block)
        np.testing.assert_allclose(m.read(), [expected])

    def test_accumulates_over_two_blocks(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        b1 = np.ones((1, 4)) * 2.0
        b2 = np.ones((1, 4)) * 4.0
        m.process(b1)
        m.process(b2)
        expected = (np.sum(b1) + np.sum(b2)) / 8
        np.testing.assert_allclose(m.read(), [expected])

    def test_reset(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        m.process(np.ones((1, 4)) * 5.0)
        m.reset()
        assert m._n_samples == 0
        np.testing.assert_array_equal(m._sum_sq, [0.0])

    def test_accumulates_after_reset(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        m.process(np.ones((1, 4)) * 9.0)
        m.reset()
        block = np.array([[3.0, 3.0, 3.0, 3.0]])
        m.process(block)
        np.testing.assert_allclose(m.read(), [3.0])  # mean of Pa² input

    def test_multichannel(self):
        p = _parent(width=2)
        m = LeqAccumulator(name="leq", parent=p)
        block = np.array([[1.0, 1.0, 1.0, 1.0],
                          [2.0, 2.0, 2.0, 2.0]])
        m.process(block)
        np.testing.assert_allclose(m.read(), [1.0, 2.0])

    def test_read_before_process_returns_zero(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        np.testing.assert_array_equal(m.read(), [0.0])


# ---------------------------------------------------------------------------
# MaxAccumulator
# ---------------------------------------------------------------------------

class TestMaxAccumulator:

    def test_running_max(self):
        p = _parent()
        m = MaxAccumulator(name="max", parent=p)
        m.process(np.array([[1.0, 2.0, 3.0, 4.0]]))
        np.testing.assert_array_equal(m.read(), [4.0])
        m.process(np.array([[10.0, 0.5, 0.5, 0.5]]))
        np.testing.assert_array_equal(m.read(), [10.0])

    def test_reset(self):
        p = _parent()
        m = MaxAccumulator(name="max", parent=p)
        m.process(np.ones((1, 4)) * 5.0)
        m.reset()
        assert m.read()[0] == -np.inf

    def test_accumulates_after_reset(self):
        p = _parent()
        m = MaxAccumulator(name="max", parent=p)
        m.process(np.ones((1, 4)) * 9.0)
        m.reset()
        m.process(np.ones((1, 4)) * 3.0)
        np.testing.assert_array_equal(m.read(), [3.0])

    def test_multichannel(self):
        p = _parent(width=2)
        m = MaxAccumulator(name="max", parent=p)
        block = np.array([[1.0, 5.0, 2.0, 3.0],
                          [4.0, 0.5, 0.5, 0.5]])
        m.process(block)
        np.testing.assert_array_equal(m.read(), [5.0, 4.0])


# ---------------------------------------------------------------------------
# MinAccumulator
# ---------------------------------------------------------------------------

class TestMinAccumulator:

    def test_running_min(self):
        p = _parent()
        m = MinAccumulator(name="min", parent=p)
        m.process(np.array([[4.0, 2.0, 3.0, 1.0]]))
        np.testing.assert_array_equal(m.read(), [1.0])
        m.process(np.array([[0.1, 5.0, 5.0, 5.0]]))
        np.testing.assert_array_equal(m.read(), [0.1])

    def test_reset(self):
        p = _parent()
        m = MinAccumulator(name="min", parent=p)
        m.process(np.ones((1, 4)) * 2.0)
        m.reset()
        assert m.read()[0] == np.inf


# ---------------------------------------------------------------------------
# MovingMeter subclasses
# ---------------------------------------------------------------------------

def _moving_parent(width=1, samplerate=48000, blocksize=4800):
    return types.SimpleNamespace(width=width, samplerate=samplerate, blocksize=blocksize)


class TestLeqMovingMeter:

    def test_single_block(self):
        # t=0.1s at 48kHz/4800 → n_blocks=1; FIFO is fully filled after one push.
        p = _moving_parent(blocksize=4800)
        m = LeqMovingMeter(name="leq", parent=p, t=0.1)
        block = np.array([[1.0] * 4800])
        m.process(block)
        np.testing.assert_allclose(m.read(), [1.0])

    def test_rolling_mean(self):
        """After the FIFO fills, old blocks are replaced."""
        p = _moving_parent(blocksize=4800)
        m = LeqMovingMeter(name="leq", parent=p, t=1.0)
        # FIFO holds 10 blocks (t=1.0s at 48000Hz / 4800)
        for _ in range(10):
            m.process(np.ones((1, 4800)) * 2.0)  # mean of Pa² = 2.0
        np.testing.assert_allclose(m.read(), [2.0])
        # Push blocks with mean=1.0 until FIFO rotates fully
        for _ in range(10):
            m.process(np.ones((1, 4800)) * 1.0)  # mean of Pa² = 1.0
        np.testing.assert_allclose(m.read(), [1.0])


class TestMaxMovingMeter:

    def test_rolling_max(self):
        p = _moving_parent(blocksize=4800)
        m = MaxMovingMeter(name="max", parent=p, t=1.0)
        # 10-block FIFO, push 5 blocks with max=3, then 5 with max=1
        for _ in range(5):
            m.process(np.array([[3.0] * 4800]))
        for _ in range(5):
            m.process(np.array([[1.0] * 4800]))
        # FIFO still has blocks with 3.0
        assert m.read()[0] == 3.0
        # Push 10 more blocks with max=1.0 to flush old blocks out
        for _ in range(10):
            m.process(np.array([[1.0] * 4800]))
        assert m.read()[0] == 1.0


class TestMinMovingMeter:

    def test_rolling_min(self):
        p = _moving_parent(blocksize=4800)
        m = MinMovingMeter(name="min", parent=p, t=1.0)
        for _ in range(10):
            m.process(np.array([[2.0] * 4800]))
        np.testing.assert_array_equal(m.read(), [2.0])
        for _ in range(10):
            m.process(np.array([[0.5] * 4800]))
        np.testing.assert_array_equal(m.read(), [0.5])


class TestLastMovingMeter:

    def test_returns_last_sample(self):
        # t = blocksize/samplerate = 4/48000 → n_blocks=1; read() returns last push.
        p = _moving_parent(samplerate=48000, blocksize=4)
        m = LastMovingMeter(name="last", parent=p, t=4 / 48000)
        block = np.array([[10.0, 20.0, 30.0, 99.0]])
        m.process(block)
        assert m.read()[0] == 99.0

    def test_tracks_most_recent_over_many_blocks(self):
        p = _moving_parent(samplerate=48000, blocksize=4800)
        m = LastMovingMeter(name="last", parent=p, t=1.0)
        for v in range(1, 12):   # push 11 blocks to cycle FIFO (n_blocks=10)
            m.process(np.full((1, 4800), float(v)))
        # last pushed value was 11.0
        assert m.read()[0] == 11.0


class TestMovingMeterExactness:
    """The partial-block moving meters must equal a per-sample reference window
    exactly, independent of blocksize — including blocksizes that do not divide
    the window length ``n = round(t·fs)`` (so the oldest block is partial)."""

    @staticmethod
    def _window(signal: np.ndarray, n: int, endpos: int) -> np.ndarray:
        """The exact last-``n`` samples ending at ``endpos`` (front zero-padded
        before the start of the stream, mirroring the zero-initialised FIFO)."""
        start = endpos - n
        if start >= 0:
            return signal[start:endpos]
        return np.concatenate([np.zeros(-start), signal[:endpos]])

    def _check(self, meter_cls, reduce_ref, blocksize, *, width=1):
        rng = np.random.default_rng(42)
        fs = 100
        t = 0.23                              # n = 23 samples
        n = round(t * fs)
        total = 50
        signal = rng.random((width, total))   # non-negative, like Pa²
        p = _moving_parent(width=width, samplerate=fs, blocksize=blocksize)
        m = meter_cls(name="m", parent=p, t=t)

        for b in range(total // blocksize):
            blk = signal[:, b * blocksize:(b + 1) * blocksize]
            m.process(blk)
            endpos = (b + 1) * blocksize
            ref = np.array([reduce_ref(self._window(signal[ch], n, endpos))
                            for ch in range(width)])
            np.testing.assert_allclose(m.read(), ref, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("blocksize", [4, 7, 10, 13, 23])
    def test_leq_matches_per_sample_reference(self, blocksize):
        self._check(LeqMovingMeter, lambda w: w.sum() / 23, blocksize)

    @pytest.mark.parametrize("blocksize", [4, 7, 10, 13])
    def test_max_matches_per_sample_reference(self, blocksize):
        self._check(MaxMovingMeter, np.max, blocksize)

    @pytest.mark.parametrize("blocksize", [4, 7, 10, 13])
    def test_min_matches_per_sample_reference(self, blocksize):
        self._check(MinMovingMeter, np.min, blocksize)

    def test_leq_multichannel(self):
        self._check(LeqMovingMeter, lambda w: w.sum() / 23, blocksize=7, width=3)

    def test_window_is_exact_n_samples(self):
        """Window length is round(t·fs), not rounded up to a whole block."""
        p = _moving_parent(samplerate=48000, blocksize=1024)
        m = LeqMovingMeter(name="m", parent=p, t=1.0)
        assert m._n == 48000
        assert m.n_blocks == 47            # ceil(48000/1024)
        assert m._o == (-48000) % 1024     # 128


# ---------------------------------------------------------------------------
# to_str coverage
# ---------------------------------------------------------------------------

class TestToStr:

    def test_accumulating_to_str(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        assert "LeqAccumulator" in m.to_str()
        assert "leq" in m.to_str()

    def test_moving_to_str(self):
        p = _moving_parent()
        m = LeqMovingMeter(name="leq", parent=p, t=1.0)
        assert "LeqMovingMeter" in m.to_str()
        assert "leq" in m.to_str()


# ---------------------------------------------------------------------------
# LastAccumulatingMeter.reset
# ---------------------------------------------------------------------------

class TestLastAccumulatingMeterReset:

    def test_reset_clears_last(self):
        from slm.meter import LastAccumulatingMeter
        p = _parent()
        m = LastAccumulatingMeter(name="last", parent=p)
        m.process(np.array([[1.0, 2.0, 3.0, 4.0]]))
        m.reset()
        np.testing.assert_array_equal(m._last, [0.0])


# ---------------------------------------------------------------------------
# Meter.get_chain
# ---------------------------------------------------------------------------

class TestMeterGetChain:

    def test_chain_includes_meter_and_upstream(self):
        from slm.io.noise_controller import NoiseController
        from slm.engine import Engine
        from slm.assembly import parse_metric, build_chain
        ctrl = NoiseController(samplerate=48_000, blocksize=1_024)
        ctrl.set_sensitivity(1.0, unit="V")
        engine = Engine(ctrl, dt=1.0)
        from slm.time_weighting import PluginSquare
        build_chain([parse_metric("LAeq")], engine)
        bus = engine._busses["A"]
        sq = next(p for p in bus.plugins if isinstance(p, PluginSquare))
        meter = sq.meters["LAeq"]
        chain = meter.get_chain()
        assert meter in chain
        assert bus in chain
