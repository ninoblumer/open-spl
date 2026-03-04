"""Unit tests for Meter subclasses."""
import types

import numpy as np
import pytest

from slm.meter import (
    LeqAccumulator, MaxAccumulator, MinAccumulator,
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
        block = np.array([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
        m.process(block)
        expected = np.mean(block ** 2)
        np.testing.assert_allclose(m.read(), [expected])

    def test_accumulates_over_two_blocks(self):
        p = _parent()
        m = LeqAccumulator(name="leq", parent=p)
        b1 = np.ones((1, 4)) * 2.0
        b2 = np.ones((1, 4)) * 4.0
        m.process(b1)
        m.process(b2)
        expected = (np.sum(b1 ** 2) + np.sum(b2 ** 2)) / 8
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
        np.testing.assert_allclose(m.read(), [9.0])  # 3²=9

    def test_multichannel(self):
        p = _parent(width=2)
        m = LeqAccumulator(name="leq", parent=p)
        block = np.array([[1.0, 1.0, 1.0, 1.0],
                          [2.0, 2.0, 2.0, 2.0]])
        m.process(block)
        np.testing.assert_allclose(m.read(), [1.0, 4.0])

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
            m.process(np.ones((1, 4800)) * 2.0)  # mean sq = 4.0
        np.testing.assert_allclose(m.read(), [4.0])
        # Push blocks with mean_sq=1.0 until FIFO rotates fully
        for _ in range(10):
            m.process(np.ones((1, 4800)) * 1.0)  # mean sq = 1.0
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
