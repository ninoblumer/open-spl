"""Unit tests for Meter types."""
import types

import numpy as np
import pytest

from slm.meter import AccumulatingMeter


def _make_accumulating_meter(width=1, block_fn=np.max, comp_fn=np.max):
    parent = types.SimpleNamespace(width=width, samplerate=48000, blocksize=4)
    return AccumulatingMeter(name="test", parent=parent, block_fn=block_fn, comp_fn=comp_fn)


class TestAccumulatingMeterReset:

    def test_reset_zeroes_accumulator(self):
        meter = _make_accumulating_meter()
        meter.process(np.ones((1, 4)) * 5.0)
        assert meter.read()[0] != 0.0
        meter.reset()
        np.testing.assert_array_equal(meter.read(), [0.0])

    def test_reset_zeroes_all_channels(self):
        meter = _make_accumulating_meter(width=2)
        meter.process(np.ones((2, 4)) * 5.0)
        meter.reset()
        np.testing.assert_array_equal(meter.read(), [0.0, 0.0])

    def test_accumulates_correctly_after_reset(self):
        """Reset must not leave stale state — accumulation restarts from zero."""
        meter = _make_accumulating_meter()
        meter.process(np.ones((1, 4)) * 9.0)
        meter.reset()
        meter.process(np.ones((1, 4)) * 3.0)
        np.testing.assert_array_equal(meter.read(), [3.0])
