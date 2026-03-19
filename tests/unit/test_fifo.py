"""Unit tests for the FIFO ring buffer."""
import numpy as np
import pytest

from slm.fifo import FIFO


class TestFIFOOrdering:

    def test_get_returns_oldest_first(self):
        fifo = FIFO((1, 3))
        fifo.push(np.array([1.0]))
        fifo.push(np.array([2.0]))
        fifo.push(np.array([3.0]))
        np.testing.assert_array_equal(fifo.get(), [[1.0, 2.0, 3.0]])

    def test_wrap_around(self):
        """After a 4th push into a size-3 FIFO, the oldest value is evicted."""
        fifo = FIFO((1, 3))
        fifo.push(np.array([1.0]))
        fifo.push(np.array([2.0]))
        fifo.push(np.array([3.0]))
        fifo.push(np.array([4.0]))       # overwrites slot 0 (value 1)
        np.testing.assert_array_equal(fifo.get(), [[2.0, 3.0, 4.0]])

    def test_multichannel_ordering(self):
        fifo = FIFO((2, 3))
        fifo.push(np.array([1.0, 10.0]))
        fifo.push(np.array([2.0, 20.0]))
        fifo.push(np.array([3.0, 30.0]))
        np.testing.assert_array_equal(
            fifo.get(),
            [[1.0, 2.0, 3.0],
             [10.0, 20.0, 30.0]],
        )


class TestFIFOMap:

    def test_map_mean(self):
        fifo = FIFO((1, 4))
        for v in [2.0, 4.0, 6.0, 8.0]:
            fifo.push(np.array([v]))
        result = fifo.map(np.mean)
        np.testing.assert_allclose(result, [5.0])

    def test_map_max(self):
        fifo = FIFO((1, 4))
        for v in [3.0, 7.0, 1.0, 5.0]:
            fifo.push(np.array([v]))
        result = fifo.map(np.max)
        np.testing.assert_allclose(result, [7.0])

    def test_map_min(self):
        fifo = FIFO((1, 4))
        for v in [3.0, 7.0, 1.0, 5.0]:
            fifo.push(np.array([v]))
        result = fifo.map(np.min)
        np.testing.assert_allclose(result, [1.0])

    def test_map_multichannel(self):
        """map applies fn independently per channel."""
        fifo = FIFO((2, 3))
        fifo.push(np.array([1.0, 10.0]))
        fifo.push(np.array([2.0, 20.0]))
        fifo.push(np.array([3.0, 30.0]))
        result = fifo.map(np.mean)
        np.testing.assert_allclose(result, [2.0, 20.0])

    def test_map_ignores_order(self):
        """map result must be the same regardless of internal ring-buffer rotation."""
        fifo = FIFO((1, 3))
        for v in [1.0, 2.0, 3.0, 4.0]:    # causes one wrap-around
            fifo.push(np.array([v]))
        np.testing.assert_allclose(fifo.map(np.mean), [3.0])   # mean(2,3,4)
        np.testing.assert_allclose(fifo.map(np.max),  [4.0])


class TestFIFOReset:

    def test_reset_zeroes_buffer(self):
        fifo = FIFO((1, 3))
        for v in [1.0, 2.0, 3.0]:
            fifo.push(np.array([v]))
        fifo.reset()
        np.testing.assert_array_equal(fifo.get(), [[0.0, 0.0, 0.0]])

    def test_reset_restores_index(self):
        fifo = FIFO((1, 3))
        fifo.push(np.array([1.0]))
        fifo.push(np.array([2.0]))
        fifo.reset()
        fifo.push(np.array([9.0]))
        # get() returns oldest-first. After reset (all zeros) + 1 push,
        # the two zero slots are "older" and the pushed value is newest → last.
        np.testing.assert_array_equal(fifo.get(), [[0.0, 0.0, 9.0]])
