from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar
from math import ceil

import numpy as np

from slm.processing_element import ProcessingElement
from slm.fifo import FIFO

if TYPE_CHECKING:
    from slm.bus import Bus
    from slm.plugin import Plugin
    from slm.plugin_meter import PluginMeter


class Meter(ProcessingElement, ABC):
    parent: PluginMeter
    samplerate: int = property(lambda self: self.parent.samplerate)
    blocksize: int = property(lambda self: self.parent.blocksize)
    width: int = property(lambda self: self.parent.width)

    def __init__(self, name: str, parent: PluginMeter, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.parent = parent

    def get_chain(self) -> list[Plugin | Bus | Meter]:
        chain = self.parent.get_chain()
        chain.append(self)
        return chain

    @abstractmethod
    def read(self) -> np.ndarray: ...

    def reset(self) -> None:
        """Clear accumulated state, leaving the parent chain's filter state intact.

        Default is a no-op: rolling :class:`MovingMeter` outputs self-flush within
        their window, so only the unbounded :class:`AccumulatingMeter` family needs
        to override this (used after a warm-up period)."""


# ---------------------------------------------------------------------------
# AccumulatingMeter family — accumulate statistics over an unbounded window
# ---------------------------------------------------------------------------

class AccumulatingMeter(Meter, ABC):

    @abstractmethod
    def process(self, block: np.ndarray): ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    @abstractmethod
    def reset(self): ...

    def to_str(self):
        return f"{type(self).__name__}(name={self.name})"


class LeqAccumulator(AccumulatingMeter):
    """Leq accumulator.

    Attaches to a squared (Pa²) source — a ``PluginSquare`` or a time-weighting
    output — so the meter never squares.  ``read()`` returns mean square
    pressure (Pa²) so that ``plugin.read_db()`` gives the correct Leq in dB SPL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sum_sq = np.zeros((self.width,))
        self._n_samples = 0

    def process(self, block: np.ndarray):
        self._sum_sq += np.sum(block, axis=-1)
        self._n_samples += block.shape[-1]

    def read(self) -> np.ndarray:
        # No data yet → 0/0 = NaN (don't report until at least one sample),
        # matching the NaN-until-full behaviour of the MovingMeter family.
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._sum_sq / self._n_samples

    def reset(self):
        self._sum_sq[:] = 0.0
        self._n_samples = 0


class MaxAccumulator(AccumulatingMeter):
    """Running maximum accumulator.

    Attaches to a time-weighting output (Pa², already squared).
    ``read()`` returns the maximum Pa² value seen so far.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._acc = np.full((self.width,), -np.inf)
        self._n_samples = 0

    def process(self, block: np.ndarray):
        self._acc = np.maximum(self._acc, np.max(block, axis=-1))
        self._n_samples += block.shape[-1]

    def read(self) -> np.ndarray:
        # The reduction identity must stay -inf (NaN would be sticky through
        # np.maximum); gate on the sample count to report NaN until data arrives.
        if self._n_samples == 0:
            return np.full((self.width,), np.nan)
        return self._acc

    def reset(self):
        self._acc[:] = -np.inf
        self._n_samples = 0


class MinAccumulator(AccumulatingMeter):
    """Running minimum accumulator.

    Attaches to a time-weighting output (Pa², already squared).
    ``read()`` returns the minimum Pa² value seen so far.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._acc = np.full((self.width,), np.inf)
        self._n_samples = 0

    def process(self, block: np.ndarray):
        self._acc = np.minimum(self._acc, np.min(block, axis=-1))
        self._n_samples += block.shape[-1]

    def read(self) -> np.ndarray:
        # The reduction identity must stay +inf (NaN would be sticky through
        # np.minimum); gate on the sample count to report NaN until data arrives.
        if self._n_samples == 0:
            return np.full((self.width,), np.nan)
        return self._acc

    def reset(self):
        self._acc[:] = np.inf
        self._n_samples = 0


class LastAccumulatingMeter(AccumulatingMeter):
    """Tracks only the last sample of the most-recent block.

    Attaches to a time-weighting output (Pa², already squared).
    ``read()`` returns the value of the final sample in the last block
    processed.  No window or FIFO is needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last = np.full((self.width,), np.nan)   # NaN until the first block

    def process(self, block: np.ndarray):
        self._last = block[:, -1]

    def read(self) -> np.ndarray:
        return self._last

    def reset(self):
        self._last[:] = np.nan


# ---------------------------------------------------------------------------
# MovingMeter family — rolling window statistics using a FIFO
# ---------------------------------------------------------------------------

class MovingMeter(Meter, ABC):
    """Rolling-window meter over exactly ``round(t·fs)`` samples.

    The engine snapshots once per block, so every ``read()`` lands on a block
    boundary.  That fixes the window's start offset: the only block partially
    inside the window is the *oldest*, contributing its suffix ``block[:, o:]``
    (``o = (-n) % blocksize``); every newer block is wholly inside.  Each block
    we store its whole-block aggregate (``_fifo_full``) and its tail aggregate
    (``_fifo_part``).  ``read()`` reduces the window as: the oldest block's tail
    combined with every newer block's whole-block aggregate — exactly ``n``
    samples wide, independent of blocksize (no rounding up to a whole block).

    Subclasses pick the reduction by setting ``_ufunc`` to a binary numpy ufunc
    (``np.add`` for Leq/LE, ``np.maximum`` / ``np.minimum`` for max/min); both
    the per-block push and the window combine use it.

    The FIFOs are NaN-initialised so an as-yet-unfilled window reads NaN rather
    than ramping up from zero: ``read()`` only returns a value once a full
    ``n``-sample window of real data has been pushed.  This relies on NaN being
    *absorbing* for ``_ufunc`` — ``np.add``/``np.maximum``/``np.minimum`` all
    propagate NaN.  Do NOT swap ``_ufunc`` for a NaN-ignoring variant
    (``np.fmax``/``np.fmin``/``np.nansum``): that would silently resurrect the
    ramp-up.  ``tests/unit/test_meter.py::TestNanWarmup`` guards this.
    """

    t: float = property(lambda self: self._t)
    _ufunc: np.ufunc

    def __init__(self, *, t: float | None = None, **kwargs):
        super().__init__(**kwargs)
        if t is None:
            t = self.parent.bus.dt
        self._t = t
        self._n = round(self._t * self.samplerate)         # window length, samples
        self.n_blocks = ceil(self._n / self.blocksize)
        self._o = (-self._n) % self.blocksize              # tail start offset within a block
        self._fifo_full = FIFO((self.width, self.n_blocks), fill=np.nan)   # whole-block aggregate
        self._fifo_part = FIFO((self.width, self.n_blocks), fill=np.nan)   # tail (block[:, o:]) aggregate

    def process(self, block: np.ndarray):
        self._fifo_full.push(self._ufunc.reduce(block, axis=-1))
        self._fifo_part.push(self._ufunc.reduce(block[:, self._o:], axis=-1))

    def _window(self) -> np.ndarray:
        """Reduce the exact n-sample window: the oldest block's tail aggregate
        combined with the whole-block aggregate of every other block.  Reduces
        over buffer *views* — no whole-buffer copy."""
        oldest = self._fifo_full.index            # slot overwritten next = current oldest
        full = self._fifo_full.buffer
        acc = self._fifo_part.buffer[:, oldest].copy()        # oldest → its tail only
        if oldest > 0:                                        # newer blocks left of oldest
            acc = self._ufunc(acc, self._ufunc.reduce(full[:, :oldest], axis=1))
        if oldest + 1 < self.n_blocks:                        # newer blocks right of oldest
            acc = self._ufunc(acc, self._ufunc.reduce(full[:, oldest + 1:], axis=1))
        return acc

    @abstractmethod
    def read(self) -> np.ndarray: ...

    def to_str(self):
        return f"{type(self).__name__}(name={self.name}, t={self._t})"


class LeqMovingMeter(MovingMeter):
    """Rolling energy-mean Leq over exactly ``round(t·fs)`` samples.

    Attaches to a squared (Pa²) source — a ``PluginSquare`` — so the meter never
    squares.  ``read()`` is the window's summed Pa² divided by ``n``.
    """

    _ufunc = np.add

    def read(self) -> np.ndarray:
        return self._window() / self._n


class MaxMovingMeter(MovingMeter):
    """Rolling maximum over exactly ``round(t·fs)`` samples (Pa² input)."""

    _ufunc = np.maximum

    def read(self) -> np.ndarray:
        return self._window()


class MinMovingMeter(MovingMeter):
    """Rolling minimum over exactly ``round(t·fs)`` samples (Pa² input)."""

    _ufunc = np.minimum

    def read(self) -> np.ndarray:
        return self._window()


class LastMovingMeter(MovingMeter):
    """Exposes only the last (most-recent) sample seen."""

    _ufunc = np.add   # unused: process/read are overridden

    def process(self, block: np.ndarray):
        self._fifo_full.push(block[:, -1])

    def read(self) -> np.ndarray:
        recent = (self._fifo_full.index - 1) % self._fifo_full.size
        return self._fifo_full.buffer[:, recent]


class LEAccumulator(LeqAccumulator):
    """Sound exposure level (LE) accumulator.

    Attaches to a squared (Pa²) source — a ``PluginSquare``.
    ``read()`` returns ``sum_sq / samplerate`` (Pa²·s) so that
    ``plugin.read_db()`` gives LE = Leq + 10·log₁₀(T / T₀) in dB (T₀ = 1 s).
    Equivalent to: 10·log₁₀(Σp² / samplerate / p₀²).
    """

    def read(self) -> np.ndarray:
        # sum_sq / samplerate = E (Pa²·s); read_db divides by p₀² → LE.
        if self._n_samples == 0:                       # NaN until data, like LeqAccumulator
            return np.full((self.width,), np.nan)
        return self._sum_sq / self.samplerate


class LEMovingMeter(LeqMovingMeter):
    """Rolling sound exposure level over exactly ``round(t·fs)`` samples.

    ``read()`` returns ``Σp² / samplerate`` (Pa²·s) over the window so that
    ``plugin.read_db()`` gives LE_window = Leq_window + 10·log₁₀(T_window / T₀)
    in dB (T₀ = 1 s).
    """

    def read(self) -> np.ndarray:
        # window sum / samplerate = E_window (Pa²·s); read_db divides by p₀² → LE_window.
        return self._window() / self.samplerate


TMeter = TypeVar("TMeter", bound=Meter)
