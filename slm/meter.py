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

    Attaches to a frequency-weighting output (linear Pa).  Squares the input
    internally.  ``read()`` returns mean square pressure (Pa²) so that
    ``plugin.read_db()`` gives the correct Leq in dB SPL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sum_sq = np.zeros((self.width,))
        self._n_samples = 0

    def process(self, block: np.ndarray):
        self._sum_sq += np.sum(block ** 2, axis=-1)
        self._n_samples += block.shape[-1]

    def read(self) -> np.ndarray:
        return self._sum_sq / max(1, self._n_samples)

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

    def process(self, block: np.ndarray):
        self._acc = np.maximum(self._acc, np.max(block, axis=-1))

    def read(self) -> np.ndarray:
        return self._acc

    def reset(self):
        self._acc[:] = -np.inf


class MinAccumulator(AccumulatingMeter):
    """Running minimum accumulator.

    Attaches to a time-weighting output (Pa², already squared).
    ``read()`` returns the minimum Pa² value seen so far.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._acc = np.full((self.width,), np.inf)

    def process(self, block: np.ndarray):
        self._acc = np.minimum(self._acc, np.min(block, axis=-1))

    def read(self) -> np.ndarray:
        return self._acc

    def reset(self):
        self._acc[:] = np.inf


class LastAccumulatingMeter(AccumulatingMeter):
    """Tracks only the last sample of the most-recent block.

    Attaches to a time-weighting output (Pa², already squared).
    ``read()`` returns the value of the final sample in the last block
    processed.  No window or FIFO is needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last = np.zeros((self.width,))

    def process(self, block: np.ndarray):
        self._last = block[:, -1]

    def read(self) -> np.ndarray:
        return self._last

    def reset(self):
        self._last[:] = 0.0


# ---------------------------------------------------------------------------
# MovingMeter family — rolling window statistics using a FIFO
# ---------------------------------------------------------------------------

class MovingMeter(Meter, ABC):

    t: float = property(lambda self: self._t)

    def __init__(self, *, t: float | None = None, **kwargs):
        super().__init__(**kwargs)
        if t is None:
            t = self.parent.bus.dt
        self._t = t
        self.n_blocks = ceil(t * self.samplerate / self.blocksize)
        self._fifo = FIFO((self.width, self.n_blocks))

    @abstractmethod
    def process(self, block: np.ndarray): ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    def to_str(self):
        return f"{type(self).__name__}(name={self.name}, t={self._t})"


class LeqMovingMeter(MovingMeter):
    """Rolling energy-mean Leq over a window of ``t`` seconds.

    Attaches to a frequency-weighting output (linear Pa).  Squares internally.
    Each FIFO slot stores the mean square for one block; ``read()`` returns
    the mean of those values — the correct energy mean over the window.
    """

    def process(self, block: np.ndarray):
        self._fifo.push(np.sum(block ** 2, axis=-1) / block.shape[-1])

    def read(self) -> np.ndarray:
        return self._fifo.map(np.mean)


class MaxMovingMeter(MovingMeter):
    """Rolling maximum over a window of ``t`` seconds."""

    def process(self, block: np.ndarray):
        self._fifo.push(np.max(block, axis=-1))

    def read(self) -> np.ndarray:
        return self._fifo.map(np.max)


class MinMovingMeter(MovingMeter):
    """Rolling minimum over a window of ``t`` seconds."""

    def process(self, block: np.ndarray):
        self._fifo.push(np.min(block, axis=-1))

    def read(self) -> np.ndarray:
        return self._fifo.map(np.min)


class LastMovingMeter(MovingMeter):
    """Exposes only the last (most-recent) sample of the rolling window."""

    def process(self, block: np.ndarray):
        self._fifo.push(block[:, -1])

    def read(self) -> np.ndarray:
        # FIFO.get() returns ordered buffer (oldest→newest); [:, -1] is most recent.
        return self._fifo.get()[:, -1]


class LEAccumulator(LeqAccumulator):
    """Sound exposure level (LE) accumulator.

    Attaches to a frequency-weighting output (linear Pa). Squares internally.
    ``read()`` returns ``sum_sq / samplerate`` (Pa²·s) so that
    ``plugin.read_db()`` gives LE = Leq + 10·log₁₀(T / T₀) in dB (T₀ = 1 s).
    Equivalent to: 10·log₁₀(Σp² / samplerate / p₀²).
    """

    def read(self) -> np.ndarray:
        # sum_sq / samplerate = E (Pa²·s); read_db divides by p₀² → LE.
        return self._sum_sq / self.samplerate


class LEMovingMeter(LeqMovingMeter):
    """Rolling sound exposure level over a window of ``t`` seconds.

    ``read()`` returns ``mean_sq * t`` (Pa²·s) so that ``plugin.read_db()``
    gives LE_window = Leq_window + 10·log₁₀(T_window / T₀) in dB (T₀ = 1 s).
    """

    def read(self) -> np.ndarray:
        # mean_sq * t = E_window (Pa²·s); read_db divides by p₀² → LE_window.
        return self._fifo.map(np.mean) * self._t


TMeter = TypeVar("TMeter", bound=Meter)
