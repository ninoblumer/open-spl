from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING, TypeVar
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

    def __init__(self, name:str, parent: PluginMeter, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.parent = parent


    def get_chain(self) -> list[Plugin | Bus | Meter]:
        chain = self.parent.get_chain()
        chain.append(self)
        return chain

    @abstractmethod
    def read(self) -> np.ndarray:
        ...


# This works well for moving averages, but:
# TODO: need a Meter for global max, min, mean... (so up to "now" (logging) or over the whole recording (reporting)
class MovingMeter(Meter):
    block_fn: Callable
    fifo_fn: Callable
    n_blocks: int
    _fifo: FIFO

    # t: float = property(lambda self: self.n_block * self.blocksize / self.samplerate)
    t: float = property(lambda self: self._t)


    # Functions
    # _block_fn_last = staticmethod(lambda a: a[:, -1])
    # _fifo_fn_last = staticmethod(lambda a: a[-1])

    @staticmethod
    def _block_fn_last(a: np.ndarray) -> np.ndarray:
        return a[:, -1]

    @staticmethod
    def _fifo_fn_last(a: np.ndarray) -> np.ndarray:
        return a[-1]

    def __init__(self, *, t: float | None = None,
                 block_fn: Callable | None = None, fifo_fn: Callable | None = None, fn: Callable | None= None,
                 **kwargs):
        super().__init__(**kwargs)
        self.block_fn = block_fn or fn
        self.fifo_fn = fifo_fn or fn
        if not self.block_fn or not self.fifo_fn:
            raise ValueError("block_fn and fifo_fn must not be None, at least block_fn must be given.")

        if t is None:
            t = self.parent.bus.dt
        self._t = t

        self.n_blocks = ceil(self._t * self.samplerate / self.blocksize)
        self._fifo = FIFO((self.width, self.n_blocks))


    def process(self, block: np.ndarray):
        self._fifo.push(self.block_fn(block))
        # result = self.block_fn(block)
        # self._fifo.push(result)

    def read(self) -> np.ndarray:
        return self._fifo.map(self.fifo_fn)

    def to_str(self):
        return f"MovingMeter(name={self.name}, block_fn={self.block_fn}, fifo_fn={self.fifo_fn})"


class AccumulatingMeter(Meter):
    block_fn: Callable
    comp_fn: Callable
    _acc: np.ndarray

    def __init__(self, *, block_fn: Callable, comp_fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self.block_fn = block_fn
        self.comp_fn = comp_fn
        self._acc = np.zeros((self.width, 1))

    def to_str(self):
        return f"AccumulatingMeter(name={self.name}, block_fn={self.block_fn}, comp_fn={self.comp_fn})"

    def process(self, block: np.ndarray):
        metric = self.block_fn(block)
        self._acc = self.comp_fn(metric, self._acc)

    def read(self) -> np.ndarray:
        return self._acc


TMeter = TypeVar("TMeter", bound=Meter)