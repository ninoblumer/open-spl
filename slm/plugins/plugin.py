from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
from typing import Callable, TypeVar, TYPE_CHECKING, NamedTuple

import numpy as np

from slm.exceptions import RequestRejected
from slm.plugins.fifo import FIFO

if TYPE_CHECKING:
    from slm.bus import Bus


# def find_plugin(function) -> type[Plugin]:
#     raise RequestRejected(
#         f"Function {function} is not implemented."
#     )

class Plugin(ABC):
    id: str
    bus: "Bus"
    input: "Plugin | Bus"
    output: np.ndarray
    subscribers: list[Plugin]

    width: int = property(lambda self: self._width)
    samplerate: int = property(lambda self: self.bus.samplerate)
    blocksize: int = property(lambda self: self.bus.blocksize)
    sensitivity: float = property(lambda self: self.bus.sensitivity)

    def __init__(self, *, bus: "Bus", id: str, input: "Plugin | Bus", width: int = 1, **_):
        self.bus = bus
        self.id = id
        self.input = input
        self._width = width
        self.subscribers = []
        if isinstance(self.input, Plugin):
            self.input.register_subscriber(self)

    def register_subscriber(self, subscriber: Plugin):
        self.subscribers.append(subscriber)

    def reset(self):
        """ must initialize output to zeros, must initialize internal states to zero """
        self.output.fill(0)

    def process(self, block):
        self.func(block)
        for sub in self.subscribers:
            sub.process(self.output)

    @abstractmethod
    def func(self, block: np.ndarray):
        ...

    def get_chain(self) -> list[Plugin | Bus | Meter]:
        chain = self.input.get_chain()
        chain.append(self)
        return chain

    # def get(self):
    #     return self.output

    # @abstractmethod
    # def implemented_function(self) -> str:
    #     ...

    @abstractmethod
    def to_str(self):
        ...

    def __str__(self):
        return self.to_str()

TPlugin = TypeVar("TPlugin", bound=Plugin)


# TODO: remove
class ReadMode(NamedTuple):
    name: str
    value: Callable

# This works well for moving averages, but:
# TODO: need a Meter for global max, min, mean... (so up to "now" (logging) or over the whole recording (reporting)
class Meter:
    name: str
    block_fn: Callable
    fifo_fn: Callable
    n_blocks: int
    fifo: FIFO
    parent: PluginMeter
    samplerate: int = property(lambda self: self.parent.samplerate)
    blocksize: int = property(lambda self: self.parent.blocksize)
    width: int = property(lambda self: self.parent.width)
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

    def __init__(self, *, name: str, parent: PluginMeter, t: float | None = None,
                 block_fn: Callable | None = None, fifo_fn: Callable | None = None, func: Callable | None= None):
        self.parent = parent
        self.name = name
        self.block_fn = block_fn or func
        self.fifo_fn = fifo_fn or func
        if not self.block_fn or not self.fifo_fn:
            raise ValueError("block_fn and fifo_fn must not be None, at least func must be given.")

        if t is None:
            t = parent.bus.dt
        self._t = t

        self.n_blocks = ceil(self._t * self.samplerate / self.blocksize)
        self.fifo = FIFO((self.width, self.n_blocks))



    def process(self, block: np.ndarray):
        self.fifo.push(self.block_fn(block))
        # result = self.block_fn(block)
        # self.fifo.push(result)

    def read(self):
        return self.fifo.map(self.fifo_fn)

    def get_chain(self) -> list[Plugin | Bus | Meter]:
        chain = self.parent.get_chain()
        chain.append(self)
        return chain

    def to_str(self):
        return f"Meter(name={self.name}, block_fn={self.block_fn}, fifo_fn={self.fifo_fn})"

    def __str__(self):
        return self.to_str()


REFERENCE_PRESSURE = 20e-6
class PluginMeter(Plugin, ABC):
    meters: dict[str, Meter]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meters = {}

    def add_meter(self, **kwargs) -> Meter:
        meter = Meter(parent=self, **kwargs)
        self.meters[meter.name] = meter
        return meter

    def process(self, block: np.ndarray):
        super().process(block)
        self.process_meters()

    def process_meters(self):
        for meter in self.meters.values():
            meter.process(self.output)

    def read_lin(self, name: str):
        return self.meters[name].read()

    def read_db(self, name):
        return 10*np.log10(self.read_lin(name)/(REFERENCE_PRESSURE*self.sensitivity)**2)
