from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, TypeVar, TYPE_CHECKING, NamedTuple

import numpy as np

from slm.exceptions import RequestRejected

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

    samplerate: int = property(lambda self: self.bus.samplerate)
    blocksize: int = property(lambda self: self.bus.blocksize)
    sensitivity: float = property(lambda self: self.bus.sensitivity)

    def __init__(self, *, bus: "Bus", id: str, input: "Plugin | Bus", **_):
        self.bus = bus
        self.id = id
        self.input = input

    def reset(self):
        """ must initialize output to zeros, must initialize internal states to zero """
        self.output.fill(0)

    @abstractmethod
    def process(self):
        """ must read the input from self.input.get() and write to self.output is guaranteed to be called once per block. """
        ...

    def get(self):
        return self.output

    # @abstractmethod
    # def implemented_function(self) -> str:
    #     ...

    @abstractmethod
    def to_str(self):
        ...

    def __str__(self):
        return self.to_str()

TPlugin = TypeVar("TPlugin", bound=Plugin)

class ReadMode(NamedTuple):
    name: str
    value: Callable

REFERENCE_PRESSURE = 20e-6
class PluginMeter(Plugin):
    #ReadModes : type[Enum]  # type hint: concrete type will be subclass-specific
    read_mode: ReadMode = property(lambda self: self._read_mode)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_lin(self):
        return self.output[:,0]

    def read_db(self):
        return 10*np.log10(self.read_lin()/(REFERENCE_PRESSURE*self.sensitivity)**2)


