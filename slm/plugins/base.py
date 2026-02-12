from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

import numpy as np

from slm.bus import Bus
from slm.engine import ExecutionContext, ExecutionError, RequestRejected

ID = str


def find_plugin(function) -> type[Plugin]:
    raise RequestRejected(
        f"Function {function} is not implemented."
    )

class Plugin(ABC):
    id: ID
    function: str = ""
    input: Plugin | None
    # outputs: list[Plugin]

    samplerate: int = property(lambda self: self.bus.samplerate)
    blocksize: int = property(lambda self: self.bus.blocksize)
    sensitivity: float = property(lambda self: self.bus.sensitivity)

    def __init__(self, *, bus: Bus, id: str, input: Plugin | None,
                 # outputs: list[Plugin] | None = None,
                 **_):
        self.bus = bus
        self.id = id
        self.input = input
        # self.outputs = outputs.copy() if outputs else []

    def process(self, ctx: ExecutionContext):
        data = ctx.cache.get(self.input.id) if self.input else ctx.block
        result = self.func(data)

        if ctx.cache.get(self.id) is not None:
            raise ExecutionError
        ctx.cache[self.id] = result

    @abstractmethod
    def func(self, block: np.ndarray) -> np.ndarray:
        ...

REFERENCE_PRESSURE = 20e-6
class PluginMeter(Plugin):
    meter: Callable[[np.ndarray], np.floating]

    def __init__(self, meter: Callable[[np.ndarray], np.floating], **kwargs):
        super().__init__(**kwargs)
        self.meter = meter

    def read_lin(self):


    def read_db(self):
        return 10*np.log10(self.read_lin()/(REFERENCE_PRESSURE*self.sensitivity)**2)

    def process(self, ctx: ExecutionContext):
