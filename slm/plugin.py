from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, TYPE_CHECKING

import numpy as np

from slm.processing_element import ProcessingElement

if TYPE_CHECKING:
    from slm.bus import Bus
    from slm.meter import Meter



class Plugin(ProcessingElement, ABC):
    # id: str
    bus: "Bus"
    input: "Plugin | Bus"
    output: np.ndarray
    subscribers: list[Plugin]

    width: int = property(lambda self: self._width)
    samplerate: int = property(lambda self: self.bus.samplerate)
    blocksize: int = property(lambda self: self.bus.blocksize)
    sensitivity: float = property(lambda self: self.bus.sensitivity)

    def __init__(self, *, input: "Plugin | Bus", width: int = 1, **kwargs):
        super().__init__(**kwargs)

        from slm.bus import Bus
        if isinstance(input, Bus):
            self.bus = input
        else:
            self.bus = input.bus

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

    @abstractmethod
    def to_str(self):
        ...

    def __str__(self):
        return self.to_str()


TPlugin = TypeVar("TPlugin", bound=Plugin)
