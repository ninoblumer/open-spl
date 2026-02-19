from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from slm.controller import Controller


class ProcessingElement(ABC):
    samplerate: int
    blocksize: int
    width: int

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def process(self, block: np.ndarray): ...

    @abstractmethod
    def get_chain(self) -> list[ProcessingElement]: ...

    @abstractmethod
    def to_str(self) -> str: ...

    def __str__(self) -> str:
        return self.to_str()