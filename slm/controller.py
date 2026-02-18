from abc import ABC, abstractmethod
import itertools
from typing import Literal

import numpy as np


class Controller(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self._counter = itertools.count(0)

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @property
    @abstractmethod
    def blocksize(self) -> int: ...

    @property
    @abstractmethod
    def sensitivity(self) -> float: ...

    @abstractmethod
    def read_block(self) -> tuple[np.ndarray, int]:
        """ read a block of audio and returns the buffer and the block_index """
        ...

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def calibrate(self, target_spl=94.0):
        ...

    def set_sensitivity(self, sensitivity: float, unit: Literal["mV", "V", "dB"]) -> None:
        if unit == "mV":
            self._sensitivity = sensitivity / 1000.0
        elif unit == "V":
            self._sensitivity = sensitivity
        elif unit == "dB":
            self._sensitivity = 10**(sensitivity/20)

