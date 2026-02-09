from abc import ABC, abstractmethod
import itertools

import numpy as np



class Controller(ABC):

    def __init__(self):
        super().__init__()
        self._counter = itertools.count(0)

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @property
    @abstractmethod
    def blocksize(self) -> int: ...

    @abstractmethod
    def read_block(self) -> tuple[np.ndarray, int]:
        """ read a block of audio and returns the data and the block_index """
        ...





