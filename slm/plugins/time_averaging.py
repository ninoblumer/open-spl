import numpy as np

from slm.plugins.base import PluginMeter

class FIFO:
    size = property(lambda self: self.data.size)

    def __init__(self, size):
        self.data = np.zeros(size)
        self.index = 0

    def push(self, value):
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size

    def get(self):
        return np.concatenate((
            self.data[self.index:],
            self.data[:self.index]
        ))

    def map(self, func):
        """
        maps a func of the contents of the buffer.
        Warning: the buffer is unordered, so only use functions that are invariant to the ordering (like min, max or mean)
        """
        return func(self.data)

class TimeAveraging(PluginMeter):
    t: float
    time_constant: str
    _buffer: FIFO

    def __init__(self, time_constant: str, t: float, **kwargs):
        super().__init__(**kwargs)
        self.time_constant = time_constant
        self.t = t

        n_blocks = self.t * self.samplerate / self.blocksize
        self._buffer = FIFO(n_blocks)

    def func(self, block: np.ndarray) -> np.ndarray:
        result = np.mean(np.square(block))
        # TODO: push into buffer, read out buffer (Average over it)
        self._buffer.push(result)

    

