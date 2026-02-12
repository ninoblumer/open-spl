from __future__ import annotations
from enum import Enum

import numpy as np

from slm.plugins.plugin import PluginMeter, ReadMode


class FIFO:
    size = property(lambda self: self.buffer.size)

    def __init__(self, size):
        self.buffer = np.zeros(size)
        self.index = 0

    def reset(self):
        self.buffer.fill(0)
        self.index = 0

    def push(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size

    def get(self):
        return np.concatenate((
            self.buffer[self.index:],
            self.buffer[:self.index]
        ))

    def map(self, func):
        """
        maps a func of the contents of the buffer.
        Warning: the buffer is unordered, so only use functions that are invariant to the ordering (like min, max or mean)
        """
        return func(self.buffer)

class PluginTimeAveraging(PluginMeter):
    # ReadModes = Enum("ReadModes", [
    #     ("mean", np.mean),
    #     ("max", np.max),
    #     ("min", np.min)
    # ])
    # class ReadModes(Enum):
    #     mean = np.mean
    #     max = np.max
    #     min = np.min

    t: float
    time_constant: str =  property(lambda self: f"{self.t:g}")
    _fifo: FIFO

    def __init__(self, time_constant: str, t: float,
                 # read_mode: Enum = ReadModes.mean,
                 read_mode: ReadMode = ReadMode("mean", np.mean),
                 **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self._read_mode = read_mode

        n_blocks = self.t * self.samplerate / self.blocksize
        self._fifo = FIFO(n_blocks)
        self.output = np.zeros((self.input.output.shape[0], 1))

    def process(self):
        self._fifo.push( # store the metric over the block
            self.read_mode.value( # apply the function (mean, max, min, ...) over the input^2
                np.square(self.input.get())
            )
        )
        self.output[:,0] = self._fifo.map(self.read_mode.value)

    def reset(self):
        super().reset()
        self._fifo.reset()

    def read_lin(self):
        return self.output[:,0] / self.t

    def to_str(self):
        return f"{type(self).__name__}(T={self.time_constant}, {self.read_mode.name})"
