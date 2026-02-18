from __future__ import annotations
from enum import Enum

import numpy as np

from slm.plugins.plugin import PluginMeter, ReadMode
from slm.plugins.fifo import FIFO

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

    time_constant: str =  property(lambda self: f"{self.t:g}")

    def __init__(self, time_constant: str,
                 # read_mode: Enum = ReadModes.mean,
                 read_mode: ReadMode = ReadMode("mean", np.mean),
                 **kwargs):
        super().__init__(**kwargs)

        self._read_mode = read_mode


        self.output = np.zeros((self.width, 1))

    def func(self, block: np.ndarray):
        self.output[:,:] = np.square(block)

    def reset(self):
        super().reset()

    def read_lin(self, name: str):
        return super().read_lin(name) / self.t

    # def to_str(self):
    #     return f"{type(self).__name__}(T={self.time_constant}, {self.read_mode.name})"

class PluginExposure(PluginTimeAveraging):
    def __init__(self, **kwargs):
        super().__init__(t=1.0, read_mode = ReadMode("mean", np.mean), **kwargs)
