from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.signal import lfilter, lfilter_zi
from numba import jit

from slm.plugins.plugin import PluginMeter, ReadMode


class PluginTimeWeighting(PluginMeter, ABC):
    time_constant: str
    # ReadModes = Enum("ReadModes", [
    #     ("last", lambda a: a[:,-1]),
    #     ("max", np.max),
    #     ("min", np.min)
    # ])

    # class ReadModes(Enum):
    #     last = lambda a: a[:,-1]
    #     max = np.max
    #     min = np.min

    def __init__(self,
                 #read_mode: Enum = ReadModes.last,
                 read_mode: ReadMode = ReadMode("last", lambda a: a[:,-1]),
                 zero_zi: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._read_mode = read_mode
        self._zero_zi = zero_zi
        self.output = np.zeros((self.input.output.shape[0], 1))

    @abstractmethod
    def _compute_filter(self) -> None: ...

    def reset(self):
        super().reset()
        self._compute_filter()

    def to_str(self):
        return f"{type(self).__name__}({self.time_constant}, {self.read_mode.name})"


class PluginSymmetricTimeWeighting(PluginTimeWeighting):
    tau: float
    # function = property(lambda self: f"{self.time_constant}-time-weighting")

    def __init__(self, *, time_constant: str, tau: float, **kwargs):
        super().__init__(**kwargs)
        self._zi = None
        self.time_constant = time_constant
        self.tau = tau
        self._compute_filter()

    def _compute_filter(self):
        alpha = 1 - np.exp(-1 / (self.samplerate * self.tau))
        self._b = [alpha]
        self._a = [1, -(1 - alpha)]
        zi = lfilter_zi(self._b, self._a)
        Nbands = self.input.output.shape[0]
        self._zi = np.tile(zi, (Nbands, 1))

        if self._zero_zi:
            self._zi.fill(0)

    def process(self):
        result, self._zi[:,:] = lfilter(self._b, self._a,
                                        np.square(self.input.get()),
                                        axis=-1, zi=self._zi)
        self.output[:,0] = self.read_mode.value(result)

    def read_lin(self):
        return self.output[:,0] / self.tau


class PluginAsymmetricTimeWeighting(PluginTimeWeighting):
    tau: tuple[float, float]
    # function = property(lambda self: f"{self.time_constant}-time-weighting")

    def __init__(self, *, time_constant: str, tau: tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.time_constant = time_constant
        self.tau = tau
        self._compute_filter()

    def _compute_filter(self):
        self._alpha_rise = 1 - np.exp(-1 / (self.samplerate * self.tau[0]))
        self._alpha_fall = 1 - np.exp(-1 / (self.samplerate * self.tau[1]))
        self._zi = np.zeros((1,1))

    def process(self):
        result, self._zi[:,:] = asymmetric_time_weighting(np.square(self.input.get()),
                                                     zi=self._zi,
                                                     alpha_rise=self._alpha_rise, alpha_fall=self._alpha_fall
                                                     )
        self.output[:,0] = self.read_mode.value(result)


class PluginFastTimeWeighting(PluginSymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="fast", tau=0.125, **kwargs)

class PluginSlowTimeWeighting(PluginSymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="slow", tau=1.0, **kwargs)

class PluginImpulseTimeWeighting(PluginAsymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="impulse", tau=(0.035, 1.500), **kwargs)


@jit(nopython=True)
def asymmetric_time_weighting(x, *, zi, alpha_rise, alpha_fall):
    """
    Process one block with IEC 61672-1 Impulse time weighting.

    Parameters
    ----------
    x : ndarray
        Input block (squared pressure)
    z : float
        Filter state (previous output sample)
    alpha_rise : float
        Rise coefficient (35 ms)
    alpha_fall : float
        Fall coefficient (1500 ms)

    Returns
    -------
    y : ndarray
        Output block
    z_new : float
        Updated filter state
    """
    y = np.zeros_like(x)
    prev = zi

    for n in range(len(x)):
        if x[n] > prev:
            a = alpha_rise
        else:
            a = alpha_fall

        yn = a * prev + (1.0 - a) * x[n]
        y[n] = yn
        prev = yn

    return y, prev
