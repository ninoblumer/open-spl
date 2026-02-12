from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.signal import lfilter, lfilter_zi
from numba import jit

from slm.plugins.base import PluginMeter


class PluginABCTimeWeighting(PluginMeter, ABC):
    class ReadModes(Enum):
        max = "max"
        min = "min"
        last = "last"

    _read_mode: ReadModes
    read_mode: ReadModes = property(lambda self: self._read_mode)

    def __init__(self, read_mode: ReadModes = ReadModes.last, zero_zi: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._read_mode = read_mode
        self._zero_zi = zero_zi

    @abstractmethod
    def _compute_filter(self) -> None: ...

    def reset(self):
        self._compute_filter()


class PluginTimeWeighting(PluginABCTimeWeighting):
    tau: float
    time_constant: str
    function = property(lambda self: f"{self.time_constant}-time-weighting")

    def __init__(self, *, time_constant: str, tau: float, **kwargs):
        super().__init__(**kwargs)
        self.time_constant = time_constant
        self.tau = tau
        self._compute_filter()

    def _compute_filter(self):
        alpha = 1 - np.exp(-1 / (self.samplerate * self.tau))
        self._b = [alpha]
        self._a = [1, -(1 - alpha)]
        self._zi = lfilter_zi(self._b, self._a)
        if self._zero_zi:
            self._zi = np.zeros_like(self._zi)

    def func(self, block: np.ndarray) -> np.ndarray:
        x_sq = np.square(block)
        result, self._zi = lfilter(self._b, self._a, x_sq, axis=-1, zi=self._zi)
        return result[-1:] # only return reading at end of block. time weighting is rate reduction


class PluginAsymmetricTimeWeighting(PluginABCTimeWeighting):
    tau: tuple[float, float]
    time_constant: str
    function = property(lambda self: f"{self.time_constant}-time-weighting")

    def __init__(self, *, time_constant: str, tau: tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.time_constant = time_constant
        self.tau = tau
        self._compute_filter()

    def _compute_filter(self):
        self._alpha_rise = 1 - np.exp(-1 / (self.samplerate * self.tau[0]))
        self._alpha_fall = 1 - np.exp(-1 / (self.samplerate * self.tau[1]))
        self._zi = np.zeros((1,1))

    def func(self, block: np.ndarray) -> np.ndarray:
        x_sq = np.square(block)
        result, self.zi = asymmetric_time_weighting(x_sq, zi=self._zi,
                                                    alpha=self._alpha_rise, alpha_fall=self._alpha_fall)
        return result[-1:] # only return reading at end of block. time weighting is rate reduction


class PluginFastTimeWeighting(PluginTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="fast", tau=0.125, **kwargs)

class PluginSlowTimeWeighting(PluginTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="slow", tau=1.0, **kwargs)

class PluginImpulseTimeWeighting(PluginAsymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="impulse", tau=(0.035, 1.500), **kwargs)


@jit(nopython=True)
def asymmetric_time_weighting(x, zi, alpha_rise, alpha_fall):
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
