from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import lfilter, lfilter_zi
from numba import jit

from slm.plugin_meter import PluginMeter


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

    def __init__(self, zero_zi: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._zero_zi = zero_zi
        self.output = np.zeros((self.width, self.blocksize))

    @abstractmethod
    def _compute_filter(self) -> None: ...

    def reset(self):
        super().reset()
        self._compute_filter()

    def to_str(self):
        return f"{type(self).__name__}({self.time_constant})"


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
        alpha = 1 - np.exp(-1 / (self.tau*self.samplerate))
        self._b = [alpha]
        self._a = [1, -(1 - alpha)]
        zi = lfilter_zi(self._b, self._a)
        self._zi = np.tile(zi, (self.width, 1))

        if self._zero_zi:
            self._zi.fill(0)

    def func(self, block: np.ndarray):
        self.output[:,:], self._zi[:,:] = lfilter(self._b, self._a,
                                        np.square(block),
                                        axis=-1, zi=self._zi)


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
        self._zi = np.zeros(self.width)

    def func(self, block: np.ndarray):
        x2 = np.square(block)
        for ch in range(self.width):
            self.output[ch], self._zi[ch] = asymmetric_time_weighting(
                x2[ch], zi=self._zi[ch],
                alpha_rise=self._alpha_rise, alpha_fall=self._alpha_fall
            )


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

        yn = (1.0 - a) * prev + a * x[n]
        y[n] = yn
        prev = yn

    return y, prev
