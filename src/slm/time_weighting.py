from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from numba import jit

from slm.plugin_meter import PluginMeter


class PluginTimeWeighting(PluginMeter, ABC):
    time_constant: str

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

    def __init__(self, *, time_constant: str, tau: float, **kwargs):
        super().__init__(**kwargs)
        self._zi = None
        self.time_constant = time_constant
        self.tau = tau
        self._compute_filter()

    def _compute_filter(self):
        self._alpha = float(1 - np.exp(-1 / (self.tau * self.samplerate)))
        # Steady-state initial condition for unit input is 1.0; zeros if zero_zi.
        init = 0.0 if self._zero_zi else 1.0
        self._zi = np.full(self.width, init, dtype=np.float64)

    def func(self, block: np.ndarray):
        _symmetric_time_weighting(block, self._zi, self._alpha, self.output)


class PluginAsymmetricTimeWeighting(PluginTimeWeighting):
    tau: tuple[float, float]

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
        _asymmetric_time_weighting(block, self._zi, self._alpha_rise, self._alpha_fall, self.output)


class PluginFastTimeWeighting(PluginSymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="fast", tau=0.125, **kwargs)

class PluginSlowTimeWeighting(PluginSymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="slow", tau=1.0, **kwargs)

class PluginImpulseTimeWeighting(PluginAsymmetricTimeWeighting):
    def __init__(self, **kwargs):
        super().__init__(time_constant="impulse", tau=(0.035, 1.500), **kwargs)


class PluginSquare(PluginTimeWeighting):
    """Instantaneous squaring — output = input².

    Used for peak-level measurements where no time constant is desired.
    Attaches to a frequency-weighting output (linear Pa); output is Pa²
    so that MaxAccumulator can be used directly (consistent with all other
    time-weighting plugins).
    """
    time_constant = "peak"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compute_filter()

    def _compute_filter(self):
        pass  # no filter state

    def func(self, block: np.ndarray):
        np.square(block, out=self.output)


#  Excluded from test coverage: Numba compiles this to machine code, so coverage.py cannot trace its body.
#  It is verified functionally against the IEC 61672-1 F/S time-weighting and tone-burst response tests
#  (tests/iec61672/test_61672_time_weightings.py, tests/iec61672/test_61672_toneburst.py).
@jit(nopython=True, cache=True)  # pragma: no cover
def _symmetric_time_weighting(x, zi, alpha, out):
    """
    Process one block with IEC 61672-1 F or S time weighting.

    Parameters
    ----------
    x   : 2-D array (channels, samples) — raw (unsquared) pressure
    zi  : 1-D float64 array (channels,) — IIR state per channel
    alpha : float — IIR coefficient  (= 1 - exp(-1 / (tau * fs)))
    out : 2-D float64 array (channels, samples) — written in-place


    """
    n_ch, n_samp = x.shape
    one_minus_alpha = 1.0 - alpha
    for ch in range(n_ch):
        state = zi[ch]
        for n in range(n_samp):
            xn = x[ch, n]
            state = one_minus_alpha * state + alpha * (xn * xn)
            out[ch, n] = state
        zi[ch] = state


#  Excluded from test coverage: Numba compiles this to machine code, so coverage.py cannot trace its body.
#  It is verified functionally against the IEC 61672-1 Impulse time-weighting tests
#  (tests/unit/test_impulse_time_weighting.py).
@jit(nopython=True, cache=True)  # pragma: no cover
def _asymmetric_time_weighting(x, zi, alpha_rise, alpha_fall, out):
    """Process one block with IEC 61672-1 Impulse time weighting.

    Parameters
    ----------
    x          : 2-D array (channels, samples) — raw (unsquared) pressure
    zi         : 1-D float64 array (channels,) — IIR state per channel
    alpha_rise : float — rise coefficient  (= 1 - exp(-1 / (tau_rise * fs)))
    alpha_fall : float — fall coefficient  (= 1 - exp(-1 / (tau_fall * fs)))
    out        : 2-D float64 array (channels, samples) — written in-place
    """
    n_ch, n_samp = x.shape
    for ch in range(n_ch):
        state = zi[ch]
        for n in range(n_samp):
            xn = x[ch, n]
            x2 = xn * xn
            a = alpha_rise if x2 > state else alpha_fall
            state = (1.0 - a) * state + a * x2
            out[ch, n] = state
        zi[ch] = state
