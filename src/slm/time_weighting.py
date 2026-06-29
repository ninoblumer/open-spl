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
        # Two cascaded stages (see :func:`_asymmetric_time_weighting`):
        #   stage 1 — symmetric exponential detector,  tau[0] = 35 ms
        #   stage 2 — peak hold decaying with           tau[1] = 1500 ms
        self._alpha_detector = 1 - np.exp(-1 / (self.samplerate * self.tau[0]))
        self._alpha_decay = 1 - np.exp(-1 / (self.samplerate * self.tau[1]))
        self._zi_detector = np.zeros(self.width)
        self._zi_hold = np.zeros(self.width)

    def func(self, block: np.ndarray):
        _asymmetric_time_weighting(
            block, self._zi_detector, self._zi_hold,
            self._alpha_detector, self._alpha_decay, self.output,
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
def _asymmetric_time_weighting(x, zi_detector, zi_hold, alpha_detector, alpha_decay, out):
    """Process one block with the IEC 60651/60804 Impulse time weighting.

    Two cascaded stages:
      1. a *symmetric* exponential detector (tau = 35 ms) on the squared signal,
         which yields the unbiased short-term mean square, then
      2. a peak hold that snaps up to the detector's value and otherwise decays
         exponentially (tau = 1500 ms, ~2.9 dB/s).

    A single asymmetric one-pole on x² (fast attack, slow release) is NOT
    equivalent: it parks near the *peaks* of x², biasing fluctuating signals
    several dB high (~+3 dB on a tone, ~+6 dB on noise).  The symmetric detector
    removes that rectification bias; the hold then captures genuine impulses.

    Parameters
    ----------
    x              : 2-D array (channels, samples) — raw (unsquared) pressure
    zi_detector    : 1-D float64 array (channels,) — stage-1 detector state per channel
    zi_hold        : 1-D float64 array (channels,) — stage-2 hold state per channel
    alpha_detector : float — detector coefficient (= 1 - exp(-1 / (tau_detector * fs)))
    alpha_decay    : float — hold-decay coefficient (= 1 - exp(-1 / (tau_decay * fs)))
    out            : 2-D float64 array (channels, samples) — written in-place
    """
    n_ch, n_samp = x.shape
    for ch in range(n_ch):
        detector = zi_detector[ch]
        hold = zi_hold[ch]
        for n in range(n_samp):
            xn = x[ch, n]
            detector = (1.0 - alpha_detector) * detector + alpha_detector * (xn * xn)
            if detector > hold:
                hold = detector                       # instantaneous attack to the peak
            else:
                hold = (1.0 - alpha_decay) * hold      # slow exponential release
            out[ch, n] = hold
        zi_detector[ch] = detector
        zi_hold[ch] = hold
