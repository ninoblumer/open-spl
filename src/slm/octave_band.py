from __future__ import annotations

import numpy as np
from numba import njit, prange

from phonometry import OctaveFilterBank

from slm.plugin_meter import PluginMeter


class PluginOctaveBand(PluginMeter):
    n_bands: int = property(lambda self: self._filter_bank.num_bands)
    center_frequencies: list[str] = property(lambda self: self._filter_bank.nominal_freq)

    _filter_bank: OctaveFilterBank

    def __init__(self, limits: tuple[float, float], bands_per_oct: float = 1.0, order: int = 6,
                 filter_type: str = "butter", ripple: float=0.1, attenuation: float=60, **kwargs):
        super().__init__(**kwargs)

        if self.input.width != 1:
            raise ValueError("OctaveBandPlugin only supports inputs of width=1")

        self._filter_bank = OctaveFilterBank(fs=self.samplerate, fraction=bands_per_oct, limits=list(limits),
                                             show=False, order=order, filter_type=filter_type,
                                             ripple=ripple, attenuation=attenuation,
                                             stateful=True, steady_ic=False, resample=False)

        self._width = self.n_bands
        self.output = np.zeros((self.n_bands, self.blocksize))

        # Stack SOS coefficients into a single contiguous array: [n_bands, n_sections, 6]
        self._sos_stack = np.ascontiguousarray(self._filter_bank.sos, dtype=np.float64)

        # Stacked filter states (zero initial condition): [n_bands, n_sections, 2]
        n_sections = self._sos_stack.shape[1]
        self._zi_stack = np.zeros((self.n_bands, n_sections, 2), dtype=np.float64)

        # Force JIT compilation now, before any real audio arrives.
        # cache=True on the function means subsequent process starts skip this.
        _sosfilt_all_bands(self._sos_stack, np.zeros(self.blocksize, dtype=np.float64),
                           self._zi_stack, self.output)
        # Reset state and output dirtied by the warmup call
        self._zi_stack[:] = 0.0
        self.output[:] = 0.0

    def func(self, block: np.ndarray):
        _sosfilt_all_bands(self._sos_stack, block[0], self._zi_stack, self.output)

    def to_str(self):
        return f"{type(self).__name__}"


#  Excluded from test coverage: Numba compiles this to machine code, so coverage.py cannot trace its body.
#  It is verified functionally, both against the IEC 61260-1:2014 class-1 filter conformance limits
#  (tests/iec61260/) and against a hardware RTA reference (integration tests).
@njit(parallel=True, cache=True)
def _sosfilt_all_bands(  # pragma: no cover
    sos_stack: np.ndarray,
    x: np.ndarray,
    zi_stack: np.ndarray,
    out: np.ndarray,
) -> None:
    """Apply SOS filter bank to mono input with all bands processed in parallel.

    sos_stack : float64[n_bands, n_sections, 6]
    x         : float64[blocksize]  1-D mono input
    zi_stack  : float64[n_bands, n_sections, 2]  updated in-place
    out       : float64[n_bands, blocksize]  written in-place
    """
    n_bands = sos_stack.shape[0]
    n_sections = sos_stack.shape[1]
    n_samples = x.shape[0]

    for b in prange(n_bands):
        # seed each band's output with the input signal
        for n in range(n_samples):
            out[b, n] = x[n]

        # apply sections sequentially (IIR data dependency within each band)
        for s in range(n_sections):
            b0 = sos_stack[b, s, 0]
            b1 = sos_stack[b, s, 1]
            b2 = sos_stack[b, s, 2]
            # sos_stack[b, s, 3] is a0, always 1.0 for normalised SOS
            a1 = sos_stack[b, s, 4]
            a2 = sos_stack[b, s, 5]
            z0 = zi_stack[b, s, 0]
            z1 = zi_stack[b, s, 1]

            for n in range(n_samples):
                xn = out[b, n]
                yn = b0 * xn + z0
                z0 = b1 * xn - a1 * yn + z1
                z1 = b2 * xn - a2 * yn
                out[b, n] = yn

            zi_stack[b, s, 0] = z0
            zi_stack[b, s, 1] = z1
