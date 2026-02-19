from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import signal as sig

from pyoctaveband import OctaveFilterBank

from slm.plugin_meter import PluginMeter

if TYPE_CHECKING:
    from slm.plugin import Plugin



class StatefulOctaveFilterBank(OctaveFilterBank):
    def __init__(self, zero_zi: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._zero_zi = zero_zi
        self._states: list[np.ndarray] = [np.expand_dims(sig.sosfilt_zi(self.sos[idx]), axis=1) for idx in range(self.num_bands)]
        # self._states = []
        # for idx in range(self.num_bands):
        #     zi = sig.sosfilt_zi(self.sos[idx])
        #     zi = zi[:, np.newaxis, :]
        #     self._states.append(zi)


        if self._zero_zi:
            for idx in range(self.num_bands):
                self._states[idx].fill(0)

    def _filter_and_resample(self, x: np.ndarray, idx: int) -> np.ndarray:
        """Resample and filter for a specific band (vectorized)."""
        if self.factor[idx] > 1:
            # axis=-1 is default for resample_poly, but being explicit is good
            sd = sig.resample_poly(x, 1, self.factor[idx], axis=-1)
        else:
            sd = x

        # sosfilt supports axis=-1 by default
        result, self._states[idx][:,:,:] = sig.sosfilt(self.sos[idx], sd, axis=-1, zi=self._states[idx])
        return cast(np.ndarray, result)


class PluginOctaveBand(PluginMeter):
    n_bands: int = property(lambda self: self._filter_bank.num_bands)
    center_frequencies: list[float] = property(lambda self: self._filter_bank.freq)

    _filter_bank: StatefulOctaveFilterBank



    def __init__(self, limits: tuple[float, float], bands_per_oct: float = 1.0, order: int = 6, detrend: bool = True,
                 filter_type: str = "butter", ripple: float=0.1, attenuation: float=60, zero_zi: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._zero_zi = zero_zi
        self._detrend = detrend

        if self.width != 1:
            raise ValueError("OctaveBandPlugin only supports inputs of width=1")

        self._filter_bank = StatefulOctaveFilterBank(zero_zi=zero_zi, fs=self.samplerate, fraction=bands_per_oct,
                                                     limits=list(limits), show=False,
                                                     order=order, filter_type=filter_type, ripple=ripple, attenuation=attenuation)

        self.output = np.zeros((self.n_bands, self.blocksize))

    def func(self, block: np.ndarray):
        spl_array, freqs, signals = self._filter_bank.filter(block, sigbands=True, detrend=self._detrend)
        self.output[:, :] = np.vstack(signals)

    def to_str(self):
        return f"{type(self).__name__}"