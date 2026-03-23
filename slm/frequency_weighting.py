import numpy as np

from pyoctaveband import WeightingFilter
from scipy.signal import butter, sosfilt, sosfilt_zi

from slm.plugin_meter import PluginMeter



class PluginFrequencyWeighting(PluginMeter):
    curve: str
    def __init__(self, *, curve: str, zero_zi: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.curve = curve
        self.output = np.zeros((1, self.blocksize))
        self._zero_zi = zero_zi
        self._compute_filter()

    def reset(self):
        super().reset()
        self._compute_filter()

    def _compute_filter(self):
        wf = WeightingFilter(fs=self.samplerate, curve=self.curve)
        self._wf = wf.sos
        self._zi = sosfilt_zi(self._wf)  # avoids ringing of filter at the start.
        shape = self._zi.shape
        self._zi = np.reshape(self._zi, (shape[0], 1, shape[1]))
        if self._zero_zi:
            self._zi = np.zeros_like(self._zi)

    def func(self, block: np.ndarray):
        self.output[0,:], self._zi[:,:] = sosfilt(self._wf, block, zi=self._zi)

    # def implemented_function(self) -> str:
    #     return f"{self.curve}-weighting"

    def to_str(self):
        return f"PluginFrequencyWeighting(curve={self.curve})"


class PluginAWeighting(PluginFrequencyWeighting):
    def __init__(self, **kwargs):
        super().__init__(curve='A', **kwargs)

    def to_str(self):
        return "PluginAWeighting()"


class PluginCWeighting(PluginFrequencyWeighting):
    def __init__(self, **kwargs):
        super().__init__(curve='C', **kwargs)

    def to_str(self):
        return "PluginCWeighting()"


class PluginZWeighting(PluginFrequencyWeighting):
    """Mathematically flat Z-weighting per IEC 61672-1 Annex E.5 (0 dB at all frequencies).

    NOTE: Real hardware SLMs typically apply a high-pass filter in their
    Z-weighted metering path (e.g. the NTi XL2 has an effective ~5 Hz 1-pole HPF).
    When comparing broadband Z-weighted results against a hardware SLM, use
    PluginHPF with an appropriate fc instead of this class.
    """

    def __init__(self, **kwargs):
        super().__init__(curve='Z', **kwargs)

    def _compute_filter(self):
        self._wf = None
        self._zi = None

    def func(self, block: np.ndarray):
        self.output[0,:] = block.copy()

    def to_str(self):
        return "PluginZWeighting()"


class PluginHPF(PluginMeter):
    """Butterworth high-pass filter with parametrized cutoff and order."""

    def __init__(self, *, fc: float, order: int = 1, zero_zi: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.fc = fc
        self.order = order
        self.output = np.zeros((1, self.blocksize))
        self._zero_zi = zero_zi
        self._compute_filter()

    def reset(self):
        super().reset()
        self._compute_filter()

    def _compute_filter(self):
        self._sos = butter(self.order, self.fc, btype='high', fs=self.samplerate, output='sos')
        self._zi = sosfilt_zi(self._sos)
        shape = self._zi.shape
        self._zi = np.reshape(self._zi, (shape[0], 1, shape[1]))
        if self._zero_zi:
            self._zi = np.zeros_like(self._zi)

    def func(self, block: np.ndarray):
        self.output[0, :], self._zi[:, :] = sosfilt(self._sos, block, zi=self._zi)

    def to_str(self):
        return f"PluginHPF(fc={self.fc}, order={self.order})"


class PluginBandpass(PluginMeter):
    """Narrow Butterworth bandpass filter (1/3-octave bandwidth around fc)."""

    def __init__(self, *, fc: float, order: int = 2, zero_zi: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.fc = fc
        self.order = order
        self.output = np.zeros((1, self.blocksize))
        self._zero_zi = zero_zi
        self._compute_filter()

    def reset(self):
        super().reset()
        self._compute_filter()

    def _compute_filter(self):
        factor = 2 ** (1 / 6)
        sos = butter(self.order, [self.fc / factor, self.fc * factor],
                     btype='bandpass', fs=self.samplerate, output='sos')
        self._sos = sos
        zi = sosfilt_zi(sos)
        self._zi = np.reshape(zi, (zi.shape[0], 1, zi.shape[1]))
        if self._zero_zi:
            self._zi = np.zeros_like(self._zi)

    def func(self, block: np.ndarray):
        self.output[0, :], self._zi[:, :] = sosfilt(self._sos, block, zi=self._zi)

    def to_str(self):
        return f"PluginBandpass(fc={self.fc}, order={self.order})"
