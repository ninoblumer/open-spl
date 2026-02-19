import numpy as np

from pyoctaveband import WeightingFilter
from scipy.signal import sosfilt, sosfilt_zi

from slm.plugin import Plugin
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
    def __init__(self, **kwargs):
        super().__init__(curve='Z', **kwargs)

    def _compute_filter(self, zero_zi:bool=False):
        self._wf = None
        self._zi = None

    def func(self, block: np.ndarray):
        self.output[0,:] = block.copy()

    def to_str(self):
        return "PluginZWeighting()"
