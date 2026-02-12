import numpy as np

from pyoctaveband import WeightingFilter
from scipy.signal import sosfilt, sosfilt_zi

from slm.plugins.plugin import Plugin



class PluginFrequencyWeighting(Plugin):
    curve: str
    def __init__(self, *, curve: str, **kwargs):
        super().__init__(**kwargs)
        self.curve = curve
        self.output = np.zeros((1, self.blocksize))
        self._compute_filter()

    def reset(self):
        super().reset()
        self._compute_filter()

    def _compute_filter(self, zero_zi:bool=False):
        wf = WeightingFilter(fs=self.samplerate, curve=self.curve)
        self._wf = wf.sos
        self._zi = sosfilt_zi(self._wf)  # avoids ringing of filter at the start.
        if zero_zi:
            self._zi = np.zeros_like(self._zi)

    def process(self):
        self.output[0,:], self._zi[:,:] = sosfilt(self._wf, self.input.get(), zi=self._zi)

    # def implemented_function(self) -> str:
    #     return f"{self.curve}-weighting"

    def to_str(self):
        return f"{type(self).__name__}(curve={self.curve})"


class PluginAWeighting(PluginFrequencyWeighting):
    def __init__(self, **kwargs):
        super().__init__(curve='A', **kwargs)


class PluginCWeighting(PluginFrequencyWeighting):
    def __init__(self, **kwargs):
        super().__init__(curve='C', **kwargs)


class PluginZWeighting(PluginFrequencyWeighting):
    def __init__(self, **kwargs):
        super().__init__(curve='Z', **kwargs)

    def _compute_filter(self, zero_zi:bool=False):
        self._wf = None
        self._zi = None

    def process(self):
        self.output[0,:] = self.input.get()
