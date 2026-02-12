import numpy as np

from pyoctaveband import WeightingFilter
from scipy.signal import sosfilt, sosfilt_zi

from slm.plugins.base import Plugin



class PluginFrequencyWeighting(Plugin):
    function = property(lambda self: f"{self.curve}-weighting")

    def __init__(self, *, curve: str, **kwargs):
        super().__init__(**kwargs)

        self.curve = curve
        self._compute_filter()

    def _compute_filter(self, zero_zi:bool=False):
        self._wf = WeightingFilter(fs=self.samplerate, curve=self.curve)
        self._zi = sosfilt_zi(self._wf)  # avoids ringing of filter at the start.
        if zero_zi:
            self._zi = np.zeros_like(self._zi)

    def func(self, block: np.ndarray) -> np.ndarray:
        result, self._zi = sosfilt(self._wf, block, zi=self._zi)
        return result


class PluginAWeighting(PluginFrequencyWeighting):
    def __init__(self, **_):
        super().__init__(curve='A')


class PluginCWeighting(PluginFrequencyWeighting):
    def __init__(self, **_):
        super().__init__(curve='C')


class PluginZWeighting(PluginFrequencyWeighting):
    def __init__(self, **_):
        super().__init__(curve='Z')

    def _compute_filter(self, zero_zi:bool=False):
        self._wf = None
        self._zi = None

    def func(self, block: np.ndarray) -> np.ndarray:
        return block.copy()
