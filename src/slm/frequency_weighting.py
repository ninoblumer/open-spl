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

    NOTE: Real hardware SLMs band-limit their input before broadband metering
    (typically a 4th-order Butterworth high-pass around 4.4 Hz and a 4th-order
    Butterworth low-pass around 23.0 kHz). When comparing broadband Z-weighted
    results against band-limited hardware, use PluginInputFilter instead of this
    flat class.
    """

    def __init__(self, **kwargs):
        super().__init__(curve='Z', **kwargs)

    def _compute_filter(self):
        self._wf = None
        self._zi = None

    def func(self, block: np.ndarray):
        self.output[0, :] = block

    def to_str(self):
        return "PluginZWeighting()"


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


class PluginInputFilter(PluginMeter):
    """Generic band-limiting analog input filter (parametrized HPF + LPF).

    Cascades an optional Butterworth high-pass with an optional Butterworth
    low-pass into a single SOS chain, band-limiting the signal to a measurement
    passband. Sits in place of the mathematically flat PluginZWeighting when
    comparing broadband results against band-limited hardware; the two stages
    are cascaded into one SOS chain so the whole filter is applied in one pass.

    Each stage's order selects its behaviour: a positive order builds that
    Butterworth stage, an order of 0 disables it, and a negative order is
    rejected. With both orders 0 the filter is a pass-through.

    Cutoffs and orders are supplied per instance; :class:`PluginXL2InputFilter`
    is a preset with typical measurement-SLM values.
    """

    def __init__(self, *, hpf_fc: float, hpf_order: int,
                 lpf_fc: float, lpf_order: int,
                 zero_zi: bool = True, **kwargs):
        super().__init__(**kwargs)
        if hpf_order < 0 or lpf_order < 0:
            raise ValueError(
                "filter orders must be >= 0 (0 disables the stage), got "
                f"hpf_order={hpf_order}, lpf_order={lpf_order}")
        self.hpf_fc = hpf_fc
        self.hpf_order = hpf_order
        self.lpf_fc = lpf_fc
        self.lpf_order = lpf_order
        self.output = np.zeros((1, self.blocksize))
        self._zero_zi = zero_zi
        self._compute_filter()

    def reset(self):
        super().reset()
        self._compute_filter()

    def _compute_filter(self):
        stages = []
        if self.hpf_order > 0:
            stages.append(butter(self.hpf_order, self.hpf_fc, btype='high',
                                 fs=self.samplerate, output='sos'))
        if self.lpf_order > 0:
            stages.append(butter(self.lpf_order, self.lpf_fc, btype='low',
                                 fs=self.samplerate, output='sos'))
        if stages:
            self._sos = np.vstack(stages)
            zi = sosfilt_zi(self._sos)
            self._zi = np.reshape(zi, (zi.shape[0], 1, zi.shape[1]))
            if self._zero_zi:
                self._zi = np.zeros_like(self._zi)
        else:
            # Both stages disabled → pass-through.
            self._sos = None
            self._zi = None

    def func(self, block: np.ndarray):
        if self._sos is None:
            self.output[0, :] = block
        else:
            self.output[0, :], self._zi[:, :] = sosfilt(self._sos, block, zi=self._zi)

    def to_str(self):
        return (f"{type(self).__name__}(hpf_fc={self.hpf_fc}, hpf_order={self.hpf_order}, "
                f"lpf_fc={self.lpf_fc}, lpf_order={self.lpf_order})")


class PluginXL2InputFilter(PluginInputFilter):
    """Preset input filter matching a common measurement-SLM analog front end.

    4th-order Butterworth high-pass at 4.4 Hz cascaded with a 4th-order
    Butterworth low-pass at 23.0 kHz. Values default to the HPF_*/LPF_* class
    attributes but can be overridden per instance for experimentation.
    """

    HPF_FC = 4.4          # Hz, 4th-order Butterworth high-pass
    HPF_ORDER = 4
    LPF_FC = 23_000.0     # Hz, 4th-order Butterworth low-pass
    LPF_ORDER = 4

    def __init__(self, *, hpf_fc: float = HPF_FC, hpf_order: int = HPF_ORDER,
                 lpf_fc: float = LPF_FC, lpf_order: int = LPF_ORDER,
                 zero_zi: bool = True, **kwargs):
        super().__init__(hpf_fc=hpf_fc, hpf_order=hpf_order,
                         lpf_fc=lpf_fc, lpf_order=lpf_order,
                         zero_zi=zero_zi, **kwargs)
