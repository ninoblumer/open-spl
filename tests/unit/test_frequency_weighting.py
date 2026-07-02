"""Unit tests for frequency weighting plugins: reset() and to_str()."""
from __future__ import annotations

import numpy as np
import pytest


SAMPLERATE = 48_000
BLOCKSIZE = 1_024


def _make_engine():
    from slm.io.noise_controller import NoiseController
    from slm.engine import Engine
    ctrl = NoiseController(samplerate=SAMPLERATE, blocksize=BLOCKSIZE)
    ctrl.set_sensitivity(1.0, unit="V")
    return Engine(ctrl, dt=1.0)


def _add_bus(engine, weighting_cls):
    bus = engine.add_bus("test", weighting_cls)
    return bus.frequency_weighting


# ---------------------------------------------------------------------------
# PluginAWeighting
# ---------------------------------------------------------------------------

class TestPluginAWeighting:

    def test_to_str(self):
        from slm.frequency_weighting import PluginAWeighting
        plugin = _add_bus(_make_engine(), PluginAWeighting)
        assert plugin.to_str() == "PluginAWeighting()"

    def test_reset_clears_output(self):
        from slm.frequency_weighting import PluginAWeighting
        engine = _make_engine()
        bus = engine.add_bus("A", PluginAWeighting)
        plugin = bus.frequency_weighting
        block = np.ones((1, BLOCKSIZE))
        plugin.process(block)
        plugin.reset()
        np.testing.assert_array_equal(plugin.output, 0.0)


# ---------------------------------------------------------------------------
# PluginCWeighting
# ---------------------------------------------------------------------------

class TestPluginCWeighting:

    def test_to_str(self):
        from slm.frequency_weighting import PluginCWeighting
        plugin = _add_bus(_make_engine(), PluginCWeighting)
        assert plugin.to_str() == "PluginCWeighting()"

    def test_reset_clears_output(self):
        from slm.frequency_weighting import PluginCWeighting
        engine = _make_engine()
        bus = engine.add_bus("C", PluginCWeighting)
        plugin = bus.frequency_weighting
        block = np.ones((1, BLOCKSIZE))
        plugin.process(block)
        plugin.reset()
        np.testing.assert_array_equal(plugin.output, 0.0)


# ---------------------------------------------------------------------------
# PluginZWeighting
# ---------------------------------------------------------------------------

class TestPluginZWeighting:

    def test_to_str(self):
        from slm.frequency_weighting import PluginZWeighting
        plugin = _add_bus(_make_engine(), PluginZWeighting)
        assert plugin.to_str() == "PluginZWeighting()"

    def test_reset_clears_output(self):
        from slm.frequency_weighting import PluginZWeighting
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        plugin = bus.frequency_weighting
        block = np.ones((1, BLOCKSIZE))
        plugin.process(block)
        plugin.reset()
        np.testing.assert_array_equal(plugin.output, 0.0)


# ---------------------------------------------------------------------------
# PluginFrequencyWeighting base to_str
# ---------------------------------------------------------------------------

class TestPluginFrequencyWeightingBaseToStr:

    def test_base_to_str_includes_curve(self):
        from slm.frequency_weighting import PluginFrequencyWeighting
        engine = _make_engine()
        bus = engine.add_bus("base", PluginFrequencyWeighting.__subclasses__()[0])
        plugin = bus.frequency_weighting
        # base class to_str still accessible via super() — test directly
        result = PluginFrequencyWeighting.to_str(plugin)
        assert "PluginFrequencyWeighting" in result or "curve=" in result


# ---------------------------------------------------------------------------
# PluginHPF
# ---------------------------------------------------------------------------

class TestPluginHPF:

    def _make_hpf(self):
        from slm.frequency_weighting import PluginZWeighting, PluginHPF
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        fw = bus.frequency_weighting
        plugin = PluginHPF(fc=5.0, order=1, input=fw)
        bus.add_plugin(plugin)
        return plugin

    def test_to_str(self):
        plugin = self._make_hpf()
        assert "PluginHPF" in plugin.to_str()
        assert "fc=5.0" in plugin.to_str()

    def test_reset_clears_output(self):
        plugin = self._make_hpf()
        block = np.ones((1, BLOCKSIZE))
        plugin.process(block)
        plugin.reset()
        np.testing.assert_array_equal(plugin.output, 0.0)


# ---------------------------------------------------------------------------
# PluginBandpass
# ---------------------------------------------------------------------------

class TestPluginBandpass:

    def _make_bp(self):
        from slm.frequency_weighting import PluginZWeighting, PluginBandpass
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        fw = bus.frequency_weighting
        plugin = PluginBandpass(fc=1000.0, order=2, input=fw)
        bus.add_plugin(plugin)
        return plugin

    def test_to_str(self):
        plugin = self._make_bp()
        assert "PluginBandpass" in plugin.to_str()
        assert "fc=1000.0" in plugin.to_str()

    def test_reset_clears_output(self):
        plugin = self._make_bp()
        block = np.ones((1, BLOCKSIZE))
        plugin.process(block)
        plugin.reset()
        np.testing.assert_array_equal(plugin.output, 0.0)


# ---------------------------------------------------------------------------
# PluginInputFilter / PluginXL2InputFilter
# ---------------------------------------------------------------------------

def _make_input_filter(cls=None, **kwargs):
    """Build a filter plugin wired behind a Z-weighting bus.

    Defaults to the XL2 preset (which supplies filter values); pass *cls* and
    explicit cutoffs/orders to exercise the generic base.
    """
    from slm.frequency_weighting import PluginZWeighting, PluginXL2InputFilter
    if cls is None:
        cls = PluginXL2InputFilter
    engine = _make_engine()
    bus = engine.add_bus("Z", PluginZWeighting)
    fw = bus.frequency_weighting
    plugin = cls(input=fw, **kwargs)
    bus.add_plugin(plugin)
    return plugin


def _steady_state(plugin, signal):
    """Feed *signal* block by block; return the last (settled) output block."""
    out = None
    for start in range(0, len(signal) - BLOCKSIZE, BLOCKSIZE):
        block = signal[start:start + BLOCKSIZE].reshape(1, BLOCKSIZE)
        plugin.process(block)
        out = plugin.output.copy()
    return out[0]


class TestPluginXL2InputFilter:

    def test_defaults_match_spec(self):
        from slm.frequency_weighting import PluginXL2InputFilter
        plugin = _make_input_filter()
        assert plugin.hpf_fc == PluginXL2InputFilter.HPF_FC == 4.4
        assert plugin.lpf_fc == PluginXL2InputFilter.LPF_FC == 23_000.0
        assert plugin.hpf_order == 4 and plugin.lpf_order == 4

    def test_to_str(self):
        plugin = _make_input_filter()
        assert "PluginXL2InputFilter" in plugin.to_str()
        assert "hpf_fc=4.4" in plugin.to_str()
        assert "lpf_fc=23000.0" in plugin.to_str()

    def test_reset_clears_output(self):
        plugin = _make_input_filter()
        plugin.process(np.ones((1, BLOCKSIZE)))
        plugin.reset()
        np.testing.assert_array_equal(plugin.output, 0.0)

    def test_passband_is_unity(self):
        """A 1 kHz tone sits in the passband → gain ≈ 0 dB."""
        t = np.arange(SAMPLERATE) / SAMPLERATE
        tone = np.sin(2 * np.pi * 1000.0 * t)
        out = _steady_state(_make_input_filter(), tone)
        rms = np.sqrt(np.mean(out ** 2))
        assert abs(20 * np.log10(rms / (1 / np.sqrt(2)))) < 0.1

    def test_rejects_dc(self):
        """DC is far below the 4.4 Hz high-pass → strongly attenuated."""
        out = _steady_state(_make_input_filter(), np.ones(SAMPLERATE))
        assert np.max(np.abs(out)) < 1e-3


class TestPluginInputFilter:
    """The generic, parametrized base filter."""

    def _generic(self, **kwargs):
        from slm.frequency_weighting import PluginInputFilter
        return _make_input_filter(cls=PluginInputFilter, **kwargs)

    def test_requires_explicit_parameters(self):
        from slm.frequency_weighting import PluginInputFilter
        with pytest.raises(TypeError):
            _make_input_filter(cls=PluginInputFilter)  # no cutoffs/orders

    def test_negative_order_rejected(self):
        with pytest.raises(ValueError):
            self._generic(hpf_fc=20.0, hpf_order=-1, lpf_fc=20_000.0, lpf_order=4)

    def test_to_str_uses_class_name(self):
        plugin = self._generic(hpf_fc=20.0, hpf_order=2, lpf_fc=20_000.0, lpf_order=2)
        s = plugin.to_str()
        assert "PluginInputFilter" in s
        assert "hpf_order=2" in s and "lpf_order=2" in s

    def test_hpf_order_zero_disables_highpass(self):
        """hpf_order=0 → DC passes (no high-pass), only the low-pass remains."""
        plugin = self._generic(hpf_fc=100.0, hpf_order=0, lpf_fc=20_000.0, lpf_order=4)
        out = _steady_state(plugin, np.ones(SAMPLERATE))
        assert np.max(np.abs(out)) > 0.5  # DC not attenuated

    def test_lpf_order_zero_disables_lowpass(self):
        """lpf_order=0 → a tone above the (disabled) LPF cutoff passes through."""
        plugin = self._generic(hpf_fc=4.4, hpf_order=4, lpf_fc=1_000.0, lpf_order=0)
        t = np.arange(SAMPLERATE) / SAMPLERATE
        tone = np.sin(2 * np.pi * 5000.0 * t)  # well above 1 kHz
        out = _steady_state(plugin, tone)
        rms = np.sqrt(np.mean(out ** 2))
        assert abs(20 * np.log10(rms / (1 / np.sqrt(2)))) < 0.1

    def test_both_orders_zero_passthrough(self):
        plugin = self._generic(hpf_fc=100.0, hpf_order=0, lpf_fc=1_000.0, lpf_order=0)
        block = np.random.default_rng(0).standard_normal((1, BLOCKSIZE))
        plugin.process(block)
        np.testing.assert_allclose(plugin.output, block)
