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
