"""Unit tests for Plugin, Bus, and Engine internals."""
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


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class TestPlugin:

    def test_reset_fills_output_with_zeros(self):
        from slm.frequency_weighting import PluginAWeighting
        engine = _make_engine()
        bus = engine.add_bus("A", PluginAWeighting)
        plugin = bus.frequency_weighting
        plugin.output[:] = 99.0
        plugin.reset()
        assert np.all(plugin.output == 0.0)

    def test_get_chain_includes_bus_and_plugin(self):
        from slm.frequency_weighting import PluginAWeighting
        engine = _make_engine()
        bus = engine.add_bus("A", PluginAWeighting)
        plugin = bus.frequency_weighting
        chain = plugin.get_chain()
        assert bus in chain
        assert plugin in chain

    def test_str_returns_to_str(self):
        from slm.frequency_weighting import PluginAWeighting
        engine = _make_engine()
        bus = engine.add_bus("A", PluginAWeighting)
        plugin = bus.frequency_weighting
        assert str(plugin) == plugin.to_str()


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class TestBus:

    def test_default_frequency_weighting_is_z(self):
        """engine.add_bus with no weighting class defaults to PluginZWeighting."""
        from slm.frequency_weighting import PluginZWeighting
        engine = _make_engine()
        bus = engine.add_bus("default")
        assert isinstance(bus.frequency_weighting, PluginZWeighting)

    def test_get_returns_block(self):
        from slm.frequency_weighting import PluginZWeighting
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        result = bus.get()
        assert isinstance(result, np.ndarray)

    def test_add_plugin_wrong_bus_raises(self):
        from slm.frequency_weighting import PluginAWeighting, PluginZWeighting
        from slm.time_weighting import PluginFastTimeWeighting
        engine = _make_engine()
        bus_a = engine.add_bus("A", PluginAWeighting)
        bus_z = engine.add_bus("Z", PluginZWeighting)
        # Create a plugin belonging to bus_a
        plugin = PluginFastTimeWeighting(input=bus_a.frequency_weighting)
        with pytest.raises(Exception):
            bus_z.add_plugin(plugin)

    def test_get_chain_returns_list_with_bus(self):
        from slm.frequency_weighting import PluginZWeighting
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        chain = bus.get_chain()
        assert bus in chain

    def test_to_str(self):
        from slm.frequency_weighting import PluginZWeighting
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        assert "Bus" in bus.to_str()
        assert "Z" in bus.to_str()

    def test_str_returns_to_str(self):
        from slm.frequency_weighting import PluginZWeighting
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        assert str(bus) == bus.to_str()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TestEngine:

    def test_get_bus_existing(self):
        from slm.frequency_weighting import PluginAWeighting
        engine = _make_engine()
        bus = engine.add_bus("A", PluginAWeighting)
        assert engine.get_bus("A") is bus

    def test_get_bus_missing_raises(self):
        engine = _make_engine()
        with pytest.raises(KeyError, match="nonexistent"):
            engine.get_bus("nonexistent")

    def test_stop_calls_controller_stop(self):
        from slm.io.noise_controller import NoiseController
        from slm.engine import Engine
        ctrl = NoiseController(samplerate=SAMPLERATE, blocksize=BLOCKSIZE)
        ctrl.set_sensitivity(1.0, unit="V")
        engine = Engine(ctrl, dt=1.0)
        engine.stop()  # should not raise


# ---------------------------------------------------------------------------
# PluginMeter.add_meter wrong parent
# ---------------------------------------------------------------------------

class TestPluginMeterAddMeter:

    def test_add_meter_wrong_parent_raises(self):
        from slm.frequency_weighting import PluginAWeighting, PluginCWeighting
        from slm.meter import LeqAccumulator
        engine = _make_engine()
        bus_a = engine.add_bus("A", PluginAWeighting)
        bus_c = engine.add_bus("C", PluginCWeighting)
        fw_a = bus_a.frequency_weighting
        fw_c = bus_c.frequency_weighting
        # Create meter on fw_a, then try adding it to fw_c
        meter = LeqAccumulator(name="leq", parent=fw_a)
        with pytest.raises(Exception):
            fw_c.add_meter(meter)


# ---------------------------------------------------------------------------
# OctaveBand width validation
# ---------------------------------------------------------------------------

class TestOctaveBandWidthCheck:

    def test_multi_channel_input_raises(self):
        from slm.frequency_weighting import PluginZWeighting
        from slm.time_weighting import PluginFastTimeWeighting
        from slm.octave_band import PluginOctaveBand
        engine = _make_engine()
        bus = engine.add_bus("Z", PluginZWeighting)
        # Build a width-2 plugin to use as input to OctaveBand
        fw = bus.frequency_weighting
        tw = PluginFastTimeWeighting(input=fw, width=2)
        bus.add_plugin(tw)
        with pytest.raises(ValueError, match="width=1"):
            PluginOctaveBand(limits=(63.0, 8000.0), input=tw)

    def test_to_str(self):
        from slm.assembly import parse_metric, build_chain
        from slm.octave_band import PluginOctaveBand
        engine = _make_engine()
        build_chain([parse_metric("LZeq:bands:63-8000")], engine)
        bus = engine._busses["Z"]
        ob = next(p for p in bus.plugins if isinstance(p, PluginOctaveBand))
        assert isinstance(ob.to_str(), str)


# ---------------------------------------------------------------------------
# ProcessingElement.__str__ via Meter (no __str__ override)
# ---------------------------------------------------------------------------

class TestProcessingElementStr:

    def test_meter_str_uses_processing_element(self):
        import types
        from slm.meter import LeqAccumulator
        # Meter inherits ProcessingElement.__str__ → calls to_str()
        p = types.SimpleNamespace(width=1, samplerate=48000, blocksize=1024)
        m = LeqAccumulator(name="leq", parent=p)
        result = str(m)
        assert "LeqAccumulator" in result
