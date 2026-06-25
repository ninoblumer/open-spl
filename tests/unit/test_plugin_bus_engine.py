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

    def _duration_engine(self):
        """An engine on a free-running (non-realtime) noise source.

        ``realtime=False`` lets the engine drive the pace, so a duration limit is
        the only thing that stops this otherwise-infinite source.
        """
        from slm.io.noise_controller import NoiseController
        from slm.engine import Engine
        from slm.frequency_weighting import PluginZWeighting
        ctrl = NoiseController(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, realtime=False)
        ctrl.set_sensitivity(1.0, unit="V")
        engine = Engine(ctrl, dt=1.0)
        engine.add_bus("Z", PluginZWeighting)
        return engine, ctrl

    def test_run_duration_stops_on_block_edge(self):
        """A fixed duration stops the (otherwise infinite) noise source, rounding
        the measured length up to the next whole block."""
        engine, ctrl = self._duration_engine()
        calls: list[tuple[float, float]] = []
        engine.on_record = lambda ts, dt: calls.append((ts.total_seconds(), dt))

        block_duration = BLOCKSIZE / SAMPLERATE
        duration = 5.5 * block_duration   # not a whole number of blocks
        with ctrl:
            engine.run(duration=duration)

        # One on_record per processed block (dt != 0); ceil(5.5) = 6 blocks.
        block_calls = [ts for ts, dt in calls if dt != 0]
        assert len(block_calls) == 6
        last_start = block_calls[-1]
        # The run covers at least `duration` ...
        assert last_start + block_duration >= duration
        # ... but stops on the first block edge that does, not later.
        assert last_start < duration

    def test_run_duration_exact_multiple(self):
        """When duration is an exact multiple of the block duration, the run stops
        at exactly that many blocks with no extra block."""
        engine, ctrl = self._duration_engine()
        n_blocks = 0

        def _count(ts, dt):
            nonlocal n_blocks
            if dt != 0:
                n_blocks += 1

        engine.on_record = _count
        block_duration = BLOCKSIZE / SAMPLERATE
        with ctrl:
            engine.run(duration=3 * block_duration)
        assert n_blocks == 3

    def test_warmup_skips_logging_and_rebases_time(self):
        """Warm-up blocks are not logged, and measurement timestamps restart at 0."""
        engine, ctrl = self._duration_engine()
        calls: list[tuple[float, float]] = []
        engine.on_record = lambda ts, dt: calls.append((ts.total_seconds(), dt))
        bd = BLOCKSIZE / SAMPLERATE
        with ctrl:
            engine.run(duration=2 * bd, warmup=3 * bd)
        block_calls = [ts for ts, dt in calls if dt != 0]
        # Warm-up (3 blocks) logged nothing; only the 2 measurement blocks recorded.
        assert len(block_calls) == 2
        # First measurement timestamp rebased to 0 (warm-up time excluded).
        assert block_calls[0] == pytest.approx(0.0, abs=1e-9)

    def test_reset_meters_clears_accumulators(self):
        """reset_meters zeroes accumulating meters (used after warm-up)."""
        from slm.io.noise_controller import NoiseController
        from slm.engine import Engine
        from slm.assembly import parse_metric, build_chain
        from slm.meter import LeqAccumulator
        ctrl = NoiseController(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, realtime=False)
        ctrl.set_sensitivity(1.0, unit="V")
        engine = Engine(ctrl, dt=1.0)
        build_chain([parse_metric("LAeq")], engine)
        with ctrl:
            for _ in range(3):
                engine._process_block()
        accs = [m for bus in engine._busses.values() for p in bus.plugins
                for m in getattr(p, "meters", {}).values() if isinstance(m, LeqAccumulator)]
        assert accs and all(m._n_samples > 0 for m in accs)
        engine.reset_meters()
        assert all(m._n_samples == 0 for m in accs)


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
