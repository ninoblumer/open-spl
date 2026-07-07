"""Unit tests for PluginOctaveBand wiring and initial conditions.

Filter conformance against IEC 61260-1 lives in tests/iec61260/; these tests
cover the plugin's construction options, output shape, and its zero
initial-condition state.
"""
from __future__ import annotations

import numpy as np

from slm.engine import Engine
from slm.frequency_weighting import PluginZWeighting
from slm.io.noise_controller import NoiseController
from slm.octave_band import PluginOctaveBand


def _make_octave(limits=(63.0, 8000.0), bands_per_oct=1.0):
    """Build a Z-weighted bus with an octave-band plugin attached."""
    ctrl = NoiseController(samplerate=48_000, blocksize=1024)
    ctrl.set_sensitivity(1.0, unit="V")
    engine = Engine(ctrl, dt=1e9)  # dt huge → log_block never fires
    bus = engine.add_bus("bus", PluginZWeighting)
    octave = bus.add_plugin(PluginOctaveBand(
        limits=limits, bands_per_oct=bands_per_oct,
        input=bus.frequency_weighting,
    ))
    return bus, octave


class TestOctaveBandInit:

    def test_output_shape_matches_n_bands(self):
        _, octave = _make_octave()
        assert octave.output.shape == (octave.n_bands, 1024)
        assert octave.width == octave.n_bands

    def test_starts_with_zero_state(self):
        _, octave = _make_octave()
        assert np.all(octave._zi_stack == 0.0)

    def test_processes_block(self):
        """A block runs through the filter bank and yields finite output."""
        bus, octave = _make_octave()
        block = np.random.default_rng(0).standard_normal((1, 1024))
        bus.process(block)
        assert octave.output.shape == (octave.n_bands, 1024)
        assert np.all(np.isfinite(octave.output))
