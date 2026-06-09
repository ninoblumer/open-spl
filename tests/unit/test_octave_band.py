"""Unit tests for PluginOctaveBand wiring and initial-condition modes.

Filter conformance against IEC 61260-1 lives in tests/iec61260/; these tests
cover the plugin's construction options and output shape, including the
``zero_zi=False`` steady-state initial-condition path.
"""
from __future__ import annotations

import numpy as np

from slm.engine import Engine
from slm.frequency_weighting import PluginZWeighting
from slm.io.noise_controller import NoiseController
from slm.octave_band import PluginOctaveBand


def _make_octave(zero_zi: bool, limits=(63.0, 8000.0), bands_per_oct=1.0):
    """Build a Z-weighted bus with an octave-band plugin attached."""
    ctrl = NoiseController(samplerate=48_000, blocksize=1024)
    ctrl.set_sensitivity(1.0, unit="V")
    engine = Engine(ctrl, dt=1e9)  # dt huge → log_block never fires
    bus = engine.add_bus("bus", PluginZWeighting)
    octave = bus.add_plugin(PluginOctaveBand(
        limits=limits, bands_per_oct=bands_per_oct,
        input=bus.frequency_weighting, zero_zi=zero_zi,
    ))
    return bus, octave


class TestOctaveBandInit:

    def test_output_shape_matches_n_bands(self):
        _, octave = _make_octave(zero_zi=True)
        assert octave.output.shape == (octave.n_bands, 1024)
        assert octave.width == octave.n_bands

    def test_zero_zi_starts_with_zero_state(self):
        _, octave = _make_octave(zero_zi=True)
        assert np.all(octave._zi_stack == 0.0)

    def test_steady_ic_starts_with_nonzero_state(self):
        """zero_zi=False seeds steady-state initial conditions (sosfilt_zi),
        so the filter state is not all-zero after construction."""
        _, octave = _make_octave(zero_zi=False)
        assert octave._zi_stack.shape[0] == octave.n_bands
        assert np.any(octave._zi_stack != 0.0)

    def test_steady_ic_processes_block(self):
        """A block runs through the steady-IC filter bank and yields finite output."""
        bus, octave = _make_octave(zero_zi=False)
        block = np.random.default_rng(0).standard_normal((1, 1024))
        bus.process(block)
        assert octave.output.shape == (octave.n_bands, 1024)
        assert np.all(np.isfinite(octave.output))
