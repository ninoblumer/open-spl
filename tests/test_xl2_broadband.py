"""
Integration tests: compare open-spl output against NTi XL2 reference measurements.

Each test loads a real WAV recording, runs it through the SLM pipeline, and
checks the result against the XL2's logged/reported values.

Tolerance: ±0.2 dB for steady-state signals.
"""
import numpy as np
import pytest

from slm.engine import Engine
from slm.file_controller import FileController
from slm.frequency_weighting import PluginAWeighting, PluginCWeighting, PluginZWeighting
from slm.meter import AccumulatingMeter
from slm.time_weighting import PluginFastTimeWeighting, PluginSlowTimeWeighting

TOLERANCE_DB = 0.2      # ±0.2 dB for steady-state comparison with XL2
TOLERANCE_Z_DB = 0.35   # IEC 61672-1 Annex E.5 defines Z(f) = 0 dB (flat passthrough).
                        # The ~0.25 dB offset vs the XL2 is due to the XL2's physical
                        # hardware (AC coupling in mic/preamp) attenuating low-frequency
                        # content, not a standardised filter requirement.


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_controller(meas, blocksize):
    controller = FileController(str(meas.wav_path), blocksize=blocksize)
    controller.set_sensitivity(meas.sensitivity, unit="V")
    return controller


def compute_leq(meas, weighting_cls, blocksize=1024):
    """Compute overall Leq (energy mean over the whole file).

    Processes every block through the frequency-weighting filter, accumulates
    the sum of squared output samples, then normalises by the true frame count
    (not padded block count) so zero-padding in the last block is irrelevant.
    """
    controller = _make_controller(meas, blocksize)
    engine = Engine(controller, dt=1e9)          # dt=1e9 s → log_block never fires
    bus = engine.add_bus("bus", weighting_cls)

    sum_sq = np.float64(0.0)
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)                     # block.T: (channels, blocksize)
        sum_sq += np.sum(bus.frequency_weighting.output ** 2)

    p_ref_sq = (20e-6 * meas.sensitivity) ** 2
    return 10 * np.log10(sum_sq / meas.n_frames / p_ref_sq)


def compute_lmax(meas, weighting_cls, tw_cls, blocksize=1024):
    """Compute Lmax: running maximum of the time-weighted level.

    e.g. weighting_cls=PluginAWeighting, tw_cls=PluginSlowTimeWeighting → LASmax.

    Uses AccumulatingMeter(block_fn=np.max, comp_fn=np.max) which tracks the
    global maximum of the time-weighted squared signal across all blocks.
    """
    controller = _make_controller(meas, blocksize)
    engine = Engine(controller, dt=1e9)
    bus = engine.add_bus("bus", weighting_cls)

    freq_w = bus.frequency_weighting
    time_w = bus.add_plugin(tw_cls(input=freq_w, zero_zi=True))
    time_w.add_meter(AccumulatingMeter(
        name="max", parent=time_w, block_fn=np.max, comp_fn=np.max,
    ))

    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)

    return time_w.read_db("max")[0]


# --------------------------------------------------------------------------- #
# SLM_000 — 1 kHz calibrator tone at 94 dB                                   #
# At 1 kHz the A, C, and Z weighting corrections are all ~0 dB, so all three #
# weighted Leq values should equal the unweighted 94.0 dB reference.         #
# --------------------------------------------------------------------------- #

class TestSLM000Calibrator:

    def test_LAeq(self, meas_000):
        ref = meas_000.report_value("LAeq")
        assert abs(compute_leq(meas_000, PluginAWeighting) - ref) <= TOLERANCE_DB

    def test_LCeq(self, meas_000):
        ref = meas_000.report_value("LCeq")
        assert abs(compute_leq(meas_000, PluginCWeighting) - ref) <= TOLERANCE_DB

    def test_LZeq(self, meas_000):
        ref = meas_000.report_value("LZeq")
        assert abs(compute_leq(meas_000, PluginZWeighting) - ref) <= TOLERANCE_Z_DB

    def test_LASmax(self, meas_000):
        """Steady-state signal → LASmax ≈ LAeq."""
        ref = meas_000.report_value("LASmax")
        assert abs(compute_lmax(meas_000, PluginAWeighting, PluginSlowTimeWeighting) - ref) <= TOLERANCE_DB

    def test_LAFmax(self, meas_000):
        ref = meas_000.report_value("LAFmax")
        assert abs(compute_lmax(meas_000, PluginAWeighting, PluginFastTimeWeighting) - ref) <= TOLERANCE_DB


# --------------------------------------------------------------------------- #
# SLM_003 — multi-frequency signal: LA=90.3, LC=92.1, LZ=94.0                #
# The 3.7 dB spread between LA and LZ tests A- and C-weighting accuracy.     #
# --------------------------------------------------------------------------- #

class TestSLM003FrequencyWeighting:

    def test_LAeq(self, meas_003):
        ref = meas_003.report_value("LAeq")
        assert abs(compute_leq(meas_003, PluginAWeighting) - ref) <= TOLERANCE_DB

    def test_LCeq(self, meas_003):
        ref = meas_003.report_value("LCeq")
        assert abs(compute_leq(meas_003, PluginCWeighting) - ref) <= TOLERANCE_DB

    def test_LZeq(self, meas_003):
        ref = meas_003.report_value("LZeq")
        assert abs(compute_leq(meas_003, PluginZWeighting) - ref) <= TOLERANCE_Z_DB

    def test_weighting_spread_LC_minus_LA(self, meas_003):
        """LC−LA spread is insensitive to sensitivity errors → tighter tolerance.

        Uses LC−LA rather than LZ−LA to avoid the ~0.25 dB hardware offset
        between the XL2 (physical AC coupling) and PluginZWeighting (mathematically
        flat per IEC 61672-1 Annex E.5).
        Tolerance is 0.15 dB (tighter than per-weighting ±0.2 dB, but A and C
        filter errors are correlated so the spread error doesn't fully cancel).
        """
        ref_spread = meas_003.report_value("LCeq") - meas_003.report_value("LAeq")
        la = compute_leq(meas_003, PluginAWeighting)
        lc = compute_leq(meas_003, PluginCWeighting)
        assert abs((lc - la) - ref_spread) <= 0.15


# --------------------------------------------------------------------------- #
# SLM_004 — low-level signal (~36–40 dB)                                     #
# Verifies accuracy well away from the calibrator level.                      #
# --------------------------------------------------------------------------- #

class TestSLM004LowLevel:

    def test_LAeq(self, meas_004):
        ref = meas_004.report_value("LAeq")
        assert abs(compute_leq(meas_004, PluginAWeighting) - ref) <= TOLERANCE_DB

    def test_LCeq(self, meas_004):
        ref = meas_004.report_value("LCeq")
        assert abs(compute_leq(meas_004, PluginCWeighting) - ref) <= TOLERANCE_DB

    def test_LZeq(self, meas_004):
        ref = meas_004.report_value("LZeq")
        assert abs(compute_leq(meas_004, PluginZWeighting) - ref) <= TOLERANCE_Z_DB
