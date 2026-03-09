"""
Integration tests: compare open-spl octave-band output against NTi XL2 RTA reference.

SLM_005 is a background-noise recording with no broadband log/report — only an
RTA report.  We compute per-band LZeq from the WAV and compare against the
XL2 RTA report's LZeq row.

Tolerance: ±0.2 dB per band. Actual max error across all 12 bands is 0.11 dB
(SLM_005). The original ±0.5 dB was set speculatively for XL2 AC-coupling
attenuation at low bands, but measurement shows no systematic HPF pattern.
"""
import numpy as np

from slm.engine import Engine
from slm.file_controller import FileController
from slm.frequency_weighting import PluginZWeighting
from slm.octave_band import PluginOctaveBand

TOLERANCE_RTA_DB = 0.2


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_controller(meas, blocksize):
    controller = FileController(str(meas.wav_path), blocksize=blocksize)
    controller.set_sensitivity(meas.sensitivity, unit="V")
    return controller


def compute_octave_leq(meas, blocksize=1024):
    """Compute per-band LZeq over the whole file using 1/1-octave filters.

    Returns (leq_array, center_frequencies) where leq_array has shape (n_bands,).
    """
    controller = _make_controller(meas, blocksize)
    engine = Engine(controller, dt=1e9)          # dt=1e9 s → log_block never fires
    bus = engine.add_bus("bus", PluginZWeighting)
    freq_w = bus.frequency_weighting
    octave = bus.add_plugin(PluginOctaveBand(
        limits=(8, 16000),
        bands_per_oct=1.0,
        input=freq_w,
        zero_zi=True,
    ))

    sum_sq = np.zeros(octave.n_bands, dtype=np.float64)
    while True:
        try:
            block, _ = controller.read_block()
        except StopIteration:
            break
        bus.process(block.T)
        sum_sq += np.sum(octave.output ** 2, axis=1)  # (n_bands,)

    p_ref_sq = (20e-6 * meas.sensitivity) ** 2
    leq = 10 * np.log10(sum_sq / meas.n_frames / p_ref_sq)
    return leq, octave.center_frequencies


# --------------------------------------------------------------------------- #
# SLM_005 — background noise, octave RTA only                                 #
# --------------------------------------------------------------------------- #

class TestSLM005OctaveBand:

    def test_lzeq_per_band(self, meas_005):
        result, bands = compute_octave_leq(meas_005)
        df = meas_005.rta_report.sections["RTA Results"].df
        ref = df.loc["LZeq"].astype(float).values   # 12 values aligned by position
        assert len(result) == len(ref), (
            f"Band count mismatch: got {len(result)}, XL2 has {len(ref)}"
        )
        for i, err in enumerate(np.abs(result - ref)):
            assert err <= TOLERANCE_RTA_DB, (
                f"Band {bands[i]} Hz: got {result[i]:.2f} dB, "
                f"XL2 {ref[i]:.2f} dB (err={err:.3f} dB)"
            )
