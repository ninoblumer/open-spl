"""Tests for ArrayController and the array form of run_measurement."""
from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from slm.app.config import SLMConfig
from slm.app.cli import run_measurement
from slm.io.array_controller import ArrayController
from slm.io.results import MeasurementResults
from slm.constants import REFERENCE_PRESSURE


# ---------------------------------------------------------------------------
# ArrayController — controller contract
# ---------------------------------------------------------------------------

class TestArrayController:

    def test_accepts_1d(self):
        ctrl = ArrayController(np.zeros(2048), samplerate=48000, blocksize=1024)
        assert ctrl.samplerate == 48000
        assert ctrl.blocksize == 1024
        assert ctrl.sensitivity == 1.0
        assert ctrl.overruns == 0

    def test_accepts_column_vector(self):
        ctrl = ArrayController(np.zeros((2048, 1)), samplerate=44100)
        block, _ = ctrl.read_block()
        assert block.shape[1] == 1

    def test_multichannel_warns_and_uses_channel_0(self):
        samples = np.zeros((1024, 2))
        samples[:, 0] = 1.0  # channel 0 all ones, channel 1 all zeros
        with pytest.warns(UserWarning, match="only mono"):
            ctrl = ArrayController(samples, samplerate=48000, blocksize=1024)
        block, _ = ctrl.read_block()
        assert block.shape == (1024, 1)
        assert np.allclose(block, 1.0)

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            ArrayController(np.zeros((2, 2, 2)), samplerate=48000)

    def test_nonpositive_samplerate_raises(self):
        with pytest.raises(ValueError, match="samplerate must be positive"):
            ArrayController(np.zeros(1024), samplerate=0)

    def test_chunks_and_indexes(self):
        ctrl = ArrayController(np.arange(2048, dtype=float), samplerate=48000,
                               blocksize=1024)
        b0, i0 = ctrl.read_block()
        b1, i1 = ctrl.read_block()
        assert i0 == 0 and i1 == 1
        assert np.array_equal(b0[:, 0], np.arange(0, 1024))
        assert np.array_equal(b1[:, 0], np.arange(1024, 2048))
        with pytest.raises(StopIteration):
            ctrl.read_block()

    def test_pads_final_partial_block(self):
        # 1500 samples, blocksize 1024 -> second block is 476 real + 548 zeros
        ctrl = ArrayController(np.ones(1500), samplerate=48000, blocksize=1024)
        ctrl.read_block()
        tail, _ = ctrl.read_block()
        assert tail.shape == (1024, 1)
        assert np.allclose(tail[:476], 1.0)
        assert np.allclose(tail[476:], 0.0)
        with pytest.raises(StopIteration):
            ctrl.read_block()

    def test_stop_sets_done(self):
        ctrl = ArrayController(np.zeros(4096), samplerate=48000, blocksize=1024)
        ctrl.stop()
        assert ctrl.done
        with pytest.raises(StopIteration):
            ctrl.read_block()

    def test_calibrate_not_implemented(self):
        ctrl = ArrayController(np.zeros(1024), samplerate=48000)
        with pytest.raises(NotImplementedError):
            ctrl.calibrate()


# ---------------------------------------------------------------------------
# run_measurement — array overload
# ---------------------------------------------------------------------------

class TestRunMeasurementArray:

    def _tone(self, fs=48000, freq=1000.0, amp=0.1, seconds=1.0):
        t = np.arange(int(fs * seconds)) / fs
        return amp * np.sin(2 * np.pi * freq * t)

    def test_returns_results_matching_analytic_level(self):
        fs = 48000
        x = self._tone(fs=fs)
        cfg = SLMConfig(metrics=["LZeq"], dt=0.1)
        res = run_measurement(x, 1.0, cfg, samplerate=fs, output="return")
        assert isinstance(res, MeasurementResults)
        rms = 0.1 / math.sqrt(2)
        expected = 20 * math.log10(rms / REFERENCE_PRESSURE)
        assert res.report["LZeq"] == pytest.approx(expected, abs=0.2)
        assert len(res.log) >= 10

    def test_array_requires_samplerate(self):
        cfg = SLMConfig(metrics=["LZeq"], dt=0.1)
        with pytest.raises(ValueError, match="samplerate is required"):
            run_measurement(self._tone(), 1.0, cfg, output="return")

    def test_invalid_output_mode_raises(self):
        cfg = SLMConfig(metrics=["LZeq"], dt=0.1)
        with pytest.raises(ValueError, match="output must be"):
            run_measurement(self._tone(), 1.0, cfg, samplerate=48000, output="nope")

    def test_output_return_writes_no_csv(self, tmp_path):
        cfg = SLMConfig(metrics=["LZeq"], dt=0.1, output=str(tmp_path / "result"))
        run_measurement(self._tone(), 1.0, cfg, samplerate=48000, output="return")
        assert not (tmp_path / "result_report.csv").exists()
        assert not (tmp_path / "result_log.csv").exists()

    def test_output_csv_writes_files(self, tmp_path):
        cfg = SLMConfig(metrics=["LZeq"], dt=0.1, output=str(tmp_path / "result"))
        res = run_measurement(self._tone(), 1.0, cfg, samplerate=48000, output="csv")
        assert (tmp_path / "result_report.csv").exists()
        assert (tmp_path / "result_log.csv").exists()
        # results object is still returned in csv mode
        assert isinstance(res, MeasurementResults)

    def test_band_metric_populates_rta(self):
        fs = 48000
        cfg = SLMConfig(metrics=["LZeq:bands:63-8000"], dt=0.5)
        res = run_measurement(self._tone(fs=fs), 1.0, cfg, samplerate=fs,
                              output="return")
        assert res.report == {}          # no broadband metric
        label = "LZeq:bands:63-8000"
        assert label in res.rta_report
        assert label in res.band_frequencies
        assert len(res.rta_report[label]) == len(res.band_frequencies[label])
        # 1 kHz tone: the 1k band dominates
        peak_idx = int(np.argmax(res.rta_report[label]))
        assert res.band_frequencies[label][peak_idx] == "1k"


# ---------------------------------------------------------------------------
# Parity: array input matches file input for the same samples
# ---------------------------------------------------------------------------

class TestArrayFileParity:

    def test_array_matches_file(self, meas_000, tmp_path):
        samples, fs = sf.read(str(meas_000.wav_path), always_2d=False)
        if samples.ndim > 1:
            samples = samples[:, 0]

        cfg_file = SLMConfig(metrics=["LAeq"], dt=1.0,
                             output=str(tmp_path / "f"))
        file_res = run_measurement(
            str(meas_000.wav_path), meas_000.sensitivity, cfg_file, output="return"
        )
        cfg_arr = SLMConfig(metrics=["LAeq"], dt=1.0)
        arr_res = run_measurement(
            samples, meas_000.sensitivity, cfg_arr, samplerate=fs, output="return"
        )
        assert arr_res.report["LAeq"] == pytest.approx(file_res.report["LAeq"], abs=1e-6)
