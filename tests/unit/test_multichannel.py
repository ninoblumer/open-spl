"""Tests for multi-channel audio handling.

Controllers warn and extract channel 0 when fed multi-channel audio.
The engine raises a ValueError if a multi-channel block slips through.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import soundfile as sf


SAMPLERATE = 48_000
BLOCKSIZE = 1_024


def _write_stereo(path, freq_ch0=1000.0, freq_ch1=2000.0,
                  duration=1.0, amplitude=0.5):
    """Write a stereo WAV with different frequencies on each channel."""
    t = np.arange(int(duration * SAMPLERATE)) / SAMPLERATE
    ch0 = (amplitude * np.sin(2 * np.pi * freq_ch0 * t)).astype(np.float32)
    ch1 = (amplitude * np.sin(2 * np.pi * freq_ch1 * t)).astype(np.float32)
    sf.write(str(path), np.column_stack([ch0, ch1]), SAMPLERATE)


def _write_mono(path, freq=1000.0, duration=1.0, amplitude=0.5):
    t = np.arange(int(duration * SAMPLERATE)) / SAMPLERATE
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLERATE)


def _run(wav_path, metric_names):
    from slm.io.file_controller import FileController
    from slm.engine import Engine
    from slm.assembly import parse_metric, build_chain

    controller = FileController(str(wav_path), blocksize=BLOCKSIZE)
    controller.set_sensitivity(1.0, unit="V")
    engine = Engine(controller, dt=1.0)
    build_chain([parse_metric(m) for m in metric_names], engine)
    engine.run()
    return engine


# ---------------------------------------------------------------------------
# FileController: stereo → warn + use channel 0
# ---------------------------------------------------------------------------

class TestFileControllerStereo:

    @pytest.mark.parametrize("metric", ["LAeq", "LCeq", "LZeq",
                                         "LAFmax", "LZeq:bands:63-8000"])
    def test_stereo_issues_warning(self, tmp_path, metric):
        """Opening a stereo file issues a UserWarning mentioning channel 0."""
        wav = tmp_path / "stereo.wav"
        _write_stereo(wav)
        with pytest.warns(UserWarning, match="channel 0"):
            _run(wav, [metric])

    def test_stereo_result_matches_channel_0(self, tmp_path):
        """LAeq from stereo file equals LAeq from an explicit channel-0 mono file."""
        stereo = tmp_path / "stereo.wav"
        mono_ch0 = tmp_path / "mono_ch0.wav"
        _write_stereo(stereo, freq_ch0=1000.0, freq_ch1=2000.0)
        _write_mono(mono_ch0, freq=1000.0)

        with pytest.warns(UserWarning):
            engine_stereo = _run(stereo, ["LAeq"])
        engine_mono = _run(mono_ch0, ["LAeq"])

        bus_s = engine_stereo._busses["A"]
        bus_m = engine_mono._busses["A"]
        result_stereo = bus_s.frequency_weighting.read_db("LAeq")
        result_mono   = bus_m.frequency_weighting.read_db("LAeq")
        assert result_stereo == pytest.approx(result_mono, abs=0.01)


# ---------------------------------------------------------------------------
# NoiseController: channels > 1 → warn + clamp to 1
# ---------------------------------------------------------------------------

class TestNoiseControllerChannels:

    def test_multichannel_issues_warning(self):
        from slm.io.noise_controller import NoiseController
        with pytest.warns(UserWarning, match="channel 0"):
            ctrl = NoiseController(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, channels=2)
        assert ctrl._channels == 1

    def test_mono_no_warning(self):
        from slm.io.noise_controller import NoiseController
        with warnings.catch_warnings():
            import warnings as _w
            _w.simplefilter("error")
            NoiseController(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, channels=1)


# ---------------------------------------------------------------------------
# Engine: ValueError guard for multi-channel blocks
# ---------------------------------------------------------------------------

class TestEngineMultichannelGuard:

    def test_engine_raises_on_multichannel_block(self):
        """Engine raises ValueError if a controller delivers a multi-channel block."""
        import itertools
        from slm.io.controller import Controller
        from slm.engine import Engine
        from slm.assembly import parse_metric, build_chain

        class FakeStereoController(Controller):
            samplerate = SAMPLERATE
            blocksize = BLOCKSIZE
            sensitivity = 1.0
            _counter = itertools.count(0)

            def read_block(self):
                return np.zeros((self.blocksize, 2), dtype=np.float32), next(self._counter)

            def stop(self): pass
            def calibrate(self, target_spl=94.0): pass

        ctrl = FakeStereoController()
        ctrl.set_sensitivity(1.0, unit="V")
        engine = Engine(ctrl, dt=1.0)
        build_chain([parse_metric("LAeq")], engine)

        with pytest.raises(ValueError, match="2-channel"):
            engine.run()
