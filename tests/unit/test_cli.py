"""Tests for slm.config and slm.cli."""
from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from slm.app.config import SLMConfig
from slm.app.cli import (
    sensitivity_from_fs_db,
    sensitivity_from_mv,
    sensitivity_from_dbv,
    _fmt_sensitivity,
    run_measurement,
    SLMShell,
)
from slm.constants import REFERENCE_PRESSURE


# ---------------------------------------------------------------------------
# SLMConfig — round-trip and validation
# ---------------------------------------------------------------------------

class TestSLMConfigToml:

    def test_round_trip(self, tmp_path):
        config = SLMConfig(metrics=["LAeq", "LAFmax"], dt=2.0, output="out/x")
        toml_path = tmp_path / "config.toml"
        config.to_toml(toml_path)
        loaded = SLMConfig.from_toml(toml_path)
        assert loaded.metrics == ["LAeq", "LAFmax"]
        assert loaded.dt == pytest.approx(2.0)
        assert loaded.output == "out/x"

    def test_empty_metrics_round_trip(self, tmp_path):
        config = SLMConfig(metrics=[], dt=1.0, output="out")
        toml_path = tmp_path / "config.toml"
        config.to_toml(toml_path)
        loaded = SLMConfig.from_toml(toml_path)
        assert loaded.metrics == []

    def test_defaults_when_sections_missing(self, tmp_path):
        toml_path = tmp_path / "minimal.toml"
        toml_path.write_text("[measurement]\n", encoding="utf-8")
        loaded = SLMConfig.from_toml(toml_path)
        assert loaded.dt == pytest.approx(1.0)
        assert loaded.output == "output/measurement"
        assert loaded.metrics == []

    def test_unknown_section_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[unknown_section]\nfoo = 1\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown"):
            SLMConfig.from_toml(toml_path)

    def test_unknown_measurement_key_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[measurement]\ndt = 1.0\nextra = "oops"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown"):
            SLMConfig.from_toml(toml_path)

    def test_unknown_metrics_key_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[metrics]\nrequire = []\nbadkey = 1\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown"):
            SLMConfig.from_toml(toml_path)

    def test_require_not_list_of_strings_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[metrics]\nrequire = [1, 2]\n', encoding="utf-8")
        with pytest.raises(ValueError, match="list of strings"):
            SLMConfig.from_toml(toml_path)

    def test_non_positive_dt_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[measurement]\ndt = -1.0\n', encoding="utf-8")
        with pytest.raises(ValueError, match="dt must be positive"):
            SLMConfig.from_toml(toml_path)

    def test_file_created(self, tmp_path):
        config = SLMConfig(metrics=["LAeq"], dt=1.0, output="out")
        toml_path = tmp_path / "sub" / "config.toml"
        config.to_toml(toml_path)   # should create parent dirs
        assert toml_path.exists()


class TestSLMConfigFromArgs:

    def test_basic(self):
        config = SLMConfig.from_args(["LAeq", "LCeq"], dt=0.5, output="out/m")
        assert config.metrics == ["LAeq", "LCeq"]
        assert config.dt == pytest.approx(0.5)
        assert config.output == "out/m"


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

class TestSensitivityHelpers:

    def test_fs_db(self):
        fs_db = 128.1
        expected = 1.0 / (10 ** (fs_db / 20) * REFERENCE_PRESSURE)
        assert sensitivity_from_fs_db(fs_db) == pytest.approx(expected, rel=1e-10)

    def test_mv(self):
        assert sensitivity_from_mv(50.0) == pytest.approx(0.05, rel=1e-10)
        assert sensitivity_from_mv(1000.0) == pytest.approx(1.0, rel=1e-10)

    def test_dbv(self):
        assert sensitivity_from_dbv(0.0) == pytest.approx(1.0, rel=1e-10)
        assert sensitivity_from_dbv(-20.0) == pytest.approx(0.1, rel=1e-8)
        assert sensitivity_from_dbv(20.0) == pytest.approx(10.0, rel=1e-8)


# ---------------------------------------------------------------------------
# CLI argument parsing (isolated — no engine run)
# ---------------------------------------------------------------------------

class TestCLIArgParsing:

    def test_measure_flag(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--measure", "LAeq", "LAFmax",
            "--file", "foo.wav", "--fs-db", "128.1",
        ])
        assert args.measure == ["LAeq", "LAFmax"]

    def test_sensitivity_fs_db(self):
        from slm.app.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--fs-db", "128.1", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) == pytest.approx(sensitivity_from_fs_db(128.1))

    def test_sensitivity_mv(self):
        from slm.app.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--sensitivity-mv", "50", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) == pytest.approx(sensitivity_from_mv(50.0))

    def test_sensitivity_dbv(self):
        from slm.app.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--sensitivity-dbv", "-20", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) == pytest.approx(sensitivity_from_dbv(-20.0))

    def test_no_sensitivity_flag_returns_none(self):
        from slm.app.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) is None

    def test_mutually_exclusive_sensitivity_flags(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--fs-db", "128.1", "--sensitivity-mv", "50"])

    def test_dt_default(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        # --dt default is None; main() applies the 1.0 fallback when building SLMConfig
        assert args.dt is None

    def test_output_default(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        # --output default is None; main() applies the "output/measurement" fallback
        assert args.output is None


# ---------------------------------------------------------------------------
# One-shot integration (SLM_000 — 1 kHz calibrator at 94 dB)
# ---------------------------------------------------------------------------

class TestRunMeasurementIntegration:

    def test_laeq_csv_exists_and_correct(self, meas_000, tmp_path):
        config = SLMConfig(metrics=["LAeq"], dt=1.0,
                           output=str(tmp_path / "result"))
        run_measurement(
            str(meas_000.wav_path), meas_000.sensitivity, config,
            print_to_console=False,
        )
        log_path = tmp_path / "result_log.csv"
        report_path = tmp_path / "result_report.csv"
        assert log_path.exists(), "log CSV not created"
        assert report_path.exists(), "report CSV not created"

        with open(report_path) as f:
            row = next(csv.DictReader(f))
        assert abs(float(row["LAeq"]) - 94.0) <= 0.18

    def test_multi_metric_csv_columns(self, meas_000, tmp_path):
        config = SLMConfig(metrics=["LAeq", "LAFmax"], dt=1.0,
                           output=str(tmp_path / "result"))
        run_measurement(
            str(meas_000.wav_path), meas_000.sensitivity, config,
            print_to_console=False,
        )
        with open(tmp_path / "result_report.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "LAeq" in row
        assert "LAFmax" in row

    def test_band_metric_creates_rta_csv(self, meas_000, tmp_path):
        """LZeq:bands:63-8000 → rta_log.csv and rta_report.csv are written."""
        config = SLMConfig(metrics=["LZeq:bands:63-8000"], dt=1.0,
                           output=str(tmp_path / "result"))
        run_measurement(
            str(meas_000.wav_path), meas_000.sensitivity, config,
            print_to_console=False,
        )
        assert (tmp_path / "result_rta_log.csv").exists()
        assert (tmp_path / "result_rta_report.csv").exists()

    def test_dt_shorter_than_block_warns_and_still_correct(self, meas_000, tmp_path):
        """When dt < blocksize/samplerate, a UserWarning is emitted and the
        overall result is still correct (resolution is clamped to one entry per block)."""
        import soundfile as sf
        info = sf.info(str(meas_000.wav_path))
        blocksize = 1024
        block_duration = blocksize / info.samplerate  # ~0.021 s at 48 kHz
        dt = block_duration / 4  # clearly shorter than one block

        config = SLMConfig(metrics=["LAeq"], dt=dt,
                           output=str(tmp_path / "result"))
        with pytest.warns(UserWarning, match="dt=.*shorter than one block"):
            run_measurement(
                str(meas_000.wav_path), meas_000.sensitivity, config,
                print_to_console=False,
            )

        # Overall LAeq must still be within tolerance
        with open(tmp_path / "result_report.csv") as f:
            row = next(csv.DictReader(f))
        assert abs(float(row["LAeq"]) - 94.0) <= 0.18

        # Log must have one row per block (every block recorded, not every dt)
        with open(tmp_path / "result_log.csv") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        expected_blocks = info.frames // blocksize
        assert n_rows == pytest.approx(expected_blocks, abs=2)


# ---------------------------------------------------------------------------
# SLMShell REPL commands
# ---------------------------------------------------------------------------

class TestSLMShellSensitivity:

    def test_sensitivity_no_args_not_set(self, capsys):
        shell = SLMShell()
        shell.do_sensitivity("")
        out = capsys.readouterr().out
        assert "not set" in out

    def test_sensitivity_no_args_prints_value(self, capsys):
        shell = SLMShell()
        shell._sensitivity_v = sensitivity_from_mv(20.0)
        shell.do_sensitivity("")
        out = capsys.readouterr().out
        assert "mV" in out
        assert "dBV" in out

    def test_sensitivity_set_fs_db(self, capsys):
        shell = SLMShell()
        shell.do_sensitivity("fs_db 128.1")
        out = capsys.readouterr().out
        assert "mV" in out
        assert shell._sensitivity_v == pytest.approx(sensitivity_from_fs_db(128.1))

    def test_sensitivity_set_mv(self, capsys):
        shell = SLMShell()
        shell.do_sensitivity("mv 20.0")
        assert shell._sensitivity_v == pytest.approx(sensitivity_from_mv(20.0))

    def test_sensitivity_set_dbv(self, capsys):
        shell = SLMShell()
        shell.do_sensitivity("dbv -34.0")
        assert shell._sensitivity_v == pytest.approx(sensitivity_from_dbv(-34.0))

    def test_sensitivity_unknown_mode(self, capsys):
        shell = SLMShell()
        shell.do_sensitivity("bad 1.0")
        out = capsys.readouterr().out
        assert "Unknown" in out
        assert shell._sensitivity_v is None

    def test_fmt_sensitivity_fields(self):
        s = _fmt_sensitivity(0.02)
        assert "mV" in s
        assert "dBV" in s
        assert "V" not in s.split("mV")[0]   # no bare "V" before "mV"


class TestSLMShellDisplay:

    def test_display_plain(self, capsys):
        shell = SLMShell()
        shell.do_display("plain")
        assert shell._display_mode == "plain"

    def test_display_bars(self, capsys):
        shell = SLMShell()
        shell.do_display("bars")
        assert shell._display_mode == "bars"

    def test_display_invalid(self, capsys):
        shell = SLMShell()
        shell.do_display("foobar")
        out = capsys.readouterr().out
        assert "Usage" in out
        assert shell._display_mode == "plain"   # unchanged

    def test_display_mode_default(self):
        shell = SLMShell()
        assert shell._display_mode == "plain"


class TestSLMShellTree:

    def test_tree_no_metrics(self, capsys):
        shell = SLMShell()
        shell.do_tree("")
        out = capsys.readouterr().out
        assert "No metrics" in out

    def test_tree_with_metrics(self, capsys):
        shell = SLMShell()
        shell._config.metrics = ["LAeq", "LAFmax", "LZeq:bands:63-8000"]
        shell.do_tree("")
        out = capsys.readouterr().out
        assert "Bus [A]" in out
        assert "Bus [Z]" in out
        assert "LAeq" in out
        assert "LAFmax" in out
        assert "LZeq:bands:63-8000" in out
        assert "LeqAccumulator" in out
        assert "MaxAccumulator" in out
        assert "PluginOctaveBand" in out

    def test_tree_moving_meter(self, capsys):
        shell = SLMShell()
        shell._config.metrics = ["LAeq_dt"]
        shell.do_tree("")
        out = capsys.readouterr().out
        assert "LeqMovingMeter" in out


class TestSLMShellInspect:

    def test_inspect_known_metric(self, capsys):
        shell = SLMShell()
        shell._config.metrics = ["LZeq:bands:63-8000"]
        shell.do_inspect("LZeq:bands:63-8000")
        out = capsys.readouterr().out
        assert "Name:" in out
        assert "Weighting:" in out
        assert "Time-wt.:" in out
        assert "Measure:" in out
        assert "Bands:" in out
        assert "Window:" in out
        assert "1/1-octave" in out

    def test_inspect_not_in_config(self, capsys):
        shell = SLMShell()
        shell.do_inspect("LAeq")
        out = capsys.readouterr().out
        assert "Not in current config" in out

    def test_inspect_no_arg(self, capsys):
        shell = SLMShell()
        shell.do_inspect("")
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_inspect_moving_metric(self, capsys):
        shell = SLMShell()
        shell._config.metrics = ["LAeq_dt"]
        shell.do_inspect("LAeq_dt")
        out = capsys.readouterr().out
        assert "LeqMovingMeter" in out
        assert "moving" in out


# ---------------------------------------------------------------------------
# SLMConfig — queue_maxsize
# ---------------------------------------------------------------------------

class TestSLMConfigQueueMaxsize:

    def test_round_trip_queue_maxsize(self, tmp_path):
        config = SLMConfig(metrics=["LAeq"], queue_maxsize=16)
        path = tmp_path / "cfg.toml"
        config.to_toml(path)
        loaded = SLMConfig.from_toml(path)
        assert loaded.queue_maxsize == 16

    def test_queue_maxsize_zero_raises(self, tmp_path):
        path = tmp_path / "cfg.toml"
        path.write_text("[measurement]\nqueue_maxsize = 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="queue_maxsize"):
            SLMConfig.from_toml(path)

    def test_queue_maxsize_negative_raises(self, tmp_path):
        path = tmp_path / "cfg.toml"
        path.write_text("[measurement]\nqueue_maxsize = -1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="queue_maxsize"):
            SLMConfig.from_toml(path)

    def test_from_args_custom_queue_maxsize(self):
        config = SLMConfig.from_args(["LAeq"], dt=1.0, output="out", queue_maxsize=8)
        assert config.queue_maxsize == 8


# ---------------------------------------------------------------------------
# CLI arg parsing — generator and queue-maxsize flags
# ---------------------------------------------------------------------------

class TestCLIArgParsingGenerator:

    def _parser(self):
        from slm.app.__main__ import _build_parser
        return _build_parser()

    def test_generator_flag_parses(self):
        args = self._parser().parse_args(
            ["--generator", "--measure", "LAeq", "--sensitivity-mv", "50"]
        )
        assert args.generator is True

    def test_generator_mutually_exclusive_with_file(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--generator", "--file", "foo.wav"])

    def test_generator_mutually_exclusive_with_device(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--generator", "--device", "0"])

    def test_queue_maxsize_flag_parses(self):
        args = self._parser().parse_args(
            ["--file", "f.wav", "--measure", "LAeq", "--queue-maxsize", "16"]
        )
        assert args.queue_maxsize == 16

    def test_queue_maxsize_default_is_none(self):
        args = self._parser().parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.queue_maxsize is None

    def test_queue_maxsize_non_integer_rejected(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--queue-maxsize", "abc"])


# ---------------------------------------------------------------------------
# SLMShell — generator command and source switching
# ---------------------------------------------------------------------------

class TestSLMShellGenerator:

    def test_generator_sets_mode(self, capsys):
        shell = SLMShell()
        shell.do_generator("")
        assert shell._generator_mode is True

    def test_generator_clears_wav_path(self):
        shell = SLMShell(wav_path="foo.wav")
        shell.do_generator("")
        assert shell._wav_path is None

    def test_generator_clears_device(self):
        shell = SLMShell()
        shell._device = 0
        shell.do_generator("")
        assert shell._device is None

    def test_generator_prints_confirmation(self, capsys):
        shell = SLMShell()
        shell.do_generator("")
        assert "generator" in capsys.readouterr().out.lower()

    def test_file_clears_generator_mode(self, tmp_path):
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"RIFF")   # fake file — just needs to exist
        shell = SLMShell()
        shell._generator_mode = True
        shell.do_file(str(wav))
        assert shell._generator_mode is False

    def test_device_clears_generator_mode(self, capsys):
        shell = SLMShell()
        shell._generator_mode = True
        shell.do_device("0")
        capsys.readouterr()
        assert shell._generator_mode is False

    def test_switch_file_then_generator(self, tmp_path):
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"RIFF")
        shell = SLMShell()
        shell.do_file(str(wav))
        assert shell._wav_path is not None
        shell.do_generator("")
        assert shell._wav_path is None
        assert shell._generator_mode is True

    def test_switch_generator_then_device(self, capsys):
        shell = SLMShell()
        shell._generator_mode = True
        shell.do_device("2")
        capsys.readouterr()
        assert shell._generator_mode is False
        assert shell._device == 2


# ---------------------------------------------------------------------------
# SLMShell — queue command
# ---------------------------------------------------------------------------

class TestSLMShellQueue:

    def test_queue_no_arg_shows_current(self, capsys):
        shell = SLMShell()
        shell.do_queue("")
        out = capsys.readouterr().out
        assert str(shell._config.queue_maxsize) in out

    def test_queue_sets_value(self, capsys):
        shell = SLMShell()
        shell.do_queue("16")
        capsys.readouterr()
        assert shell._config.queue_maxsize == 16

    def test_queue_one_is_valid(self):
        shell = SLMShell()
        shell.do_queue("1")
        assert shell._config.queue_maxsize == 1

    def test_queue_zero_rejected(self, capsys):
        shell = SLMShell()
        original = shell._config.queue_maxsize
        shell.do_queue("0")
        out = capsys.readouterr().out
        assert "Invalid" in out
        assert shell._config.queue_maxsize == original  # unchanged

    def test_queue_negative_rejected(self, capsys):
        shell = SLMShell()
        shell.do_queue("-1")
        out = capsys.readouterr().out
        assert "Invalid" in out

    def test_queue_float_rejected(self, capsys):
        shell = SLMShell()
        shell.do_queue("3.5")
        out = capsys.readouterr().out
        assert "Invalid" in out

    def test_queue_text_rejected(self, capsys):
        shell = SLMShell()
        shell.do_queue("lots")
        out = capsys.readouterr().out
        assert "Invalid" in out

    def test_show_includes_queue_max(self, capsys):
        shell = SLMShell()
        shell._config.queue_maxsize = 8
        shell.do_show("")
        out = capsys.readouterr().out
        assert "Queue max" in out
        assert "8" in out


# ---------------------------------------------------------------------------
# SLMShell — do_show source display
# ---------------------------------------------------------------------------

class TestSLMShellShowSource:

    def test_show_no_source(self, capsys):
        shell = SLMShell()
        shell.do_show("")
        assert "not set" in capsys.readouterr().out

    def test_show_generator_source(self, capsys):
        shell = SLMShell()
        shell._generator_mode = True
        shell.do_show("")
        assert "generator" in capsys.readouterr().out.lower()

    def test_show_file_source(self, capsys):
        shell = SLMShell(wav_path="/some/file.wav")
        shell.do_show("")
        assert "/some/file.wav" in capsys.readouterr().out

    def test_show_device_source(self, capsys):
        shell = SLMShell()
        shell._device = 3
        shell.do_show("")
        assert "3" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# SLMShell — do_start error paths
# ---------------------------------------------------------------------------

class TestSLMShellStartErrors:

    def test_start_no_source(self, capsys):
        shell = SLMShell()
        shell._sensitivity_v = 0.05
        shell._config.metrics = ["LAeq"]
        shell.do_start("")
        out = capsys.readouterr().out
        assert "No source" in out

    def test_start_no_sensitivity(self, capsys):
        shell = SLMShell()
        shell._generator_mode = True
        shell._config.metrics = ["LAeq"]
        shell.do_start("")
        out = capsys.readouterr().out
        assert "sensitivity" in out.lower()

    def test_start_no_metrics(self, capsys):
        shell = SLMShell()
        shell._generator_mode = True
        shell._sensitivity_v = 0.05
        shell.do_start("")
        out = capsys.readouterr().out
        assert "metric" in out.lower()


# ---------------------------------------------------------------------------
# run_noise_measurement — invalid sensitivity
# ---------------------------------------------------------------------------

class TestRunNoiseMeasurement:
    from slm.app.cli import run_noise_measurement as _fn

    def test_zero_sensitivity_raises(self, tmp_path):
        from slm.app.cli import run_noise_measurement
        config = SLMConfig(metrics=["LAeq"], output=str(tmp_path / "out"))
        with pytest.raises(ValueError, match="sensitivity_v"):
            run_noise_measurement(0.0, config)

    def test_negative_sensitivity_raises(self, tmp_path):
        from slm.app.cli import run_noise_measurement
        config = SLMConfig(metrics=["LAeq"], output=str(tmp_path / "out"))
        with pytest.raises(ValueError, match="sensitivity_v"):
            run_noise_measurement(-1.0, config)

    def test_runs_and_writes_output(self, tmp_path):
        """Smoke test: generator processes a handful of blocks then stops normally."""
        from unittest.mock import patch
        from slm.app.cli import run_noise_measurement
        from slm.engine import Engine

        config = SLMConfig(metrics=["LAeq"], dt=1.0,
                           output=str(tmp_path / "result"))

        def _short_run(self):
            for _ in range(10):
                self._process_block()
            if self._last_timestamp is not None:
                self.on_record(self._last_timestamp, 0)

        with patch.object(Engine, "run", _short_run):
            run_noise_measurement(0.05, config, print_to_console=False)

        assert (tmp_path / "result_report.csv").exists()


# ---------------------------------------------------------------------------
# display._fmt_status
# ---------------------------------------------------------------------------

class TestFmtStatus:

    def test_none_controller_returns_empty(self):
        from slm.io.display import _fmt_status
        assert _fmt_status(None) == ""

    def test_no_data_shows_dashes(self):
        from slm.io.display import _fmt_status
        from slm.io.noise_controller import NoiseController
        ctrl = NoiseController(samplerate=48_000, blocksize=1_024)
        result = _fmt_status(ctrl)
        assert "---%" in result       # load not yet populated
        assert "Q=" in result
        assert "missed blocks" not in result   # no overruns yet

    def test_with_data_shows_numeric_load(self):
        from slm.io.display import _fmt_status
        from slm.io.noise_controller import NoiseController
        ctrl = NoiseController(samplerate=48_000, blocksize=1_024, queue_maxsize=8)
        ctrl.start()
        try:
            for _ in range(5):
                ctrl.read_block()
        finally:
            ctrl.stop()
        result = _fmt_status(ctrl)
        assert "---%" not in result
        assert "Load=" in result
        assert "%" in result
        assert "Q=" in result
        assert "/8" in result

    def test_missed_blocks_hidden_when_zero(self):
        from slm.io.display import _fmt_status
        from slm.io.noise_controller import NoiseController
        ctrl = NoiseController(samplerate=48_000, blocksize=1_024)
        assert ctrl.overruns == 0
        assert "missed blocks" not in _fmt_status(ctrl)

    def test_missed_blocks_shown_when_nonzero(self):
        from slm.io.display import _fmt_status
        from slm.io.noise_controller import NoiseController
        ctrl = NoiseController(samplerate=48_000, blocksize=1_024)
        ctrl._overruns = 3
        assert "missed blocks=3" in _fmt_status(ctrl)


class TestSLMShellCompleteFile:

    def test_complete_file_returns_list(self, tmp_path):
        shell = SLMShell()
        # Create a temp file and complete against its partial name
        f = tmp_path / "test_sound.wav"
        f.write_bytes(b"")
        results = shell.complete_file(str(tmp_path / "test_"), "", 0, 0)
        assert isinstance(results, list)

    def test_complete_file_no_match_returns_empty(self):
        shell = SLMShell()
        results = shell.complete_file("/nonexistent_path_xyz/", "", 0, 0)
        assert results == []
