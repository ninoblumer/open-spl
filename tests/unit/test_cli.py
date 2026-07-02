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
    parse_duration,
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

    def test_warmup_round_trip(self, tmp_path):
        config = SLMConfig(metrics=["LAeq"], warmup=2.5)
        toml_path = tmp_path / "config.toml"
        config.to_toml(toml_path)
        assert SLMConfig.from_toml(toml_path).warmup == pytest.approx(2.5)

    def test_warmup_defaults_to_zero(self, tmp_path):
        toml_path = tmp_path / "minimal.toml"
        toml_path.write_text("[measurement]\n", encoding="utf-8")
        assert SLMConfig.from_toml(toml_path).warmup == pytest.approx(0.0)

    def test_negative_warmup_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[measurement]\nwarmup = -1.0\n', encoding="utf-8")
        with pytest.raises(ValueError, match="warmup must be non-negative"):
            SLMConfig.from_toml(toml_path)

    def test_file_created(self, tmp_path):
        config = SLMConfig(metrics=["LAeq"], dt=1.0, output="out")
        toml_path = tmp_path / "sub" / "config.toml"
        config.to_toml(toml_path)   # should create parent dirs
        assert toml_path.exists()

    def test_signal_conditioning_round_trip(self, tmp_path):
        config = SLMConfig(metrics=["LAeq"], signal_conditioning="xl2")
        toml_path = tmp_path / "config.toml"
        config.to_toml(toml_path)
        assert SLMConfig.from_toml(toml_path).signal_conditioning == "xl2"

    def test_custom_signal_conditioning_round_trip(self, tmp_path):
        config = SLMConfig(metrics=["LAeq"], signal_conditioning="4.4 4 20000.0 4")
        toml_path = tmp_path / "config.toml"
        config.to_toml(toml_path)
        assert SLMConfig.from_toml(toml_path).signal_conditioning == "4.4 4 20000.0 4"

    def test_custom_signal_conditioning_negative_order_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="orders must be"):
            SLMConfig(metrics=["LAeq"]).from_args(
                ["LAeq"], dt=1.0, output="out", signal_conditioning="20 2 23000 -1")

    def test_signal_conditioning_defaults_to_none(self, tmp_path):
        toml_path = tmp_path / "minimal.toml"
        toml_path.write_text("[measurement]\n", encoding="utf-8")
        assert SLMConfig.from_toml(toml_path).signal_conditioning is None

    def test_signal_conditioning_none_normalizes(self, tmp_path):
        toml_path = tmp_path / "c.toml"
        toml_path.write_text('[measurement]\nsignal_conditioning = "none"\n', encoding="utf-8")
        assert SLMConfig.from_toml(toml_path).signal_conditioning is None

    def test_unknown_signal_conditioning_raises(self, tmp_path):
        toml_path = tmp_path / "bad.toml"
        toml_path.write_text('[measurement]\nsignal_conditioning = "bogus"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="signal.conditioning"):
            SLMConfig.from_toml(toml_path)


class TestSignalConditioning:

    def test_resolve_xl2_returns_input_filter(self):
        from slm.app.cli import resolve_signal_conditioning
        from slm.frequency_weighting import PluginXL2InputFilter
        assert resolve_signal_conditioning("xl2") is PluginXL2InputFilter

    def test_resolve_custom_returns_configured_filter(self):
        from slm.app.cli import resolve_signal_conditioning
        from slm.frequency_weighting import PluginInputFilter
        factory = resolve_signal_conditioning("4.4 0 20000 4")
        # A functools.partial over PluginInputFilter with the parsed spec bound.
        assert factory.func is PluginInputFilter
        assert factory.keywords == {
            "hpf_fc": 4.4, "hpf_order": 0, "lpf_fc": 20000.0, "lpf_order": 4}

    def test_resolve_custom_negative_order_raises(self):
        from slm.app.cli import resolve_signal_conditioning
        with pytest.raises(ValueError, match="orders must be"):
            resolve_signal_conditioning("20 2 23000 -2")

    def test_resolve_none_returns_none(self):
        from slm.app.cli import resolve_signal_conditioning
        assert resolve_signal_conditioning(None) is None

    def test_resolve_unknown_raises(self):
        from slm.app.cli import resolve_signal_conditioning
        with pytest.raises(ValueError, match="Unknown signal conditioning"):
            resolve_signal_conditioning("bogus")


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
# parse_duration
# ---------------------------------------------------------------------------

class TestParseDuration:

    def test_seconds_only(self):
        assert parse_duration("30") == pytest.approx(30.0)

    def test_mm_ss(self):
        assert parse_duration("1:30") == pytest.approx(90.0)

    def test_hh_mm_ss(self):
        assert parse_duration("01:02:03") == pytest.approx(3723.0)

    def test_fractional_fields(self):
        assert parse_duration("1.5") == pytest.approx(1.5)
        assert parse_duration("0:0.5") == pytest.approx(0.5)

    def test_surrounding_whitespace(self):
        assert parse_duration("  90 ") == pytest.approx(90.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("")

    def test_too_many_fields_raises(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("1:2:3:4")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_duration("ab")

    def test_negative_field_raises(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("-5")

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("0")


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

    def test_duration_default_is_none(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.duration is None

    def test_duration_flag_parses_as_string(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["--file", "f.wav", "--measure", "LAeq", "--duration", "1:30"]
        )
        # Stored raw; parse_duration is applied in main()
        assert args.duration == "1:30"

    def test_warmup_default_is_none(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.warmup is None

    def test_warmup_flag_parses(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["--file", "f.wav", "--measure", "LAeq", "--warmup", "2.5"]
        )
        assert args.warmup == pytest.approx(2.5)

    def test_negative_warmup_rejected(self):
        from slm.app.__main__ import _build_parser, main
        import sys
        parser = _build_parser()
        # Parsing succeeds; main() rejects via parser.error -> SystemExit
        argv = ["slm", "--generator", "--measure", "LAeq", "--sensitivity-mv", "50",
                "--warmup", "-1"]
        with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
            main()

    def test_blocksize_default(self):
        from slm.app.__main__ import _build_parser
        from slm.io.realtime_controller import DEFAULT_BLOCKSIZE
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.blocksize == DEFAULT_BLOCKSIZE

    def test_blocksize_flag_parses(self):
        from slm.app.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["--file", "f.wav", "--measure", "LAeq", "--blocksize", "4800"]
        )
        assert args.blocksize == 4800

    def test_nonpositive_blocksize_rejected(self):
        from slm.app.__main__ import main
        import sys
        # Parsing succeeds; main() rejects via parser.error -> SystemExit
        argv = ["slm", "--generator", "--measure", "LAeq", "--sensitivity-mv", "50",
                "--blocksize", "0"]
        with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
            main()

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

    def test_queue_maxsize_zero_valid(self, tmp_path):
        # 0 means an unbounded queue, so it is a valid setting.
        path = tmp_path / "cfg.toml"
        path.write_text("[measurement]\nqueue_maxsize = 0\n", encoding="utf-8")
        loaded = SLMConfig.from_toml(path)
        assert loaded.queue_maxsize == 0

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

    def test_queue_zero_valid(self):
        # 0 means an unbounded queue, so the shell accepts it.
        shell = SLMShell()
        shell.do_queue("0")
        assert shell._config.queue_maxsize == 0

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


class TestSLMShellOutputName:
    """`output` sets the directory; `name` sets the file stem; each keeps the other."""

    def test_output_sets_directory_keeps_name(self):
        shell = SLMShell()   # default output "output/measurement"
        shell.do_output("results/2026")
        assert shell._config.output == str(Path("results/2026") / "measurement")

    def test_name_sets_stem_keeps_directory(self):
        shell = SLMShell()
        shell.do_name("street_01")
        assert shell._config.output == str(Path("output") / "street_01")

    def test_output_then_name_compose(self):
        shell = SLMShell()
        shell.do_output("results/2026")
        shell.do_name("street_01")
        assert shell._config.output == str(Path("results/2026") / "street_01")

    def test_output_no_arg_shows_current_dir(self, capsys):
        shell = SLMShell()
        shell.do_output("")
        assert "Output dir" in capsys.readouterr().out

    def test_name_no_arg_shows_current_name(self, capsys):
        shell = SLMShell()
        shell.do_name("")
        out = capsys.readouterr().out
        assert "Measurement" in out and "measurement" in out


class TestSLMShellWarmup:

    def test_warmup_sets_value(self, capsys):
        shell = SLMShell()
        shell.do_warmup("2.5")
        assert shell._config.warmup == pytest.approx(2.5)

    def test_warmup_no_arg_shows_current(self, capsys):
        shell = SLMShell()
        shell.do_warmup("")
        assert "Warm-up" in capsys.readouterr().out

    def test_warmup_negative_rejected(self, capsys):
        shell = SLMShell()
        shell.do_warmup("-1")
        assert "Invalid" in capsys.readouterr().out
        assert shell._config.warmup == pytest.approx(0.0)   # unchanged

    def test_warmup_text_rejected(self, capsys):
        shell = SLMShell()
        shell.do_warmup("soon")
        assert "Invalid" in capsys.readouterr().out


class TestSLMShellConditioning:

    def test_preset_sets_value(self, capsys):
        shell = SLMShell()
        shell.do_conditioning("xl2")
        assert shell._config.signal_conditioning == "xl2"
        assert "xl2" in capsys.readouterr().out

    def test_custom_spec_sets_canonical_value(self, capsys):
        shell = SLMShell()
        shell.do_conditioning("20 2 23000 4")
        assert shell._config.signal_conditioning == "20.0 2 23000.0 4"

    def test_none_clears_value(self, capsys):
        shell = SLMShell()
        shell._config.signal_conditioning = "xl2"
        shell.do_conditioning("none")
        assert shell._config.signal_conditioning is None

    def test_no_arg_shows_current(self, capsys):
        shell = SLMShell()
        shell.do_conditioning("")
        assert "Conditioning" in capsys.readouterr().out

    def test_invalid_rejected_and_unchanged(self, capsys):
        shell = SLMShell()
        shell.do_conditioning("bogus")
        assert "Invalid signal conditioning" in capsys.readouterr().out
        assert shell._config.signal_conditioning is None

    def test_negative_order_rejected(self, capsys):
        shell = SLMShell()
        shell.do_conditioning("23000 4 20 -1")
        assert "Invalid signal conditioning" in capsys.readouterr().out
        assert shell._config.signal_conditioning is None

    def test_show_includes_conditioning(self, capsys):
        shell = SLMShell()
        shell._config.signal_conditioning = "xl2"
        shell.do_show("")
        assert "Conditioning" in capsys.readouterr().out


class TestSLMShellStartDuration:
    """do_start forwards a parsed duration to the runner; bad input is rejected."""

    def _ready_shell(self):
        shell = SLMShell()
        shell._generator_mode = True
        shell._sensitivity_v = 0.05
        shell._config.metrics = ["LAeq"]
        return shell

    def test_no_arg_passes_none(self):
        shell = self._ready_shell()
        with patch("slm.app.cli.run_noise_measurement") as m:
            shell.do_start("")
        assert m.call_args.kwargs["duration"] is None

    def test_duration_parsed_and_forwarded(self):
        shell = self._ready_shell()
        with patch("slm.app.cli.run_noise_measurement") as m:
            shell.do_start("1:30")
        assert m.call_args.kwargs["duration"] == pytest.approx(90.0)

    def test_invalid_duration_does_not_run(self, capsys):
        shell = self._ready_shell()
        with patch("slm.app.cli.run_noise_measurement") as m:
            shell.do_start("nonsense")
        m.assert_not_called()
        assert "Invalid duration" in capsys.readouterr().out


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

        def _short_run(self, duration=None, warmup=0.0):
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


# ---------------------------------------------------------------------------
# CLI arg parsing — --display flag
# ---------------------------------------------------------------------------

class TestCLIArgParsingDisplay:

    def _parser(self):
        from slm.app.__main__ import _build_parser
        return _build_parser()

    def test_display_default_is_plain(self):
        args = self._parser().parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.display == "plain"

    def test_display_bars_parses(self):
        args = self._parser().parse_args(
            ["--file", "f.wav", "--measure", "LAeq", "--display", "bars"]
        )
        assert args.display == "bars"

    def test_display_invalid_choice_rejected(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(
                ["--file", "f.wav", "--measure", "LAeq", "--display", "fancy"]
            )

    def test_display_forwarded_to_runner(self, tmp_path):
        """main() one-shot generator run forwards --display to the runner."""
        from slm.app.__main__ import main
        import sys
        argv = ["slm", "--generator", "--measure", "LAeq", "--sensitivity-mv", "50",
                "--display", "bars", "--output", str(tmp_path / "r")]
        with patch("slm.app.cli.run_noise_measurement") as m, \
                patch.object(sys, "argv", argv):
            main()
        assert m.call_args.kwargs["display_mode"] == "bars"


# ---------------------------------------------------------------------------
# CLI arg parsing — --samplerate / --blocksize
# ---------------------------------------------------------------------------

class TestCLIArgParsingSamplerate:

    def _parser(self):
        from slm.app.__main__ import _build_parser
        return _build_parser()

    def test_samplerate_default(self):
        from slm.io.realtime_controller import DEFAULT_SAMPLERATE
        args = self._parser().parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.samplerate == DEFAULT_SAMPLERATE

    def test_samplerate_flag_parses(self):
        args = self._parser().parse_args(
            ["--generator", "--measure", "LAeq", "--samplerate", "44100"]
        )
        assert args.samplerate == 44100


# ---------------------------------------------------------------------------
# SLMShell — samplerate / blocksize commands
# ---------------------------------------------------------------------------

class TestSLMShellSamplerate:

    def test_default_matches_controller(self):
        from slm.io.realtime_controller import DEFAULT_SAMPLERATE
        shell = SLMShell()
        assert shell._samplerate == DEFAULT_SAMPLERATE

    def test_no_arg_shows_current(self, capsys):
        shell = SLMShell()
        shell.do_samplerate("")
        assert str(shell._samplerate) in capsys.readouterr().out

    def test_sets_value(self, capsys):
        shell = SLMShell()
        shell.do_samplerate("44100")
        capsys.readouterr()
        assert shell._samplerate == 44100

    def test_zero_rejected(self, capsys):
        shell = SLMShell()
        original = shell._samplerate
        shell.do_samplerate("0")
        assert "Invalid" in capsys.readouterr().out
        assert shell._samplerate == original

    def test_text_rejected(self, capsys):
        shell = SLMShell()
        shell.do_samplerate("fast")
        assert "Invalid" in capsys.readouterr().out


class TestSLMShellBlocksize:

    def test_default_matches_controller(self):
        from slm.io.realtime_controller import DEFAULT_BLOCKSIZE
        shell = SLMShell()
        assert shell._blocksize == DEFAULT_BLOCKSIZE

    def test_no_arg_shows_current(self, capsys):
        shell = SLMShell()
        shell.do_blocksize("")
        assert str(shell._blocksize) in capsys.readouterr().out

    def test_sets_value(self, capsys):
        shell = SLMShell()
        shell.do_blocksize("4096")
        capsys.readouterr()
        assert shell._blocksize == 4096

    def test_zero_rejected(self, capsys):
        shell = SLMShell()
        original = shell._blocksize
        shell.do_blocksize("0")
        assert "Invalid" in capsys.readouterr().out
        assert shell._blocksize == original

    def test_show_includes_samplerate_and_blocksize(self, capsys):
        shell = SLMShell()
        shell._samplerate = 44100
        shell._blocksize = 4096
        shell.do_show("")
        out = capsys.readouterr().out
        assert "Sample rate" in out and "44100" in out
        assert "Block size" in out and "4096" in out

    def test_start_forwards_samplerate_and_blocksize(self):
        """do_start (generator) forwards the shell's samplerate/blocksize."""
        shell = SLMShell()
        shell._generator_mode = True
        shell._sensitivity_v = 0.05
        shell._config.metrics = ["LAeq"]
        shell._samplerate = 44100
        shell._blocksize = 4096
        with patch("slm.app.cli.run_noise_measurement") as m:
            shell.do_start("")
        assert m.call_args.kwargs["samplerate"] == 44100
        assert m.call_args.kwargs["blocksize"] == 4096
