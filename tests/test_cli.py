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

from slm.config import SLMConfig
from slm.cli import (
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
        from slm.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--measure", "LAeq", "LAFmax",
            "--file", "foo.wav", "--fs-db", "128.1",
        ])
        assert args.measure == ["LAeq", "LAFmax"]

    def test_sensitivity_fs_db(self):
        from slm.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--fs-db", "128.1", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) == pytest.approx(sensitivity_from_fs_db(128.1))

    def test_sensitivity_mv(self):
        from slm.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--sensitivity-mv", "50", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) == pytest.approx(sensitivity_from_mv(50.0))

    def test_sensitivity_dbv(self):
        from slm.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--sensitivity-dbv", "-20", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) == pytest.approx(sensitivity_from_dbv(-20.0))

    def test_no_sensitivity_flag_returns_none(self):
        from slm.__main__ import _build_parser, _resolve_sensitivity
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert _resolve_sensitivity(args) is None

    def test_mutually_exclusive_sensitivity_flags(self):
        from slm.__main__ import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--fs-db", "128.1", "--sensitivity-mv", "50"])

    def test_dt_default(self):
        from slm.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.dt == pytest.approx(1.0)

    def test_output_default(self):
        from slm.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--file", "f.wav", "--measure", "LAeq"])
        assert args.output == "output/measurement"


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
