"""Tests for slm.assembly: MetricSpec parsing and build_chain wiring."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from slm.assembly import MetricSpec, parse_metric, build_chain, assemble_engine
from slm.engine import Engine
from slm.io.file_controller import FileController
from slm.meter import (
    LeqAccumulator, LeqMovingMeter, MaxAccumulator, MinAccumulator,
    LEAccumulator, LEMovingMeter, LastAccumulatingMeter,
)
from slm.octave_band import PluginOctaveBand
from slm.time_weighting import PluginFastTimeWeighting, PluginSquare
from slm.io.reporter import Reporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sine(path: Path, freq: float = 1000.0, duration: float = 1.0,
                samplerate: int = 48000, amplitude: float = 0.5) -> None:
    t = np.arange(int(duration * samplerate)) / samplerate
    signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, samplerate)


def _run_chain(tmp_path: Path, metric_names: list[str],
               amplitude: float = 0.5, sensitivity_v: float = 1.0,
               dt: float = 10.0, blocksize: int = 1024) -> tuple[Engine, Reporter]:
    """Write a 1 kHz sine, build chain, run, return engine and reporter."""
    wav = tmp_path / "sine.wav"
    _write_sine(wav, amplitude=amplitude)

    controller = FileController(str(wav), blocksize=blocksize)
    controller.set_sensitivity(sensitivity_v, unit="V")

    specs = [parse_metric(n) for n in metric_names]
    engine, bindings = assemble_engine(specs, controller, dt=dt)
    reporter = Reporter()
    reporter.add_columns(bindings)
    engine.on_record = reporter.record
    engine.run()
    return engine, reporter


# ---------------------------------------------------------------------------
# Parsing — valid names
# ---------------------------------------------------------------------------

class TestParseMetricValid:

    @pytest.mark.parametrize("name,weighting,tw,measure,window_is_dt,window_secs,bands,bpo", [
        ("LAeq",   "A", None, "eq",  False, None, None, 1.0),
        ("LCeq",   "C", None, "eq",  False, None, None, 1.0),
        ("LZeq",   "Z", None, "eq",  False, None, None, 1.0),
        ("LAFmax", "A", "F",  "max", False, None, None, 1.0),
        ("LASmax", "A", "S",  "max", False, None, None, 1.0),
        ("LAImin", "A", "I",  "min", False, None, None, 1.0),
        ("LCFmax", "C", "F",  "max", False, None, None, 1.0),
        ("LZSmin", "Z", "S",  "min", False, None, None, 1.0),
        ("LAeq_dt",   "A", None, "eq", True,  None, None, 1.0),
        ("LAeq_5s",   "A", None, "eq", False, 5.0,  None, 1.0),
        ("LAeq_1m",   "A", None, "eq", False, 60.0, None, 1.0),
        ("LZeq_1h",   "Z", None, "eq", False, 3600.0, None, 1.0),
        ("LAFmax_1m", "A", "F", "max", False, 60.0, None, 1.0),
        (
            "LZeq:bands:63-8000",
            "Z", None, "eq", False, None, (63.0, 8000.0), 1.0,
        ),
        (
            "LAeq:bands:1/3:31-16000",
            "A", None, "eq", False, None, (31.0, 16000.0), 3.0,
        ),
        (
            "LAeq:bands:1/6:63-8000",
            "A", None, "eq", False, None, (63.0, 8000.0), 6.0,
        ),
        (
            "LAeq:bands:1/12:63-8000",
            "A", None, "eq", False, None, (63.0, 8000.0), 12.0,
        ),
        (
            "LAeq:bands:2/3:63-8000",
            "A", None, "eq", False, None, (63.0, 8000.0), 1.5,
        ),
        (
            "LAeq_dt:bands:63-8000",
            "A", None, "eq", True, None, (63.0, 8000.0), 1.0,
        ),
        ("LAE",          "A", None, "E", False, None,  None, 1.0),
        ("LCE",          "C", None, "E", False, None,  None, 1.0),
        ("LZE",          "Z", None, "E", False, None,  None, 1.0),
        ("LAE_10s",      "A", None, "E", False, 10.0,  None, 1.0),
        ("LAE_dt",       "A", None, "E", True,  None,  None, 1.0),
        (
            "LAE:bands:63-8000",
            "A", None, "E", False, None, (63.0, 8000.0), 1.0,
        ),
        # bare metrics: time-weighting letter optional; measure defaults to "last"
        ("LZF",               "Z", "F",  "last", False, None, None,          1.0),
        ("LAF",               "A", "F",  "last", False, None, None,          1.0),
        ("LZ",                "Z", None, "last", False, None, None,          1.0),
        ("LA",                "A", None, "last", False, None, None,          1.0),
        ("LZF:bands:63-8000", "Z", "F",  "last", False, None, (63.0, 8000.0), 1.0),
        ("LZ:bands:63-8000",  "Z", None, "last", False, None, (63.0, 8000.0), 1.0),
    ])
    def test_valid(self, name, weighting, tw, measure,
                   window_is_dt, window_secs, bands, bpo):
        spec = parse_metric(name)
        assert spec.name == name
        assert spec.weighting == weighting
        assert spec.time_weighting == tw
        assert spec.measure == measure
        assert spec.window_is_dt == window_is_dt
        assert spec.window_seconds == pytest.approx(window_secs) if window_secs else spec.window_seconds == window_secs
        assert spec.bands == bands
        assert spec.bands_per_oct == pytest.approx(bpo)

    def test_window_5s(self):
        assert parse_metric("LAeq_5s").window_seconds == pytest.approx(5.0)

    def test_window_2m(self):
        assert parse_metric("LAeq_2m").window_seconds == pytest.approx(120.0)

    def test_window_2h(self):
        assert parse_metric("LZeq_2h").window_seconds == pytest.approx(7200.0)


# ---------------------------------------------------------------------------
# Parsing — invalid names
# ---------------------------------------------------------------------------

class TestParseMetricInvalid:

    @pytest.mark.parametrize("name", [
        "LAFeq",       # time-weighting letter on Leq
        "LAmax",       # max without time-weighting letter
        "LDmax",       # unknown weighting letter D
        "LAeq_5x",     # unknown window unit 'x'
        "leq",         # lowercase
        "LAeq:",       # trailing colon, no bands
        "",            # empty
        "LAeq:bands:", # bands prefix, no range
        "LAeq:bands:0/3:63-8000",  # zero numerator in octave fraction
    ])
    def test_invalid(self, name):
        with pytest.raises(ValueError):
            parse_metric(name)

    def test_eq_with_time_weighting_message(self):
        with pytest.raises(ValueError, match="time-weighting"):
            parse_metric("LAFeq")

    def test_max_without_time_weighting_message(self):
        with pytest.raises(ValueError, match="time-weighting"):
            parse_metric("LAmax")

    def test_le_with_time_weighting_message(self):
        with pytest.raises(ValueError, match="LE does not use a time-weighting letter"):
            parse_metric("LAFE")

    def test_bare_metric_with_window_suffix_raises(self):
        with pytest.raises(ValueError, match="window suffix"):
            parse_metric("LAF_5s")


# ---------------------------------------------------------------------------
# Structural tests (synthetic WAV)
# ---------------------------------------------------------------------------

class TestSignalConditioning:
    """The optional bus input filter (signal conditioning) sits in front of the
    frequency weighting on every bus."""

    def _engine(self, tmp_path, metrics, input_filter):
        wav = tmp_path / "sine.wav"
        _write_sine(wav)
        controller = FileController(str(wav), blocksize=1024)
        controller.set_sensitivity(1.0, unit="V")
        specs = [parse_metric(n) for n in metrics]
        engine, _ = assemble_engine(specs, controller, dt=10.0, input_filter=input_filter)
        return engine

    def test_no_input_filter_by_default(self, tmp_path):
        engine = self._engine(tmp_path, ["LAeq", "LCeq"], input_filter=None)
        for bus in engine._busses.values():
            assert bus.input_filter is None
            assert bus.frequency_weighting.input is bus   # weighting reads the raw bus

    def test_input_filter_inserted_in_front_of_weighting(self, tmp_path):
        from slm.frequency_weighting import PluginXL2InputFilter
        engine = self._engine(tmp_path, ["LAeq", "LZeq"], input_filter=PluginXL2InputFilter)
        assert set(engine._busses.keys()) == {"A", "Z"}
        for bus in engine._busses.values():
            assert isinstance(bus.input_filter, PluginXL2InputFilter)
            assert bus.input_filter.input is bus                    # filter reads the raw bus
            assert bus.frequency_weighting.input is bus.input_filter  # weighting reads the filter

    def test_conditioned_run_produces_finite_output(self, tmp_path):
        from slm.frequency_weighting import PluginXL2InputFilter
        engine = self._engine(tmp_path, ["LAeq"], input_filter=PluginXL2InputFilter)
        engine.run()
        bus = engine._busses["A"]
        sq = next(p for p in bus.plugins if isinstance(p, PluginSquare))
        assert np.isfinite(sq.read_db("LAeq")[0])


class TestBuildChainStructure:

    def test_shared_bus_for_same_weighting(self, tmp_path):
        """LAeq + LAFmax → exactly one Bus named 'A'."""
        engine, _ = _run_chain(tmp_path, ["LAeq", "LAFmax"])
        assert list(engine._busses.keys()) == ["A"]

    def test_two_metrics_two_buses(self, tmp_path):
        """LAeq + LCeq → two buses ('A' and 'C')."""
        engine, _ = _run_chain(tmp_path, ["LAeq", "LCeq"])
        assert set(engine._busses.keys()) == {"A", "C"}

    def test_leq_and_leq_dt_share_freq_weighting(self, tmp_path):
        """LAeq + LAeq_dt → both meters share one PluginSquare on the A bus."""
        engine, _ = _run_chain(tmp_path, ["LAeq", "LAeq_dt"])
        bus = engine._busses["A"]
        freq_w = bus.frequency_weighting
        # The only extra plugin is the shared PluginSquare; both meters live on it.
        non_fw = [p for p in bus.plugins if p is not freq_w]
        assert len(non_fw) == 1
        sq = non_fw[0]
        assert isinstance(sq, PluginSquare)
        assert sq.input is freq_w
        assert isinstance(sq.meters["LAeq"], LeqAccumulator)
        assert isinstance(sq.meters["LAeq_dt"], LeqMovingMeter)

    def test_time_weighting_plugin_created_for_max(self, tmp_path):
        """LAFmax → one time-weighting plugin on the A bus."""
        engine, _ = _run_chain(tmp_path, ["LAFmax"])
        bus = engine._busses["A"]
        non_fw = [p for p in bus.plugins if p is not bus.frequency_weighting]
        assert len(non_fw) == 1
        assert "LAFmax" in non_fw[0].meters
        assert isinstance(non_fw[0].meters["LAFmax"], MaxAccumulator)

    def test_shared_time_weighting_plugin(self, tmp_path):
        """LAFmax + LAFmin → same F time-weighting plugin, two meters."""
        engine, _ = _run_chain(tmp_path, ["LAFmax", "LAFmin"])
        bus = engine._busses["A"]
        non_fw = [p for p in bus.plugins if p is not bus.frequency_weighting]
        assert len(non_fw) == 1
        tw = non_fw[0]
        assert "LAFmax" in tw.meters
        assert "LAFmin" in tw.meters

    def test_octave_band_plugin_created(self, tmp_path):
        """LZeq:bands:63-8000 → one PluginOctaveBand on the Z bus."""
        engine, _ = _run_chain(tmp_path, ["LZeq:bands:63-8000"])
        bus = engine._busses["Z"]
        obs = [p for p in bus.plugins if isinstance(p, PluginOctaveBand)]
        assert len(obs) == 1

    def test_octave_band_width_equals_n_bands(self, tmp_path):
        """PluginOctaveBand.width must equal n_bands (set in __init__, not patched)."""
        engine, _ = _run_chain(tmp_path, ["LZeq:bands:63-8000"])
        bus = engine._busses["Z"]
        ob = next(p for p in bus.plugins if isinstance(p, PluginOctaveBand))
        assert ob.width == ob.n_bands
        assert ob.width > 1

    def test_octave_band_goes_to_rta_reporter(self, tmp_path):
        """LZeq:bands:63-8000 → reporter band_columns, not broadband_columns."""
        _, reporter = _run_chain(tmp_path, ["LZeq:bands:63-8000"])
        assert len(reporter._band_columns) == 1
        assert len(reporter._broadband_columns) == 0

    def test_shared_octave_band_plugin(self, tmp_path):
        """Two metrics with same bands → single PluginOctaveBand, two meters."""
        # We need two band metrics on the same weighting/limits/bpo.
        # LAeq_dt:bands:63-8000 and LAeq:bands:63-8000 share the OB plugin.
        engine, _ = _run_chain(tmp_path, ["LAeq:bands:63-8000", "LAeq_dt:bands:63-8000"])
        bus = engine._busses["A"]
        obs = [p for p in bus.plugins if isinstance(p, PluginOctaveBand)]
        assert len(obs) == 1
        # Both band-eq meters share the single PluginSquare downstream of the band.
        sq = next(p for p in bus.plugins if isinstance(p, PluginSquare))
        assert sq.input is obs[0]
        assert "LAeq:bands:63-8000" in sq.meters
        assert "LAeq_dt:bands:63-8000" in sq.meters

    def test_reporter_broadband_columns_for_broadband_metrics(self, tmp_path):
        """LAeq + LAFmax → two entries in broadband_columns."""
        _, reporter = _run_chain(tmp_path, ["LAeq", "LAFmax"])
        labels = [col[0] for col in reporter._broadband_columns]
        assert "LAeq" in labels
        assert "LAFmax" in labels

    def test_run_produces_rows(self, tmp_path):
        """After engine.run(), reporter broadband_rows is non-empty."""
        _, reporter = _run_chain(tmp_path, ["LAeq"], dt=0.5)
        assert len(reporter._broadband_rows) >= 1

    def test_lae_creates_le_accumulator_shares_bus(self, tmp_path):
        """LAE creates LEAccumulator; shares A-bus freq-weighting with LAeq."""
        engine, _ = _run_chain(tmp_path, ["LAeq", "LAE"])
        bus = engine._busses["A"]
        freq_w = bus.frequency_weighting
        # LAeq + LAE share the single PluginSquare on the A bus.
        non_fw = [p for p in bus.plugins if p is not freq_w]
        assert len(non_fw) == 1
        sq = non_fw[0]
        assert isinstance(sq, PluginSquare)
        assert isinstance(sq.meters["LAeq"], LeqAccumulator)
        assert isinstance(sq.meters["LAE"], LEAccumulator)

    def test_lae_dt_creates_le_moving_meter(self, tmp_path):
        """LAE_dt creates LEMovingMeter on freq-weighting plugin."""
        engine, _ = _run_chain(tmp_path, ["LAE_dt"])
        bus = engine._busses["A"]
        sq = next(p for p in bus.plugins if isinstance(p, PluginSquare))
        assert isinstance(sq.meters["LAE_dt"], LEMovingMeter)

    # -- band + time-weighting chain (regression: previously NaN) -----------

    def test_band_tw_chain_structure(self, tmp_path):
        """LZF:bands:63-8000 → OctaveBand → FTW (not OctaveBand → LastMeter directly).

        Regression: before the fix, build_chain ignored time_weighting when
        bands was set, chaining OctaveBand directly to LastAccumulatingMeter.
        LastAccumulatingMeter then read raw linear Pa (signed) → log10(negative) = NaN.
        """
        engine, _ = _run_chain(tmp_path, ["LZF:bands:63-8000"])
        bus = engine._busses["Z"]
        freq_w = bus.frequency_weighting
        non_fw = [p for p in bus.plugins if p is not freq_w]

        ob = next(p for p in non_fw if isinstance(p, PluginOctaveBand))
        tw = next(p for p in non_fw if isinstance(p, PluginFastTimeWeighting))

        assert tw.input is ob, "FTW must be downstream of OctaveBand, not freq_w"
        assert tw.width == ob.width > 1, "FTW width must equal n_bands"
        assert "LZF:bands:63-8000" in tw.meters
        assert isinstance(tw.meters["LZF:bands:63-8000"], LastAccumulatingMeter)

    def test_band_tw_width_not_one(self, tmp_path):
        """FTW plugin on band output must have width == n_bands, not 1.

        Regression: without explicit width= propagation, PluginFastTimeWeighting
        defaulted to width=1 and crashed with a broadcast error on the first block.
        """
        engine, _ = _run_chain(tmp_path, ["LZF:bands:63-8000"])
        bus = engine._busses["Z"]
        tw = next(p for p in bus.plugins if isinstance(p, PluginFastTimeWeighting))
        ob = next(p for p in bus.plugins if isinstance(p, PluginOctaveBand))
        assert tw.width == ob.n_bands

    def test_band_tw_shared_plugin(self, tmp_path):
        """LZF:bands + LZFmax:bands same limits → one band-TW plugin, two meters."""
        engine, _ = _run_chain(tmp_path, ["LZF:bands:63-8000", "LZFmax:bands:63-8000"])
        bus = engine._busses["Z"]
        ftw_plugins = [p for p in bus.plugins if isinstance(p, PluginFastTimeWeighting)]
        assert len(ftw_plugins) == 1
        assert "LZF:bands:63-8000" in ftw_plugins[0].meters
        assert "LZFmax:bands:63-8000" in ftw_plugins[0].meters

    def test_band_tw_no_nan(self, tmp_path):
        """LZF:bands:63-8000 must not produce NaN — core regression check."""
        _, reporter = _run_chain(tmp_path, ["LZF:bands:63-8000"])
        _, plugin, meter_name, _ = reporter._band_columns[0]
        vals = plugin.read_db(meter_name)
        assert not np.any(np.isnan(vals)), f"NaN in LZF:bands output: {vals}"

    def test_band_tw_some_bands_finite(self, tmp_path):
        """At least one band has a finite level for a 1 kHz sine (1 kHz octave band)."""
        _, reporter = _run_chain(tmp_path, ["LZF:bands:63-8000"])
        _, plugin, meter_name, _ = reporter._band_columns[0]
        vals = plugin.read_db(meter_name)
        assert np.any(np.isfinite(vals)), f"No finite values in LZF:bands output: {vals}"

    # -- band + no-TW (PluginSquare) chain ----------------------------------

    def test_band_sq_chain_structure(self, tmp_path):
        """LZ:bands:63-8000 → OctaveBand → PluginSquare with correct width."""
        engine, _ = _run_chain(tmp_path, ["LZ:bands:63-8000"])
        bus = engine._busses["Z"]
        freq_w = bus.frequency_weighting
        non_fw = [p for p in bus.plugins if p is not freq_w]

        ob = next(p for p in non_fw if isinstance(p, PluginOctaveBand))
        sq = next(p for p in non_fw if isinstance(p, PluginSquare))

        assert sq.input is ob
        assert sq.width == ob.width > 1
        assert "LZ:bands:63-8000" in sq.meters
        assert isinstance(sq.meters["LZ:bands:63-8000"], LastAccumulatingMeter)

    def test_band_sq_no_nan(self, tmp_path):
        """LZ:bands:63-8000 must not produce NaN."""
        _, reporter = _run_chain(tmp_path, ["LZ:bands:63-8000"])
        _, plugin, meter_name, _ = reporter._band_columns[0]
        vals = plugin.read_db(meter_name)
        assert not np.any(np.isnan(vals)), f"NaN in LZ:bands output: {vals}"

    # -- broadband bare metric (PluginSquare) --------------------------------

    def test_broadband_sq_chain_structure(self, tmp_path):
        """LZ → PluginSquare on bus, LastAccumulatingMeter."""
        engine, _ = _run_chain(tmp_path, ["LZ"])
        bus = engine._busses["Z"]
        freq_w = bus.frequency_weighting
        non_fw = [p for p in bus.plugins if p is not freq_w]

        sq = next(p for p in non_fw if isinstance(p, PluginSquare))
        assert sq.input is freq_w
        assert sq.width == 1
        assert "LZ" in sq.meters
        assert isinstance(sq.meters["LZ"], LastAccumulatingMeter)

    def test_broadband_sq_no_nan(self, tmp_path):
        """LZ on a 1 kHz sine must not produce NaN (may be -inf at zero crossings)."""
        _, reporter = _run_chain(tmp_path, ["LZ"])
        _, plugin, meter_name = reporter._broadband_columns[0]
        val = plugin.read_db(meter_name)[0]
        assert not np.isnan(val)


# ---------------------------------------------------------------------------
# Numerical tests (XL2 fixtures)
# ---------------------------------------------------------------------------

TOLERANCE_DB = 0.18


class TestAssemblyNumerical:

    def _run(self, meas, metric_names, blocksize=1024):
        """Build chain via assembly, run, return final report row as dict."""
        from slm.app.config import SLMConfig
        from slm.app.cli import run_measurement

        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "result")
            config = SLMConfig(metrics=metric_names, dt=1.0, output=out)
            run_measurement(
                str(meas.wav_path), meas.sensitivity, config,
                print_to_console=False, blocksize=blocksize,
            )
            import csv
            report_path = Path(td) / "result_report.csv"
            with open(report_path) as f:
                row = next(csv.DictReader(f))
        return row

    def test_laeq_slm000(self, meas_000):
        row = self._run(meas_000, ["LAeq"])
        ref = meas_000.report_value("LAeq")
        assert abs(float(row["LAeq"]) - ref) <= TOLERANCE_DB

    def test_lceq_slm000(self, meas_000):
        row = self._run(meas_000, ["LCeq"])
        ref = meas_000.report_value("LCeq")
        assert abs(float(row["LCeq"]) - ref) <= TOLERANCE_DB

    def test_lafmax_slm000(self, meas_000):
        row = self._run(meas_000, ["LAFmax"])
        ref = meas_000.report_value("LAFmax")
        assert abs(float(row["LAFmax"]) - ref) <= TOLERANCE_DB

    def test_lasmax_slm000(self, meas_000):
        row = self._run(meas_000, ["LASmax"])
        ref = meas_000.report_value("LASmax")
        assert abs(float(row["LASmax"]) - ref) <= TOLERANCE_DB

    def test_laeq_slm003(self, meas_003):
        row = self._run(meas_003, ["LAeq"])
        ref = meas_003.report_value("LAeq")
        assert abs(float(row["LAeq"]) - ref) <= TOLERANCE_DB

    def test_lceq_slm003(self, meas_003):
        row = self._run(meas_003, ["LCeq"])
        ref = meas_003.report_value("LCeq")
        assert abs(float(row["LCeq"]) - ref) <= TOLERANCE_DB

    def test_multi_metric_same_run(self, meas_000):
        """LAeq + LAFmax in one run → both within tolerance."""
        row = self._run(meas_000, ["LAeq", "LAFmax"])
        ref_leq = meas_000.report_value("LAeq")
        ref_max = meas_000.report_value("LAFmax")
        assert abs(float(row["LAeq"]) - ref_leq) <= TOLERANCE_DB
        assert abs(float(row["LAFmax"]) - ref_max) <= TOLERANCE_DB

    def test_lae_equals_laeq_plus_log_T(self, meas_003):
        """LAE = LAeq + 10*log10(T) — pure arithmetic identity, ±0.01 dB."""
        import soundfile as sf
        info = sf.info(str(meas_003.wav_path))
        T = info.frames / info.samplerate

        row = self._run(meas_003, ["LAeq", "LAE"])
        laeq = float(row["LAeq"])
        lae = float(row["LAE"])
        expected_lae = laeq + 10 * np.log10(T)
        assert abs(lae - expected_lae) <= 0.01


# ---------------------------------------------------------------------------
# Sensitivity conversion helpers
# ---------------------------------------------------------------------------

class TestSensitivityHelpers:

    def test_fs_db_formula(self):
        from slm.app.cli import sensitivity_from_fs_db
        from slm.constants import REFERENCE_PRESSURE
        fs_db = 128.1
        expected = 1.0 / (10 ** (fs_db / 20) * REFERENCE_PRESSURE)
        assert sensitivity_from_fs_db(fs_db) == pytest.approx(expected, rel=1e-10)

    def test_mv_conversion(self):
        from slm.app.cli import sensitivity_from_mv
        assert sensitivity_from_mv(50.0) == pytest.approx(0.05, rel=1e-10)

    def test_dbv_conversion(self):
        from slm.app.cli import sensitivity_from_dbv
        # 0 dBV → 1.0 V
        assert sensitivity_from_dbv(0.0) == pytest.approx(1.0, rel=1e-10)
        # -20 dBV → 0.1 V
        assert sensitivity_from_dbv(-20.0) == pytest.approx(0.1, rel=1e-8)


# ---------------------------------------------------------------------------
# plan_chain — structural intermediate representation
# ---------------------------------------------------------------------------

def _collect_meters(engine) -> dict:
    """Map meter name → meter instance across every bus/plugin in *engine*."""
    found: dict = {}
    for bus in engine._busses.values():
        for plugin in bus.plugins:
            for mname, meter in getattr(plugin, "meters", {}).items():
                found[mname] = meter
    return found


def _build_only(tmp_path: Path, names: list[str]):
    """Write a sine, build the chain (without running), return the engine."""
    wav = tmp_path / "sine.wav"
    _write_sine(wav)
    controller = FileController(str(wav), blocksize=1024)
    controller.set_sensitivity(1.0, unit="V")
    engine = Engine(controller, dt=10.0)
    build_chain([parse_metric(n) for n in names], engine)
    return engine


class TestPlanChain:

    @pytest.mark.parametrize("name,kinds", [
        ("LAeq",                    ["freq_weighting", "square"]),          # eq, no TW → square
        ("LAeq_dt",                 ["freq_weighting", "square"]),
        ("LAFmax",                  ["freq_weighting", "time_weighting"]),
        ("LAF",                     ["freq_weighting", "time_weighting"]),  # bare + TW → TW node
        ("LA",                      ["freq_weighting", "square"]),          # bare, no TW → square
        ("LZeq:bands:63-8000",      ["freq_weighting", "band", "square"]),
        ("LZFmax:bands:63-8000",    ["freq_weighting", "band", "time_weighting"]),
        ("LZ:bands:63-8000",        ["freq_weighting", "band", "square"]),  # bare per-band → square
    ])
    def test_node_kinds(self, name, kinds):
        from slm.assembly import plan_chain
        plan = plan_chain(parse_metric(name))
        assert [n.kind for n in plan.nodes] == kinds
        assert plan.name == name

    def test_meter_fields(self):
        from slm.assembly import plan_chain
        # accumulating broadband eq
        m = plan_chain(parse_metric("LAeq")).meter
        assert (m.measure, m.moving, m.is_band) == ("eq", False, False)
        # moving via _dt
        m = plan_chain(parse_metric("LAeq_dt")).meter
        assert m.moving and m.window_is_dt and m.window_seconds is None
        # moving via explicit window
        m = plan_chain(parse_metric("LAeq_30s")).meter
        assert m.moving and not m.window_is_dt and m.window_seconds == 30.0
        # band metric flags is_band
        assert plan_chain(parse_metric("LZeq:bands:63-8000")).meter.is_band

    def test_dedup_keys_shared(self):
        from slm.assembly import plan_chain
        bus_a = plan_chain(parse_metric("LAeq")).nodes[0].key
        # LAFmax and LAF share both the bus node and the fast-TW node
        lafmax = plan_chain(parse_metric("LAFmax")).nodes
        laf = plan_chain(parse_metric("LAF")).nodes
        assert lafmax[0].key == bus_a == laf[0].key
        assert lafmax[1].key == laf[1].key            # shared PluginFastTimeWeighting
        # a band node is shared between an eq and a max metric on the same band
        band_eq = plan_chain(parse_metric("LZeq:bands:63-8000")).nodes[1].key
        band_max = plan_chain(parse_metric("LZFmax:bands:63-8000")).nodes[1].key
        assert band_eq == band_max

    def test_meter_class_name_matches_build_chain(self, tmp_path):
        """Oracle: the IR's meter class equals what build_chain actually wires."""
        from slm.assembly import plan_chain, meter_class_name
        names = ["LAeq", "LAeq_dt", "LAeq_30s", "LAFmax", "LASmin",
                 "LAF", "LZeq:bands:63-8000", "LZFmax:bands:63-8000"]
        engine = _build_only(tmp_path, names)
        meters = _collect_meters(engine)
        for name in names:
            plan = plan_chain(parse_metric(name))
            assert type(meters[name]).__name__ == meter_class_name(plan.meter), name

    def test_node_and_meter_labels(self):
        from slm.assembly import plan_chain, node_label, meter_class_name
        plan = plan_chain(parse_metric("LZFmax:bands:1/3:31-16000"))
        labels = [node_label(n) for n in plan.nodes]
        assert labels[0] == "Bus [Z]  PluginZWeighting"
        assert labels[1].startswith("PluginOctaveBand") and "bpo=1/3" in labels[1]
        assert labels[2] == "PluginFastTimeWeighting"
        assert meter_class_name(plan.meter) == "MaxAccumulator"

    def test_square_node_label(self):
        """A bare metric (no time-weighting) ends in a square node."""
        from slm.assembly import plan_chain, node_label
        plan = plan_chain(parse_metric("LA"))   # bare → freq_weighting + square
        assert node_label(plan.nodes[-1]) == "PluginSquare"

    def test_bpo_label_variants(self):
        from slm.assembly import _bpo_label
        assert _bpo_label(None) == ""            # no band fraction
        assert _bpo_label(1.0) == "1/1"          # whole-octave
        assert _bpo_label(3.0) == "1/3"          # third-octave
        assert _bpo_label(1.5) == "1.5/oct"      # non-integer fraction


# ---------------------------------------------------------------------------
# assemble_engine factory + on_record boundary
# ---------------------------------------------------------------------------

def _file_controller(tmp_path: Path, duration: float = 1.0):
    wav = tmp_path / "sine.wav"
    _write_sine(wav, duration=duration)
    controller = FileController(str(wav), blocksize=1024)
    controller.set_sensitivity(1.0, unit="V")
    return controller


class TestAssembleEngine:

    def test_returns_engine_and_bindings(self, tmp_path):
        from slm.assembly import ColumnBinding
        controller = _file_controller(tmp_path)
        specs = [parse_metric("LAeq"), parse_metric("LZeq:bands:63-8000")]
        engine, bindings = assemble_engine(specs, controller, dt=10.0)
        assert isinstance(engine, Engine)
        assert all(isinstance(b, ColumnBinding) for b in bindings)
        assert [b.label for b in bindings] == ["LAeq", "LZeq:bands:63-8000"]
        # broadband binding carries no centre freqs; the band one does
        assert bindings[0].center_frequencies is None
        assert bindings[1].center_frequencies is not None

    def test_on_record_default_is_noop(self, tmp_path):
        """An engine with no sink attached runs and still updates its meters."""
        controller = _file_controller(tmp_path)
        engine, _ = assemble_engine([parse_metric("LAeq")], controller, dt=10.0)
        engine.run()   # no on_record wired → must not raise
        bus = engine._busses["A"]
        sq = next(p for p in bus.plugins if isinstance(p, PluginSquare))
        val = sq.read_db("LAeq")
        assert np.isfinite(val)

    def test_reporter_wiring_populates_rows(self, tmp_path):
        controller = _file_controller(tmp_path)
        engine, bindings = assemble_engine([parse_metric("LAeq")], controller, dt=0.1)
        reporter = Reporter()
        reporter.add_columns(bindings)
        engine.on_record = reporter.record
        engine.run()
        assert len(reporter._broadband_rows) >= 1
        assert np.isfinite(reporter._broadband_rows[-1]["LAeq"])

    def test_on_record_ticks_each_block_then_final_snapshot(self, tmp_path):
        controller = _file_controller(tmp_path, duration=0.5)
        engine, _ = assemble_engine([parse_metric("LAeq")], controller, dt=10.0)
        seen_dts: list[float] = []
        engine.on_record = lambda ts, dt: seen_dts.append(dt)
        engine.run()
        assert len(seen_dts) > 1     # one tick per processed block
        assert seen_dts[-1] == 0     # final forced snapshot passes dt=0
