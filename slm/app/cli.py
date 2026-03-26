"""High-level CLI helpers: sensitivity conversions, calibration, measurement, REPL."""
from __future__ import annotations

import cmd
import math
from pathlib import Path
from typing import TYPE_CHECKING

from slm.constants import REFERENCE_PRESSURE

if TYPE_CHECKING:
    from slm.app.config import SLMConfig


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

def sensitivity_from_fs_db(fs_db: float) -> float:
    """Convert a WAV full-scale annotation (dBSPL at 0 dBFS) to controller sensitivity.

    Matches the formula used in ``tests/conftest.py``::

        sensitivity = 1 / (10^(fs_db/20) * P_ref)
    """
    return 1.0 / (10 ** (fs_db / 20) * REFERENCE_PRESSURE)


def sensitivity_from_mv(mv: float) -> float:
    """Convert microphone sensitivity from mV/Pa to V/Pa."""
    return mv / 1000.0


def sensitivity_from_dbv(dbv: float) -> float:
    """Convert microphone sensitivity from dBV (re 1 V/Pa) to V/Pa."""
    return 10 ** (dbv / 20)


def _fmt_device_table(devices: list[dict]) -> str:
    """Format a list of audio input devices as a wrapped-name table string."""
    import textwrap
    NAME_WIDTH = 44
    lines = [
        f"  {'IDX':>4}  {'NAME':<{NAME_WIDTH}}  {'CH':>3}  {'FS / Hz':>8}",
        f"  {'-' * 4}  {'-' * NAME_WIDTH}  {'-' * 3}  {'-' * 8}",
    ]
    for d in devices:
        name_lines = textwrap.wrap(d["name"], NAME_WIDTH) or [""]
        lines.append(
            f"  {d['index']:>4}  {name_lines[0]:<{NAME_WIDTH}}  "
            f"{d['max_input_channels']:>3}  {d['default_samplerate']:>8.0f}"
        )
        for cont in name_lines[1:]:
            lines.append(f"  {'':>4}  {cont:<{NAME_WIDTH}}")
    return "\n".join(lines)


def _fmt_sensitivity(sens_v: float) -> str:
    """Format sensitivity value in mV and dBV."""
    mv = sens_v * 1000.0
    dbv = 20.0 * math.log10(sens_v)
    return f"{mv:.4g} mV  |  {dbv:.2f} dBV"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_from_file(
    wav_path: str | Path,
    cal_freq: float = 1000.0,
    cal_level: float = 94.0,
    blocksize: int = 1024,
) -> float:
    """Derive controller sensitivity from a calibrator-tone WAV recording.

    Creates a FileController, sets a unity sensitivity, then delegates to
    ``slm.calibration.calibrate_sensitivity`` which applies a bandpass filter
    at *cal_freq* so only the fundamental tone contributes to the estimate.

    Returns a value suitable for ``controller.set_sensitivity(result, unit="V")``.
    """
    from slm.io.file_controller import FileController
    from slm.calibration import calibrate_sensitivity

    controller = FileController(str(wav_path), blocksize=blocksize)
    controller.set_sensitivity(1.0, unit="V")   # dummy — just need raw WAV values
    return calibrate_sensitivity(controller, cal_freq=cal_freq, cal_level=cal_level)


# ---------------------------------------------------------------------------
# Device calibration
# ---------------------------------------------------------------------------

def calibrate_from_device(
    device: int | str | None = None,
    samplerate: int = 48_000,
    blocksize: int = 1_024,
    cal_freq: float = 1000.0,
    cal_level: float = 94.0,
    stability_window: int = 10,
    stability_threshold: float = 0.1,
) -> float:
    """Derive sensitivity from a live calibrator tone via a real-time input device.

    Opens the audio stream, waits until the bandpass-filtered Leq has converged
    (rolling std-dev < *stability_threshold* dB over *stability_window* half-second
    readings), then stops automatically.

    Returns a value suitable for ``controller.set_sensitivity(result, unit="V")``.
    """
    from slm.io.sounddevice_controller import SounddeviceController
    from slm.calibration import calibrate_sensitivity

    controller = SounddeviceController(
        device=device, samplerate=samplerate, blocksize=blocksize
    )
    controller.set_sensitivity(1.0, unit="V")
    controller.start()
    try:
        sens = calibrate_sensitivity(
            controller,
            cal_freq=cal_freq,
            cal_level=cal_level,
            stability_window=stability_window,
            stability_threshold=stability_threshold,
        )
    finally:
        controller.stop()
    return sens


# ---------------------------------------------------------------------------
# One-shot measurement
# ---------------------------------------------------------------------------

def run_measurement(
    wav_path: str | Path,
    sensitivity_v: float,
    config: "SLMConfig",
    print_to_console: bool = False,
    blocksize: int = 1024,
    display_mode: str = "plain",
    realtime: bool = False,
) -> None:
    """Parse *config.metrics*, build the plugin chain, run the engine, write results."""
    if sensitivity_v <= 0:
        raise ValueError(f"sensitivity_v must be positive, got {sensitivity_v}")
    from slm.assembly import parse_metric, build_chain
    from slm.io.file_controller import FileController
    from slm.engine import Engine
    from slm.io.reporter import Reporter
    from slm.io.display import make_display_fn

    specs = [parse_metric(m) for m in config.metrics]

    controller = FileController(str(wav_path), blocksize=blocksize, realtime=realtime)
    controller.set_sensitivity(sensitivity_v, unit="V")

    display_fn = make_display_fn(display_mode, precision=2) if print_to_console else None
    reporter = Reporter(precision=2, print_to_console=print_to_console, display_fn=display_fn)
    engine = Engine(controller, dt=config.dt, reporter=reporter)

    build_chain(specs, engine)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("Measurement interrupted.")
    finally:
        reporter.write(config.output)


# ---------------------------------------------------------------------------
# Real-time measurement
# ---------------------------------------------------------------------------

def run_realtime_measurement(
    sensitivity_v: float,
    config: "SLMConfig",
    device: int | str | None = None,
    samplerate: int = 48_000,
    blocksize: int = 1_024,
    print_to_console: bool = False,
    display_mode: str = "plain",
) -> None:
    """Start a live measurement from a real-time audio input device.

    The engine runs until ``KeyboardInterrupt`` (Ctrl+C), at which point the
    stream is stopped and results are written to *config.output*.
    """
    if sensitivity_v <= 0:
        raise ValueError(f"sensitivity_v must be positive, got {sensitivity_v}")
    from slm.assembly import parse_metric, build_chain
    from slm.io.sounddevice_controller import SounddeviceController
    from slm.engine import Engine
    from slm.io.reporter import Reporter
    from slm.io.display import make_display_fn

    specs = [parse_metric(m) for m in config.metrics]

    controller = SounddeviceController(
        device=device, samplerate=samplerate, blocksize=blocksize
    )
    controller.set_sensitivity(sensitivity_v, unit="V")
    controller.start()

    display_fn = make_display_fn(display_mode, precision=2) if print_to_console else None
    reporter = Reporter(precision=2, print_to_console=print_to_console, display_fn=display_fn)
    engine = Engine(controller, dt=config.dt, reporter=reporter)

    build_chain(specs, engine)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nMeasurement interrupted.")
        controller.stop()
    finally:
        if controller.overruns:
            print(f"Warning: {controller.overruns} block(s) dropped (engine too slow).")
        reporter.write(config.output)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

class SLMShell(cmd.Cmd):
    """Interactive SLM REPL.

    Commands: add, remove, file, sensitivity, calibrate, output, dt,
              show, save, load, start, display, tree, inspect, exit/quit/EOF.
    """

    intro = (
        "open-spl  Copyright (C) 2026  Nino Blumer\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see LICENSE for details.\n"
        "\n"
        "SLM interactive shell.  Type 'help' for a list of commands."
    )
    prompt = "slm> "

    def __init__(
        self,
        *,
        wav_path: str | None = None,
        sensitivity_v: float | None = None,
        config: "SLMConfig | None" = None,
    ) -> None:
        super().__init__()
        from slm.app.config import SLMConfig
        self._config = config if config is not None else SLMConfig()
        self._wav_path = wav_path
        self._sensitivity_v = sensitivity_v
        self._display_mode: str = "plain"
        self._realtime: bool = False
        self._device: int | str | None = None

    # ------------------------------------------------------------------
    # Metric management
    # ------------------------------------------------------------------

    def do_add(self, arg: str) -> None:
        """add METRIC — add a metric to the current configuration.

Metric name syntax:
  L<W>[<T>](eq|max|min)[_<window>][:bands:[1/3:]<fmin>-<fmax>]

  W  weighting : A  C  Z
  T  time-wtg  : F (fast 125 ms)  S (slow 1 s)  I (impulse)
                 required for max/min; forbidden for eq
  window       : dt  5s  1m  2h  (omit -> accumulate whole file)
  bands        : :bands:63-8000        (1/1-oct, Hz)
                 :bands:1/3:31-16000   (1/3-oct, Hz)

Examples:
  add LAeq                     overall A-weighted Leq
  add LAeq_dt                  A-weighted Leq logged every dt seconds
  add LAFmax                   A-weighted fast-time-weighted maximum
  add LZeq:bands:63-8000       Z-weighted 1/1-oct octave bands 63-8000 Hz
  add LAeq:bands:1/3:31-16000  A-weighted 1/3-oct bands
"""
        from slm.assembly import parse_metric
        metric = arg.strip()
        if not metric:
            print("Usage: add METRIC")
            return
        try:
            parse_metric(metric)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        if metric not in self._config.metrics:
            self._config.metrics.append(metric)
            print(f"Added: {metric}")
        else:
            print(f"Already present: {metric}")

    def do_remove(self, arg: str) -> None:
        """remove METRIC — remove a previously added metric."""
        metric = arg.strip()
        if metric in self._config.metrics:
            self._config.metrics.remove(metric)
            print(f"Removed: {metric}")
        else:
            print(f"Not found: {metric}")

    # ------------------------------------------------------------------
    # File and sensitivity
    # ------------------------------------------------------------------

    def do_device(self, arg: str) -> None:
        """device [INDEX_OR_NAME] — list devices or select one for real-time input.

With no argument, prints all available input devices.
With an argument, sets the active input device (index or name substring).

Examples:
  device            list all input devices
  device 0          select device 0
  device Focusrite  select first device whose name contains 'Focusrite'
"""
        from slm.io.sounddevice_controller import SounddeviceController
        arg = arg.strip()
        if not arg:
            devices = SounddeviceController.list_devices()
            if not devices:
                print("No input devices found.")
                return
            print(_fmt_device_table(devices))
            return
        # Try to parse as integer first, else treat as name substring
        try:
            self._device = int(arg)
        except ValueError:
            self._device = arg
        print(f"Device: {self._device!r}")

    def do_file(self, arg: str) -> None:
        """file PATH — set the WAV file to measure."""
        path = arg.strip()
        if not path:
            print("Usage: file PATH")
            return
        if not Path(path).exists():
            print(f"File not found: {path}")
            return
        self._wav_path = path
        print(f"File: {path}")

    def complete_file(self, text, line, begidx, endidx):
        """Tab-complete file paths for the 'file' command."""
        import glob
        pattern = text + "*"
        return glob.glob(pattern) or []

    def do_sensitivity(self, arg: str) -> None:
        """sensitivity [fs_db VALUE | dbv VALUE | mv VALUE]

With no arguments, prints the current sensitivity in V, mV, and dBV.
With arguments, sets the sensitivity from the specified value:

  sensitivity fs_db VALUE   from WAV full-scale annotation (dB SPL at 0 dBFS)
  sensitivity dbv VALUE     from microphone sensitivity in dBV (re 1 V/Pa)
  sensitivity mv VALUE      from microphone sensitivity in mV/Pa

Units:
  mV  = millivolts per pascal (common in microphone datasheets)
  dBV = 20*log10(V/Pa)  (e.g. -34 dBV for a 20 mV/Pa microphone)
"""
        parts = arg.split()
        if not parts:
            if self._sensitivity_v is None:
                print("Sensitivity not set.  Use: sensitivity fs_db VALUE | dbv VALUE | mv VALUE")
            else:
                print(f"  Sensitivity: {_fmt_sensitivity(self._sensitivity_v)}")
            return
        if len(parts) != 2:
            print("Usage: sensitivity fs_db VALUE | dbv VALUE | mv VALUE")
            return
        mode, val_str = parts
        try:
            val = float(val_str)
        except ValueError:
            print(f"Invalid value: {val_str!r}")
            return
        if mode == "fs_db":
            self._sensitivity_v = sensitivity_from_fs_db(val)
        elif mode == "dbv":
            self._sensitivity_v = sensitivity_from_dbv(val)
        elif mode == "mv":
            self._sensitivity_v = sensitivity_from_mv(val)
        else:
            print(f"Unknown mode {mode!r}.  Use fs_db, dbv, or mv.")
            return
        print(f"  Sensitivity: {_fmt_sensitivity(self._sensitivity_v)}")

    def do_calibrate(self, arg: str) -> None:
        """calibrate [LEVEL_DB [FREQ_HZ]] — derive sensitivity from a calibrator-tone WAV.

Runs the engine on the currently-set WAV file, applying a 1/3-octave bandpass
filter at FREQ_HZ (default 1000.0 Hz) and treating the filtered signal as a
pure calibrator tone at LEVEL_DB (default 94.0 dB SPL).

The returned sensitivity is the controller sensitivity in V/Pa — NOT the
raw mV/Pa figure from the microphone datasheet.

Use this when you have a physical calibrator and a recording of it; use
'sensitivity mv VALUE' when you know the microphone sensitivity directly.
"""
        if not self._wav_path and self._device is None:
            print("No source set.  Use: file PATH  or  device INDEX")
            return
        cal_level = 94.0
        cal_freq = 1000.0
        parts = arg.split()
        if len(parts) >= 1:
            try:
                cal_level = float(parts[0])
            except ValueError:
                print(f"Invalid calibration level: {parts[0]!r}")
                return
        if len(parts) >= 2:
            try:
                cal_freq = float(parts[1])
            except ValueError:
                print(f"Invalid calibration frequency: {parts[1]!r}")
                return
        print(f"Calibrating against {cal_level} dB SPL at {cal_freq} Hz ...")
        if self._wav_path:
            sens = calibrate_from_file(self._wav_path, cal_freq=cal_freq, cal_level=cal_level)
        else:
            print("(Listening for calibrator tone — will stop automatically when stable)")
            sens = calibrate_from_device(
                device=self._device, cal_freq=cal_freq, cal_level=cal_level
            )
        print(f"  Sensitivity: {_fmt_sensitivity(sens)}")
        try:
            answer = input("Set as current sensitivity? [Y/n]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("", "y"):
            self._sensitivity_v = sens
            print("Sensitivity set.")
        else:
            print("Sensitivity not changed.")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def do_output(self, arg: str) -> None:
        """output PATH — set the output file base path."""
        path = arg.strip()
        if not path:
            print("Usage: output PATH")
            return
        self._config.output = path
        print(f"Output: {path}")

    def do_dt(self, arg: str) -> None:
        """dt SECONDS — set the logging interval."""
        try:
            self._config.dt = float(arg.strip())
            print(f"dt: {self._config.dt} s")
        except ValueError:
            print(f"Invalid dt: {arg.strip()!r}")

    def do_show(self, _: str) -> None:
        """show — display the current configuration."""
        print(f"  File:        {self._wav_path or '(not set)'}")
        print(f"  Device:      {self._device if self._device is not None else '(not set — file mode)'}")
        print(f"  Sensitivity: {'(not set)' if self._sensitivity_v is None else self._sensitivity_v}")
        print(f"  dt:          {self._config.dt} s")
        print(f"  Output:      {self._config.output}")
        print(f"  Metrics:     {self._config.metrics or '(none)'}")
        print(f"  Display:     {self._display_mode}")
        print(f"  Realtime:    {'on' if self._realtime else 'off'}")

    def do_save(self, arg: str) -> None:
        """save PATH.toml — save the current configuration to a TOML file."""
        path = arg.strip()
        if not path:
            print("Usage: save PATH.toml")
            return
        try:
            self._config.to_toml(path)
            print(f"Saved: {path}")
        except Exception as exc:
            print(f"Error: {exc}")

    def do_load(self, arg: str) -> None:
        """load PATH.toml — load configuration from a TOML file."""
        from slm.app.config import SLMConfig
        path = arg.strip()
        if not path:
            print("Usage: load PATH.toml")
            return
        try:
            self._config = SLMConfig.from_toml(path)
            print(f"Loaded: {path}")
        except Exception as exc:
            print(f"Error: {exc}")

    def do_display(self, arg: str) -> None:
        """display plain|bars — set display mode for measurements.

  plain  scrolling plain-text output (default)
  bars   live-updating bar graph (requires a TTY; falls back to plain)
"""
        mode = arg.strip().lower()
        if mode not in ("plain", "bars"):
            print("Usage: display plain | bars")
            return
        self._display_mode = mode
        print(f"Display mode: {mode}")

    def do_realtime(self, arg: str) -> None:
        """realtime [on|off] — toggle simulated real-time playback.

With no argument, shows the current state.
With 'on' or 'off', enables or disables real-time pacing.

When enabled, the engine processes each audio block at the same rate
as it was recorded, so dt-interval updates arrive every dt real seconds.
When disabled (default), the file is processed as fast as possible.
"""
        arg = arg.strip().lower()
        if not arg:
            print(f"  Realtime: {'on' if self._realtime else 'off'}")
            return
        if arg == "on":
            self._realtime = True
        elif arg == "off":
            self._realtime = False
        else:
            print("Usage: realtime [on|off]")
            return
        print(f"  Realtime: {arg}")

    # ------------------------------------------------------------------
    # Chain inspector
    # ------------------------------------------------------------------

    def do_tree(self, _: str) -> None:
        """tree — print the planned plugin chain for the current metrics."""
        from slm.assembly import parse_metric

        if not self._config.metrics:
            print("No metrics added.  Use: add METRIC")
            return

        specs = []
        for name in self._config.metrics:
            try:
                specs.append(parse_metric(name))
            except ValueError as exc:
                print(f"  Error parsing {name!r}: {exc}")
                return

        print(f"Planned chain  (dt={self._config.dt} s)")

        _w_plugin = {
            "A": "PluginAWeighting",
            "C": "PluginCWeighting",
            "Z": "PluginZWeighting",
        }
        _tw_plugin = {
            "F": "PluginFastTimeWeighting",
            "S": "PluginSlowTimeWeighting",
            "I": "PluginImpulseTimeWeighting",
        }
        _acc_cls = {"eq": "LeqAccumulator", "max": "MaxAccumulator", "min": "MinAccumulator",
                    "last": "LastAccumulatingMeter", "E": "LEAccumulator"}
        _mov_cls = {"eq": "LeqMovingMeter", "max": "MaxMovingMeter", "min": "MinMovingMeter",
                    "E": "LEMovingMeter"}

        # Group by weighting
        by_weight: dict[str, list] = {}
        for spec in specs:
            by_weight.setdefault(spec.weighting, []).append(spec)

        def _print_meter(spec, prefix):
            is_moving = spec.window_is_dt or spec.window_seconds is not None
            if not is_moving:
                cls_name = _acc_cls[spec.measure]
                detail = ""
            else:
                cls_name = _mov_cls[spec.measure]
                if spec.window_is_dt:
                    detail = f"   t=dt={self._config.dt} s"
                else:
                    detail = f"   t={spec.window_seconds} s"
            print(f"{prefix} {spec.name:<32} {cls_name}{detail}")

        weight_keys = list(by_weight.keys())
        for wi, w in enumerate(weight_keys):
            is_last_bus = wi == len(weight_keys) - 1
            bus_pfx = "└──" if is_last_bus else "├──"
            child_pfx = "    " if is_last_bus else "│   "
            print(f"{bus_pfx} Bus [{w}]  {_w_plugin[w]}")

            w_specs = by_weight[w]

            # Split specs into groups by upstream plugin type
            freq_specs = [s for s in w_specs
                          if s.time_weighting is None and s.bands is None and s.measure != "last"]
            sq_specs = [s for s in w_specs
                        if s.time_weighting is None and s.bands is None and s.measure == "last"]
            tw_groups: dict[str, list] = {}
            for s in w_specs:
                if s.time_weighting is not None and s.bands is None:
                    tw_groups.setdefault(s.time_weighting, []).append(s)
            # band_groups: keyed by (bands, bpo); value is dict tw_letter→[specs]
            band_groups: dict[tuple, dict] = {}
            for s in w_specs:
                if s.bands is not None:
                    key = (s.bands, s.bands_per_oct)
                    tw_key = s.time_weighting or ""
                    band_groups.setdefault(key, {}).setdefault(tw_key, []).append(s)

            groups: list[tuple[str, list]] = []
            if freq_specs:
                groups.append(("freq_weighting", freq_specs))
            if sq_specs:
                groups.append(("PluginSquare", sq_specs))
            for tw_letter, tw_list in tw_groups.items():
                groups.append((_tw_plugin[tw_letter], tw_list))

            n_band_keys = len(band_groups)
            n_non_band = len(groups)
            total_groups = n_non_band + n_band_keys

            for gi, (group_name, group_specs) in enumerate(groups):
                is_last_group = gi == total_groups - 1
                grp_pfx = child_pfx + ("└──" if is_last_group else "├──")
                met_pfx = child_pfx + ("    " if is_last_group else "│   ")
                print(f"{grp_pfx} {group_name}")
                for si, spec in enumerate(group_specs):
                    is_last = si == len(group_specs) - 1
                    m_pfx = met_pfx + ("└──" if is_last else "├──")
                    _print_meter(spec, m_pfx)

            for bi, ((bands, bpo), tw_dict) in enumerate(band_groups.items()):
                gi = n_non_band + bi
                is_last_group = gi == total_groups - 1
                grp_pfx = child_pfx + ("└──" if is_last_group else "├──")
                band_pfx = child_pfx + ("    " if is_last_group else "│   ")
                bpo_label = "1/3" if bpo == 3.0 else "1/1"
                print(f"{grp_pfx} PluginOctaveBand  limits=({bands[0]:.0f}, {bands[1]:.0f} Hz)  bpo={bpo_label}")

                tw_keys = list(tw_dict.keys())
                for ti, tw_key in enumerate(tw_keys):
                    tw_specs = tw_dict[tw_key]
                    is_last_tw = ti == len(tw_keys) - 1
                    if tw_key:
                        # band + time-weighting: extra level
                        tw_pfx = band_pfx + ("└──" if is_last_tw else "├──")
                        met_pfx2 = band_pfx + ("    " if is_last_tw else "│   ")
                        print(f"{tw_pfx} {_tw_plugin[tw_key]}")
                        for si, spec in enumerate(tw_specs):
                            is_last = si == len(tw_specs) - 1
                            m_pfx = met_pfx2 + ("└──" if is_last else "├──")
                            _print_meter(spec, m_pfx)
                    else:
                        # band only: check if bare "last" metrics need PluginSquare level
                        last_specs = [s for s in tw_specs if s.measure == "last"]
                        other_specs = [s for s in tw_specs if s.measure != "last"]
                        all_sub = []
                        if other_specs:
                            all_sub.append(("", other_specs))
                        if last_specs:
                            all_sub.append(("sq", last_specs))
                        for subi, (sub_key, sub_specs) in enumerate(all_sub):
                            is_last_sub = subi == len(all_sub) - 1 and is_last_tw
                            if sub_key == "sq":
                                sq_pfx = band_pfx + ("└──" if is_last_sub else "├──")
                                sq_met_pfx = band_pfx + ("    " if is_last_sub else "│   ")
                                print(f"{sq_pfx} PluginSquare")
                                for si, spec in enumerate(sub_specs):
                                    is_last = si == len(sub_specs) - 1
                                    m_pfx = sq_met_pfx + ("└──" if is_last else "├──")
                                    _print_meter(spec, m_pfx)
                            else:
                                for si, spec in enumerate(sub_specs):
                                    is_last = si == len(sub_specs) - 1 and is_last_sub
                                    m_pfx = band_pfx + ("└──" if is_last else "├──")
                                    _print_meter(spec, m_pfx)

    def do_inspect(self, arg: str) -> None:
        """inspect METRIC — show detailed human-readable info for a metric."""
        from slm.assembly import parse_metric

        name = arg.strip()
        if not name:
            print("Usage: inspect METRIC")
            return
        if name not in self._config.metrics:
            print(f"Not in current config: {name!r}.  Use 'show' to list added metrics.")
            return
        try:
            spec = parse_metric(name)
        except ValueError as exc:
            print(f"Error: {exc}")
            return

        _w_desc = {
            "A": "PluginAWeighting — A-weighting per IEC 61672-1",
            "C": "PluginCWeighting — C-weighting per IEC 61672-1",
            "Z": "PluginZWeighting — flat (Z-weighting), IEC 61672-1 Annex E.5",
        }
        _tw_desc = {
            "F": "F (fast, tau=0.125 s)",
            "S": "S (slow, tau=1.0 s)",
            "I": "I (impulse)",
        }
        _acc_cls = {"eq": "LeqAccumulator", "max": "MaxAccumulator", "min": "MinAccumulator",
                    "last": "LastAccumulatingMeter", "E": "LEAccumulator"}
        _mov_cls = {"eq": "LeqMovingMeter", "max": "MaxMovingMeter", "min": "MinMovingMeter",
                    "E": "LEMovingMeter"}

        is_moving = spec.window_is_dt or spec.window_seconds is not None
        if is_moving:
            meter_cls = _mov_cls[spec.measure]
            if spec.window_is_dt:
                window_str = f"t=dt={self._config.dt} s"
            else:
                window_str = f"t={spec.window_seconds} s"
        else:
            meter_cls = _acc_cls[spec.measure]
            window_str = "accumulates whole file"

        print(f"  Name:         {spec.name}")
        print(f"  Weighting:    {spec.weighting}  ({_w_desc[spec.weighting]})")
        print(f"  Time-wt.:     {_tw_desc.get(spec.time_weighting, 'none')}")
        print(f"  Measure:      {spec.measure} -> {meter_cls}  ({window_str})")
        if spec.bands is not None:
            bpo_str = "1/3-octave" if spec.bands_per_oct == 3.0 else "1/1-octave"
            print(f"  Bands:        {bpo_str}, {spec.bands[0]:.0f} - {spec.bands[1]:.0f} Hz")
        else:
            print(f"  Bands:        broadband")
        print(f"  Window:       {'moving' if is_moving else 'accumulating'}")

    # ------------------------------------------------------------------
    # Workflow help
    # ------------------------------------------------------------------

    def help_workflow(self) -> None:
        print(
            "Typical workflow:\n"
            "  1. file PATH          — set the WAV file\n"
            "  2. sensitivity ...    — set sensitivity (or: calibrate)\n"
            "  3. add METRIC ...     — add one or more metrics\n"
            "  4. dt SECONDS         — set logging interval (default 1.0 s)\n"
            "  5. output PATH        — set output base path\n"
            "  6. start              — run the measurement\n"
            "  7. save FILE.toml     — save config for next time"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def do_start(self, _: str) -> None:
        """start — run the measurement with the current configuration."""
        if not self._wav_path and self._device is None:
            print("No source set.  Use: file PATH  or  device INDEX")
            return
        if self._sensitivity_v is None:
            print("No sensitivity set.  Use: sensitivity ... or calibrate")
            return
        if not self._config.metrics:
            print("No metrics set.  Use: add METRIC")
            return
        if self._wav_path:
            run_measurement(
                self._wav_path,
                self._sensitivity_v,
                self._config,
                print_to_console=True,
                display_mode=self._display_mode,
                realtime=self._realtime,
            )
        else:
            run_realtime_measurement(
                self._sensitivity_v,
                self._config,
                device=self._device,
                print_to_console=True,
                display_mode=self._display_mode,
            )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def do_exit(self, _: str) -> bool:
        """exit — exit the shell."""
        return True

    def do_quit(self, _: str) -> bool:
        """quit — exit the shell."""
        return True

    def do_EOF(self, _: str) -> bool:
        print()
        return True
