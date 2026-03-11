"""High-level CLI helpers: sensitivity conversions, calibration, measurement, REPL."""
from __future__ import annotations

import cmd
import math
from pathlib import Path
from typing import TYPE_CHECKING

from slm.constants import REFERENCE_PRESSURE

if TYPE_CHECKING:
    from slm.config import SLMConfig


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


def _fmt_sensitivity(sens_v: float) -> str:
    """Format sensitivity value in mV and dBV."""
    mv = sens_v * 1000.0
    dbv = 20.0 * math.log10(sens_v)
    return f"{mv:.4g} mV  |  {dbv:.2f} dBV"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_sensitivity(
    wav_path: str | Path,
    cal_level: float = 94.0,
    blocksize: int = 1024,
) -> float:
    """Derive controller sensitivity from a calibrator-tone WAV recording.

    The calibrator is assumed to emit *cal_level* dB SPL.  Returns a value
    suitable for ``controller.set_sensitivity(result, unit="V")``.
    """
    from slm.file_controller import FileController
    from slm.engine import Engine
    from slm.frequency_weighting import PluginZWeighting
    from slm.meter import LeqAccumulator

    controller = FileController(str(wav_path), blocksize=blocksize)
    controller.set_sensitivity(1.0, unit="V")   # dummy — just need raw WAV values

    engine = Engine(controller, dt=1e9)         # dt=1e9 → reporter never fires
    bus = engine.add_bus("Z", PluginZWeighting)
    freq_w = bus.frequency_weighting
    freq_w.create_meter(LeqAccumulator, name="leq")

    engine.run()

    mean_sq = freq_w.read_lin("leq")[0]         # mean(s²) over the whole file
    rms = mean_sq ** 0.5
    return rms / (REFERENCE_PRESSURE * 10 ** (cal_level / 20))


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
) -> None:
    """Parse *config.metrics*, build the plugin chain, run the engine, write results."""
    from slm.assembly import parse_metric, build_chain
    from slm.file_controller import FileController
    from slm.engine import Engine
    from slm.reporter import Reporter
    from slm.display import make_display_fn

    specs = [parse_metric(m) for m in config.metrics]

    controller = FileController(str(wav_path), blocksize=blocksize)
    controller.set_sensitivity(sensitivity_v, unit="V")

    engine = Engine(controller, dt=config.dt)
    display_fn = make_display_fn(display_mode, precision=2) if print_to_console else None
    reporter = Reporter(precision=2, print_to_console=print_to_console, display_fn=display_fn)
    engine.reporter = reporter

    build_chain(specs, engine, reporter)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("Measurement interrupted.")
    finally:
        reporter.write(config.output)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

class SLMShell(cmd.Cmd):
    """Interactive SLM REPL.

    Commands: add, remove, file, sensitivity, calibrate, output, dt,
              show, save, load, start, display, tree, inspect, exit/quit/EOF.
    """

    intro = "SLM interactive shell.  Type 'help' for a list of commands."
    prompt = "slm> "

    def __init__(self) -> None:
        super().__init__()
        from slm.config import SLMConfig
        self._config = SLMConfig()
        self._wav_path: str | None = None
        self._sensitivity_v: float | None = None
        self._display_mode: str = "plain"

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
        """calibrate [LEVEL_DB] — derive controller sensitivity from a calibrator-tone WAV.

Runs the engine on the currently-set WAV file, treating the signal as a
pure calibrator tone at LEVEL_DB (default 94.0 dB SPL).

The returned sensitivity is the controller sensitivity in V/Pa — NOT the
raw mV/Pa figure from the microphone datasheet.

Use this when you have a physical calibrator and a recording of it; use
'sensitivity mv VALUE' when you know the microphone sensitivity directly.
"""
        if not self._wav_path:
            print("No file set.  Use: file PATH")
            return
        cal_level = 94.0
        if arg.strip():
            try:
                cal_level = float(arg.strip())
            except ValueError:
                print(f"Invalid calibration level: {arg.strip()!r}")
                return
        print(f"Calibrating against {cal_level} dB SPL ...")
        sens = calibrate_sensitivity(self._wav_path, cal_level=cal_level)
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
        print(f"  Sensitivity: {self._sensitivity_v or '(not set)'}")
        print(f"  dt:          {self._config.dt} s")
        print(f"  Output:      {self._config.output}")
        print(f"  Metrics:     {self._config.metrics or '(none)'}")
        print(f"  Display:     {self._display_mode}")

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
        from slm.config import SLMConfig
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
        _acc_cls = {"eq": "LeqAccumulator", "max": "MaxAccumulator", "min": "MinAccumulator"}
        _mov_cls = {"eq": "LeqMovingMeter", "max": "MaxMovingMeter", "min": "MinMovingMeter"}

        # Group by weighting
        by_weight: dict[str, list] = {}
        for spec in specs:
            by_weight.setdefault(spec.weighting, []).append(spec)

        weight_keys = list(by_weight.keys())
        for wi, w in enumerate(weight_keys):
            is_last_bus = wi == len(weight_keys) - 1
            bus_pfx = "└──" if is_last_bus else "├──"
            child_pfx = "    " if is_last_bus else "│   "
            print(f"{bus_pfx} Bus [{w}]  {_w_plugin[w]}")

            w_specs = by_weight[w]

            # Split specs into groups by upstream plugin type
            freq_specs = [s for s in w_specs if s.time_weighting is None and s.bands is None]
            tw_groups: dict[str, list] = {}
            for s in w_specs:
                if s.time_weighting is not None and s.bands is None:
                    tw_groups.setdefault(s.time_weighting, []).append(s)
            band_groups: dict[tuple, list] = {}
            for s in w_specs:
                if s.bands is not None:
                    band_groups.setdefault((s.bands, s.bands_per_oct), []).append(s)

            groups: list[tuple[str, list]] = []
            if freq_specs:
                groups.append(("freq_weighting", freq_specs))
            for tw_letter, tw_list in tw_groups.items():
                groups.append((_tw_plugin[tw_letter], tw_list))
            for (bands, bpo), band_list in band_groups.items():
                bpo_label = "1/3" if bpo == 3.0 else "1/1"
                tag = f"PluginOctaveBand  limits=({bands[0]:.0f}, {bands[1]:.0f} Hz)  bpo={bpo_label}"
                groups.append((tag, band_list))

            for gi, (group_name, group_specs) in enumerate(groups):
                is_last_group = gi == len(groups) - 1
                grp_pfx = child_pfx + ("└──" if is_last_group else "├──")
                met_pfx = child_pfx + ("    " if is_last_group else "│   ")
                print(f"{grp_pfx} {group_name}")

                for si, spec in enumerate(group_specs):
                    is_last = si == len(group_specs) - 1
                    m_pfx = met_pfx + ("└──" if is_last else "├──")
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
                    print(f"{m_pfx} {spec.name:<32} {cls_name}{detail}")

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
        _acc_cls = {"eq": "LeqAccumulator", "max": "MaxAccumulator", "min": "MinAccumulator"}
        _mov_cls = {"eq": "LeqMovingMeter", "max": "MaxMovingMeter", "min": "MinMovingMeter"}

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
        if not self._wav_path:
            print("No file set.  Use: file PATH")
            return
        if not self._sensitivity_v:
            print("No sensitivity set.  Use: sensitivity ... or calibrate")
            return
        if not self._config.metrics:
            print("No metrics set.  Use: add METRIC")
            return
        run_measurement(
            self._wav_path,
            self._sensitivity_v,
            self._config,
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
