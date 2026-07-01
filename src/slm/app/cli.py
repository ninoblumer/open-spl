"""High-level CLI helpers: sensitivity conversions, calibration, measurement, REPL."""
from __future__ import annotations

import cmd
import gc
import math
from pathlib import Path
from typing import TYPE_CHECKING

from slm.constants import CALIBRATION_FREQ_HZ, CALIBRATION_LEVEL_DB, REFERENCE_PRESSURE
from slm.io.controller import Controller
from slm.io.realtime_controller import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLERATE
from slm.io.thread_priority import high_priority

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


def parse_duration(text: str) -> float:
    """Parse a colon-separated duration into seconds.

    Accepts ``hh:mm:ss``, ``mm:ss``, or ``ss``; each field may be fractional.
    The last field is seconds, the next minutes, the next hours::

        parse_duration("90")        -> 90.0
        parse_duration("1:30")      -> 90.0
        parse_duration("01:02:03")  -> 3723.0

    Raises :class:`ValueError` if the text is empty, has more than three fields,
    contains a non-numeric or negative field, or sums to a non-positive value.
    """
    fields = text.strip().split(":")
    if not text.strip() or len(fields) > 3:
        raise ValueError(f"Invalid duration: {text!r}")
    seconds = 0.0
    for field in fields:
        try:
            value = float(field)
        except ValueError:
            raise ValueError(f"Invalid duration: {text!r}") from None
        if value < 0:
            raise ValueError(f"Invalid duration: {text!r}")
        seconds = seconds * 60 + value
    if seconds <= 0:
        raise ValueError(f"Invalid duration: {text!r}")
    return seconds


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
    cal_freq: float = CALIBRATION_FREQ_HZ,
    cal_level: float = CALIBRATION_LEVEL_DB,
    blocksize: int = DEFAULT_BLOCKSIZE,
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
    samplerate: int = DEFAULT_SAMPLERATE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    cal_freq: float = CALIBRATION_FREQ_HZ,
    cal_level: float = CALIBRATION_LEVEL_DB,
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
# Shared engine runner
# ---------------------------------------------------------------------------

def _build_and_run_engine(
    controller: Controller,
    config: "SLMConfig",
    print_to_console: bool = False,
    display_mode: str = "plain",
    duration: float | None = None,
) -> None:
    """Build and run the engine for *controller*; write results on exit.

    Branch-free across source types: the controller's uniform interface handles
    start/stop (context manager), dropped-block counting (``overruns``), and live
    telemetry (``load_status``); file sources use the inert defaults.
    """
    from slm.assembly import parse_metric, assemble_engine
    from slm.io.reporter import Reporter
    from slm.io.display import make_display_fn

    specs = [parse_metric(m) for m in config.metrics]
    display_fn = make_display_fn(display_mode, precision=2, controller=controller) if print_to_console else None
    reporter = Reporter(precision=2, print_to_console=print_to_console, display_fn=display_fn)
    engine, bindings = assemble_engine(specs, controller, dt=config.dt)
    reporter.add_columns(bindings)
    engine.on_record = reporter.record

    # start execution

    gc.collect()
    gc.disable()
    try:
        with high_priority(), controller:   # controller.__enter__/__exit__ → start/stop
            try:
                engine.run(duration=duration, warmup=config.warmup)
            except KeyboardInterrupt:
                print("\nMeasurement interrupted.")
    finally:
        gc.enable()
        if controller.overruns:
            print(f"Warning: {controller.overruns} block(s) dropped (engine too slow).")
        reporter.write(config.output)


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
    duration: float | None = None,
) -> None:
    """Parse *config.metrics*, build the plugin chain, run the engine, write results.

    *duration* (seconds), if given, stops the run after that much signal has been
    processed (rounded up to the next whole block); otherwise the whole file is read.
    """
    if sensitivity_v <= 0:
        raise ValueError(f"sensitivity_v must be positive, got {sensitivity_v}")
    from slm.io.file_controller import FileController

    controller = FileController(str(wav_path), blocksize=blocksize, realtime=realtime)
    controller.set_sensitivity(sensitivity_v, unit="V")
    _build_and_run_engine(controller, config, print_to_console=print_to_console,
                          display_mode=display_mode, duration=duration)


# ---------------------------------------------------------------------------
# Real-time measurement
# ---------------------------------------------------------------------------

def run_noise_measurement(
    sensitivity_v: float,
    config: "SLMConfig",
    samplerate: int = DEFAULT_SAMPLERATE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    print_to_console: bool = False,
    display_mode: str = "plain",
    duration: float | None = None,
) -> None:
    """Run a measurement driven by white-noise input (no audio hardware required).

    Uses :class:`~slm.io.noise_controller.NoiseController` in real-time mode.
    Useful for testing the processing pipeline and the status display.

    *duration* (seconds), if given, stops the run after that much signal has been
    processed (rounded up to the next whole block); otherwise it runs until Ctrl+C.
    """
    if sensitivity_v <= 0:
        raise ValueError(f"sensitivity_v must be positive, got {sensitivity_v}")
    from slm.io.noise_controller import NoiseController

    controller = NoiseController(
        samplerate=samplerate, blocksize=blocksize,
        realtime=True, queue_maxsize=config.queue_maxsize, dt=config.dt,
    )
    controller.set_sensitivity(sensitivity_v, unit="V")
    print(
        f"  Source: white-noise generator  |  "
        f"Sample rate: {samplerate} Hz  |  "
        f"Block size: {blocksize}  |  "
        f"Queue max: {config.queue_maxsize} blocks"
    )
    _build_and_run_engine(controller, config, print_to_console=print_to_console,
                          display_mode=display_mode, duration=duration)


def run_realtime_measurement(
    sensitivity_v: float,
    config: "SLMConfig",
    device: int | str | None = None,
    samplerate: int = DEFAULT_SAMPLERATE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    print_to_console: bool = False,
    display_mode: str = "plain",
    duration: float | None = None,
) -> None:
    """Start a live measurement from a real-time audio input device.

    The engine runs until *duration* seconds have elapsed (rounded up to the next
    whole block) or until ``KeyboardInterrupt`` (Ctrl+C), at which point the stream
    is stopped and results are written to *config.output*.  If *duration* is None,
    the run is unbounded and stops only on Ctrl+C.
    """
    if sensitivity_v <= 0:
        raise ValueError(f"sensitivity_v must be positive, got {sensitivity_v}")
    from slm.io.sounddevice_controller import SounddeviceController

    controller = SounddeviceController(
        device=device, samplerate=samplerate, blocksize=blocksize,
        queue_maxsize=config.queue_maxsize, dt=config.dt,
    )
    controller.set_sensitivity(sensitivity_v, unit="V")
    print(
        f"  Sample rate: {samplerate} Hz  |  "
        f"Block size: {blocksize}  |  "
        f"Queue max: {config.queue_maxsize} blocks"
    )
    _build_and_run_engine(controller, config, print_to_console=print_to_console,
                          display_mode=display_mode, duration=duration)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

class SLMShell(cmd.Cmd):
    """Interactive SLM REPL.

    Commands: add, remove, file, device, generator, sensitivity, calibrate,
              output, name, warmup, dt, queue, samplerate, blocksize, show, save,
              load, start, display, realtime, tree, inspect, exit/quit/EOF.
    """

    intro = (
        "soundlevelmeter  Copyright (C) 2026  Nino Blumer\n"
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
        self._generator_mode: bool = False
        self._samplerate: int = Controller.DEFAULT_SAMPLERATE
        self._blocksize: int = Controller.DEFAULT_BLOCKSIZE

    def emptyline(self) -> bool:
        """Do nothing on an empty line.

        The :class:`cmd.Cmd` default repeats the last command; instead we
        just print a fresh prompt.
        """
        return False

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
        self._generator_mode = False
        print(f"Device: {self._device!r}")

    def do_generator(self, _: str) -> None:
        """generator — use the white-noise generator as input source.

Produces Gaussian white noise in real-time without requiring any audio
hardware.  Useful for testing the processing pipeline and the status display.

Clears any previously set file or device source.
"""
        self._generator_mode = True
        self._wav_path = None
        self._device = None
        print("  Source: white-noise generator")

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
        self._generator_mode = False
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
        cal_level = CALIBRATION_LEVEL_DB
        cal_freq = CALIBRATION_FREQ_HZ
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
        """output DIR — set the directory where result files are written.

The measurement name (see `name`) is kept; files are written to
DIR/NAME_report.csv etc.
"""
        directory = arg.strip()
        if not directory:
            print(f"Output dir: {Path(self._config.output).parent}")
            return
        name = Path(self._config.output).name
        self._config.output = str(Path(directory) / name)
        print(f"Output dir: {directory}")

    def do_name(self, arg: str) -> None:
        """name NAME — set the measurement name (the output file stem).

The output directory (see `output`) is kept; files are written to
DIR/NAME_report.csv etc.
"""
        name = arg.strip()
        if not name:
            print(f"Measurement: {Path(self._config.output).name}")
            return
        directory = Path(self._config.output).parent
        self._config.output = str(directory / name)
        print(f"Measurement: {name}")
        print(f"  -> {self._config.output}_report.csv")

    def do_warmup(self, arg: str) -> None:
        """warmup SECONDS — settle the chain for SECONDS before measuring (default 0).

During warm-up the signal is processed but not logged; the meters are then
reset so the accumulators (Leq, max/min) start from a settled state rather
than from the initial filter transient.  Set 0 to disable.
"""
        arg = arg.strip()
        if not arg:
            print(f"Warm-up: {self._config.warmup} s")
            return
        try:
            warmup = float(arg)
            if warmup < 0:
                raise ValueError
        except ValueError:
            print(f"Invalid warmup: {arg!r}  (must be a non-negative number)")
            return
        self._config.warmup = warmup
        print(f"Warm-up: {warmup} s")

    def do_dt(self, arg: str) -> None:
        """dt SECONDS — set the logging interval.

Note: log rows are written on block edges, so the cadence is quantized to the
block duration (blocksize/fs).  A dt that is an exact integer multiple of the
block duration logs at the requested interval; otherwise each interval is
rounded up to the next block edge and the effective dt drifts slightly above
the value set here.
"""
        try:
            self._config.dt = float(arg.strip())
            print(f"dt: {self._config.dt} s")
        except ValueError:
            print(f"Invalid dt: {arg.strip()!r}")

    def do_queue(self, arg: str) -> None:
        """queue N — set the real-time block queue depth (default: 0 = unbounded).

N blocks are buffered between the audio driver and the engine.  A larger
finite value absorbs OS scheduling jitter at the cost of higher latency.
0 means an unbounded queue (no latency bound, no dropped blocks).  The
engine reports overruns when a finite buffer is full.

Examples:
  queue      show current setting
  queue 0    unbounded — no dropped blocks, latency may grow (default)
  queue 16   generous — absorbs bursts; ~85 ms latency at bs=4096/48 kHz
"""
        arg = arg.strip()
        if not arg:
            print(f"  Queue max: {self._config.queue_maxsize} blocks")
            return
        try:
            n = int(arg)
            if n < 0:
                raise ValueError
        except ValueError:
            print(f"Invalid value: {arg!r}  (must be a non-negative integer; 0 = unbounded)")
            return
        self._config.queue_maxsize = n
        print(f"  Queue max: {n} blocks")

    def do_samplerate(self, arg: str) -> None:
        """samplerate HZ — set the sample rate for real-time (device/generator) input.

Ignored for file input, where the sample rate is read from the WAV header.

Examples:
  samplerate         show current setting
  samplerate 48000   request 48 kHz (default)
"""
        arg = arg.strip()
        if not arg:
            print(f"  Sample rate: {self._samplerate} Hz")
            return
        try:
            hz = int(arg)
            if hz < 1:
                raise ValueError
        except ValueError:
            print(f"Invalid value: {arg!r}  (must be a positive integer)")
            return
        self._samplerate = hz
        print(f"  Sample rate: {hz} Hz")

    def do_blocksize(self, arg: str) -> None:
        """blocksize SAMPLES — set the processing block size in samples.

Smaller blocks lower latency for real-time input; larger blocks reduce
per-block overhead.  Does not affect measured levels.

Examples:
  blocksize        show current setting
  blocksize 1024   default
"""
        arg = arg.strip()
        if not arg:
            print(f"  Block size: {self._blocksize} samples")
            return
        try:
            n = int(arg)
            if n < 1:
                raise ValueError
        except ValueError:
            print(f"Invalid value: {arg!r}  (must be a positive integer)")
            return
        self._blocksize = n
        print(f"  Block size: {n} samples")

    def do_show(self, _: str) -> None:
        """show — display the current configuration."""
        if self._generator_mode:
            source = "white-noise generator"
        elif self._wav_path:
            source = f"file: {self._wav_path}"
        elif self._device is not None:
            source = f"device: {self._device!r}"
        else:
            source = "(not set)"
        out_path = Path(self._config.output)
        print(f"  Source:      {source}")
        print(f"  Sensitivity: {'(not set)' if self._sensitivity_v is None else self._sensitivity_v}")
        print(f"  dt:          {self._config.dt} s")
        print(f"  Warm-up:     {self._config.warmup} s")
        print(f"  Sample rate: {self._samplerate} Hz")
        print(f"  Block size:  {self._blocksize} samples")
        print(f"  Queue max:   {self._config.queue_maxsize} blocks")
        print(f"  Output dir:  {out_path.parent}")
        print(f"  Name:        {out_path.name}")
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
        from slm.assembly import parse_metric, plan_chain, node_label, meter_class_name

        if not self._config.metrics:
            print("No metrics added.  Use: add METRIC")
            return

        plans = []
        for name in self._config.metrics:
            try:
                plans.append(plan_chain(parse_metric(name)))
            except ValueError as exc:
                print(f"  Error parsing {name!r}: {exc}")
                return

        print(f"Planned chain  (dt={self._config.dt} s)")

        # Merge the plans into a trie keyed by node dedup key: metrics sharing an
        # upstream node share a branch — the same prefix-sharing build_chain uses
        # to dedup plugins.  Each entry holds its NodeReq, child nodes, and the
        # meters of metrics that terminate at this node.
        root: dict = {}
        for plan in plans:
            children = root
            entry: dict = {}
            for node in plan.nodes:
                entry = children.setdefault(
                    node.key, {"node": node, "children": {}, "meters": []}
                )
                children = entry["children"]
            entry["meters"].append(plan.meter)

        def _meter_line(meter) -> str:
            cls = meter_class_name(meter)
            if not meter.moving:
                detail = ""
            elif meter.window_is_dt:
                detail = f"   t=dt={self._config.dt} s"
            else:
                detail = f"   t={meter.window_seconds} s"
            return f"{meter.name:<32} {cls}{detail}"

        def _render(node_entry: dict, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + node_label(node_entry["node"]))
            child_prefix = prefix + ("    " if is_last else "│   ")
            # Meters terminating here come first, then downstream plugin nodes.
            items = ([("meter", m) for m in node_entry["meters"]]
                     + [("node", c) for c in node_entry["children"].values()])
            for i, (kind, obj) in enumerate(items):
                item_last = i == len(items) - 1
                if kind == "meter":
                    leaf = "└── " if item_last else "├── "
                    print(child_prefix + leaf + _meter_line(obj))
                else:
                    _render(obj, child_prefix, item_last)

        bus_entries = list(root.values())
        for i, bus_entry in enumerate(bus_entries):
            _render(bus_entry, "", i == len(bus_entries) - 1)

    def do_inspect(self, arg: str) -> None:
        """inspect METRIC — show detailed human-readable info for a metric."""
        from slm.assembly import parse_metric, plan_chain, meter_class_name

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

        meter = plan_chain(spec).meter
        meter_cls = meter_class_name(meter)
        if meter.moving:
            if meter.window_is_dt:
                window_str = f"t=dt={self._config.dt} s"
            else:
                window_str = f"t={meter.window_seconds} s"
        else:
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
        print(f"  Window:       {'moving' if meter.moving else 'accumulating'}")

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
            "  5. output DIR         — set output directory\n"
            "  6. name NAME          — set measurement name (output file stem)\n"
            "  7. warmup SECONDS     — optional settle time before measuring\n"
            "  8. start [DURATION]   — run the measurement (optionally fixed-length)\n"
            "  9. save FILE.toml     — save config for next time"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def do_start(self, arg: str) -> None:
        """start [DURATION] — run the measurement with the current configuration.

DURATION, if given, runs a fixed-length measurement and stops automatically.
It is accepted as hh:mm:ss, mm:ss, or just ss (fields may be fractional):

  start            run until end of file / Ctrl+C
  start 30         run for 30 seconds
  start 1:30       run for 1 minute 30 seconds
  start 01:00:00   run for 1 hour

The run stops on a block edge, so the measured length is rounded up to the
next whole block (blocksize/fs).
"""
        duration: float | None = None
        if arg.strip():
            try:
                duration = parse_duration(arg)
            except ValueError as exc:
                print(exc)
                return
        if not self._wav_path and self._device is None and not self._generator_mode:
            print("No source set.  Use: file PATH  |  device INDEX  |  generator")
            return
        if self._sensitivity_v is None:
            print("No sensitivity set.  Use: sensitivity ... or calibrate")
            return
        if not self._config.metrics:
            print("No metrics set.  Use: add METRIC")
            return
        if self._generator_mode:
            run_noise_measurement(
                self._sensitivity_v,
                self._config,
                samplerate=self._samplerate,
                blocksize=self._blocksize,
                print_to_console=True,
                display_mode=self._display_mode,
                duration=duration,
            )
        elif self._wav_path:
            run_measurement(
                self._wav_path,
                self._sensitivity_v,
                self._config,
                print_to_console=True,
                blocksize=self._blocksize,
                display_mode=self._display_mode,
                realtime=self._realtime,
                duration=duration,
            )
        else:
            run_realtime_measurement(
                self._sensitivity_v,
                self._config,
                device=self._device,
                samplerate=self._samplerate,
                blocksize=self._blocksize,
                print_to_console=True,
                display_mode=self._display_mode,
                duration=duration,
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
