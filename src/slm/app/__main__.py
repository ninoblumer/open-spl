"""Entry point: ``python -m slm``.

Modes
-----
Interactive REPL (default when no action flags are given)::

    python -m slm

Interactive REPL with pre-populated state (combine -i with other flags)::

    python -m slm -i --file PATH --sensitivity-dbv DBV [--measure METRIC ...] [...]

Calibration (derive sensitivity from a calibrator-tone recording)::

    python -m slm --calibrate --file PATH [--cal-level DB] [--cal-freq HZ]

One-shot measurement::

    python -m slm --file PATH --measure METRIC [METRIC ...] [--fs-db DB] [...]
    python -m slm --file PATH --config FILE.toml [--fs-db DB] [...]
"""
from __future__ import annotations

import argparse

from slm.constants import CALIBRATION_FREQ_HZ, CALIBRATION_LEVEL_DB
from slm.io.controller import Controller
from slm.io.realtime_controller import RealtimeController


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slm",
        description="Sound Level Meter — IEC 61672-1 compliant measurement tool.",
    )

    # Input source: --file, --device, --generator are mutually exclusive
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--file", metavar="PATH", help="Input WAV file")
    source_group.add_argument(
        "--device", metavar="INDEX_OR_NAME",
        help="Real-time audio input device index or name substring (use --list-devices to see options)",
    )
    source_group.add_argument(
        "--generator", action="store_true",
        help="Use the white-noise generator as input (no audio hardware required)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--samplerate", type=int, default=Controller.DEFAULT_SAMPLERATE, metavar="HZ",
        help="Sample rate for real-time input (default: 48000)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start the interactive REPL (default when no action flags are given)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibration mode: compute sensitivity from a calibrator-tone recording",
    )
    parser.add_argument(
        "--cal-level", type=float, default=CALIBRATION_LEVEL_DB, metavar="DB",
        help="Calibrator level in dB SPL (default: 94.0)",
    )
    parser.add_argument(
        "--cal-freq", type=float, default=CALIBRATION_FREQ_HZ, metavar="HZ",
        help="Calibrator tone frequency in Hz (default: 1000.0)",
    )
    parser.add_argument(
        "--measure", nargs="+", metavar="METRIC",
        help="One or more metric names to compute (e.g. LAeq LAFmax LZeq:bands:63-8000)",
    )
    parser.add_argument(
        "--config", metavar="FILE.toml",
        help="Load measurement configuration from a TOML file",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Output file base path (default: output/measurement)",
    )
    parser.add_argument(
        "--dt", type=float, default=None, metavar="SECONDS",
        help="Logging interval in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--realtime", "-r", action="store_true",
        help="Simulate real-time playback: pace processing so each dt interval takes dt real seconds",
    )
    parser.add_argument(
        "--queue-maxsize", type=int, default=None, metavar="N",
        help="Real-time block queue depth (default: 4). Larger values absorb scheduling jitter.",
    )

    sens_group = parser.add_mutually_exclusive_group()
    sens_group.add_argument(
        "--fs-db", type=float, metavar="DB",
        help="WAV full-scale annotation in dBSPL (from XL2 filename, e.g. 128.1)",
    )
    sens_group.add_argument(
        "--sensitivity-dbv", type=float, metavar="DBV",
        help="Microphone sensitivity in dBV (re 1 V/Pa)",
    )
    sens_group.add_argument(
        "--sensitivity-mv", type=float, metavar="MV",
        help="Microphone sensitivity in mV/Pa",
    )

    return parser


def _resolve_sensitivity(args: argparse.Namespace) -> float | None:
    """Return sensitivity in V from the CLI flags, or None if none were given."""
    from slm.app.cli import sensitivity_from_fs_db, sensitivity_from_dbv, sensitivity_from_mv
    if args.fs_db is not None:
        return sensitivity_from_fs_db(args.fs_db)
    if args.sensitivity_dbv is not None:
        return sensitivity_from_dbv(args.sensitivity_dbv)
    if args.sensitivity_mv is not None:
        return sensitivity_from_mv(args.sensitivity_mv)
    return None


def _require_sounddevice(parser: argparse.ArgumentParser) -> None:
    """Exit with a friendly message if sounddevice is not installed."""
    try:
        import slm.io.sounddevice_controller  # noqa: F401
    except ImportError as exc:
        parser.error(str(exc))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_devices or args.device is not None:
        _require_sounddevice(parser)

    if args.generator and not (args.measure or args.config):
        parser.error("--generator requires --measure METRIC [...] or --config FILE.toml")

    # ------------------------------------------------------------------ #
    # Interactive REPL                                                     #
    # ------------------------------------------------------------------ #
    # --list-devices: print table and exit
    if args.list_devices:
        from slm.io.sounddevice_controller import SounddeviceController
        devices = SounddeviceController.list_devices()
        if not devices:
            print("No input devices found.")
        else:
            from slm.app.cli import _fmt_device_table
            print(_fmt_device_table(devices))
        return

    # --realtime requires --file
    if args.realtime and not args.file:
        parser.error("--realtime requires --file")

    no_action = (not args.file and args.device is None and not args.generator
                 and not args.calibrate and not args.measure and not args.config)
    if no_action:
        # Bare invocation — open an empty shell
        from slm.app.cli import SLMShell
        SLMShell().cmdloop()
        return

    if args.interactive:
        # --interactive alongside other flags: pre-populate shell state
        from slm.app.cli import SLMShell
        from slm.app.config import SLMConfig

        if args.config:
            config = SLMConfig.from_toml(args.config)
            if args.measure:
                config.metrics = list(args.measure)
            if args.output is not None:
                config.output = args.output
            if args.dt is not None:
                config.dt = args.dt
            if args.queue_maxsize is not None:
                config.queue_maxsize = args.queue_maxsize
        else:
            config = SLMConfig.from_args(
                metrics=list(args.measure) if args.measure else [],
                dt=args.dt if args.dt is not None else 1.0,
                output=args.output if args.output is not None else "output/measurement",
                queue_maxsize=args.queue_maxsize if args.queue_maxsize is not None else RealtimeController.DEFAULT_QUEUE_MAXSIZE,
            )

        # Parse device: try int, fall back to string
        device = None
        if args.device is not None:
            try:
                device = int(args.device)
            except ValueError:
                device = args.device

        shell = SLMShell(
            wav_path=args.file,
            sensitivity_v=_resolve_sensitivity(args),
            config=config,
        )
        shell._device = device
        if args.generator:
            shell._generator_mode = True
        shell.cmdloop()
        return

    # ------------------------------------------------------------------ #
    # Calibration mode                                                     #
    # ------------------------------------------------------------------ #
    if args.calibrate:
        if not args.file and args.device is None:
            parser.error("--calibrate requires --file or --device")
        if _resolve_sensitivity(args) is not None:
            parser.error("--calibrate cannot be combined with a sensitivity flag")
        from slm.app.cli import calibrate_from_file, calibrate_from_device, _fmt_sensitivity
        if args.file:
            sens = calibrate_from_file(args.file, cal_freq=args.cal_freq, cal_level=args.cal_level)
        else:
            print("Listening for calibrator tone — will stop automatically when stable ...")
            sens = calibrate_from_device(
                device=args.device,
                samplerate=args.samplerate,
                cal_freq=args.cal_freq,
                cal_level=args.cal_level,
            )
        print(f"Sensitivity: {_fmt_sensitivity(sens)}")
        return

    # ------------------------------------------------------------------ #
    # One-shot measurement                                                 #
    # ------------------------------------------------------------------ #
    from slm.app.config import SLMConfig
    from slm.app.cli import run_measurement

    if args.config:
        config = SLMConfig.from_toml(args.config)
        # CLI flags override loaded values
        if args.measure:
            config.metrics = list(args.measure)
        if args.output is not None:
            config.output = args.output
        if args.dt is not None:
            config.dt = args.dt
        if args.queue_maxsize is not None:
            config.queue_maxsize = args.queue_maxsize
    else:
        if not args.measure:
            parser.error(
                "One of --measure METRIC [...] or --config FILE.toml is required "
                "for one-shot mode"
            )
        config = SLMConfig.from_args(
            metrics=list(args.measure),
            dt=args.dt if args.dt is not None else 1.0,
            output=args.output if args.output is not None else "output/measurement",
            queue_maxsize=args.queue_maxsize if args.queue_maxsize is not None else RealtimeController.DEFAULT_QUEUE_MAXSIZE,
        )

    if not args.file and args.device is None and not args.generator:
        parser.error("--file, --device, or --generator is required for one-shot measurement")

    sens = _resolve_sensitivity(args)
    if sens is None:
        parser.error(
            "A sensitivity flag is required: --fs-db DB, "
            "--sensitivity-dbv DBV, or --sensitivity-mv MV"
        )

    if args.file:
        run_measurement(args.file, sens, config, print_to_console=True, realtime=args.realtime)
    elif args.generator:
        from slm.app.cli import run_noise_measurement
        run_noise_measurement(
            sens, config,
            samplerate=args.samplerate,
            print_to_console=True,
        )
    else:
        from slm.app.cli import run_realtime_measurement
        run_realtime_measurement(
            sens, config,
            device=args.device,
            samplerate=args.samplerate,
            print_to_console=True,
        )


if __name__ == "__main__":
    main()
