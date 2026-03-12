"""Entry point: ``python -m slm``.

Modes
-----
Interactive REPL (default when no action flags are given)::

    python -m slm

Interactive REPL with pre-populated state (combine -i with other flags)::

    python -m slm -i --file PATH --sensitivity-dbv DBV [--measure METRIC ...] [...]

Calibration (derive sensitivity from a calibrator-tone recording)::

    python -m slm --calibrate --file PATH [--cal-level DB]

One-shot measurement::

    python -m slm --file PATH --measure METRIC [METRIC ...] [--fs-db DB] [...]
    python -m slm --file PATH --config FILE.toml [--fs-db DB] [...]
"""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm",
        description="Sound Level Meter — IEC 61672-1 compliant measurement tool.",
    )

    parser.add_argument("--file", metavar="PATH", help="Input WAV file")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start the interactive REPL (default when no action flags are given)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibration mode: compute sensitivity from a calibrator-tone recording",
    )
    parser.add_argument(
        "--cal-level", type=float, default=94.0, metavar="DB",
        help="Calibrator level in dB SPL (default: 94.0)",
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
        "--output", default="output/measurement", metavar="PATH",
        help="Output file base path (default: output/measurement)",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0, metavar="SECONDS",
        help="Logging interval in seconds (default: 1.0)",
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
    from slm.cli import sensitivity_from_fs_db, sensitivity_from_dbv, sensitivity_from_mv
    if args.fs_db is not None:
        return sensitivity_from_fs_db(args.fs_db)
    if args.sensitivity_dbv is not None:
        return sensitivity_from_dbv(args.sensitivity_dbv)
    if args.sensitivity_mv is not None:
        return sensitivity_from_mv(args.sensitivity_mv)
    return None


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Interactive REPL                                                     #
    # ------------------------------------------------------------------ #
    no_action = not args.file and not args.calibrate and not args.measure and not args.config
    if no_action:
        # Bare invocation — open an empty shell
        from slm.cli import SLMShell
        SLMShell().cmdloop()
        return

    if args.interactive:
        # --interactive alongside other flags: pre-populate shell state
        from slm.cli import SLMShell
        from slm.config import SLMConfig

        if args.config:
            config = SLMConfig.from_toml(args.config)
            if args.measure:
                config.metrics = list(args.measure)
            if args.output != "output/measurement":
                config.output = args.output
            if args.dt != 1.0:
                config.dt = args.dt
        else:
            config = SLMConfig.from_args(
                metrics=list(args.measure) if args.measure else [],
                dt=args.dt,
                output=args.output,
            )

        SLMShell(
            wav_path=args.file,
            sensitivity_v=_resolve_sensitivity(args),
            config=config,
        ).cmdloop()
        return

    # ------------------------------------------------------------------ #
    # Calibration mode                                                     #
    # ------------------------------------------------------------------ #
    if args.calibrate:
        if not args.file:
            parser.error("--calibrate requires --file")
        if _resolve_sensitivity(args) is not None:
            parser.error("--calibrate cannot be combined with a sensitivity flag")
        from slm.cli import calibrate_sensitivity
        sens = calibrate_sensitivity(args.file, cal_level=args.cal_level)
        print(f"Sensitivity: {sens:.6g} V")
        return

    # ------------------------------------------------------------------ #
    # One-shot measurement                                                 #
    # ------------------------------------------------------------------ #
    from slm.config import SLMConfig
    from slm.cli import run_measurement

    if args.config:
        config = SLMConfig.from_toml(args.config)
        # CLI flags override loaded values
        if args.measure:
            config.metrics = list(args.measure)
        if args.output != "output/measurement":
            config.output = args.output
        if args.dt != 1.0:
            config.dt = args.dt
    else:
        if not args.measure:
            parser.error(
                "One of --measure METRIC [...] or --config FILE.toml is required "
                "for one-shot mode"
            )
        config = SLMConfig.from_args(
            metrics=list(args.measure),
            dt=args.dt,
            output=args.output,
        )

    if not args.file:
        parser.error("--file is required for one-shot measurement")

    sens = _resolve_sensitivity(args)
    if sens is None:
        parser.error(
            "A sensitivity flag is required: --fs-db DB, "
            "--sensitivity-dbv DBV, or --sensitivity-mv MV"
        )

    run_measurement(args.file, sens, config, print_to_console=True)


if __name__ == "__main__":
    main()
