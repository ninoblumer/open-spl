"""Public API for slm.app — configuration, CLI helpers, and measurement runners."""
from slm.app.config import SLMConfig
from slm.io.results import MeasurementResults
from slm.app.cli import (
    SLMShell,
    sensitivity_from_fs_db,
    sensitivity_from_mv,
    sensitivity_from_dbv,
    parse_duration,
    calibrate_from_file,
    calibrate_from_device,
    run_measurement,
    run_realtime_measurement,
    normalize_signal_conditioning,
    resolve_signal_conditioning,
)

__all__ = [
    "SLMConfig",
    "MeasurementResults",
    "SLMShell",
    "sensitivity_from_fs_db",
    "sensitivity_from_mv",
    "sensitivity_from_dbv",
    "parse_duration",
    "calibrate_from_file",
    "calibrate_from_device",
    "run_measurement",
    "run_realtime_measurement",
    "normalize_signal_conditioning",
    "resolve_signal_conditioning",
]
