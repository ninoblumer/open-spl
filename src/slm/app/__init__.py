"""Public API for slm.app — configuration, CLI helpers, and measurement runners."""
from slm.app.config import SLMConfig
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
)

__all__ = [
    "SLMConfig",
    "SLMShell",
    "sensitivity_from_fs_db",
    "sensitivity_from_mv",
    "sensitivity_from_dbv",
    "parse_duration",
    "calibrate_from_file",
    "calibrate_from_device",
    "run_measurement",
    "run_realtime_measurement",
]
