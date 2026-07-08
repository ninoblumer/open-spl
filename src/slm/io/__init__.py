"""Public API for slm.io — I/O controllers, reporter, and display helpers."""
from slm.io.controller import Controller
from slm.io.file_controller import FileController
from slm.io.array_controller import ArrayController
from slm.io.realtime_controller import (
    DEFAULT_BLOCKSIZE,
    DEFAULT_QUEUE_MAXSIZE,
    DEFAULT_SAMPLERATE,
    RealtimeController,
)
from slm.io.noise_controller import NoiseController
from slm.io.reporter import Reporter
from slm.io.results import MeasurementResults
from slm.io.display import make_display_fn

try:
    from slm.io.sounddevice_controller import SounddeviceController
    _has_sounddevice = True
except ImportError: # pragma: no cover
    _has_sounddevice = False

__all__ = [
    "DEFAULT_SAMPLERATE",
    "DEFAULT_BLOCKSIZE",
    "DEFAULT_QUEUE_MAXSIZE",
    "Controller",
    "FileController",
    "ArrayController",
    "RealtimeController",
    "NoiseController",
    "Reporter",
    "MeasurementResults",
    "make_display_fn",
    *( ["SounddeviceController"] if _has_sounddevice else [] ),
]