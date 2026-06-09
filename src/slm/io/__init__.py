"""Public API for slm.io — I/O controllers, reporter, and display helpers."""
from slm.io.controller import Controller
from slm.io.file_controller import FileController
from slm.io.realtime_controller import RealtimeController
from slm.io.noise_controller import NoiseController
from slm.io.reporter import Reporter
from slm.io.display import make_display_fn

try:
    from slm.io.sounddevice_controller import SounddeviceController
    _has_sounddevice = True
except ImportError: # pragma: no cover
    _has_sounddevice = False

__all__ = [
    "Controller",
    "FileController",
    "RealtimeController",
    "NoiseController",
    "Reporter",
    "make_display_fn",
    *( ["SounddeviceController"] if _has_sounddevice else [] ),
]