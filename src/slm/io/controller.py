from abc import ABC, abstractmethod
import itertools
from typing import Literal

import numpy as np

from slm.constants import CALIBRATION_LEVEL_DB


class Controller(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self._counter = itertools.count(0)

    @property
    @abstractmethod
    def samplerate(self) -> int: ...

    @property
    @abstractmethod
    def blocksize(self) -> int: ...

    @property
    @abstractmethod
    def sensitivity(self) -> float: ...

    @property
    @abstractmethod
    def overruns(self) -> int:
        """Blocks dropped because the engine fell behind."""
        ...

    @abstractmethod
    def read_block(self) -> tuple[np.ndarray, int]:
        """ read a block of audio and returns the buffer and the block_index """
        ...

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def calibrate(self, target_spl=CALIBRATION_LEVEL_DB):
        ...

    def set_sensitivity(self, sensitivity: float, unit: Literal["mV", "V", "dB"]) -> None:
        if unit == "mV":
            self._sensitivity = sensitivity / 1000.0
        elif unit == "V":
            self._sensitivity = sensitivity
        elif unit == "dB":
            self._sensitivity = 10**(sensitivity/20)
        else:
            raise ValueError(f"Unknown sensitivity unit: {unit!r}. Expected 'mV', 'V', or 'dB'.")

    # ------------------------------------------------------------------
    # Lifecycle / telemetry — uniform across sources so the caller never has to
    # special-case the source type.  Real-time controllers override these; the
    # defaults suit pull-based sources (e.g. files): start is a no-op and there
    # is no live load telemetry.
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin producing audio.  No-op for pull-based sources (e.g. files);
        real-time sources override this to launch their producer."""

    def load_status(self) -> str | None:
        """One-line load/queue telemetry for the live display, or ``None`` when
        the source has none (e.g. file playback)."""
        return None

    def __enter__(self) -> "Controller":
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

