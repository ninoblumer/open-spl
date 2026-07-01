"""Abstract base class for queue-backed real-time audio controllers."""
from __future__ import annotations

import queue
import threading
import time
from abc import abstractmethod
from collections import deque

import numpy as np

from slm.io.controller import Controller

# ---------------------------------------------------------------------------
# Shared audio defaults — the single source of truth for these values.
# Every other module (controllers, config, CLI, tests) imports them from here
# instead of hardcoding literals.
# ---------------------------------------------------------------------------
DEFAULT_SAMPLERATE: int = 48_000
DEFAULT_BLOCKSIZE: int = 1_024
DEFAULT_QUEUE_MAXSIZE: int = 0  # zero means infinite queue size -> no bound on latency


class RealtimeController(Controller):
    """Base class for queue-backed real-time audio controllers.

    Owns the block queue, stop event, overrun counter, sensitivity, and load
    monitoring so subclasses only need to implement the producer side
    (``start`` / ``stop`` / ``calibrate`` / ``list_devices``) and push blocks
    into ``self._queue``.

    Parameters
    ----------
    samplerate:
        Sample rate in Hz.
    blocksize:
        Samples per block.
    queue_maxsize:
        Maximum queued blocks before the producer drops and counts an overrun.
    """

    def __init__(
        self,
        samplerate: int,
        blocksize: int,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        dt: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._sensitivity: float = 1.0
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._overruns: int = 0

        # Load monitoring: rolling window sized to dt (the logging interval)
        _window = max(1, round(dt * samplerate / blocksize))
        self._rho_buf: deque[float] = deque(maxlen=_window)
        self._qdepth_buf: deque[int] = deque(maxlen=_window)
        self._last_return_t: float | None = None

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def read_block(self) -> tuple[np.ndarray, int]:
        """Pop the next block from the queue and return ``(block, index)``.

        Records per-block processing utilisation rho = t_proc / t_budget in a
        ~1 s rolling window, where t_proc is the time elapsed since the
        previous block was returned (i.e. the engine's processing time for
        the previous block).

        Raises :exc:`StopIteration` once :meth:`stop` has been called and the
        queue is drained.
        """
        t_now = time.perf_counter()
        if self._last_return_t is not None:
            t_proc = t_now - self._last_return_t
            self._rho_buf.append(t_proc / (self._blocksize / self._samplerate))
            self._qdepth_buf.append(self._queue.qsize())

        while True:
            try:
                block = self._queue.get(timeout=0.5)
                self._last_return_t = time.perf_counter()
                return block, next(self._counter)
            except queue.Empty:
                if self._stop_event.is_set():
                    raise StopIteration

    def start(self) -> None:
        """Reset monitoring state and clear the stop flag.

        Subclasses must call ``super().start()`` before starting their producer.
        """
        self._stop_event.clear()
        self._last_return_t = None
        self._rho_buf.clear()
        self._qdepth_buf.clear()

    def stop(self) -> None:
        """Signal the queue consumer to stop.

        Subclasses that own a producer thread or stream must tear it down
        first, then call ``super().stop()``.
        """
        self._stop_event.set()

    # ------------------------------------------------------------------
    # RealtimeController interface — subclasses must implement
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def list_devices() -> list[dict]:
        """Return available input devices (empty list if not applicable)."""
        ...

    # ------------------------------------------------------------------
    # Load monitoring
    # ------------------------------------------------------------------

    @property
    def overruns(self) -> int:
        """Blocks dropped because the engine fell behind (queue full or stream error)."""
        return self._overruns

    @property
    def rho_mean(self) -> float | None:
        """Mean processing utilisation over the last ~1 s, or ``None`` if no data yet.

        rho = t_proc / t_budget. Values above 1.0 indicate the engine is
        missing real-time deadlines.
        """
        if not self._rho_buf:
            return None
        return sum(self._rho_buf) / len(self._rho_buf)

    @property
    def queue_depth_max(self) -> int:
        """Maximum queue depth observed over the last ~1 s.

        A sustained non-zero value means the engine is consistently slower
        than real-time and blocks are accumulating in the queue.
        """
        return max(self._qdepth_buf, default=0)

    def load_status(self) -> str:
        """Compact live telemetry: ``Load=NN%  Q=depth/max  [missed blocks=N]``."""
        rho = self.rho_mean
        load_str = f"{rho * 100:.0f}%" if rho is not None else "---%"
        parts = [f"Load={load_str}", f"Q={self.queue_depth_max}/{self._queue.maxsize}"]
        if self.overruns > 0:
            parts.append(f"missed blocks={self.overruns}")
        return "  ".join(parts)
