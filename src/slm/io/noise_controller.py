"""White-noise real-time controller for engine load testing."""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from slm.io.realtime_controller import RealtimeController


class NoiseController(RealtimeController):
    """Real-time controller that generates white noise without a file or audio device.

    A background thread produces Gaussian white-noise blocks and pushes them
    into the shared queue.  Runs until :meth:`stop` is called.

    Two pacing modes:

    * ``realtime=True`` (default) — producer sleeps to maintain
      ``blocksize / samplerate`` seconds per block, matching a live audio
      stream.  Excess blocks are dropped and counted in :attr:`overruns`.
    * ``realtime=False`` — producer runs as fast as possible; it blocks when
      the queue is full so the engine drives the pace.  Use this for
      Quantity-A load measurements where wall-clock time should not limit
      throughput.

    Parameters
    ----------
    samplerate:
        Sample rate in Hz (default 48 000).
    blocksize:
        Samples per block (default 1 024).
    channels:
        Number of channels per block (default 1).
    realtime:
        Enable real-time pacing (default ``True``).
    queue_maxsize:
        Queue depth before excess blocks are dropped in realtime mode
        (default 16).
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        samplerate: int = 48_000,
        blocksize: int = 1_024,
        channels: int = 1,
        realtime: bool = True,
        queue_maxsize: int = 16,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(samplerate=samplerate, blocksize=blocksize,
                         queue_maxsize=queue_maxsize, **kwargs)
        self._channels = channels
        self._realtime = realtime
        self._rng = np.random.default_rng(seed)
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # RealtimeController interface
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict]:
        return []

    def start(self) -> None:
        """Start the noise-producer thread."""
        super().start()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the noise-producer thread."""
        super().stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def calibrate(self, target_spl: float = 94.0) -> None:
        raise NotImplementedError("NoiseController has no calibration")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _produce(self) -> None:
        t_budget = self._blocksize / self._samplerate
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            block = self._rng.standard_normal(
                (self._blocksize, self._channels)
            ).astype(np.float32)
            if self._realtime:
                try:
                    self._queue.put_nowait(block)
                except queue.Full:
                    self._overruns += 1
                elapsed = time.perf_counter() - t0
                sleep_for = t_budget - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            else:
                # Non-realtime: block until the consumer frees space.
                # Use a short timeout so the stop event is still checked.
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(block, timeout=0.1)
                        break
                    except queue.Full:
                        pass
