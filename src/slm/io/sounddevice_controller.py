"""Real-time audio controller backed by sounddevice (PortAudio).

Works on Windows (WASAPI/MME/DirectSound/ASIO), macOS (CoreAudio),
and Linux (ALSA/PulseAudio/JACK) without any additional dependencies
beyond the ``sounddevice`` package.
"""
from __future__ import annotations

import queue
import warnings

import numpy as np
try:
    import sounddevice as sd
except ImportError as _exc:
    raise ImportError(
        "Real-time audio requires the sounddevice package. "
        "Install it with: pip install sounddevice"
    ) from _exc

from slm.io.realtime_controller import RealtimeController


class SounddeviceController(RealtimeController):
    """Cross-platform real-time audio controller using PortAudio via sounddevice.

    The PortAudio callback runs on a dedicated OS audio thread and pushes
    blocks into the queue owned by :class:`RealtimeController`.
    :meth:`read_block` pops from the queue on the engine's main thread.

    Parameters
    ----------
    device:
        PortAudio device index or substring of a device name.  ``None``
        uses the system default input device.
    samplerate:
        Sample rate in Hz (default 48 000).
    blocksize:
        Samples per block delivered to the engine (default 1 024).
    channels:
        Number of input channels (default 1).
    dtype:
        Sample format passed to sounddevice (default ``'float32'``).
    queue_maxsize:
        Maximum number of blocks buffered between the callback and
        :meth:`read_block`.  If the engine falls behind, excess blocks are
        dropped and :attr:`overruns` is incremented (default 16).
    """

    def __init__(
        self,
        device: int | str | None = None,
        samplerate: int = 48_000,
        blocksize: int = 1_024,
        channels: int = 1,
        dtype: str = "float32",
        queue_maxsize: int = 16,
        **kwargs,
    ) -> None:
        if channels > 1:
            warnings.warn(
                f"SounddeviceController was given channels={channels}; only mono is supported. "
                "Only channel 0 will be used.",
                UserWarning,
                stacklevel=2,
            )
            channels = 1
        super().__init__(samplerate=samplerate, blocksize=blocksize,
                         queue_maxsize=queue_maxsize, **kwargs)
        self._device = device
        self._channels = channels
        self._dtype = dtype
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # RealtimeController interface
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict]:
        """Return all input-capable devices reported by PortAudio."""
        return [
            {
                "index": i,
                "name": d["name"],
                "max_input_channels": d["max_input_channels"],
                "default_samplerate": d["default_samplerate"],
            }
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0
        ]

    def start(self) -> None:
        """Open and start the PortAudio input stream."""
        super().start()
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            channels=self._channels,
            dtype=self._dtype,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the PortAudio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        super().stop()

    def calibrate(self, target_spl: float = 94.0) -> None:
        """Derive and set sensitivity from a live calibrator tone.

        Starts the stream, runs :func:`~slm.calibration.calibrate_sensitivity`
        with stability detection, then stops the stream and stores the result.
        """
        from slm.calibration import calibrate_sensitivity

        self.set_sensitivity(1.0, unit="V")
        self.start()
        try:
            sens = calibrate_sensitivity(self, stability_window=10)
        finally:
            self.stop()
        self.set_sensitivity(sens, unit="V")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self._overruns += 1
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            self._overruns += 1
