"""In-memory numpy-array controller for driving measurements from Python."""
from __future__ import annotations

import warnings

import numpy as np

from slm.io.controller import Controller
from slm.io.realtime_controller import DEFAULT_BLOCKSIZE


class ArrayController(Controller):
    """Pull-based controller that feeds blocks from an in-memory numpy array.

    A sibling of :class:`~slm.io.file_controller.FileController` for library use:
    instead of streaming a WAV file it chunks a numpy array of samples into
    blocks.  Because there is no WAV header, the sample rate must be supplied
    explicitly.  Only mono is supported; a multichannel array is reduced to its
    first channel with a warning, matching ``FileController``.

    The final partial block is zero-padded to a full ``blocksize`` so the
    engine's block/duration accounting is identical to the file path
    (``FileController`` uses ``fill_value=0.0`` with ``always_2d=True``).

    Parameters
    ----------
    samples:
        Sample data, shape ``(n,)`` or ``(n, 1)`` (mono).  A ``(n, channels)``
        array with more than one channel is reduced to channel 0 with a warning.
    samplerate:
        Sample rate in Hz (must be positive).
    blocksize:
        Samples per block (default :data:`DEFAULT_BLOCKSIZE`).
    """

    blocksize: int = property(lambda self: self._blocksize)
    samplerate: int = property(lambda self: self._samplerate)
    sensitivity: float = property(lambda self: self._sensitivity)
    done: bool = property(lambda self: self._done)

    # fields
    _samples: np.ndarray
    _samplerate: int
    _blocksize: int
    _pos: int
    _done: bool
    _sensitivity: float = 1.0
    _overruns: int

    def __init__(self, samples, samplerate: int, blocksize: int = DEFAULT_BLOCKSIZE,
                 **kwargs):
        super().__init__(**kwargs)
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim == 2:
            if arr.shape[1] > 1:
                warnings.warn(
                    f"ArrayController was given {arr.shape[1]} channels; only mono "
                    "is supported. Only channel 0 will be analysed.",
                    UserWarning,
                    stacklevel=2,
                )
                arr = arr[:, 0:1]
        else:
            raise ValueError(
                f"samples must be 1-D or 2-D, got a {arr.ndim}-D array"
            )
        if samplerate <= 0:
            raise ValueError(f"samplerate must be positive, got {samplerate}")

        self._samples = arr
        self._samplerate = int(samplerate)
        self._blocksize = blocksize
        self._pos = 0
        self._done = False
        self._sensitivity = 1.0
        self._overruns = 0

    def read_block(self) -> tuple[np.ndarray, int]:
        if self._done or self._pos >= self._samples.shape[0]:
            self._done = True
            raise StopIteration
        block = self._samples[self._pos:self._pos + self._blocksize]
        self._pos += self._blocksize
        # Zero-pad the final partial block to a full blocksize (parity with
        # FileController's fill_value=0.0 / always_2d=True).
        if block.shape[0] < self._blocksize:
            pad = np.zeros((self._blocksize - block.shape[0], 1), dtype=block.dtype)
            block = np.concatenate([block, pad], axis=0)
        return block, next(self._counter)

    @property
    def overruns(self) -> int:
        """Always 0 — a pull-based array source never drops blocks."""
        return self._overruns

    def calibrate(self, target_spl=94.0):
        raise NotImplementedError()

    def stop(self):
        self._done = True
