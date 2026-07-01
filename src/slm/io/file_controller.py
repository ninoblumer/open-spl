import time
import warnings
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf

from slm.io.controller import Controller
from slm.io.realtime_controller import DEFAULT_BLOCKSIZE


class FileController(Controller):
    blocksize: int = property(lambda self: self._blocksize)
    samplerate: int = property(lambda self: self._sf.samplerate)
    sensitivity: float = property(lambda self: self._sensitivity)
    done: bool = property(lambda self: self._done)

     # fields
    _blocksize: int
    _overlap: int
    _sensitivity: float = 1.0
    _sf: sf.SoundFile | None
    _filename: Path | str
    _stream: Generator[np.ndarray, None, None]
    _done: bool
    _overruns: int


    def __init__(self, filename: str | Path, blocksize: int = DEFAULT_BLOCKSIZE, overlap: int = 0,
                 realtime: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._sf = None
        self._realtime = realtime
        self._next_block_time: float | None = None
        self._overruns: int = 0
        self.open(filename, blocksize=blocksize, overlap=overlap)

    def open(self, filename: str | Path, *, blocksize: int, overlap: int = 0):
        if self._sf and not self.done:
            raise RuntimeError("File has not been finished.")
        self._done = False

        if not isinstance(filename, str):
            filename = str(filename)

        self._blocksize = blocksize
        self._overlap = overlap
        self._filename = filename
        self._sf = sf.SoundFile(filename)
        if self._sf.channels > 1:
            warnings.warn(
                f"Audio file '{Path(filename).name}' has {self._sf.channels} channels; "
                "only mono is supported. Only channel 0 will be analysed.",
                UserWarning,
                stacklevel=2,
            )
        self._multichannel = self._sf.channels > 1
        self._stream = self._sf.blocks(blocksize=self._blocksize, overlap=self._overlap,
                                       fill_value=0.0, always_2d=True)
        self._next_block_time = None  # reset on (re-)open

    def read_block(self) -> tuple[np.ndarray, int]:
        if self._done:
            raise StopIteration
        if self._realtime:
            now = time.monotonic()
            if self._next_block_time is None:
                self._next_block_time = now
            sleep_for = self._next_block_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                self._overruns += 1
            self._next_block_time += self._blocksize / self._sf.samplerate
        try:
            block = next(self._stream)
            if self._multichannel:
                block = block[:, 0:1]
            return block, next(self._counter)
        except StopIteration:
            self._done = True
            raise

    @property
    def overruns(self) -> int:
        """Number of blocks where processing exceeded the real-time block period."""
        return self._overruns

    def calibrate(self, target_spl=94.0):
        raise NotImplementedError()

    def stop(self):
        self._done = True
        self._sf.close()
