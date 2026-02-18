from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf

from slm.controller import Controller


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


    def __init__(self, filename: str | Path, blocksize: int, overlap: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._sf = None
        self.open(filename, blocksize=blocksize, overlap=overlap)

    def open(self, filename: str | Path, *, blocksize: int, overlap: int = 0):
        if self._sf and not self.done:
            raise RuntimeError("File has not been finished.")
        self._done = False

        self._filename = filename
        if not isinstance(filename, str):
            filename = str(filename)

        self._blocksize = blocksize
        self._overlap = overlap
        self._filename = filename
        self._sf = sf.SoundFile(filename)
        self._stream = self._sf.blocks(blocksize=self._blocksize, overlap=self._overlap,
                                       fill_value=0.0, always_2d=True)

    def read_block(self) -> tuple[np.ndarray, int]:
        try:
            return next(self._stream), next(self._counter)
        except StopIteration as e:
            self._done = True
            raise e

    def calibrate(self, target_spl=94.0):
        raise NotImplementedError()

    def stop(self):
        self._done = True
        self._sf.close()
